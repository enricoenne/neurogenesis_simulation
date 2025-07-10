import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage import uniform_filter1d
import scipy.integrate as integrate
import math
from scipy.stats import norm
from tqdm import tqdm
import os
from matplotlib import colormaps
import pandas as pd
import sympy as sp

def clipped(p):
    return max(min(p, 1), 0)

def is_constant_function(f, test_inputs=None):
    if test_inputs is None:
        test_inputs = [np.array([0.0]), np.array([2.0]), np.array([5.0]), np.array([8.0])]

    first_value = f(test_inputs[0])
    
    for x in test_inputs[1:]:
        if not np.allclose(f(x), first_value):
            return False
    return True

def arrayed_f(f):
    if is_constant_function(f):
        constant_value = np.asarray(f(np.array([0.0])))

        if constant_value.size ==1:
            const_scalar = constant_value.item()
        else:
            const_scalar = constant_value
        return lambda x: np.full_like(x, const_scalar, dtype=np.float64)
    else:
        return f


class CellSimulation:
    def __init__(self, n_aP0, n_bP0,
                 a_cycle_length = lambda x: 2.2028571428571415 * x + 7.942857142857142,
                 b_cycle_length = lambda x: 2.2028571428571415 * x + 7.942857142857142,
                 pB_to_N_fc = lambda x: 1,
                 pE_fc = None,
                 pA_to_B_fc = None,
                 perc_A_fc = None,
                 min_per_frame = 60,
                 smoothing_perc = 0.2,
                 layers_dist = None,
                 layers_perc = None,
                 pA_to_N_fc = None,
                 clip_prob = False,
                 clip_eq = 4):
        
        #  input values
        self.n_aP0 = n_aP0
        self.n_bP0 = n_bP0

        self.a_cycle_length = a_cycle_length
        self.b_cycle_length = b_cycle_length

        self.pB_to_N_fc = arrayed_f(pB_to_N_fc)

        self.standard_pE_fc = lambda x: 1/(1 + np.exp(-0.9620962961790901*(x-3.4620613854762183)))
        if pE_fc is None:
            self.pE_fc = self.standard_pE_fc
        else:
            self.pE_fc = arrayed_f(pE_fc)

        # if it's none, we use the apical/basal progenitor ratio constraint
        self.pA_to_B_fc = pA_to_B_fc

        self.standard_perc_A_fc = lambda x: -0.0330568898* x + 0.8658511766
        if perc_A_fc is None:
            self.perc_A_fc = self.standard_perc_A_fc
        else:
            self.perc_A_fc = arrayed_f(perc_A_fc)

        self.min_per_frame = min_per_frame

        self.smoothing_perc = smoothing_perc

        self.layers_dist = layers_dist
        if self.layers_dist is not None:
            self.n_layers = layers_dist.shape[0]
        else:
            self.n_layers = 0

        self.layers_perc = layers_perc

        if pA_to_N_fc is None:
            self.pA_to_N_fc = pA_to_N_fc
        else:
            self.pA_to_N_fc = arrayed_f(pA_to_N_fc)


        # what to do if one probability gets negative
        # do somethin or not
        self.clip_prob = clip_prob
        # if clip_prob == True, which equation to ignore
        self.clip_eq = clip_eq


        # constants / derived values
        self.R = 60 / min_per_frame     # time resolution
        self.D = 8                      # 8 days of total simulation, t = 0 is E11
        self.tot_frames = int(self.D * 24 * self.R)

        # simulation results
        self.days = np.array([range(self.tot_frames)])[0] / 24 / self.R  + 11      # array of every frame in days of simulation
        self.n_cells = np.zeros((4, self.tot_frames))                     # how many cells of each type                               aP, bP, aN, bN
        self.new_cells = np.zeros((4, self.tot_frames))                   # how many cells of each type are born at that frame        aP, bP, aN, bN
        self.n_mitos = np.zeros((2, self.tot_frames))                     # how many mitosis at that time frame                       aP, bP
        self.probs = np.zeros((6, self.tot_frames))                       # probabilities in time                                     pA_to_A, pA_to_B, pA_to_N, pB_to_B, pB_to_N, E (cell cycle exit)
        self.percs = np.zeros((4, self.tot_frames))                       # percentages of progenitors and neurons                    perc_A, perc_B, perc_Na, perc_Nb
        
        self.layers = np.zeros((2, self.n_layers, self.tot_frames))       # first dimension is apical or basal                        each layer in the order defined by layers_dist
        self.layer_spread = np.zeros((self.n_layers, self.tot_frames))    # spread function for the layers
        
        self.test = np.zeros((1, self.tot_frames))



        self.colors = ['#1c9d40', '#c21745', '#23a3ae', '#e46b0f', '#666666']
        self.labels = ['apical P', 'basal P', 'apical N', 'basal N']


        self.layer_colors = ['#ae0000', '#e33e00', '#ffc000', '#0088ab', '#03286b']
        self.layer_labels = ['l6', 'l5', 'l4', 'l23', 'l1']


        self.distr_func = self.distribute_mitosis_gaussian

        

        self.t = 0

        # adding the starting cells
        self.n_cells[0, 0] = self.n_aP0
        c0_a = self.a_cycle_length(0.5/(24 * self.R))*self.R
        self.n_cells[1, 0] = self.n_bP0
        c0_b = self.b_cycle_length(0.5/(24 * self.R))*self.R


        ''' starting cell distribution (mitosis)'''
        # cell birth distributed following a preexisting exponential growth

        ''' apical '''
        if self.n_aP0 != 0:
            bleeding_range = int(3*self.smoothing_perc*c0_a)
            mitoses_a = np.zeros((3, self.tot_frames+bleeding_range))
            # future mitosis at maximum are at the cell_cycle_length + the smoothing distance
            furthest_mitosis_a = min(int(c0_a)+1, self.tot_frames)
            # we need to consider also negative times
            negative_mitosis_a = -bleeding_range

            # I need to add mitosis until the cell cycle (further ones are going to interfere with the ones calculated in simulation)
            for i, t in enumerate(range(negative_mitosis_a, furthest_mitosis_a)):
                mitoses_a[0, i] = t
                mitoses_a[1, i] = 2 ** ((t+0.5) / c0_a)


            # I need to smooth everything
            # I lose some on t<cycle, but then I get it back with left tails of simulation mitosis
            # I add a right tail, but is what is needed later
            for i, t in enumerate(tqdm(mitoses_a[0, bleeding_range:], desc="apical setup")):
                self.distr_func(mitoses_a[2], mitoses_a[1, i], i+0.5, c0_a)

            tot_mitos_a = np.sum(mitoses_a[2, bleeding_range:])
            if tot_mitos_a > 0:
                # normalize at 1
                mitoses_a[2] /= tot_mitos_a
                # normalize at starting aP
                mitoses_a[2] *= self.n_aP0
            
            self.n_mitos[0,:furthest_mitosis_a+bleeding_range] = mitoses_a[2, bleeding_range:2*bleeding_range + furthest_mitosis_a]
            

        ''' basal '''
        if self.n_bP0 != 0:
            bleeding_range = int(3*self.smoothing_perc*c0_b)
            mitoses_b = np.zeros((3, self.tot_frames+bleeding_range))
            # future mitosis at maximum are at the cell_cycle_length + the smoothing distance
            furthest_mitosis_b = min(int(c0_b)+1, self.tot_frames)
            # we need to consider also negative times
            negative_mitosis_b = -bleeding_range

            # I need to add mitosis until the cell cycle (further ones are going to interfere with the ones calculated in simulation)
            for i, t in enumerate(range(negative_mitosis_b, furthest_mitosis_b)):
                mitoses_b[0, i] = t
                mitoses_b[1, i] = 2 ** ((t+0.5) / c0_b)

            # I need to smooth everything
            # I lose some on t<cycle, but then I get it back with left tails of simulation mitosis
            # I add a right tail, but is what is needed later
            for i, t in enumerate(tqdm(mitoses_b[0, bleeding_range:], desc="basal setup")):
                self.distr_func(mitoses_b[2], mitoses_b[1, i], i+0.5, c0_b)

            tot_mitos_b = np.sum(mitoses_b[2, bleeding_range:])
            if tot_mitos_b > 0:
                # normalize at 1
                mitoses_b[2] /= tot_mitos_b
                # normalize at starting aP
                mitoses_b[2] *= self.n_bP0
            
            self.n_mitos[1,:furthest_mitosis_b+bleeding_range] = mitoses_b[2, bleeding_range:2*bleeding_range + furthest_mitosis_b]



    def gaussian(self, x, m, s, k):
        return k * np.exp(-((x-m)/s)**2)
    

    '''probability calculator'''
    def resolve_with_costraints(self, equations, variables, fix_var=None, fix_value=0, remove_eq_idx=None):
        modified_eqs = equations.copy()

        if remove_eq_idx is not None:
            del modified_eqs[remove_eq_idx]
            if fix_var is not None:
                modified_eqs.append(sp.Eq(fix_var, fix_value))
        
        return sp.solve(modified_eqs, variables, dict = True)

    def prob_perc_test(self, t, nA, nB, mA, mB):
        
        output = {}

        perc_A = nA / (nA + nB)
        perc_B = 1 - perc_A

        if self.pA_to_N_fc is None:
            #  cell cycle exit
            E = self.pE_fc(t/(24 * self.R))
        else:
            # if we have pA_to_N function, we fixed that and we don't care about cell cycle exit
            E = self.pA_to_N_fc(t/(24 * self.R))

        # target perc_A
        x = self.perc_A_fc(t /(24 * self.R))

        pA_to_A, pA_to_B, pA_to_N = sp.symbols('pA_to_A pA_to_B pA_to_N')
        pB_to_B, pB_to_N = sp.symbols('pB_to_B pB_to_N')
        
        # probabilities sum to 1
        eq0 = sp.Eq(pA_to_A + pA_to_B + pA_to_N, 1)
        eq1 = sp.Eq(pB_to_B + pB_to_N, 1)

        # cell cycle exit condition
        if self.pA_to_N_fc:
            eq2 = sp.Eq(E, pA_to_N)
        else:
            eq2 = sp.Eq(E, perc_A * pA_to_N + perc_B * pB_to_N)

        if self.pA_to_B_fc is None:
            # target ratio between aP and bP condition
            eq3 = sp.Eq(nA + 2*mA*pA_to_A - mA, x*(nA + nB + mA*(2*(pA_to_A + pA_to_B) - 1) + mB*(2*pB_to_B - 1)))
        else:
            # percentage of proliferative daughters of A going to B
            current_pA_to_B = self.pA_to_B_fc(t/(24 * self.R))
            eq3 = sp.Eq(pA_to_B, current_pA_to_B * (1-pA_to_N))

        # bP behavior
        current_pB_to_N = self.pB_to_N_fc(t/(24 * self.R))
        eq4 = sp.Eq(pB_to_N, current_pB_to_N)

        equations = [eq0, eq1, eq2, eq3, eq4]
        variables = (pA_to_A, pA_to_B, pA_to_N, pB_to_B, pB_to_N)

        solution = sp.solve(equations, variables, dict=True)

        # if any of the values is zero
        if solution:
            result = solution[0]
            for var in variables:
                if result[var] < 0:
                    if self.clip_prob:
                        eq_to_remove = self.clip_eq
                    else:
                        eq_to_remove = None
                    solution = self.resolve_with_costraints(equations, variables, fix_var=var, remove_eq_idx=eq_to_remove)
                    break


        

        # keys are not strings yet
        output = {str(k): float(v) for k, v in solution[0].items()}


        output['E'] = E
        output['target_percA'] = x
        return output
    

    ''' spreading / smoothing functions'''
    # spread the new mitosis to the previos/following frame depending on the mitosis_time relative to the center of the frame
    def distribute_mitosis(self, n_mitos_row, new_cells, mitosis_time, cycle_length):

        idx_main = int(mitosis_time)
        # the planned mitosis can be on the first or second half of the frame
        offset = mitosis_time - idx_main - 0.5
        # if it's in the first half, we need to add correction to the previous frame
        # if it's in the second half, we add it to the following frame

        # if the mitosis time is on the right or on the left of the center of the frame
        correction_t = int(np.sign(offset))
        correction_cells = abs(offset)

        if idx_main < self.tot_frames:
            n_mitos_row[idx_main] += new_cells * (1- correction_cells)
        if 0 <= idx_main + correction_t < self.tot_frames:
            n_mitos_row[idx_main + correction_t] += new_cells * correction_cells

    def distribute_mitosis_gaussian(self, n_mitos_row, new_cells, mitosis_time, cycle_length):
        # the sigma of the distribution is a fraction of the cell cycle length
        sigma = self.smoothing_perc * cycle_length

        # no uncertainty
        if sigma == 0:
            t = int(mitosis_time)
            if 0 <= t < self.tot_frames:
                n_mitos_row[t] += new_cells
            return
        
        t_start = max(0, int(np.floor(mitosis_time - 3*sigma)))
        t_end = min(self.tot_frames, int(np.ceil(mitosis_time + 3*sigma)))

        for t in range(t_start, t_end):
            prob = norm.cdf(t+1, loc=mitosis_time, scale=sigma) - norm.cdf(t, loc=mitosis_time, scale=sigma)
            n_mitos_row[t] += new_cells * prob

    
    '''simulation computation'''
    def step(self):
        if self.t == self.tot_frames:
            raise IndexError('simulation is over')

        # copying the value of each cell type from the previous time
        if self.t == 0:
            self.n_cells[0, self.t] = self.n_aP0
            self.n_cells[1, self.t] = self.n_bP0
            self.n_cells[2:3, self.t] = 0

        else:
            self.n_cells[:, self.t] = self.n_cells[:, self.t - 1]
        
        # tot amount of mitosis at time t
        tot_mitosis = self.n_mitos[0, self.t] + self.n_mitos[1, self.t]
        # percentage of progenitors that are apical
        perc_A = self.n_cells[0, self.t] / (self.n_cells[0, self.t] + self.n_cells[1, self.t])

        prob_dict = self.prob_perc_test(self.t+0.5, self.n_cells[0, self.t], self.n_cells[1, self.t], self.n_mitos[0, self.t], self.n_mitos[1, self.t])

        # probabilities
        pA_to_A = self.probs[0, self.t] = prob_dict['pA_to_A']
        pA_to_B = self.probs[1, self.t] = prob_dict['pA_to_B']
        pA_to_N = self.probs[2, self.t] = prob_dict['pA_to_N']
        pB_to_B = self.probs[3, self.t] = prob_dict['pB_to_B']
        pB_to_N = self.probs[4, self.t] = prob_dict['pB_to_N']



        if tot_mitosis != 0:
            # number of cells = number of current cells + new cells - mitosis

            # apical progenitors         api -> api
            new_aP = self.n_mitos[0, self.t] * 2 * pA_to_A
            self.new_cells[0, self.t] = new_aP
            self.n_cells[0, self.t] += new_aP - self.n_mitos[0, self.t]

            # basal progenitors          bas -> bas                    api -> bas
            new_bP = self.n_mitos[1, self.t] * 2 * pB_to_B + self.n_mitos[0, self.t] * 2 * pA_to_B
            self.new_cells[1, self.t] = new_bP
            self.n_cells[1, self.t] += new_bP - self.n_mitos[1, self.t]

            # apical neurons             api -> neu
            new_aN = self.n_mitos[0, self.t] * 2 * pA_to_N
            self.new_cells[2, self.t] = new_aN
            self.n_cells[2, self.t] += new_aN

            # basal neurons              bas -> neu
            new_bN = self.n_mitos[1, self.t] * 2 * pB_to_N
            self.new_cells[3, self.t] = new_bN
            self.n_cells[3, self.t] += new_bN


            # layers calculations
            if self.layers_dist is not None:
                spreading = np.zeros(self.n_layers)
                for i in range(self.n_layers):
                    m = self.layers_dist.iloc[i]['m']
                    s = self.layers_dist.iloc[i]['s']
                    k = self.layers_dist.iloc[i]['k']

                    spreading[i] = self.gaussian(self.t / (24 * self.R), m, s, k)
                
                spreading = spreading / np.sum(spreading)
                self.layer_spread[:, self.t] = spreading
            
                self.layers[0,:,self.t] = new_aN * spreading
                self.layers[1,:,self.t] = new_bN * spreading


            ''' mitosis are spread to the earlier/later
            this doesn't depend on the frame resolution'''
            # the cells are considered doing mitosis in the middle of the frame
            mitosis_time_a = self.t + 0.5 + self.a_cycle_length((self.t + 0.5)/(24 * self.R))*self.R
            self.distr_func(self.n_mitos[0], new_aP, mitosis_time_a, self.a_cycle_length((self.t + 0.5)/(24 * self.R))*self.R)

            mitosis_time_b = self.t + 0.5 + self.b_cycle_length((self.t + 0.5)/(24 * self.R))*self.R
            self.distr_func(self.n_mitos[1], new_bP, mitosis_time_b, self.b_cycle_length((self.t + 0.5)/(24 * self.R))*self.R)
        
        perc_A = self.n_cells[0, self.t] / (self.n_cells[0, self.t] + self.n_cells[1, self.t])
        self.percs[0, self.t] = perc_A
        self.percs[1, self.t] = 1 - perc_A

        perc_Na = self.n_cells[2, self.t] / (self.n_cells[2, self.t] + self.n_cells[3, self.t])
        self.percs[2, self.t] = perc_Na
        self.percs[3, self.t] = 1 - perc_Na

        # self.test[0, self.t] = self.prob_test(self.t)

        E = self.probs[5, self.t] = self.probs[2, self.t]*self.percs[0, self.t] + self.probs[4, self.t]*self.percs[1, self.t]

        self.t += 1

    def run(self, steps = None):
        if steps == None:
            steps = self.tot_frames - self.t
        
        for i in tqdm(range(self.t, self.t + steps), desc='calculation'):
            self.step()
        


    '''obtain result tables'''
    def get_tables(self):
        return {
            'days': self.days,
            'n_cells': self.n_cells,
            'new_cells': self.new_cells,
            'n_mitos': self.n_mitos,
            'probs': self.probs,
            'layers': self.layers,
            'layer_spread': self.layer_spread
        }

    def get_t(self):
        return self.t
    
    def get_metrics(self):
        # summing neurons from aP and bP
        layers_sum = np.sum(self.layers, axis = 0)

        layers_cumsum = np.cumsum(layers_sum, axis = 1)
        # neurons by layers (absolute)
        layers_n = layers_cumsum[:, -1]
        # layers by layers (percent)
        layers_p = layers_n / np.sum(layers_n)
        
        layers_metrics_n = {}
        layers_metrics_p = {}
        for i in range(self.n_layers):
            layer = self.layers_dist['l'][i]
            layers_metrics_n[layer + '_n'] = layers_n[i].item()
            layers_metrics_p[layer + '_p'] = layers_p[i].item()
        

        neuro_threshold = 0.995
        tot_neurons = self.n_cells[2] + self.n_cells[3]
        final_neurons = tot_neurons[-1]

        # moment when we reach 99.5% of total neurons produced
        end_point = self.days[np.argmax(tot_neurons > final_neurons * neuro_threshold)]


        start_point = self.days[np.argmax(tot_neurons > final_neurons * (1-neuro_threshold))]

        metrics = {
            'peak_aP': self.days[np.argmax(self.n_cells[0])].tolist(),
            'peak_bP': self.days[np.argmax(self.n_cells[1])].tolist(),
            'peak_aN_prod': self.days[np.argmax(self.new_cells[2])].tolist(),
            'peak_bN_prod': self.days[np.argmax(self.new_cells[3])].tolist(),
            'peak_N_prod': self.days[np.argmax(self.new_cells[2] + self.new_cells[3])].tolist(),
            'N_ratio': (self.n_cells[2,-1] / (self.n_cells[2,-1] + self.n_cells[3,-1])).tolist(),            # final ratio of neurons from apical
            'start_point': float(start_point),
            'end_point': float(end_point)
        }

        metrics = metrics | layers_metrics_n | layers_metrics_p
        return metrics

    '''plots'''
    def plot_cells(self, metric = False, ax=None, negative = False, debug=False):
        if ax is None:
            fig, ax = plt.subplots()
            own_fig = True
        else:
            own_fig = False
        
        if negative:
            neg_positions = np.any(self.n_cells < 0, axis=0)
            x_values = self.days
            neg_xs = x_values[neg_positions]
            for x in neg_xs:
                ax.axvline(x, color='red', alpha=0.1)
        
        ax.set_title('cell number')
        ax.axhline(y=0, color='#bbbbbb', linestyle='-')
        for i in range(4):
            ax.plot(self.days, self.n_cells[i,], label=self.labels[i], color=self.colors[i])

        if metric:
            ax.axvline(x=self.get_metrics()['peak_aP'], color = self.colors[0], linestyle=':')
            ax.axvline(x=self.get_metrics()['peak_bP'], color = self.colors[1], linestyle=':')
        
        # ax.axvline(x=self.get_metrics()['end_point'], color = '#ff0000')

        if debug:
            ax.set_xlim([11, 19])
        else:
            ax.set_xlim([11, self.get_metrics()['end_point']])

        ax.legend()
        ax.grid()

        if own_fig:
            plt.tight_layout()
            plt.show()

    def plot_new_cells(self, metric = False, ax=None, negative = True, debug = False):
        if ax is None:
            fig, ax = plt.subplots()
            own_fig = True
        else:
            own_fig = False
        
        
        ax.set_title('cell production')
        ax.axhline(y=0, color='#bbbbbb', linestyle='-')
        ax.plot(self.days, self.new_cells[2,:]+self.new_cells[3,:], label='tot neurogenesis', linestyle = 'dashed', color=self.colors[4])
        for i in range(4):
            ax.plot(self.days, self.new_cells[i,:], label=self.labels[i], color=self.colors[i])

        if metric:
            ax.axvline(x=self.get_metrics()['peak_aN_prod'], color = self.colors[2], linestyle=':')
            ax.axvline(x=self.get_metrics()['peak_bN_prod'], color = self.colors[3], linestyle=':')
            ax.axvline(x=self.get_metrics()['peak_N_prod'], color = self.colors[4], linestyle=':')
        
        ax.legend()
        ax.grid()
        if debug:
            ax.set_xlim([11, 19])
        else:
            ax.set_xlim([11, self.get_metrics()['end_point']])

        if own_fig:
            plt.tight_layout()
            plt.show()

    def plot_prog(self, metric = False, ax=None, negative = True, debug = False):
        if ax is None:
            fig, ax = plt.subplots()
            own_fig = True
        else:
            own_fig = False
        
        tot_prog = self.n_cells[0] + self.n_cells[1]
        perc_A = self.n_cells[0] / tot_prog
        perc_B = self.n_cells[1] / tot_prog


        ax.set_title('progenitor percentage')
        ax.axhline(y=0, color='#bbbbbb', linestyle='-')

        if self.perc_A_fc != self.standard_perc_A_fc:
            print(len(self.perc_A_fc(self.days-11)))
            ax.plot(self.days, self.perc_A_fc(self.days-11), label=' apical prog (assumed)', color=self.colors[0], linestyle='--')
            ax.plot(self.days, 1-self.perc_A_fc(self.days-11), label='basal prog (assumed)', color=self.colors[1], linestyle='--')

        ax.plot(self.days, self.standard_perc_A_fc(self.days-11), label=' apical prog (exp)', color=self.colors[0], linestyle=':')
        ax.plot(self.days, 1-self.standard_perc_A_fc(self.days-11), label='basal prog (exp)', color=self.colors[1], linestyle=':')

        ax.plot(self.days, perc_A, label='apical prog (sim)', color=self.colors[0])
        ax.plot(self.days, perc_B, label='basal prog (sim)', color=self.colors[1])



        ax.set_ylim([0,1])
        ax.legend()
        ax.grid()
        if debug:
            ax.set_xlim([11, 19])
        else:
            ax.set_xlim([11, self.get_metrics()['end_point']])

        if own_fig:
            plt.tight_layout()
            plt.show()

    def plot_probsA(self, metric = False, ax=None, debug = False):
        if ax is None:
            fig, ax = plt.subplots()
            own_fig = True
        else:
            own_fig = False
        
        ax.set_title('apical daughter prob')
        ax.axhline(y=0, color='#bbbbbb', linestyle='-')
        ax.axhline(y=1, color='#bbbbbb', linestyle='-')

        # if assumed cycle exit is already standard, we don't need to plot the assumed
        if self.pE_fc != self.standard_pE_fc:
            # cell cycle exit given as a function in the assumption
            ax.plot(self.days, self.pE_fc(self.days - 11), label="cycle exit (assumed)", linestyle = 'dashed', color=self.colors[4])

        # this is the takahashi standard curve
        ax.plot(self.days, self.standard_pE_fc(self.days - 11), label="cycle exit (Takahashi)", linestyle = ':', color=self.colors[4])

        # cell cycle exit coming from simulation
        ax.plot(self.days, self.probs[5], label="cycle exit (sim)", color='black')


        ax.plot(self.days, self.probs[0], label="prob A to A", color=self.colors[0])
        ax.plot(self.days, self.probs[1], label="prob A to B", color=self.colors[1])
        ax.plot(self.days, self.probs[2], label="prob A to N", color=self.colors[2])
        
        ax.set_ylim([-0.1,1.1])
        ax.legend()
        ax.grid()

        if debug:
            ax.set_xlim([11, 19])
        else:
            ax.set_xlim([11, self.get_metrics()['end_point']])

        if own_fig:
            plt.tight_layout()
            plt.show()

    def plot_probsB(self, metric = False, ax=None, debug = False):
        if ax is None:
            fig, ax = plt.subplots()
            own_fig = True
        else:
            own_fig = False
        
        ax.set_title('basal daughter prob')
        ax.axhline(y=0, color='#bbbbbb', linestyle='-')
        ax.axhline(y=1, color='#bbbbbb', linestyle='-')

        # if assumed cycle exit is already standard, we don't need to plot the assumed
        if self.pE_fc != self.standard_pE_fc:
            # cell cycle exit given as a function in the assumption
            ax.plot(self.days, self.pE_fc(self.days - 11), label="cycle exit (assumed)", linestyle = 'dashed', color=self.colors[4])

        # this is the takahashi standard curve
        ax.plot(self.days, self.standard_pE_fc(self.days - 11), label="cycle exit (Takahashi)", linestyle = ':', color=self.colors[4])

        # cell cycle exit coming from simulation
        ax.plot(self.days, self.probs[5], label="cycle exit (sim)", color='black')


        ax.plot(self.days, self.probs[3], label="prob B to B", color=self.colors[1])
        ax.plot(self.days, self.probs[4], label="prob B to N", color=self.colors[2])

        
        ax.set_ylim([-0.1,1.1])
        ax.legend()
        ax.grid()
        if debug:
            ax.set_xlim([11, 19])
        else:
            ax.set_xlim([11, self.get_metrics()['end_point']])

        if own_fig:
            plt.tight_layout()
            plt.show()
    
    def plot_mitosis_perc(self, metric = False, ax=None, debug = False):
        if ax is None:
            fig, ax = plt.subplots()
            own_fig = True
        else:
            own_fig = False
        
        ax.set_title('mitosis perc')
        ax.axhline(y=0, color='#bbbbbb', linestyle='-')
        ax.axhline(y=1, color='#bbbbbb', linestyle='-')

        ax.plot(self.days, self.n_mitos[0] / self.n_cells[0], label="perc aP in mitosis", color=self.colors[0])
        ax.plot(self.days, self.n_mitos[1] / self.n_cells[1], label="perc bP in mitosis", color=self.colors[1])

        # ax.plot(self.days, self.n_mitos[0], label="aP mitosis")
        # ax.plot(self.days, self.n_cells[0], label="aP cells")

        ax.set_ylim([0,1])
        ax.legend()
        ax.grid()
        if debug:
            ax.set_xlim([11, 19])
        else:
            ax.set_xlim([11, self.get_metrics()['end_point']])

        if own_fig:
            plt.tight_layout()
            plt.show()

    def plot_neur(self, metric = False, ax=None, debug = False):
        if ax is None:
            fig, ax = plt.subplots()
            own_fig = True
        else:
            own_fig = False
        
        tot_neur = self.new_cells[2] + self.new_cells[3]
        perc_nA = self.new_cells[2] / tot_neur
        perc_nB = self.new_cells[3] / tot_neur

        ax.set_title('neurogenesis percentage')
        ax.axhline(y=0, color='#bbbbbb', linestyle='-')
        ax.plot(self.days, perc_nA, label="apical neur", color=self.colors[2])
        ax.plot(self.days, perc_nB, label="basal neur", color=self.colors[3])
        ax.set_ylim([0,1])

        ax.grid()
        if debug:
            ax.set_xlim([11, 19])
        else:
            ax.set_xlim([11, self.get_metrics()['end_point']])

        if own_fig:
            ax.plot(self.days, (self.new_cells[2,:]+self.new_cells[3,:])/(np.max(self.new_cells[2,:]+self.new_cells[3,:])*1.1), label='tot neurogenesis (scaled)', linestyle = 'dashed', color=self.colors[4])
        
        ax.legend()
            
        if own_fig:
            plt.tight_layout()
            plt.show()
        
        

    def plot_layers(self, metric = False, save = False, folder_path = None, show = True, debug = False):

        plt.figure(figsize=(12,6))

        ax_cumsum = plt.subplot(1,2,1)
        ax_sum = plt.subplot(2,2,2)
        ax_spread = plt.subplot(2,2,4)

        axs = [ax_cumsum, ax_sum, ax_spread]
        
        ax_cumsum.set_title('tot neurons')
        ax_sum.set_title('neuron production')
        ax_spread.set_title('spread functions')

        # summing neurons from aP and bP
        layers_sum = np.sum(self.layers, axis = 0)

        layers_cumsum = np.cumsum(layers_sum, axis = 1)

        tot_neurons = np.sum(layers_cumsum[:, -1])

        for l in range(self.n_layers):
            current_layer = self.layer_labels[l]
            current_color = self.layer_colors[l]


            current_expected_perc = self.layers_perc[self.layers_perc['layer'] == current_layer]['percent'].item()
            current_expected_n = current_expected_perc * tot_neurons


            ax_cumsum.plot(self.days, layers_cumsum[l], label=current_layer, color=current_color)
            if self.layers_perc is not None:
                ax_cumsum.axhline(y= current_expected_n, color=current_color, linestyle='--', label = current_layer + ' experimental')

            ax_sum.plot(self.days, layers_sum[l], label=current_layer, color=current_color)

            ax_spread.plot(self.days, self.layer_spread[l], label=current_layer, color=current_color)

        tot_neurogenesis = self.new_cells[2,:]+self.new_cells[3,:]

        ax_spread.plot(self.days, tot_neurogenesis / np.max(tot_neurogenesis), label='neuro', linestyle = 'dashed', color=self.colors[4])

        # for p in range(1,3):
        #     axs[p].axhline(y=0, color='#bbbbbb', linestyle='-')

        #     for l in range(self.n_layers):
        #         axs[p].plot(self.days, self.layers[p-1, l], label=self.layer_labels[l], color=self.layer_colors[l])

        #     axs[p].legend()
        #     axs[p].grid()

        for p in axs:
            p.legend()
            p.grid()
            if debug:
                p.set_xlim([11, 19])
            else:
                p.set_xlim([11, self.get_metrics()['end_point']])

        if save:
            plt.savefig(f'{folder_path}_plot_layers.png')
        
        if show:
            plt.show()

    def plot_test(self):
        plt.plot(self.days, np.sum(self.n_cells[0:2,:], axis = 0))
        # plt.ylim([0,1])
        plt.show()

    def plot(self, metric = False, save = False, folder_path = None, show = True, debug = False):

        fig, axs = plt.subplots(2, 3, figsize=(18, 9))

        self.plot_cells(metric = metric, ax = axs[0, 0], debug = debug)
        self.plot_new_cells(metric = metric, ax = axs[1, 0], debug = debug)

        self.plot_probsA(metric = metric, ax = axs[0, 1], debug = debug)
        self.plot_probsB(metric = metric, ax = axs[1, 1], debug = debug)

        self.plot_prog(metric = metric, ax = axs[0, 2], debug = debug)
        self.plot_neur(metric = metric, ax = axs[1, 2], debug = debug)

        plt.tight_layout()
        if save:
            plt.savefig(f'{folder_path}_plot.png')
        
        if show:
            plt.show()
