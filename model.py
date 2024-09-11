import math
import numpy as np
import polars as pl
import cubewalkers as cw
import roomyrobot as rr
from scipy import integrate
from cana.boolean_node import BooleanNode

class Model:
    
    '''Measures the Robustness of Cellular Automata Rule'''
    
    def __init__(self, rule: str, # rule in hexadecimal representation
                 noise_range: tuple = None,                                 
                 noise_interval: float = None,
                 noise_value: int = 0.0,  
                 lattice_size: int = 149, 
                 test_rate: int = 5, # for time steps based on lattice size
                 deviations: int = 3,
                 n_walkers: int = 100_000):
        
        self.rule = rule
        self.noise_range = noise_range        
        self.noise_interval = noise_interval
        self.noise_value = noise_value
        self.lattice_size = lattice_size
        self.test_rate = test_rate
        self.deviations = deviations
        self.n_walkers = n_walkers

    def frange(self, start: float, end: float, step: float):
        
        '''Generates a list of floats between the start and end values'''
        
        while start <= end:
            yield start
            start += step

    def get_noise_list(self):
        
        '''Generates the list of noise values'''
        
        if self.noise_range and self.noise_interval:    
            start, end = self.noise_range
            noise_values = [round(i, len(str(self.noise_interval).split('.')[1])) for i in self.frange(start, end, self.noise_interval)]
            return noise_values
        else:
            return None
    
    def get_robustness(self) -> pl.DataFrame:
           
        '''Computes the robustness measure of the rule for given noise range'''
        
        noise_levels = self.get_noise_list() if self.get_noise_list() is not None else self.noise_value
        results_dict = {}
        
        for noise in noise_levels:     
               
            model = cw.Model(
                rr.rule_construction.template_to_network(rr.rule_construction.hex_to_algebraic(self.rule), self.lattice_size, noise = noise),
                n_time_steps = self.lattice_size * self.test_rate,
                n_walkers = self.n_walkers,
            )
            
            model.initialize_walkers()
            model.simulate_ensemble(maskfunction = cw.update_schemes.synchronous_PBN, T_window = 1)
            accuracy = rr.metrics.compute_accuracy(model, noise = noise, deviations = self.deviations)
            results_dict.append({"noise": noise, "accuracy": accuracy})
        
        results_df = pl.DataFrame(results_dict)

        x = np.array(results_df['noise'].to_list(), dtype=float) 
        y = results_df["accuracy"].to_numpy()
        robustness_measure = integrate.trapezoid(y=y, x=x)
        
        return f"Robustness: {robustness_measure:.4f}"
    
    def get_lut(rule):        
        
        '''Generates the Look Up Table for the rule'''
    
        return [int(x) for x in bin(int(rule, 16))[2:]]
    
    def get_cana_measures(self):

        '''Computes the canalization measures of the rule'''
        
        lut = self.get_lut(self.rule)
        
        node = BooleanNode(k=int(math.log2(len(lut))), 
                           outputs=lut)
        
        ks = node.input_symmetry()
        ke = node.effective_connectivity()
        kr = 1/ke
    
        return f"Input Symmetry: {ks:.4f}, Input Redundancy: {kr:.4f}"