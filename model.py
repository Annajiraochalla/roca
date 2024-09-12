from __future__ import annotations

from typing import Generator, List, Optional, Tuple, Union

import math
import numpy as np
import polars as pl
import cubewalkers as cw
import roomyrobot as rr
from scipy import integrate
from cana.boolean_node import BooleanNode

class Model:
    """
    Measures the Robustness of Cellular Automata Rule
    """
    
    def __init__(self, rule: str,
                 noise_range: Tuple = None,
                 noise_interval: float = None,
                 noise_value: int = 0.0,
                 lattice_size: int = 149,
                 test_rate: int = 5,
                 deviations: int = 3,
                 n_walkers: int = 100_000) -> None:
        """
        Parameters
        ----------
            rule : str
                Cellular Automata Rule in hexadecimal representation
                Example: '0x5f005f005f005f005fff5f005fff5f' for GKL Rule
            noise_range: Tuple, optional
                Noise Probability - Usually between (0.0, 1.0). Defaults to None.
            noise_interval : float, optional
                Step Size of the Noise Range. Defaults to None.
            noise_value : int, optional
                Test Cellular Automata Rule at any Noise Level between the range. Defaults to 0.0.
            lattice_size : int, optional
                Lattize Size of the Automata. Defaults to 149.
            test_rate (int, optional): 
                Rate of measurement for time steps based on Lattice Size 
                Time Steps = Lattice Size * Test Rate
                Defaults to 5.
            deviations: int, optional
                Deviations that influences the neighboring cells. Defaults to 3.
            n_walkers : int, optional
                Number of ensemble walkers to simulate, Defaults to 100_000.
        """
        
        self.rule = rule
        self.noise_range = noise_range        
        self.noise_interval = noise_interval
        self.noise_value = noise_value
        self.lattice_size = lattice_size
        self.test_rate = test_rate
        self.deviations = deviations
        self.n_walkers = n_walkers

    def frange(self, start: float, end: float, step: float) -> Generator[float, None, None]:
        """
        Generates a sequence of floats between start (inclusive) and end (inclusive) with a given step size.

        Parameters
        ----------
            start (float): The starting value of the sequence.
            end (float): The end value of the sequence.
            step (float): The step size between consecutive values.

        Yields
            float: The next value in the sequence.
        """
        while start <= end:
            yield start
            start += step

    def get_noise_list(self) -> Optional[List[float]]:
        """
        Generates a list of noise values based on the noise range and interval.

        Returns:
            Optional[List[float]]: A list of rounded noise values if both noise_range and noise_interval are set; otherwise, None.
        """
        
        if self.noise_range and self.noise_interval:
            start, end = self.noise_range
            
            if not (0 <= start <= 1) or not (0 <= end <= 1):
                raise ValueError("Noise range values must be between 0 and 1, inclusive.")
            
            if start > end:
                raise ValueError("Start of the noise range cannot be greater than the end.")
            
            decimal_places = len(str(self.noise_interval).split('.')[1])
            noise_values = [round(i, decimal_places) for i in self.frange(start, end, self.noise_interval)]
            return noise_values
        else:
            return None
        
    def get_robustness(self) -> str:
        """
        Computes the robustness measure of the rule for the given noise range.

        Returns:
            str: A formatted string representing the robustness measure.
        """
        
        noise_levels: Union[List[float], float] = self.get_noise_list() or [self.noise_value]
        
        results_dict = []
        
        for noise in noise_levels:
            
            model = cw.Model(
                rr.rule_construction.template_to_network(rr.rule_construction.hex_to_algebraic(self.rule), self.lattice_size, noise=noise),
                n_time_steps=self.lattice_size * self.test_rate,
                n_walkers=self.n_walkers,
            )
            
            model.initialize_walkers()
            model.simulate_ensemble(maskfunction=cw.update_schemes.synchronous_PBN, T_window=1)
            
            accuracy = rr.metrics.compute_accuracy(model, noise=noise, deviations=self.deviations)
            results_dict.append({"noise": noise, "accuracy": accuracy})
        
        results_df = pl.DataFrame(results_dict)
        x = np.array(results_df['noise'].to_list(), dtype=float)
        y = results_df["accuracy"].to_numpy()
        robustness_measure = integrate.trapezoid(y=y, x=x)
        
        return f"Robustness: {robustness_measure:.4f}"

    
    def get_lut(self, rule: str) -> List[int]:
        """
        Generates the Look-Up Table (LUT) for a given rule.

        Parameters
        ----------
            rule (str): A hexadecimal string representing the rule.

        Returns:
            List[int]: A list of integers representing the LUT derived from the binary representation of the rule.
        """
    
        return [int(x) for x in bin(int(rule, 16))[2:]]
    
    def get_cana_measures(self) -> str:
        """
        Computes the canalization measures of the rule.

        Returns:
            str: A formatted string representing the input symmetry and input redundancy of the rule.
        """
        
        lut = self.get_lut(self.rule)
        
        lut_length = len(lut)
        if lut_length == 0 or (lut_length & (lut_length - 1)) != 0:
            raise ValueError("LUT length must be a power of 2 and greater than 0.")
    
        n_inputs = int(math.log2(lut_length))
        node = BooleanNode(k=n_inputs, outputs=lut)
        
        ks = node.input_symmetry()
        ke = node.effective_connectivity()
        kr = n_inputs - ke
        
        return f"Input Symmetry: {ks:.4f}, Input Redundancy: {kr:.4f}"