# ROCA
Measure the Robustness of Cellular Automata Rule

# Citation
If you use `ROCA` in your research, please cite us and check out our paper! 

> A Challa, D Hao, JC Rozum, and LM Rocha, "The Effect of Noise on the Density Classification Task for Various Cellular Automata Rules," in ALIFE 2024: Proceedings of the 2024 Artificial Life Conference, 2024.


# Requirements and Installation
Requires `CuPy` (https://cupy.dev/), which in turn requires the `CUDA Toolkit` (https://developer.nvidia.com/cuda-toolkit) and a compatable GPU (https://developer.nvidia.com/cuda-gpus). See the `CuPy` installation guide (https://docs.cupy.dev/en/stable/install.html) for further information.

After installing these prerequisites, install `ROCA` via
```
pip install git+https://github.com/Annajiraochalla/RoboCA
```
Note that this `pip` command ***WILL NOT*** install the `CuPy` and `CUDA Toolkit` dependencies automatically. These are hardware-specific and must be installed manually.