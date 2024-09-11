def example_rule():
    '''GKL Rule'''
    model = RoboCA(rule = '0x5f005f005f005f005fff5f005fff5f', noise_range = (0.0, 1.0),
                    noise_interval = 0.002, lattice_size = 149, test_rate = 5, deviations = 3, n_walkers = 100_000)
    return model.get_robustness()