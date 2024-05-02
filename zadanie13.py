import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, pareto, gamma

# Parametry rozkładów
pareto_b = 2.62
gamma_shape = 2
gamma_scale = 2

# Liczebności próbek
sample_sizes = [50, 100, 500, 1000, 5000]

# Przechowaj wyniki
results_pareto = []
results_gamma = []

# Przeprowadź symulacje
for n in sample_sizes:
    rejections_pareto = 0
    rejections_gamma = 0
    simulations = 1000
    for _ in range(simulations):
        # Test dla rozkładu Pareto
        sample_pareto1 = pareto.rvs(pareto_b, size=n)
        sample_pareto2 = pareto.rvs(pareto_b, size=n)
        statistic, pvalue = ks_2samp(sample_pareto1, sample_pareto2)
        if pvalue < 0.05:
            rejections_pareto += 1

        # Test dla rozkładu Gamma
        sample_gamma1 = gamma.rvs(gamma_shape, scale=gamma_scale, size=n)
        sample_gamma2 = gamma.rvs(gamma_shape, scale=gamma_scale, size=n)
        statistic, pvalue = ks_2samp(sample_gamma1, sample_gamma2)
        if pvalue < 0.05:
            rejections_gamma += 1

    results_pareto.append(rejections_pareto / simulations)
    results_gamma.append(rejections_gamma / simulations)

# Wykreśl wyniki
plt.figure(figsize=(10, 6))
plt.plot(sample_sizes, results_pareto, marker='o', label='Pareto')
plt.plot(sample_sizes, results_gamma, marker='o', label='Gamma')
plt.title('Odsetek odrzuceń prawdziwej hipotezy głównej')
plt.xlabel('Liczba danych')
plt.ylabel('Odsetek odrzuceń')
plt.legend()
plt.grid(True)
plt.show()
