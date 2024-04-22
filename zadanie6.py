import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import lilliefors

# Parametry
dfs = [i for i in range(1, 31, 1)] + [i for i in range(35, 101, 10)]  # stopnie swobody
SampleSizes = [i for i in range(10, 101, 10)]  # rozmiary próbek
simulations_number = 100  # liczba symulacji

# Inicjalizacja wyników
results = []

for Sample in SampleSizes:
    for df in dfs:
        # Inicjalizacja liczników
        ks_counter = 0
        lf_counter = 0
        ad_counter = 0

        for _ in range(simulations_number):
            # Generowanie danych z rozkladu t-studenta
            data = np.random.standard_t(df, size=Sample)

            # Standaryzacja danych
            data = (data - np.mean(data)) / np.std(data)

            # Test Kołmogorowa-Smirnowa
            ks_stat, p_val_ks = stats.kstest(data, 'norm')
            if p_val_ks < 0.05:
                ks_counter += 1

            # Test Lillieforsa
            lf_stat, p_val_lf = lilliefors(data)
            if p_val_lf < 0.05:
                lf_counter += 1

            # Test Andersona-Darlinga
            ad_stat, critical_values, significance_levels = stats.anderson(data, dist='norm')
            if ad_stat > critical_values[2]:  # Porównanie statystyki testowej z krytycznymi wartościami dla poziomu istotności 0.05
                ad_counter += 1

        # Zapisanie wyników
        results.append([Sample, df, ks_counter/simulations_number, lf_counter/simulations_number, ad_counter/simulations_number])

# Wyświetlanie wyników
for result in results:
    print(f"Rozmiar próbki: {result[0]}, stopnie swobody: {result[1]}, moc testu Kołmogorowa: {result[2]}, moc testu Lillieforsa: {result[3]}, moc testu Andersona-Darlinga: {result[4]}")
