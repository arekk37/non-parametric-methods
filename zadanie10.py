import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Parametry
dfs = [i for i in range(1, 11, 1)] # stopnie swobody
SampleSizes = [i for i in range(10, 101, 10)]  # rozmiary próbek
num_bins = [i for i in range(5, 51, 5)]  # liczba klas
simulations_number = 100  # liczba symulacji

# Inicjalizacja wyników
results = []

for df in dfs:
    for Sample in SampleSizes:
        for bins in num_bins:
            # Inicjalizacja licznika
            chi2_counter = 0

            for _ in range(simulations_number):
                # Generowanie danych
                data = np.random.standard_t(df, size=Sample)

                # Standaryzacja danych
                data = (data - np.mean(data)) / np.std(data)

                # Test chi-kwadrat
                observed_values, _ = np.histogram(data, bins=bins)
                expected_values = np.full(bins, Sample / bins)
                chi2_stat, p_val_chi2 = stats.chisquare(observed_values, expected_values)
                if p_val_chi2 < 0.05:
                    chi2_counter += 1

            # Zapisanie wyników
            results.append([df, Sample, bins, chi2_counter/simulations_number])

# Wyświetlanie wyników
for result in results:
    print(f"Stopnie swobody: {result[0]}, rozmiar próbki: {result[1]}, liczba klas: {result[2]}, odsetek odrzuceń prawdziwej hipotezy głównej: {result[3]}")
