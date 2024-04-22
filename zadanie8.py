# wybrałem że będziemy generować dane z rozkładu chi2 bo wcześniej już wiele razy powtarzał się rozkład tstudenta :)
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Parametry
dfs = [i for i in range(1, 11, 1)]  # stopnie swobody
SampleSizes = [i for i in range(10, 101, 10)]  # rozmiary próbek
simulations_number = 100  # liczba symulacji

# Inicjalizacja wyników
results = []

for df in dfs:
    for Sample in SampleSizes:
        # Inicjalizacja liczników
        ks_counter = 0
        pit_ks_counter = 0

        for _ in range(simulations_number):
            # Generowanie danych
            data = np.random.chisquare(df, size=Sample)

            # Standaryzacja danych
            data = (data - np.mean(data)) / np.std(data)

            # Test Kołmogorowa-Smirnowa
            ks_stat, p_val_ks = stats.kstest(data, 'norm')
            if p_val_ks < 0.05:
                ks_counter += 1

            # Transformacja PIT
            pit_data = stats.norm.cdf(data)

            # Test Kołmogorowa-Smirnowa na danych po transformacji PIT
            pit_ks_stat, p_val_pit_ks = stats.kstest(pit_data, 'uniform')
            if p_val_pit_ks < 0.05:
                pit_ks_counter += 1

        # Zapisanie wyników
        results.append([df, Sample, ks_counter/simulations_number, pit_ks_counter/simulations_number])

# Wyświetlanie wyników
for result in results:
    print(f"Stopnie swobody: {result[0]}, rozmiar próbki: {result[1]}, moc testu Kołmogorowa: {result[2]}, moc testu Kołmogorowa z PIT: {result[3]}")
