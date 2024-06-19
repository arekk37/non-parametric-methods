import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import lilliefors

# Parametry
stds = [i for i in range(1, 11, 1)]
means = [i for i in range(1, 11, 2)]
SampleSizes = [i for i in range(10, 101, 10)]  # rozmiary próbek
simulations_number = 100  # liczba symulacji

# Inicjalizacja wyników
results = []


for mean in means:
    for std_dev1 in stds:
        for std_dev2 in stds:
            if std_dev1 != std_dev2:  # upewniamy się, że odchylenia standardowe są różne
                for Sample in SampleSizes:
                    # Inicjalizacja licznika
                    ks_counter = 0

                    for _ in range(simulations_number):
                        # Generowanie danych
                        data1 = np.random.normal(loc=mean, scale=std_dev1, size=Sample)
                        data2 = np.random.normal(loc=mean, scale=std_dev2, size=Sample)

                        # Test Kołmogorowa-Smirnowa
                        ks_stat, p_val_ks = stats.ks_2samp(data1, data2)
                        if p_val_ks < 0.05:
                            ks_counter += 1

                    # Zapisanie wyników
                    results.append([std_dev1, std_dev2, Sample, ks_counter/simulations_number])

    # Wyświetlanie wyników
for result in results:
    print(f"Odchylenie standardowe 1: {result[0]}, Odchylenie standardowe 2: {result[1]}, rozmiar próbki: {result[2]}, moc testu Kołmogorowa: {result[3]}")