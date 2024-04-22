import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Parametry
means = [0, 1, 2]  # wartości oczekiwane
std_devs = [1, 2, 3]  # odchylenia standardowe
SampleSizes = [i for i in range(10, 101, 10)]  # rozmiary próbek
simulations_number = 100  # liczba symulacji

# Inicjalizacja wyników
results = []

for mean1 in means:
    for mean2 in means:
        if mean1 == mean2:  # pomijamy przypadek, gdy wartości oczekiwane są takie same
            continue
        for std_dev in std_devs:
            for Sample in SampleSizes:
                # Inicjalizacja licznika
                ks_counter = 0

                for _ in range(simulations_number):
                    # Generowanie danych
                    data1 = np.random.normal(loc=mean1, scale=std_dev, size=Sample)
                    data2 = np.random.normal(loc=mean2, scale=std_dev, size=Sample)  # dane z tą samą wartością odchylenia standardowego, ale inną wartością oczekiwaną

                    # Test Kołmogorowa-Smirnowa
                    ks_stat, p_val_ks = stats.ks_2samp(data1, data2)
                    if p_val_ks < 0.05:
                        ks_counter += 1

                # Zapisanie wyników
                results.append([mean1, mean2, std_dev, Sample, ks_counter/simulations_number])

# Wyświetlanie wyników
for result in results:
    print(f"Wartość oczekiwana 1: {result[0]}, Wartość oczekiwana 2: {result[1]}, odchylenie standardowe: {result[2]}, rozmiar próbki: {result[3]}, moc testu Kołmogorowa: {result[4]}")
