import numpy as np
import scipy.stats as stats
from statsmodels.stats.diagnostic import lilliefors

# Parametry
dfs = [i for i in range(1, 11, 1)]   # stopnie swobody
SampleSizes = [i for i in range(10, 101, 10)]  # rozmiary próbek
simulations_number = 100  # liczba symulacji

# Inicjalizacja wyników
results = []

for df in dfs:
    for Sample in SampleSizes:
        # Inicjalizacja liczników
        jb_counter = 0
        sw_counter = 0
        lf_counter = 0

        for _ in range(simulations_number):
            # Generowanie danych
            data = np.random.standard_t(df, size=Sample)

            # Test Jarque-Bera
            jb_stat, p_val_jb = stats.jarque_bera(data)
            if p_val_jb < 0.05:
                jb_counter += 1

            # Test Shapiro-Wilka
            sw_stat, p_val_sw = stats.shapiro(data)
            if p_val_sw < 0.05:
                sw_counter += 1

            # Test Lillieforsa
            lf_stat, p_val_lf = lilliefors(data)
            if p_val_lf < 0.05:
                lf_counter += 1

        # Zapisanie wyników
        results.append([df, Sample, jb_counter/simulations_number, sw_counter/simulations_number, lf_counter/simulations_number])

# Wyświetlanie wyników
for result in results:
    print(f"Stopnie swobody: {result[0]}, rozmiar próbki: {result[1]}, moc testu Jarque-Bera: {result[2]}, moc testu Shapiro-Wilka: {result[3]}, moc testu Lillieforsa: {result[4]}")
