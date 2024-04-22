import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math

# generuje różne stopnie swobody dla prowadzonych testów, uznałem, że będą okej od 1 do 30, a potem 35 40 45 do 100.
dfs = [i for i in range(1, 31, 5)] + [i for i in range(35, 101, 10)]
print(dfs, 'stopnie swobody')

# teraz różne ilości danych uznalem, że okej będą 10, 20 i tak do 500

SampleSize = [i for i in range(10, 101, 10)]
print(SampleSize, 'ilosci danych')

# wyniki będę zapisywał w tablicy: [ [],[],[] ] czyli będzie się tutaj zapisywał wynik trzech testów.
results = []


mean = 0  # Średnia
std_dev = 1  # Odchylenie standardowe
simulations_number = 100  # Liczba symulacji jakie przeprowadzam


for Sample in SampleSize:
    for df in dfs:
        # generujemy dane z rozkladu tstudenta dla df stopni swobody i liczby danych size = Sample
        
        # countery sa zeby zliczyc ile razy odrzucamy prawdziwą hipotezę h0
        chi2_counter = 0
        kologomorov_counter = 0
        pit_chi2_counter = 0
        pit_kologomorov_counter = 0

        # ilość przedziałów dyskretyzacji danych , w internecie przeczytalem, że powinien on zależeć od liczby próbek i okej powinno być 1 przedział dla 5 próbek, więc tak będę robił
        bins = math.floor(Sample / 5)

        for i in range(simulations_number): # przeprowadzamy dla kazdych wygenerowanych danych symulacje 100 razy
            data = np.random.standard_t(df, size=Sample)

            # zamieniamy dane na jednostajny rozklad do uzycia w metodzie PIT
            pit_data = stats.t.cdf(data, df)
            
            # Dyskretyzacja danych dla testu chi2 poniewaz wymaga on tego
            bin_counts, bin_edges = np.histogram(pit_data, bins=bins)

            

            # Oczekiwane liczności
            expected_counts = np.full(bins, Sample / bins)
            chi2_stat, p_val_chi2 = stats.chisquare(data)
            ks_stat, p_val_ks = stats.kstest(data, 't', args=(df,))
            pit_ks_stat, pit_p_val_ks = stats.kstest(pit_data, 'uniform')
                # Test chi-kwadrat
            pit_chi2_stat, pit_p_val_chi2 = stats.chisquare(bin_counts, expected_counts)
            if(p_val_chi2 <= 0.05):
                chi2_counter = chi2_counter + 1
            if(p_val_ks <= 0.05):
                kologomorov_counter = kologomorov_counter + 1
            if(pit_p_val_ks <= 0.05):
                pit_kologomorov_counter = pit_kologomorov_counter + 1
            if(pit_p_val_chi2 <= 0.05):
                pit_chi2_counter = pit_chi2_counter + 1

        results.append(["sample size:", Sample, "df:", df, chi2_counter/100, pit_chi2_counter/100, kologomorov_counter/100, pit_kologomorov_counter/100])

print(results)