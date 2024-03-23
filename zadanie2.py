import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# generuje różne stopnie swobody dla prowadzonych testów, uznałem, że będą okej od 1 do 30, a potem 35 40 45 do 100.
# w przypadku tego zadania testów będzie po 1000 by dokładniej sprawdzić czy są różnice pomiędzy klasycznym sposobem testowania a metodą PIT
df = [i for i in range(1, 31, 10)] + [i for i in range(35, 101, 5)]
print(df)

# teraz różne ilości danych uznalem, że okej będą 10, 20 i tak do 500

SampleSize = [i for i in range(1, 101, 10)]
print(SampleSize)


# wyniki będę zapisywał w tablicy: [ [],[],[] ] czyli będzie się tutaj zapisywał wynik trzech testów.
results = []
counter = 0
test_range = 200
#dodałem sobie ile ukonczylem testow zebym widzial ile zostalo do konca symulacji mniej wiecej
total_tests = len(SampleSize) * len(df) * test_range
current_tests  = 0



for size in SampleSize:
    num_bins = int(np.sqrt(size))
    for df_x in df:

        print(f"Postęp symulacji: {current_tests/total_tests*test_range:.2f}%")

        kologomorov_counter = 0
        kologomorov_pit_counter = 0
        chi2_counter = 0
        chi2_pit_counter = 0


        for i in range(test_range):
 
            # generowanie danych z rozkladu chi2
            data = np.random.chisquare(df_x, size)
            # zamieniam metodą PIT dane
            data_pit = stats.chi2.cdf(data, df=df_x)
            current_tests += 1

             # Przeprowadzasz test Kołmogorowa
            d, p_ks = stats.kstest(data, 'chi2', args=(df_x,))

            d_pit, p_ks_pit = stats.kstest(data_pit, 'uniform')

            # Przeprowadzasz test chi-kwadrat
            chi2, p_chi2 = stats.chisquare(data)

         
           # Obliczanie częstości dla przekształconych danych
            bins = np.linspace(0, 1, num=num_bins+1)
            counts, _ = np.histogram(data_pit, bins=bins)
            expected_counts = np.full_like(counts, fill_value=len(data_pit)/num_bins)

            # Normalizacja oczekiwanych częstości
            expected_counts = expected_counts / np.sum(expected_counts) * np.sum(counts)

            # Przeprowadzenie testu chi-kwadrat dla przekształconych danych
            chi2_pit, p_chi2_pit = stats.chisquare(counts, expected_counts)

            # print("kologomorow: ", p_ks, " chi2: ", p_chi2)

            if(p_ks <= 0.05):
                kologomorov_counter+=1
            if(p_chi2 <= 0.05):
                chi2_counter+=1
            if(p_ks_pit <= 0.05):
                kologomorov_pit_counter+=1
            if(p_chi2_pit <= 0.05):
                chi2_pit_counter+=1

        results.append([kologomorov_counter/test_range, kologomorov_pit_counter/test_range, chi2_counter/test_range, chi2_pit_counter/test_range])    
            


print(results)