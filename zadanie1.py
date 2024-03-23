import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# generuje różne stopnie swobody dla prowadzonych testów, uznałem, że będą okej od 1 do 30, a potem 35 40 45 do 100.
df = [i for i in range(1, 31)] + [i for i in range(35, 101, 5)]
print(df)

# teraz różne ilości danych uznalem, że okej będą 10, 20 i tak do 500

SampleSize = [i for i in range(10, 501, 10)]
print(SampleSize)


# wyniki będę zapisywał w tablicy: [ [],[],[] ] czyli będzie się tutaj zapisywał wynik trzech testów.
results = []
counter = 0

# Test będzie wyglądał w następujący sposób: będę generował dane rozkładu studenta (0,1)
# np 10 probek i dla tych danych przeprowadzę po 100 testów każdego rodzaju na tych samych danych żeby obliczyć prawdopodobieństwo odrzucenia h0.
for size in SampleSize:
    for df_x in df:
        results.append([])
        data = np.random.standard_t(df_x, size)
        for i in range(100):
            w, p_shapiro = stats.shapiro(data)
            d, p_ks = stats.kstest((data - np.mean(data)) / np.std(data, ddof=1), 'norm')
            chi2, p_chi2 = stats.chisquare(data)
            print("shapiro:" , p_shapiro ," kologomorow: ", p_ks, " chi2: " , p_chi2)



