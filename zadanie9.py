# Za pomocą odpowiednich symulacji zbadać odsetek odrzuceń prawdziwej hipotezy głównej (tzn. błąd
# I rodzaju) w teście Kołmogorowa w przypadku weryfikacji zgodności z rozkładem NIG oraz z
# rozkładem gamma o różnych wartościach parametrów. Należy porównać wyniki klasycznego sposobu
# testowania i testowania z wykorzystaniem PIT (probability integral transform). Uzyskane wyniki
# należy przedstawić na odpowiednich wykresach ilustrujących moce testów z uwzględnieniem:
# - liczby danych,
# - parametrów generowanego rozkładu

import numpy as np
import scipy.stats as stats
from scipy.special import gamma
from numpy.random import default_rng

# Parametry
SampleSizes = [i for i in range(10, 101, 10)]  # rozmiary próbek
simulations_number = 100  # liczba symulacji

# w internecie przeczytalem, że takie prametry mogą przyjąć rozkłady gamma oraz NIG, jednak nie mogłem znaleźć informacji jak bardzo parametr wpływa na rozkład więc wykorzystam "małe" wartości parametrów

# Parametry dla rozkładu Gamma
shape_params = [1, 2, 3, 4, 5]  # parametry kształtu
scale_params = [1, 2]  # parametry skali

# Parametry dla rozkładu NIG
mu_params = [0, 0.5, 1]  # parametry mu
alpha_params = [1, 2, 3]  # parametry alpha
beta_params = [0, 1, 2]  # parametry beta
delta_params = [1, 2, 3]  # parametry delta

# Inicjalizacja wyników
results = []

rng = default_rng()

for Sample in SampleSizes:
    for shape in shape_params:
        for scale in scale_params:
            # Inicjalizacja licznika
            ks_counter = 0
            pit_ks_counter = 0

            for _ in range(simulations_number):
                # Generowanie danych z rozkładu Gamma
                data = np.random.gamma(shape, scale, Sample)

                # Test Kołmogorowa-Smirnowa
                ks_stat, p_val_ks = stats.kstest(data, 'gamma', args=(shape, scale))
                if p_val_ks < 0.05:
                    ks_counter += 1

                # Transformacja PIT
                pit_data = stats.gamma.cdf(data, shape, scale)

                # Test Kołmogorowa-Smirnowa na danych po transformacji PIT
                pit_ks_stat, p_val_pit_ks = stats.kstest(pit_data, 'uniform')
                if p_val_pit_ks < 0.05:
                    pit_ks_counter += 1

            # Zapisanie wyników
            results.append(['Gamma', shape, scale, Sample, ks_counter/simulations_number, pit_ks_counter/simulations_number])

    for mu in mu_params:
        for alpha in alpha_params:
            for beta in beta_params:
                for delta in delta_params:
                    # Inicjalizacja licznika
                    ks_counter = 0
                    pit_ks_counter = 0

                    for _ in range(simulations_number):
                        # Generowanie danych z rozkładu NIG
                        data = mu + delta * (beta + np.sqrt(np.random.gamma(alpha, 2, Sample)) * np.random.normal(0, 1, Sample)) / np.sqrt(np.random.gamma(alpha, 2, Sample))

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
                    results.append(['NIG', mu, alpha, beta, delta, Sample, ks_counter/simulations_number, pit_ks_counter/simulations_number])

# Wyświetlanie wyników
for result in results:
    if result[0] == 'Gamma':
        print(f"Rozkład: {result[0]}, kształt: {result[1]}, skala: {result[2]}, rozmiar próbki: {result[3]}, moc testu Kołmogorowa: {result[4]}, moc testu Kołmogorowa z PIT: {result[5]}")
    else:
        print(f"Rozkład: {result[0]}, mu: {result[1]}, alpha: {result[2]}, beta: {result[3]}, delta: {result[4]}, rozmiar próbki: {result[5]}, moc testu Kołmogorowa: {result[6]}, moc testu Kołmogorowa z PIT: {result[7]}")
