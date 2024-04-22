import numpy as np
from scipy.stats import chi2

def simulate_chi_squared(num_classes, num_samples, df):
    # Generowanie danych z rozkładu chi-kwadrat
    data = np.random.chisquare(df, size=(num_classes, num_samples))

    # Obliczanie oczekiwanych częstości na podstawie założonego rozkładu chi-kwadrat
    expected_counts = chi2.pdf(np.arange(num_classes), df)

    # Normalizacja oczekiwanych częstości do sumy 1
    expected_counts /= expected_counts.sum()

    # Przeskalowanie oczekiwanych częstości do liczby próbek
    expected_counts *= num_samples

    # Przekształcenie oczekiwanych częstości do kształtu (num_classes, num_samples)
    expected_counts = np.tile(expected_counts, (num_samples, 1)).T

    # Dodanie małej wartości do oczekiwanych częstości, aby uniknąć dzielenia przez zero
    expected_counts += 1e-10

    # Obliczanie statystyki chi-kwadrat
    chi2_stat = np.sum((data - expected_counts)**2 / expected_counts)

    # Obliczanie wartości p
    p_val = 1 - chi2.cdf(chi2_stat, df=num_classes-1)

    # Sprawdzenie, czy odrzucamy hipotezę zerową
    reject_null = p_val < 0.05

    return reject_null, chi2_stat, p_val

def main():
    num_classes_list = [2, 3, 5, 10]  # Różne liczby klas
    num_samples = 1000  # Liczba danych
    df = 5  # Stopnie swobody generowanego rozkładu chi-kwadrat

    print("Liczba klas | Odsetek odrzuceń hipotezy zerowej | Statystyka chi-kwadrat | Wartość p")
    print("-" * 80)

    for num_classes in num_classes_list:
        # Powtórz symulację wielokrotnie dla każdej liczby klas
        rejection_rates = []
        chi2_stats = []
        p_values = []
        for _ in range(100):
            reject_null, chi2_stat, p_val = simulate_chi_squared(num_classes, num_samples, df)
            rejection_rates.append(reject_null)
            chi2_stats.append(chi2_stat)
            p_values.append(p_val)

        rejection_rate_mean = np.mean(rejection_rates)
        chi2_stat_mean = np.mean(chi2_stats)
        p_val_mean = np.mean(p_values)

        print(f"{num_classes:^12} | {rejection_rate_mean:^40.4f} | {chi2_stat_mean:^24.4f} | {p_val_mean:^9.4f}")

if __name__ == "__main__":
    main()
