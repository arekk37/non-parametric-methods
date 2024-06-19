import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt

def simulate_chi_squared(num_classes, num_samples, df):
    # Generowanie danych z rozkładu chi-kwadrat
    data = np.random.chisquare(df, size=num_samples)

    # Przypisywanie danych do klas na podstawie ich wartości
    class_labels = np.floor(data / (2*df) * num_classes).astype(int)
    class_labels[class_labels >= num_classes] = num_classes - 1  # Przypisanie największych wartości do ostatniej klasy

    # Obliczanie liczby danych w każdej klasie
    observed_counts = np.bincount(class_labels, minlength=num_classes)

    # Obliczanie oczekiwanych częstości na podstawie założonego rozkładu chi-kwadrat
    expected_probs = chi2.cdf((np.arange(1, num_classes+1) / num_classes) * 2*df, df) - chi2.cdf((np.arange(0, num_classes) / num_classes) * 2*df, df)

    # Przeskalowanie oczekiwanych częstości do liczby próbek
    expected_counts = expected_probs * num_samples

    # Dodanie małej wartości do oczekiwanych częstości, aby uniknąć dzielenia przez zero
    expected_counts += 1e-10

    # Obliczanie statystyki chi-kwadrat
    chi2_stat = np.sum((observed_counts - expected_counts)**2 / expected_counts)

    # Obliczanie wartości p
    p_val = 1 - chi2.cdf(chi2_stat, df=num_classes-1)

    # Sprawdzenie, czy odrzucamy hipotezę zerową
    reject_null = p_val < 0.05

    return reject_null


def main():
    num_classes_list = [2, 3, 5, 10, 20, 30]  # Różne liczby klas
    num_samples = 100  # Liczba danych
    df_list = range(1, 11)  # Stopnie swobody generowanego rozkładu chi-kwadrat

    plt.figure(figsize=(10, 7))

    for df in df_list:
        rejection_rates = []
        for num_classes in num_classes_list:
            # Powtórz symulację wielokrotnie dla każdej liczby klas
            rejections = []
            for _ in range(100):
                reject_null = simulate_chi_squared(num_classes, num_samples, df)
                rejections.append(reject_null)

            # Obliczanie odsetka odrzuceń
            rejection_rate = np.mean(rejections)
            rejection_rates.append(rejection_rate)

        # Dodanie linii do wykresu dla danego stopnia swobody
        plt.plot(num_classes_list, rejection_rates, marker='o', label=f'df = {df}')

    plt.title('Odsetek odrzuceń hipotezy zerowej dla różnych stopni swobody')
    plt.xlabel('Liczba klas')
    plt.ylabel('Odsetek odrzuceń')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
