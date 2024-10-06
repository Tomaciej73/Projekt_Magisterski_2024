import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Importujemy funkcje z exoplanet_deep_learning.py
from exoplanet_classification.exoplanet_deep_learning import load_data, preprocess_data, impute_missing_values, \
    remove_rare_classes

# Ustawienia stylu wykresów
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Wczytujemy dane
filepath = './exoplanet_classification/exoplanets.csv'
df = load_data(filepath)

# Przetwarzamy dane z modyfikacją, aby nie usuwać kolumny 'Mass'
def preprocess_data_modified(df):
    """
    Przetwarza dane: usuwa niepotrzebne kolumny, analizuje i imputuje brakujące wartości oraz koduje zmienne kategoryczne.
    """
    # Usunięcie kolumn, które nie są użyteczne lub zawierają wiele braków
    df = df.drop(['No.', 'Planet Name', 'Planet Host', 'Num Stars', 'Num Planets'], axis=1)

    # Analiza brakujących wartości
    df, cols_dropped = analyze_missing_values_modified(df)

    # Imputacja brakujących wartości
    df = impute_missing_values(df)

    # Usunięcie klas z zbyt małą liczbą próbek
    target_col = 'Discovery Method'  # Upewnij się, że to jest prawidłowa kolumna celu
    df = remove_rare_classes(df, target_col=target_col, min_samples=10)

    # Sprawdzenie, czy pozostają co najmniej dwie klasy
    if df[target_col].nunique() < 2:
        print("\nZbyt mała liczba klas po usunięciu rzadkich klas. Przerwanie procesu.")
        exit()

    # Kodowanie zmiennych kategorycznych
    categorical_cols = df.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    print("\nZakodowane zmienne kategoryczne:")
    print(list(categorical_cols))

    return df, label_encoders

def analyze_missing_values_modified(df):
    """
    Analizuje brakujące wartości i decyduje, które kolumny usunąć, a które imputować.
    Modyfikacja: Nie usuwa kolumny 'Mass', nawet jeśli ma dużo braków.
    """
    missing_counts = df.isnull().sum()
    missing_percentage = (missing_counts / len(df)) * 100
    missing_df = pd.DataFrame({
        'Column': df.columns,
        'Missing Values': missing_counts,
        'Missing Percentage': missing_percentage
    }).sort_values(by='Missing Percentage', ascending=False)

    print("\n--- Analiza brakujących wartości ---")
    print(missing_df)

    # Usunięcie kolumn z bardzo wysokim procentem braków, np. powyżej 50%, z wyjątkiem 'Mass'
    cols_to_drop = missing_df[
        (missing_df['Missing Percentage'] > 50) & (missing_df['Column'] != 'Mass')
    ]['Column']
    df = df.drop(cols_to_drop, axis=1)
    print(f"\nUsunięto kolumny z >50% braków (bez 'Mass'): {list(cols_to_drop)}")

    return df, cols_to_drop

# Zamiast oryginalnej funkcji, użyjemy zmodyfikowanej wersji
df_processed, label_encoders = preprocess_data_modified(df)

# Reszta kodu pozostaje bez zmian, ponieważ kolumna 'Mass' jest zachowana

# 1. Histogram mas egzoplanet
def plot_mass_histogram(df):
    """
    Tworzy histogram mas egzoplanet.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Mass'], bins=50, kde=True, color='blue')
    plt.xlabel('Masa (w masach Ziemi)')
    plt.ylabel('Liczba egzoplanet')
    plt.title('Histogram mas egzoplanet')
    plt.show()

plot_mass_histogram(df_processed)

# 2. Wykres rozrzutu masy względem okresu orbitalnego
def plot_mass_vs_orbital_period(df):
    """
    Tworzy wykres rozrzutu masy egzoplanet względem okresu orbitalnego.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Orbital Period Days', y='Mass', hue='Discovery Method', alpha=0.7)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Okres orbitalny (dni)')
    plt.ylabel('Masa (w masach Ziemi)')
    plt.title('Masa egzoplanet względem okresu orbitalnego')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

plot_mass_vs_orbital_period(df_processed)

# Pozostałe funkcje wizualizacji również będą działać poprawnie

# 3. Wykres słupkowy metod odkrycia
def plot_discovery_method_counts(df):
    """
    Tworzy wykres słupkowy liczby egzoplanet odkrytych różnymi metodami.
    """
    plt.figure(figsize=(10, 6))
    method_counts = df['Discovery Method'].value_counts()
    sns.barplot(x=method_counts.index, y=method_counts.values, palette='viridis')
    plt.xlabel('Metoda odkrycia')
    plt.ylabel('Liczba egzoplanet')
    plt.title('Liczba egzoplanet odkrytych różnymi metodami')
    plt.xticks(rotation=45)
    plt.show()

plot_discovery_method_counts(df_processed)

# 4. Wykres pudełkowy masy dla każdej metody odkrycia
def plot_mass_boxplot_by_method(df):
    """
    Tworzy wykres pudełkowy masy egzoplanet dla każdej metody odkrycia.
    """
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Discovery Method', y='Mass', palette='Set3')
    plt.yscale('log')
    plt.xlabel('Metoda odkrycia')
    plt.ylabel('Masa (w masach Ziemi)')
    plt.title('Rozkład mas egzoplanet w zależności od metody odkrycia')
    plt.xticks(rotation=45)
    plt.show()

plot_mass_boxplot_by_method(df_processed)

# 5. Macierz korelacji zmiennych numerycznych
def plot_correlation_matrix(df):
    """
    Tworzy macierz korelacji zmiennych numerycznych.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Macierz korelacji')
    plt.show()

plot_correlation_matrix(df_processed)

# 6. Wykres rozrzutu temperatury gwiazdy względem masy egzoplanety
def plot_stellar_temp_vs_planet_mass(df):
    """
    Tworzy wykres rozrzutu temperatury efektywnej gwiazdy względem masy egzoplanety.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Stellar Effective Temperature', y='Mass', hue='Discovery Method', alpha=0.7)
    plt.xlabel('Temperatura efektywna gwiazdy (K)')
    plt.ylabel('Masa egzoplanety (w masach Ziemi)')
    plt.title('Temperatura gwiazdy vs masa egzoplanety')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

plot_stellar_temp_vs_planet_mass(df_processed)

# 7. Wykres gęstości orbitalnego półosi głównej
def plot_orbit_semi_major_axis_density(df):
    """
    Tworzy wykres gęstości orbitalnego półosi głównej egzoplanet.
    """
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df['Orbit Semi-Major Axis'], shade=True, color='green')
    plt.xlabel('Orbitalna półoś główna (AU)')
    plt.ylabel('Gęstość')
    plt.title('Gęstość orbitalnej półosi głównej egzoplanet')
    plt.show()

plot_orbit_semi_major_axis_density(df_processed)
