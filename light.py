import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from astropy.io import fits
from astropy.table import Table

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

import warnings

warnings.filterwarnings('ignore')

# Ustawienia wykresów
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Krok 1: Wczytanie i Połączenie Danych z Plików FITS
fits_files = [f'{i}.fits' for i in range(1, 4)]

# Inicjalizacja list do przechowywania danych
time_list = []
flux_list = []

for filename in fits_files:
    #print(f"Przetwarzanie pliku: {filename}")

    # Otwieranie pliku FITS
    with fits.open(filename) as hdulist:
        # Sprawdzenie, czy dane są w drugim HDU
        if len(hdulist) > 1 and isinstance(hdulist[1], fits.BinTableHDU):
            data = hdulist[1].data
            time = data['TIME']
            flux = data['PDCSAP_FLUX']

            # Filtracja nanów
            mask = ~np.isnan(time) & ~np.isnan(flux)
            time = time[mask]
            flux = flux[mask]

            time_list.append(time)
            flux_list.append(flux)
        else:
            print(f"Brak danych w pliku {filename}")

# Połączenie wszystkich danych
time_all = np.concatenate(time_list)
flux_all = np.concatenate(flux_list)

# Sortowanie danych po czasie
sorted_indices = np.argsort(time_all)
time_all = time_all[sorted_indices]
flux_all = flux_all[sorted_indices]

# Tworzenie DataFrame
df = pd.DataFrame({'TIME': time_all, 'FLUX': flux_all})

#print("\nPierwsze 5 rekordów:")
#print(df.head())

# Krok 2: Eksploracyjna Analiza Danych (EDA)
# Wizualizacja krzywej blasku
plt.figure(figsize=(12, 6))
plt.plot(df['TIME'], df['FLUX'], color='blue', lw=0.5)
plt.xlabel('Momenty pomiarów w juliańskiej dacie barycentrycznej (BTJD)')
plt.ylabel('Miara jasności obiektu (elektrony/sekundę)')
plt.title('Krzywa blasku obiektu')
plt.show()

# Statystyki opisowe
print("\nStatystyki opisowe FLUX:")
print(df['FLUX'].describe())

# Wykres pudełkowy strumienia
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['FLUX'])
plt.title('Wykres pudełkowy jasności obiektu astronomicznego')
plt.show()

# Krok 3: Przetwarzanie i Przygotowanie Danych
# Usunięcie wartości odstających (3 sigma)
flux_mean = np.mean(df['FLUX'])
flux_std = np.std(df['FLUX'])
threshold = 3
outlier_mask = (df['FLUX'] > flux_mean - threshold * flux_std) & (df['FLUX'] < flux_mean + threshold * flux_std)
df_clean = df[outlier_mask]

print(f"\nUsunięto {len(df) - len(df_clean)} wartości odstających.")

# Normalizacja danych
scaler = MinMaxScaler(feature_range=(0, 1))
flux_scaled = scaler.fit_transform(df_clean['FLUX'].values.reshape(-1, 1))

# Aktualizacja DataFrame
df_clean['FLUX_SCALED'] = flux_scaled

print("\nPierwsze 5 rekordów po normalizacji:")
print(df_clean.head())

# Krok 4: Przygotowanie Danych do Modelowania
# Funkcja do tworzenia sekwencji
def create_sequences(data, sequence_length=100):
    X = []
    y = []

    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])

    return np.array(X), np.array(y)

# Tworzenie sekwencji
sequence_length = 100  # Długość okna czasowego

X, y = create_sequences(df_clean['FLUX_SCALED'].values, sequence_length)

print(f"\nKształt X: {X.shape}")
print(f"Kształt y: {y.shape}")

# Podział na zestawy treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print(f"\nRozmiar zbioru treningowego: {X_train.shape[0]}")
print(f"Rozmiar zbioru testowego: {X_test.shape[0]}")

# Krok 5: Budowa i Trenowanie Modelu LSTM
# Reshape danych do formatu (samples, time_steps, features)
X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Definicja modelu
model = Sequential()
model.add(LSTM(64, input_shape=(X_train_reshaped.shape[1], 1), return_sequences=True))
model.add(LSTM(32))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Definicja Early Stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Trenowanie modelu
history = model.fit(
    X_train_reshaped, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# Krok 6: Ewaluacja Modelu
# Wizualizacja funkcji strat
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Strata treningowa')
plt.plot(history.history['val_loss'], label='Strata walidacyjna')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.title('Funkcja strat modelu')
plt.legend()
plt.show()

# Prognozy na zbiorze testowym
y_pred = model.predict(X_test_reshaped)

# Odwrócenie skalowania
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_inv = scaler.inverse_transform(y_pred)

# Wizualizacja wyników
plt.figure(figsize=(12, 6))
plt.plot(range(len(y_test_inv)), y_test_inv, label='Rzeczywiste')
plt.plot(range(len(y_pred_inv)), y_pred_inv, label='Prognozowane')
plt.xlabel('Próbka')
plt.ylabel('Miara jasności obiektu (elektrony/sekundę)')
plt.title('Porównanie wartości rzeczywistych i prognozowanych')
plt.legend()
plt.show()

# Obliczenie metryk ewaluacji
from sklearn.metrics import mean_squared_error, mean_absolute_error

mse = mean_squared_error(y_test_inv, y_pred_inv)
mae = mean_absolute_error(y_test_inv, y_pred_inv)

print(f"Średni błąd kwadratowy: {mse:.3f}")
print(f"Średni błąd bezwzględny: {mae:.3f}")

# Obliczenie średniego bezwzględnego błędu procentowego (MAPE)
mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100
print(f"Średni bezwzględny błąd procentowy: {mape:.2f}%")
