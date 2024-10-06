import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from scipy import stats
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping

import warnings

warnings.filterwarnings('ignore')

# Ustawienia stylu wykresów
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


def load_data(url):
    """
    Wczytuje dane z URL.
    """
    # Ustawienie low_memory=False, aby uniknąć DtypeWarning
    df = pd.read_csv(url, low_memory=False)
    print(f"Liczba rekordów: {df.shape[0]}")
    print(f"Liczba kolumn: {df.shape[1]}")
    print("\n--- Pierwsze 5 rekordów ---")
    print(df.head())
    return df


def perform_eda(df):
    """
    Wykonuje Eksploracyjną Analizę Danych (EDA).
    """
    print("\n--- Informacje o danych ---")
    print(df.info())

    print("\n--- Statystyki opisowe ---")
    print(df.describe())

    print("\n--- Liczba brakujących wartości ---")
    print(df.isnull().sum())

    # Sprawdzenie, czy kolumna 'pl_bmasse' (masa planety w masach Ziemi) istnieje
    if 'pl_bmasse' in df.columns:
        # Rozkład masy egzoplanet
        plt.figure(figsize=(10, 6))
        sns.histplot(df['pl_bmasse'], bins=30, kde=True, color='skyblue')
        plt.title('Rozkład masy egzoplanet')
        plt.xlabel('Masa [Earth Mass]')
        plt.ylabel('Liczba')
        plt.show()
    else:
        print("Kolumna 'pl_bmasse' nie istnieje w danych.")

    # Zależność masy od okresu orbitalnego
    if 'pl_orbper' in df.columns and 'pl_bmasse' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='pl_orbper', y='pl_bmasse', hue='discoverymethod', palette='viridis', alpha=0.6)
        plt.title('Masa egzoplanet a okres orbitalny')
        plt.xlabel('Okres orbitalny [dni]')
        plt.ylabel('Masa [Earth Mass]')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()
    else:
        print("Kolumny 'pl_orbper' lub 'pl_bmasse' nie istnieją w danych.")

    # Heatmapa korelacji tylko dla zmiennych numerycznych
    #plt.figure(figsize=(12, 10))
    #numeric_cols = df.select_dtypes(include=[np.number]).columns
    #corr_matrix = df[numeric_cols].corr()
    #sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f")
    #plt.title('Heatmapa korelacji')
    #plt.show()


def analyze_missing_values(df):
    """
    Analizuje brakujące wartości i decyduje, które kolumny usunąć, a które imputować.
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

    # Usunięcie kolumn z bardzo wysokim procentem braków, np. powyżej 80%
    cols_to_drop = missing_df[missing_df['Missing Percentage'] > 80]['Column']
    df = df.drop(cols_to_drop, axis=1)
    print(f"\nUsunięto kolumny z >80% braków: {list(cols_to_drop)}")

    return df, cols_to_drop


def impute_missing_values(df):
    """
    Imputuje brakujące wartości w danych.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Imputacja dla zmiennych numerycznych za pomocą mediany
    num_imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

    # Imputacja dla zmiennych kategorycznych za pomocą najczęściej występującej wartości
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    print("\nImputacja brakujących wartości zakończona.")
    return df


def remove_rare_classes(df, target_col, min_samples=10):
    """
    Usuwa klasy, które mają mniej niż min_samples próbek.
    """
    class_counts = df[target_col].value_counts()
    classes_to_keep = class_counts[class_counts >= min_samples].index
    removed_classes = class_counts[class_counts < min_samples].index

    if len(removed_classes) > 0:
        print(f"\nUsunięto klasy z mniej niż {min_samples} próbkami: {list(removed_classes)}")
        df = df[df[target_col].isin(classes_to_keep)]
    else:
        print("\nWszystkie klasy mają wystarczającą liczbę próbek.")

    return df


def preprocess_data(df):
    """
    Przetwarza dane: usuwa niepotrzebne kolumny, analizuje i imputuje brakujące wartości oraz koduje zmienne kategoryczne.
    """
    # Usunięcie kolumn, które nie są użyteczne lub zawierają wiele braków
    columns_to_drop = ['pl_name', 'hostname', 'default_flag']  # Możesz dodać więcej kolumn do usunięcia
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Analiza brakujących wartości
    df, cols_dropped = analyze_missing_values(df)

    # Imputacja brakujących wartości
    df = impute_missing_values(df)

    # Usunięcie klas z zbyt małą liczbą próbek
    target_col = 'discoverymethod'  # Upewnij się, że to jest prawidłowa kolumna celu
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


def remove_outliers(df, threshold=3):
    """
    Usuwa wartości odstające na podstawie z-score dla zmiennych numerycznych.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Zmiana: Obliczamy z-score dla każdej kolumny osobno i usuwamy tylko te wartości, które są odstające w wielu kolumnach
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    # Ustawmy próg na 4 i wymagajmy, aby z-score był powyżej progu w co najmniej 2 kolumnach
    threshold = 4
    outliers = (z_scores > threshold).sum(axis=1) >= 2
    df_clean = df[~outliers]
    print(f"\nUsunięto {outliers.sum()} wartości odstających.")
    return df_clean


def scale_features(X_train, X_test):
    """
    Standaryzuje cechy za pomocą StandardScaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def apply_pca(X_train, X_test, variance_ratio=0.95):
    """
    Redukuje wymiarowość za pomocą PCA, zachowując określony procent wariancji.
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=variance_ratio, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f"\nLiczba komponentów PCA: {pca.n_components_}")
    return X_train_pca, X_test_pca, pca


def build_cnn_model(input_shape, num_classes):
    """
    Buduje model Convolutional Neural Network (CNN).
    """
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_lstm_model(input_shape, num_classes):
    """
    Buduje model Long Short-Term Memory (LSTM).
    """
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.5),
        LSTM(128),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_cnn_lstm_model(input_shape, num_classes):
    """
    Buduje hybrydowy model CNN-LSTM.
    """
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        LSTM(128, return_sequences=True),
        Dropout(0.5),
        LSTM(128),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def evaluate_deep_learning_model(model, X_test, y_test, model_name):
    """
    Ewaluacja modelu głębokiego uczenia: raport klasyfikacji, AUC-ROC, macierz pomyłek, krzywa ROC.
    """
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Raport klasyfikacji
    print(f"--- {model_name} Classification Report ---")
    print(classification_report(y_true, y_pred))

    # AUC-ROC dla każdej klasy
    if y_test.shape[1] > 1:
        fpr = {}
        tpr = {}
        roc_auc = {}
        num_classes = y_test.shape[1]
        for i in range(num_classes):
            if len(np.unique(y_test[:, i])) < 2:
                print(f"Klasa {i} nie jest reprezentowana w zestawie testowym. Pomijanie obliczania AUC-ROC dla tej klasy.")
                continue
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_prob[:, i])
            roc_auc[i] = roc_auc_score(y_test[:, i], y_pred_prob[:, i])

        # Średnia krzywa ROC
        all_fpr = np.unique(np.concatenate([fpr[i] for i in fpr.keys()]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in fpr.keys():
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= len(fpr.keys())
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        try:
            roc_auc["macro"] = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
            print(f"AUC-ROC (macro-average): {roc_auc['macro']:.4f}\n")
        except ValueError as e:
            print(f"AUC-ROC nie może zostać obliczone: {e}\n")

        plt.figure(figsize=(8, 6))
        plt.plot(fpr["macro"], tpr["macro"],
                 label=f'Krzywa makro-średnia ROC (AUC = {roc_auc["macro"]:.2f})',
                 color='navy', linestyle=':', linewidth=4)

        for i in fpr.keys():
            plt.plot(fpr[i], tpr[i], lw=2, label=f'Krzywa ROC klasy {i} (AUC = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('Współczynnik wyników fałszywie pozytywnych')
        plt.ylabel('Współczynnik wyników prawdziwie pozytywnych')
        plt.title(f'Krzywa ROC - {model_name}')
        plt.legend(loc='lower right')
        plt.show()
    else:
        print("Krzywa ROC jest dostępna tylko dla problemów binarnych lub wieloklasowych.")

    # Macierz pomyłek
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Macierz pomyłek - {model_name}')
    plt.xlabel('Przewidywane')
    plt.ylabel('Rzeczywiste')
    plt.show()

    # Przykładowe predykcje
    print(f"Przykładowe predykcje - {model_name}:")
    sample_size = min(10, len(y_true))
    sample_pred = y_pred[:sample_size]
    sample_true = y_true[:sample_size]
    predictions_df = pd.DataFrame({
        "Przewidywany wynik": sample_pred,
        "Rzeczywisty wynik": sample_true
    })
    print(predictions_df.to_markdown())


def evaluate_traditional_model(model, X_test, y_test, model_name, target_classes):
    """
    Ewaluacja tradycyjnych modeli: raport klasyfikacji, AUC-ROC, macierz pomyłek, krzywa ROC.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    y_true = np.argmax(y_test, axis=1)

    # Raport klasyfikacji
    print(f"--- {model_name} Classification Report ---")
    print(classification_report(y_true, y_pred))

    # AUC-ROC
    if y_test.shape[1] > 1:
        fpr = {}
        tpr = {}
        roc_auc = {}
        num_classes = y_test.shape[1]
        for i in range(num_classes):
            if len(np.unique(y_test[:, i])) < 2:
                print(f"Klasa {i} nie jest reprezentowana w zestawie testowym. Pomijanie obliczania AUC-ROC dla tej klasy.")
                continue
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_proba[:, i])
            roc_auc[i] = roc_auc_score(y_test[:, i], y_pred_proba[:, i])

        # Średnia krzywa ROC
        all_fpr = np.unique(np.concatenate([fpr[i] for i in fpr.keys()]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in fpr.keys():
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= len(fpr.keys())
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        try:
            roc_auc["macro"] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            print(f"AUC-ROC (macro-average): {roc_auc['macro']:.4f}\n")
        except ValueError as e:
            print(f"AUC-ROC nie może zostać obliczone: {e}\n")

        plt.figure(figsize=(8, 6))
        plt.plot(fpr["macro"], tpr["macro"],
                 label=f'Krzywa makro-średnia ROC (AUC = {roc_auc["macro"]:.2f})',
                 color='navy', linestyle=':', linewidth=4)

        for i in fpr.keys():
            plt.plot(fpr[i], tpr[i], lw=2, label=f'Krzywa ROC klasy {i} (AUC = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('Współczynnik wyników fałszywie pozytywnych')
        plt.ylabel('Współczynnik wyników prawdziwie pozytywnych')
        plt.title(f'Krzywa ROC - {model_name}')
        plt.legend(loc='lower right')
        plt.show()
    else:
        print("Krzywa ROC jest dostępna tylko dla problemów binarnych lub wieloklasowych.")

    # Macierz pomyłek
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Macierz pomyłek - {model_name}')
    plt.xlabel('Przewidywane')
    plt.ylabel('Rzeczywiste')
    plt.show()

    # Analiza Ważności Cech
    if isinstance(model, RandomForestClassifier):
        feature_importances = model.feature_importances_
        indices = np.argsort(feature_importances)[::-1]
        feature_names = target_classes

        plt.figure(figsize=(12, 8))
        sns.barplot(x=feature_importances[indices], y=feature_names[indices], palette='viridis')
        plt.title(f'Ważność cech - {model_name}')
        plt.xlabel('Ważność')
        plt.ylabel('Cechy')
        plt.show()

    # Przykładowe Predykcje
    print(f"\nPrzykładowe predykcje - {model_name}:")
    sample_size = min(10, len(y_true))
    sample_pred = y_pred[:sample_size]
    sample_true = y_true[:sample_size]
    predictions_df = pd.DataFrame({
        "Przewidywany wynik": sample_pred,
        "Rzeczywisty wynik": sample_true
    })
    print(predictions_df.to_markdown())


def main():
    # URL do danych z NASA Exoplanet Archive
    data_url = 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+ps&format=csv'

    # 1. Importowanie danych
    df = load_data(data_url)

    # 2. EDA
    perform_eda(df)

    # 3. Preprocessing
    df_processed, label_encoders = preprocess_data(df)

    # 4. Usuwanie wartości odstających (opcjonalnie)
    # Możesz zdecydować się na pominięcie tego kroku lub dostosowanie go
    remove_outliers_flag = False  # Ustaw True, jeśli chcesz usunąć wartości odstające
    if remove_outliers_flag:
        df_clean = remove_outliers(df_processed)
    else:
        df_clean = df_processed.copy()
        print("\nPominięto usuwanie wartości odstających.")

    # 5. Definiowanie zmiennej celu i cech
    target_col = 'discoverymethod'  # Upewnij się, że to jest prawidłowa kolumna celu
    X = df_clean.drop(target_col, axis=1)
    y = df_clean[target_col]

    # 6. Kodowanie zmiennej celu
    ohe = OneHotEncoder(sparse_output=False)
    y_encoded = ohe.fit_transform(y.values.reshape(-1, 1))
    num_classes = y_encoded.shape[1]
    print(f"\nLiczba klas w zmiennej celu: {num_classes}")

    # 7. Sprawdzenie rozkładu klas przed podziałem
    class_distribution = np.sum(y_encoded, axis=0)
    print("\nRozkład klas przed podziałem:")
    for idx, count in enumerate(class_distribution):
        print(f"Klasa {idx}: {int(count)} próbki")

    # 8. Podział na zestawy treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
    )

    # Sprawdzenie rozkładu klas po podziale
    train_class_distribution = np.sum(y_train, axis=0)
    test_class_distribution = np.sum(y_test, axis=0)
    print("\nRozkład klas w zestawie treningowym:")
    for idx, count in enumerate(train_class_distribution):
        print(f"Klasa {idx}: {int(count)} próbki")
    print("\nRozkład klas w zestawie testowym:")
    for idx, count in enumerate(test_class_distribution):
        print(f"Klasa {idx}: {int(count)} próbki")

    # 9. Skalowanie cech
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # 10. Redukcja wymiarowości (opcjonalnie)
    apply_pca_flag = False  # Ustaw True, jeśli chcesz zastosować PCA
    if apply_pca_flag:
        X_train_pca, X_test_pca, pca = apply_pca(X_train_scaled, X_test_scaled, variance_ratio=0.95)
    else:
        X_train_pca, X_test_pca = X_train_scaled, X_test_scaled

    # 11. Trenowanie i Ewaluacja Modeli Głębokiego Uczenia
    # Przygotowanie danych dla modeli CNN, LSTM, CNN-LSTM
    X_train_cnn = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
    X_test_cnn = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

    X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
    X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

    X_train_cnn_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
    X_test_cnn_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

    # Budowa modeli
    cnn_model = build_cnn_model(input_shape=(X_train_cnn.shape[1], X_train_cnn.shape[2]), num_classes=num_classes)
    lstm_model = build_lstm_model(input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), num_classes=num_classes)
    cnn_lstm_model = build_cnn_lstm_model(input_shape=(X_train_cnn_lstm.shape[1], X_train_cnn_lstm.shape[2]),
                                          num_classes=num_classes)

    # Podsumowanie modeli
    print("\n--- Model CNN ---")
    cnn_model.summary()

    print("\n--- Model LSTM ---")
    lstm_model.summary()

    print("\n--- Model CNN-LSTM ---")
    cnn_lstm_model.summary()

    # Definicja Early Stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Trenowanie modeli
    print("\nTrenowanie modelu CNN...")
    history_cnn = cnn_model.fit(
        X_train_cnn, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    print("\nTrenowanie modelu LSTM...")
    history_lstm = lstm_model.fit(
        X_train_lstm, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    print("\nTrenowanie modelu CNN-LSTM...")
    history_cnn_lstm = cnn_lstm_model.fit(
        X_train_cnn_lstm, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    # Ewaluacja modeli
    print("\nEwaluacja modelu CNN...")
    evaluate_deep_learning_model(cnn_model, X_test_cnn, y_test, "CNN")

    print("\nEwaluacja modelu LSTM...")
    evaluate_deep_learning_model(lstm_model, X_test_lstm, y_test, "LSTM")

    print("\nEwaluacja modelu CNN-LSTM...")
    evaluate_deep_learning_model(cnn_lstm_model, X_test_cnn_lstm, y_test, "CNN-LSTM")

    # 12. Trenowanie i Ewaluacja Tradycyjnych Modeli
    print("\n--- Trenowanie i Ewaluacja Tradycyjnych Modeli ---")

    # Zastosowanie SMOTE do balansowania danych
    smote = SMOTE(random_state=42)
    X_res_rf, y_res_rf = smote.fit_resample(X_train_scaled, y_train)
    print(f"\nRozmiar zbioru po SMOTE: {X_res_rf.shape}")

    # Trenowanie Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_res_rf, np.argmax(y_res_rf, axis=1))

    # Ewaluacja Random Forest
    evaluate_traditional_model(
        rf_model,
        X_test_scaled,
        y_test,
        model_name="Random Forest",
        target_classes=X.columns
    )

    # Automatyczna Krótka Analiza Wyników
    print("\n--- Automatyczna Krótka Analiza Wyników ---")

    def generate_summary():
        # Sprawdzenie rozkładu klas po podziale
        train_class_counts = np.sum(y_train, axis=0)
        test_class_counts = np.sum(y_test, axis=0)
        rf_class_counts = np.sum(y_test, axis=0)

        summary = f"""
### **Podsumowanie Wyników**

**Zbiór Danych:**
- Początkowa liczba rekordów: {df.shape[0]}
- Po usunięciu kolumn z >80% braków: {df_processed.shape[0]}
- Po imputacji braków i usunięciu rzadkich klas: {df_clean.shape[0]} rekordów

**Rozkład Klas:**
"""
        for idx, count in enumerate(class_distribution):
            summary += f"- Klasa {idx}: {int(count)} próbki\n"

        # Modele głębokiego uczenia
        cnn_accuracy = classification_report(np.argmax(y_test, axis=1), np.argmax(cnn_model.predict(X_test_cnn), axis=1), output_dict=True)['accuracy']
        lstm_accuracy = classification_report(np.argmax(y_test, axis=1), np.argmax(lstm_model.predict(X_test_lstm), axis=1), output_dict=True)['accuracy']
        cnn_lstm_accuracy = classification_report(np.argmax(y_test, axis=1), np.argmax(cnn_lstm_model.predict(X_test_cnn_lstm), axis=1), output_dict=True)['accuracy']

        summary += f"""
**Modele Głębokiego Uczenia:**
- **CNN:**
  - Dokładność: {round(cnn_accuracy, 2)}
- **LSTM:**
  - Dokładność: {round(lstm_accuracy, 2)}
- **CNN-LSTM:**
  - Dokładność: {round(cnn_lstm_accuracy, 2)}

**Tradycyjny Model: Random Forest:**
"""
        rf_accuracy = classification_report(np.argmax(y_test, axis=1), rf_model.predict(X_test_scaled), output_dict=True)['accuracy']
        summary += f"- Dokładność: {round(rf_accuracy, 2)}\n"

        summary += """
"""

        print(summary)

    generate_summary()


if __name__ == "__main__":
    main()
