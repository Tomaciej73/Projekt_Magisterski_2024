import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gradio import Image
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import io
from PIL import Image as PILImage


def load_and_preprocess_data():
    # Wczytanie danych
    df = pd.read_csv('egzoplanety_pl.csv', encoding='utf-8')

    # Usunięcie niepotrzebnych kolumn
    df = df.drop(columns=['Nr', 'Nazwa_Planety', 'Gospodarz_Planety', 'Instytucja_Odkrywajaca',
                          'Stosunek_Metalicznosci_Gwiazdy'], errors='ignore')

    # Lista kolumn do konwersji na typ numeryczny
    cols_to_convert = [
        'Liczba_Gwiazd', 'Liczba_Planet', 'Rok_Odkrycia',
        'Okres_Orbitalny', 'Pol_Os_Orbity', 'Masa', 'Mimosrod',
        'Strumien_Insolacji', 'Temperatura_Rownowagi', 'Efektywna_Temperatura_Gwiazdy',
        'Promien_Gwiazdy', 'Masa_Gwiazdy', 'Metalicznosc_Gwiazdy',
        'Grawitacja_Powierzchniowa_Gwiazdy', 'Odleglosc', 'Wielkosc_Magnitudy'
    ]

    # Konwersja kolumn na typ numeryczny
    for col in cols_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Lista zmiennych docelowych
    target_variables = ['Temperatura_Rownowagi', 'Okres_Orbitalny', 'Masa', 'Mimosrod', 'Strumien_Insolacji',
                        'Metalicznosc_Gwiazdy']

    # Wypełnienie brakujących wartości w cechach (z wyłączeniem zmiennych docelowych)
    features_to_fill = [col for col in cols_to_convert if col not in target_variables]
    for col in features_to_fill:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Label Encoding dla kolumn kategorycznych
    categorical_cols = ['Metoda_Odkrycia', 'Typ_Spektralny']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Skalowanie danych liczbowych
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Wykluczenie kolumny 'Metoda_Odkrycia' z listy kolumn do skalowania
    if 'Metoda_Odkrycia' in numeric_cols:
        numeric_cols.remove('Metoda_Odkrycia')
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df, label_encoders, scaler

def classify_metoda_odkrycia(df, n_estimators):
    # Usunięcie kolumn tekstowych
    df_class = df.drop(columns=['Typ_Spektralny'], errors='ignore')
    X = df_class.drop('Metoda_Odkrycia', axis=1)
    y = df_class['Metoda_Odkrycia']
    # Upewnij się, że wszystkie cechy są numeryczne
    X = X.select_dtypes(include=[np.number])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    clf = RandomForestClassifier(n_estimators=int(n_estimators), random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return f"Dokładność modelu klasyfikacyjnego: {accuracy:.2f}"

def model_okres_orbitalny(df, n_estimators):
    # Funkcja regresji dla Okresu Orbitalnego
    features = ['Pol_Os_Orbity', 'Masa', 'Mimosrod']
    target = 'Okres_Orbitalny'
    df_model = df[features + [target]].dropna().reset_index(drop=True)
    X = df_model[features]
    y = df_model[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    model = RandomForestRegressor(n_estimators=int(n_estimators), random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return f"RMSE: {rmse:.2f}, R²: {r2:.2f}"

def model_masa_planety(df, n_estimators):
    # Funkcja regresji dla Masy Planety
    features = ['Masa_Gwiazdy', 'Promien_Gwiazdy', 'Metalicznosc_Gwiazdy', 'Pol_Os_Orbity', 'Mimosrod']
    target = 'Masa'
    df_model = df[features + [target]].dropna().reset_index(drop=True)
    X = df_model[features]
    y = df_model[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    model = RandomForestRegressor(n_estimators=int(n_estimators), random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return f"RMSE: {rmse:.2f}, R²: {r2:.2f}"

def model_mimosrod(df, n_estimators):
    # Funkcja regresji dla Mimosrodu
    features = ['Okres_Orbitalny', 'Pol_Os_Orbity', 'Masa_Gwiazdy', 'Masa']
    target = 'Mimosrod'
    df_model = df[features + [target]].dropna().reset_index(drop=True)
    X = df_model[features]
    y = df_model[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    model = RandomForestRegressor(n_estimators=int(n_estimators), random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return f"RMSE: {rmse:.2f}, R²: {r2:.2f}"

def model_temperatura_rownowagi(df, n_estimators):
    # Funkcja regresji dla Temperatury Równowagi
    features = ['Strumien_Insolacji', 'Pol_Os_Orbity', 'Typ_Spektralny', 'Efektywna_Temperatura_Gwiazdy']
    target = 'Temperatura_Rownowagi'
    df_model = df[features + [target]].dropna().reset_index(drop=True)
    X = df_model[features]
    y = df_model[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    model = RandomForestRegressor(n_estimators=int(n_estimators), random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return f"RMSE: {rmse:.2f}, R²: {r2:.2f}"

def model_strumien_insolacji(df, n_estimators):
    # Funkcja regresji dla Strumienia Insolacji
    features = ['Typ_Spektralny', 'Efektywna_Temperatura_Gwiazdy', 'Pol_Os_Orbity']
    target = 'Strumien_Insolacji'
    df_model = df[features + [target]].dropna().reset_index(drop=True)
    X = df_model[features]
    y = df_model[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    model = RandomForestRegressor(n_estimators=int(n_estimators), random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return f"RMSE: {rmse:.2f}, R²: {r2:.2f}"

def model_metalicznosc_gwiazdy(df, n_estimators):
    # Funkcja regresji dla Metaliczności Gwiazdy
    features = ['Typ_Spektralny', 'Efektywna_Temperatura_Gwiazdy', 'Masa_Gwiazdy', 'Promien_Gwiazdy', 'Okres_Orbitalny', 'Pol_Os_Orbity']
    target = 'Metalicznosc_Gwiazdy'
    df_model = df[features + [target]].dropna().reset_index(drop=True)
    X = df_model[features]
    y = df_model[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    model = RandomForestRegressor(n_estimators=int(n_estimators), random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return f"RMSE: {rmse:.2f}, R²: {r2:.2f}"

def plot_visualization(plot_type):
    df = pd.read_csv('egzoplanety_pl.csv', encoding='utf-8')
    sns.set_style('whitegrid')
    plt.figure(figsize=(10, 6))

    if plot_type == 'Masa planety vs Odległość od gwiazdy':
        data = df[['Pol_Os_Orbity', 'Masa']].dropna()
        sns.scatterplot(data=data, x='Pol_Os_Orbity', y='Masa')
        plt.xlabel('Pół Oś Orbity (au)')
        plt.ylabel('Masa Planety ($M_{Jup}$)')
        plt.title('Zależność masy planety od odległości od gwiazdy')
    elif plot_type == 'Masa planety vs Masa gwiazdy':
        data = df[['Masa_Gwiazdy', 'Masa']].dropna()
        sns.scatterplot(data=data, x='Masa_Gwiazdy', y='Masa')
        plt.xlabel('Masa Gwiazdy ($M_{Sun}$)')
        plt.ylabel('Masa Planety ($M_{Jup}$)')
        plt.title('Zależność masy planety od masy gwiazdy')
    elif plot_type == 'Masa planety vs Okres orbitalny':
        data = df[['Okres_Orbitalny', 'Masa']].dropna()
        sns.scatterplot(data=data, x='Okres_Orbitalny', y='Masa')
        plt.xlabel('Okres Orbitalny (dni)')
        plt.ylabel('Masa Planety ($M_{Jup}$)')
        plt.title('Zależność masy planety od okresu orbitalnego')
    elif plot_type == 'Okres orbitalny vs Odległość (Prawa Keplera)':
        data = df[['Pol_Os_Orbity', 'Okres_Orbitalny']].dropna()
        sns.scatterplot(data=data, x='Pol_Os_Orbity', y='Okres_Orbitalny')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Pół Oś Orbity (au)')
        plt.ylabel('Okres Orbitalny (dni)')
        plt.title('Zależność okresu orbitalnego od pół osi orbity (skala log-log)')
    elif plot_type == 'Masa planety vs Temperatura gwiazdy':
        data = df[['Efektywna_Temperatura_Gwiazdy', 'Masa']].dropna()
        data['Efektywna_Temperatura_Gwiazdy'] = data['Efektywna_Temperatura_Gwiazdy'] - 273.15  # Konwersja na °C
        sns.scatterplot(data=data, x='Efektywna_Temperatura_Gwiazdy', y='Masa')
        plt.xlabel('Efektywna Temperatura Gwiazdy (°C)')
        plt.ylabel('Masa Planety ($M_{Jup}$)')
        plt.title('Zależność masy planety od temperatury gwiazdy')
    elif plot_type == 'Rozkład temperatury równowagi':
        data = df['Temperatura_Rownowagi'].dropna() - 273.15  # Konwersja na °C
        sns.histplot(data=data, bins=30)
        plt.ylabel('Liczba egzoplanet')
        plt.xlabel('Temperatura Równowagi (°C)')
        plt.title('Rozkład temperatury równowagi na egzoplanetach')
    else:
        return "Nieznany typ wizualizacji."

    plt.tight_layout()
    # Zapisz wykres do bufora w pamięci
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    # Odczytaj obraz z bufora
    img = PILImage.open(buf)
    return img

# Tworzenie interfejsu Gradio
def main():
    df, label_encoders, scaler = load_and_preprocess_data()

    # Definiujemy funkcje dla Gradio
    def classify_model(n_estimators):
        result = classify_metoda_odkrycia(df, n_estimators)
        return result

    def regression_model(target, n_estimators):
        if target == 'Okres_Orbitalny':
            return model_okres_orbitalny(df, n_estimators)
        elif target == 'Masa':
            return model_masa_planety(df, n_estimators)
        elif target == 'Mimosrod':
            return model_mimosrod(df, n_estimators)
        elif target == 'Temperatura_Rownowagi':
            return model_temperatura_rownowagi(df, n_estimators)
        elif target == 'Strumien_Insolacji':
            return model_strumien_insolacji(df, n_estimators)
        elif target == 'Metalicznosc_Gwiazdy':
            return model_metalicznosc_gwiazdy(df, n_estimators)
        else:
            return "Nieznany cel regresji."

    # Tworzenie interfejsu dla wizualizacji
    visualization_interface = gr.Interface(
        fn=plot_visualization,
        inputs=gr.Dropdown([
            'Masa planety vs Odległość od gwiazdy',
            'Masa planety vs Masa gwiazdy',
            'Masa planety vs Okres orbitalny',
            'Okres orbitalny vs Odległość (Prawa Keplera)',
            'Masa planety vs Temperatura gwiazdy',
            'Rozkład temperatury równowagi'
        ], label="Wybierz wizualizację"),
        outputs="image",
        title="Wizualizacje egzoplanet"
    )

    classifier_interface = gr.Interface(
        fn=classify_model,
        inputs=gr.Number(value=100, label="Liczba drzew w lesie losowym"),
        outputs="text",
        title="Klasyfikacja Metody Odkrycia Planety"
    )

    regression_interface = gr.Interface(
        fn=regression_model,
        inputs=[
            gr.Dropdown([
                "Okres_Orbitalny",
                "Masa",
                "Mimosrod",
                "Temperatura_Rownowagi",
                "Strumien_Insolacji",
                "Metalicznosc_Gwiazdy"
            ], label="Cel regresji"),
            gr.Number(value=100, label="Liczba drzew w lesie losowym")
        ],
        outputs="text",
        title="Regresja dla różnych wartości planet"
    )

    # Uruchamianie interfejsu z zakładkami i ustawienie share=True
    gr.TabbedInterface(
        [classifier_interface, regression_interface, visualization_interface],
        ["Klasyfikacja", "Regresja", "Wizualizacje"]
    ).launch(share=True)


if __name__ == "__main__":
    main()
