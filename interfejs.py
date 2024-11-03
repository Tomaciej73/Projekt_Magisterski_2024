import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image as PILImage

def load_data():
    df = pd.read_csv('pulsar_data_train_pl.csv', encoding='utf-8')
    return df

def plot_visualization(plot_type):
    df = load_data()
    sns.set_style('whitegrid')
    plt.figure(figsize=(10, 6))

    if plot_type == 'Średnia profilu zintegrowanego vs Klasa sygnału':
        sns.boxplot(data=df, x='Klasa_docelowa', y='Srednia_profili')
        plt.xlabel('Klasa docelowa (0 = inne sygnały, 1 = pulsar)')
        plt.ylabel('Średnia profilu zintegrowanego')
        plt.title('Średnia profilu zintegrowanego w zależności od klasy sygnału')
    elif plot_type == 'Odchylenie standardowe krzywej DM-SNR vs Klasa sygnału':
        sns.boxplot(data=df, x='Klasa_docelowa', y='Odchylenie_standardowe_krzywej_DM-SNR')
        plt.xlabel('Klasa docelowa (0 = inne sygnały, 1 = pulsar)')
        plt.ylabel('Odchylenie standardowe krzywej DM-SNR')
        plt.title('Zmienność DM-SNR w zależności od klasy sygnału')
    elif plot_type == 'Skosnosc profili zintegrowanego vs Klasa sygnału':
        sns.histplot(data=df, x='Skosnosc_profili', hue='Klasa_docelowa', bins=30, kde=True)
        plt.xlabel('Skośność profilu zintegrowanego')
        plt.ylabel('Liczba sygnałów')
        plt.title('Rozkład skośności profilu zintegrowanego dla różnych klas')
    else:
        return "Nieznany typ wizualizacji."

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img = PILImage.open(buf)
    return img

visualization_interface = gr.Interface(
    fn=plot_visualization,
    inputs=gr.Dropdown([
        'Średnia profilu zintegrowanego vs Klasa sygnału',
        'Odchylenie standardowe krzywej DM-SNR vs Klasa sygnału',
        'Skosnosc profili zintegrowanego vs Klasa sygnału'
    ], label="Wybierz wizualizację"),
    outputs="image",
    title="Wizualizacja danych pulsarów"
)

visualization_interface.launch(share=True)
