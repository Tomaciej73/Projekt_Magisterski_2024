import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
from PIL import Image as PILImage

def load_data():
    df = pd.read_csv('dataset.csv', encoding='utf-8')
    return df

def plot_visualization(plot_type):
    df = load_data()
    sns.set_style('whitegrid')
    plt.figure(figsize=(10, 6))

    if plot_type == 'Rozkład obiektów na niebie':
        sns.scatterplot(data=df, x='ra', y='decl', hue='target')
        plt.xlabel('Rektascensja (RA)')
        plt.ylabel('Deklinacja (Dec)')
        plt.title('Rozkład obiektów na niebie (ra vs dec)')
    elif plot_type == 'Rozkład obiektów w Galaktyce':
        sns.scatterplot(data=df, x='gal_l', y='gal_b', hue='target')
        plt.xlabel('Galaktyczna długość (Galaktyczne l)')
        plt.ylabel('Galaktyczna szerokość (Galaktyczne b)')
        plt.title('Rozkład obiektów w Galaktyce (gal_l vs gal_b)')
    elif plot_type == 'Redshift fotometryczny vs Moduł odległości':
        sns.scatterplot(data=df, x='hostgal_photoz', y='distmod', hue='target')
        plt.xlabel('Redshift fotometryczny galaktyki')
        plt.ylabel('Moduł odległości')
        plt.title('Redshift fotometryczny vs Moduł odległości')
    elif plot_type == 'Moduł odległości dla różnych typów obiektów':
        sns.boxplot(data=df, x='target', y='distmod')
        plt.xlabel('Klasa docelowa (target)')
        plt.ylabel('Moduł odległości')
        plt.title('Moduł odległości dla różnych typów obiektów')
    else:
        return "Nieznany typ wizualizacji."

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img = PILImage.open(buf)
    return img
port = int(os.environ.get('PORT', 7862))
visualization_interface = gr.Interface(
    fn=plot_visualization,
    inputs=gr.Dropdown([
        'Rozkład obiektów na niebie',
        'Rozkład obiektów w Galaktyce',
        'Redshift fotometryczny vs Moduł odległości',
        'Moduł odległości dla różnych typów obiektów'
    ], label="Wybierz wizualizację"),
    outputs="image",
    title="Wizualizacja analizy czasowej szeregów światła"
)

visualization_interface.launch(server_name="0.0.0.0", server_port=port)
