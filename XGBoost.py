import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc
import seaborn as sn
from astroquery.gaia import Gaia
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from xgboost import XGBClassifier

# 1. Definiowanie zapytania SQL
query = """
SELECT TOP 1000 
    source_id, ra, dec, parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error, phot_g_mean_mag, bp_rp
FROM gaiadr3.gaia_source
WHERE parallax IS NOT NULL
AND pmra IS NOT NULL
AND pmdec IS NOT NULL
AND phot_g_mean_mag IS NOT NULL
AND bp_rp IS NOT NULL
"""

# 2. Wykonanie zapytania i pobranie danych
job = Gaia.launch_job(query)
gaia_data = job.get_results().to_pandas()

# 3. Usunięcie kolumny SOURCE_ID
gaia_data = gaia_data.drop(columns=['SOURCE_ID'])

# 4. Dodanie losowej kolumny 'Pulsar_Star' jako przykład
np.random.seed(42)
gaia_data['Pulsar_Star'] = np.random.randint(0, 2, gaia_data.shape[0])

# 5. Usuwanie wartości odstających za pomocą z-score
def remove_outliers(df, columns, threshold=3):
    z_scores = np.abs(stats.zscore(df[columns]))
    return df[(z_scores < threshold).all(axis=1)]

# Kolumny do sprawdzenia na outliery
feature_columns = gaia_data.columns.drop('Pulsar_Star')
gaia_data_clean = remove_outliers(gaia_data, feature_columns)

print(f"Usunięto {gaia_data.shape[0] - gaia_data_clean.shape[0]} wartości odstających.")

# 6. Przygotowanie danych do trenowania/testowania
X = gaia_data_clean.drop("Pulsar_Star", axis=1)
y = gaia_data_clean["Pulsar_Star"]

# 7. Standaryzacja cech
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 8. Redukcja wymiarowości za pomocą PCA
pca = PCA(n_components=0.95, random_state=42)  # Zachowanie 95% wariancji
X_pca = pca.fit_transform(X_scaled)
print(f"Liczba komponentów PCA: {pca.n_components_}")

# 9. Podział na zestaw treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.25, random_state=42, stratify=y
)

# 10. Trening modelu XGBoost
xgb = XGBClassifier(
    n_estimators=200,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
xgb.fit(X_train, y_train)

# 11. Ocena modelu
y_pred = xgb.predict(X_test)
y_pred_proba = xgb.predict_proba(X_test)[:,1]

# 12. Podstawowe metryki
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
roc_auc = metrics.roc_auc_score(y_test, y_pred_proba)

print('--- XGBoost ---')
print(f'Dokładność: {accuracy:.4f}')
print(f'Precyzja: {precision:.4f}')
print(f'Czułość (Recall): {recall:.4f}')
print(f'F1-score: {f1:.4f}')
print(f'AUC-ROC: {roc_auc:.4f}\n')

# 13. Macierz pomyłek
conf_matrix = pd.crosstab(y_test, y_pred, rownames=['Rzeczywiste'], colnames=['Przewidywane'], margins=True)
plt.figure(figsize=(6,4))
sn.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues')
plt.title('Macierz pomyłek - XGBoost')
plt.show()

# 14. Krzywa ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc_val = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_val:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Odsetek fałszywie pozytywnych (FPR)')
plt.ylabel('Odsetek prawdziwie pozytywnych (TPR)')
plt.title('Krzywa ROC - XGBoost')
plt.legend(loc="lower right")
plt.show()

# 15. Raport klasyfikacji
print("Raport klasyfikacji - XGBoost:\n", classification_report(y_test, y_pred))

# 16. Ważność cech (po PCA)
importances = xgb.feature_importances_
forest_importances = pd.Series(importances, index=[f'PC{i+1}' for i in range(pca.n_components_)])
forest_importances = forest_importances.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sn.barplot(x=forest_importances, y=forest_importances.index)
plt.title('Ważność cech po PCA - XGBoost')
plt.xlabel('Ważność')
plt.ylabel('Komponenty PCA')
plt.tight_layout()
plt.show()

# 17. Przykładowe predykcje
print(f"Pulsar TAK: 1 , Pulsar NIE: 0")
print(f"------------------------------------------------------")
predictions_df_xgb = pd.DataFrame({
    "Przewidywany wynik": y_pred[:10],
    "Rzeczywisty wynik": y_test[:10].values
})
print(predictions_df_xgb.to_markdown())

# 18. Statystyki zmiennych
dataStatDF = pd.DataFrame({
    "Max_Input": X.max(),
    "Mean_Input": X.mean(),
    "Min_Input": X.min()
})

# 19. Przykładowe dane do predykcji
data = {
    "ra": [np.max(X['ra']), np.mean(X['ra']), np.min(X['ra'])],
    "dec": [np.max(X['dec']), np.mean(X['dec']), np.min(X['dec'])],
    "parallax": [np.max(X['parallax']), np.mean(X['parallax']), np.min(X['parallax'])],
    "parallax_error": [np.max(X['parallax_error']), np.mean(X['parallax_error']), np.min(X['parallax_error'])],
    "pmra": [np.max(X['pmra']), np.mean(X['pmra']), np.min(X['pmra'])],
    "pmra_error": [np.max(X['pmra_error']), np.mean(X['pmra_error']), np.min(X['pmra_error'])],
    "pmdec": [np.max(X['pmdec']), np.mean(X['pmdec']), np.min(X['pmdec'])],
    "pmdec_error": [np.max(X['pmdec_error']), np.mean(X['pmdec_error']), np.min(X['pmdec_error'])],
    "phot_g_mean_mag": [np.max(X['phot_g_mean_mag']), np.mean(X['phot_g_mean_mag']), np.min(X['phot_g_mean_mag'])],
    "bp_rp": [np.max(X['bp_rp']), np.mean(X['bp_rp']), np.min(X['bp_rp'])]
}

mockDF = pd.DataFrame(data)

# 20. Standaryzacja i PCA dla nowych danych
mock_scaled = scaler.transform(mockDF)
mock_pca = pca.transform(mock_scaled)

# 21. Predykcja na podstawie przykładowych danych dla XGBoost
predictions_xgb = xgb.predict(mock_pca)
mockDF["Pulsar Star Predictions - XGBoost"] = predictions_xgb

print("\nPrzykładowe predykcje dla nowych danych - XGBoost:")
print(mockDF.to_markdown())

# 22. Podsumowanie wyników
true_positives_xgb = np.sum((y_test == 1) & (y_pred == 1))
false_positives_xgb = np.sum((y_test == 0) & (y_pred == 1))
true_negatives_xgb = np.sum((y_test == 0) & (y_pred == 0))
false_negatives_xgb = np.sum((y_test == 1) & (y_pred == 0))

summary_df_xgb = pd.DataFrame({
    "Nazwa": ["Prawdziwie pozytywne", "Fałszywie pozytywne", "Prawdziwie negatywne", "Fałszywie negatywne"],
    "Zliczone wartości": [true_positives_xgb, false_positives_xgb, true_negatives_xgb, false_negatives_xgb]
})

print("\nPodsumowanie wyników - XGBoost:")
print(summary_df_xgb.to_markdown())
