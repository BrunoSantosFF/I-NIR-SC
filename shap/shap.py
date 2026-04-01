import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import shap.shap as shap
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
from pathlib import Path

warnings.filterwarnings("ignore")

#================================  Dados e defines de leitura ====================== 
INPUT_PATH = '/home/von-neumann/Bruno_Santos/data/NIR-SC-UFES/coluna_imagem.csv'
OUTPUT_DIR = '/home/von-neumann/Bruno_Santos/results/shap_nir/'

TEST_SIZE = 0.2
PERCENTUAL_FEATURES = 0.3 # porcentagem de features capturadas
NUM_FEATURES_BEESWARM = 20
BACKGROUND_SIZE = 50
TEST_SAMPLE_SIZE = 500
RANDOM_STATE = 42
HIDDEN_LAYERS = (100, 50)
MAX_ITER = 500

# comprimentos das ondas de espectros 
WAVELENGTHS = [
    908.1, 914.294, 920.489, 926.683, 932.877, 939.072, 945.266, 951.46, 957.655, 963.849,
    970.044, 976.238, 982.432, 988.627, 994.821, 1001.015, 1007.21, 1013.404, 1019.598, 1025.793,
    1031.987, 1038.181, 1044.376, 1050.57, 1056.764, 1062.959, 1069.153, 1075.348, 1081.542, 1087.736,
    1093.931, 1100.125, 1106.319, 1112.514, 1118.708, 1124.902, 1131.097, 1137.291, 1143.485, 1149.68,
    1155.874, 1162.069, 1168.263, 1174.457, 1180.652, 1186.847, 1193.04, 1199.235, 1205.429, 1211.623,
    1217.818, 1224.012, 1230.206, 1236.401, 1242.595, 1248.789, 1254.984, 1261.178, 1267.373, 1273.567,
    1279.761, 1285.956, 1292.15, 1298.344, 1304.539, 1310.733, 1316.927, 1323.122, 1329.316, 1335.51,
    1341.705, 1347.899, 1354.094, 1360.288, 1366.482, 1372.677, 1378.871, 1385.065, 1391.26, 1397.454,
    1403.648, 1409.843, 1416.037, 1422.231, 1428.426, 1434.62, 1440.814, 1447.009, 1453.203, 1459.398,
    1465.592, 1471.786, 1477.981, 1484.175, 1490.369, 1496.564, 1502.758, 1508.952, 1515.147, 1521.341,
    1527.535, 1533.73, 1539.924, 1546.119, 1552.313, 1558.507, 1564.702, 1570.896, 1577.09, 1583.285,
    1589.479, 1595.673, 1601.868, 1608.062, 1614.256, 1620.451, 1626.645, 1632.839, 1639.034, 1645.228,
    1651.423, 1657.617, 1663.811, 1670.006, 1676.2
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


#================================  Carregando e preparando  dados ======================
df_nir = pd.read_csv(INPUT_PATH)
data_columns = [f'data{i}' for i in range(1, 126)]

for col in data_columns:
    df_nir[col] = df_nir[col].astype(str).str.replace(',', '.').astype(float)

df_nir['img_id'] = df_nir['Imagem'].str.split('.').str[0]

print(f"Total samples: {df_nir.shape[0]}")
print(f"Classes: {sorted(df_nir['y'].unique())}")

X_raw = df_nir[data_columns].copy()
y = df_nir['y'].values


X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"Training samples: {len(X_train_raw)}")
print(f"Test samples: {len(X_test_raw)}")


# =========================== Normalizando dados ========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)


# =============================== Classificador =========================
mlp = MLPClassifier(
    hidden_layer_sizes=HIDDEN_LAYERS,
    activation='relu',
    solver='adam',
    max_iter=MAX_ITER,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=RANDOM_STATE
)

with tqdm(total=1, desc="Training MLP") as pbar:
    mlp.fit(X_train_scaled, y_train)
    pbar.update(1)


# =========================== Avalição de modelo ===============================

y_pred_train = mlp.predict(X_train_scaled)
y_pred_test = mlp.predict(X_test_scaled)
y_pred_proba_test = mlp.predict_proba(X_test_scaled)[:, 1]

train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)
test_auc = roc_auc_score(y_test, y_pred_proba_test)

# ======================== Valores SHAP ===================================
np.random.seed(RANDOM_STATE)

background_size = min(BACKGROUND_SIZE, X_train_scaled.shape[0])
background_indices = np.random.choice(X_train_scaled.shape[0], background_size, replace=False)
background = X_train_scaled[background_indices]

print(f"Background size: {background_size}")
print(f"Test sample size: {min(TEST_SAMPLE_SIZE, X_test_scaled.shape[0])}")

with tqdm(total=1, desc="Computing SHAP") as pbar:
    explainer = shap.Explainer(mlp.predict_proba, background)
    explain_sample_size = min(TEST_SAMPLE_SIZE, X_test_scaled.shape[0])
    X_explain = X_test_scaled[:explain_sample_size]
    shap_values = explainer(X_explain)
    pbar.update(1)

shap_values_class1 = shap_values[..., 1]


# ======================== Features importantes ============================

shap_importance = np.abs(shap_values_class1.values).mean(axis=0)

feature_importance_df = pd.DataFrame({
    'feature': data_columns,
    'wavelength': WAVELENGTHS,
    'shap_importance': shap_importance
}).sort_values(by='shap_importance', ascending=False)

NUM_FEATURES = int(PERCENTUAL_FEATURES * len(data_columns))
top_features = feature_importance_df.head(NUM_FEATURES)['feature'].tolist()

# ============================= Grafico Beeswarm =============================

plt.figure(figsize=(10, 6))
top_n_features = feature_importance_df.head(NUM_FEATURES_BEESWARM)['feature'].tolist()
top_n_indices = [data_columns.index(f) for f in top_n_features]

shap_values_top = shap_values_class1[:, top_n_indices]

shap_values_top_obj = shap.Explanation(
    values=shap_values_top.values,
    base_values=shap_values_class1.base_values,
    data=shap_values_class1.data[:, top_n_indices],
    feature_names=[f"{w:.1f} nm" for w in feature_importance_df.head(NUM_FEATURES_BEESWARM)['wavelength']]
)

shap.plots.beeswarm(shap_values_top_obj, max_display=NUM_FEATURES_BEESWARM, show=False)
plt.title(f"Top {NUM_FEATURES_BEESWARM} Features - SHAP Beeswarm", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'shap_beeswarm_top{NUM_FEATURES_BEESWARM}.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"Beeswarm plot saved: {OUTPUT_DIR}/shap_beeswarm_top{NUM_FEATURES_BEESWARM}.png")


# ====================== Top 5 comprimentos dos espectros ============================

plt.figure(figsize=(14, 8))

espectro_medio_classe0 = df_nir[df_nir['y'] == 0][data_columns].mean()
espectro_medio_classe1 = df_nir[df_nir['y'] == 1][data_columns].mean()

plt.plot(WAVELENGTHS, espectro_medio_classe0.values, 'b-', label='Class 0', linewidth=2, alpha=0.7)
plt.plot(WAVELENGTHS, espectro_medio_classe1.values, 'r-', label='Class 1', linewidth=2, alpha=0.7)

top5_features = feature_importance_df.head(5)['feature'].tolist()
top5_wavelengths = feature_importance_df.head(5)['wavelength'].tolist()
top5_importance = feature_importance_df.head(5)['shap_importance'].tolist()

for i, (feature, wavelength, importance) in enumerate(zip(top5_features, top5_wavelengths, top5_importance)):
    idx = data_columns.index(feature)
    valor_classe0 = espectro_medio_classe0.iloc[idx]
    valor_classe1 = espectro_medio_classe1.iloc[idx]
    
    plt.scatter(wavelength, valor_classe0, c='blue', s=150, edgecolors='black', linewidth=2, zorder=5, marker='o')
    plt.scatter(wavelength, valor_classe1, c='red', s=150, edgecolors='black', linewidth=2, zorder=5, marker='s')
    
    plt.annotate(f'{wavelength:.1f} nm\nimp:{importance:.4f}',
                (wavelength, max(valor_classe0, valor_classe1)),
                xytext=(10, 10), textcoords='offset points', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel('Mean Intensity', fontsize=12)
plt.title('Mean NIR Spectrum by Class with Top 5 Wavelengths', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)

info_text = "Top 5 features:\n"
for i, (wavelength, importance) in enumerate(zip(top5_wavelengths, top5_importance), 1):
    info_text += f"{i}. {wavelength:.1f} nm: {importance:.6f}\n"
plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'espectro_com_top5_comprimentos.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"Spectrum plot saved: {OUTPUT_DIR}/espectro_com_top5_comprimentos.png")


# =============================== Salvando as features =======================================
output_df = pd.DataFrame()
output_df['img_id'] = df_nir['Imagem']
output_df['label'] = df_nir['y']

for i, (feature, wavelength) in enumerate(zip(top_features, feature_importance_df.head(NUM_FEATURES)['wavelength']), 1):
    output_df[f'f_{i}_{wavelength:.1f}nm'] = df_nir[feature]

output_csv = os.path.join(OUTPUT_DIR, f'top_{NUM_FEATURES}_features_nir_com_comprimentos.csv')
output_df.to_csv(output_csv, index=False)


# =============================== Relatório ===================================

with open(os.path.join(OUTPUT_DIR, "relatorio_modelo.txt"), "w") as f:
    f.write("MODEL REPORT - NIR SPECTRA + SHAP\n")
    f.write("="*60 + "\n\n")
    f.write(f"Configuration:\n")
    f.write(f"  - Features in CSV: {PERCENTUAL_FEATURES*100:.0f}% ({NUM_FEATURES} features)\n")
    f.write(f"  - Train/test split: {TEST_SIZE*100:.0f}% test\n")
    f.write(f"  - Training samples: {len(X_train_raw)}\n")
    f.write(f"  - Test samples: {len(X_test_raw)}\n\n")
    
    f.write(f"Model Performance:\n")
    f.write(f"  - Train accuracy: {train_acc:.4f}\n")
    f.write(f"  - Test accuracy: {test_acc:.4f}\n")
    f.write(f"  - Test AUC: {test_auc:.4f}\n\n")
    
    f.write(f"Top 20 Features (based on test set):\n")
    for i, row in feature_importance_df.head(20).iterrows():
        f.write(f"{row['feature']} ({row['wavelength']:.1f} nm) -> {row['shap_importance']:.6f}\n")
    
    f.write(f"\nClass Distribution:\n")
    for label in sorted(df_nir['y'].unique()):
        count = (df_nir['y'] == label).sum()
        f.write(f"  - Class {label}: {count} samples ({count/len(df_nir)*100:.1f}%)\n")

print(f"Report saved: {OUTPUT_DIR}/relatorio_modelo.txt")

# ============================= Resultado Final ===================================
print("\n" + "=" * 60)
print(f"Results saved in: {OUTPUT_DIR}")
print("=" * 60)
print(f"\nTop 5 wavelengths:")
for i, (wavelength, importance) in enumerate(zip(top5_wavelengths, top5_importance), 1):
    print(f"  {i}. {wavelength:.1f} nm: {importance:.6f}")
