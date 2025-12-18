import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

# ==========================================
# KONFIGURASI
# ==========================================
SELECTED_FEATURES = [
    'lembartotal', 
    'jumlahcetak', 
    'hargabuku', 
    'kategoriutama', 
    'servicefee', 
    'tahunterbit', 
    'kategorisekunder', 
    'dayofyear',          
    'marketplacename',    
    'tipetransaksi'       
]

CATEGORICAL_FEATURES = [
    'kategoriutama',
    'kategorisekunder',
    'marketplacename',
    'tipetransaksi'
]

# File Input & Output
PROCESSED_FILE = "data/processed.csv"
RAW_FILE = "data/data.csv"

# Lokasi Output
OUTPUT_MODEL = "models/model_balanced.pkl"
OUTPUT_ENCODER = "models/encoder_final.pkl"
OUTPUT_MATRIX_IMG = "reports/matrix_scenario_balanced.png"
OUTPUT_REPORT_TXT = "reports/report_balanced.txt"

# ==========================================
# 1. FUNGSI LOAD & PERSIAPAN
# ==========================================
def load_data(path):
    print(f"--- Memuat data dari {path} ---")
    df = pd.read_csv(path)
    if 'stock' in df.columns:
        bins = [-float('inf'), 5, 50, float('inf')]
        labels = ['Rendah', 'Sedang', 'Tinggi']
        df['stock_class'] = pd.cut(df['stock'], bins=bins, labels=labels)
        df = df.drop(columns=['stock']) 
    return df

def balance_training_data(X_train, y_train):
    print("\n>>> PROSES BALANCING (OVERSAMPLING) <<<")
    train_data = pd.concat([X_train, y_train], axis=1)
    
    rendah = train_data[train_data.stock_class == 'Rendah']
    sedang = train_data[train_data.stock_class == 'Sedang']
    tinggi = train_data[train_data.stock_class == 'Tinggi']
    
    print(f"   Original -> Rendah: {len(rendah)}, Sedang: {len(sedang)}, Tinggi: {len(tinggi)}")
    
    n_majority = len(rendah)
    sedang_up = resample(sedang, replace=True, n_samples=n_majority, random_state=42)
    tinggi_up = resample(tinggi, replace=True, n_samples=n_majority, random_state=42)
    
    train_balanced = pd.concat([rendah, sedang_up, tinggi_up])
    print(f"   Balanced -> Rendah: {len(rendah)}, Sedang: {len(sedang_up)}, Tinggi: {len(tinggi_up)}")
    
    return train_balanced.drop('stock_class', axis=1), train_balanced['stock_class']

def generate_encoder():
    print("\n>>> MEMBUAT ENCODER KHUSUS BALANCE <<<")
    try:
        df_raw = pd.read_csv(RAW_FILE, sep=';')
        encoders = {}
        for col in CATEGORICAL_FEATURES:
            if col in df_raw.columns:
                df_raw[col] = df_raw[col].astype(str).fillna("Unknown")
                le = LabelEncoder()
                le.fit(df_raw[col])
                encoders[col] = le
        joblib.dump(encoders, OUTPUT_ENCODER)
        print(f"   Encoder disimpan ke: {OUTPUT_ENCODER}")
    except Exception as e:
        print(f"   Error membuat encoder: {e}")

# ==========================================
# 2. FUNGSI EVALUASI (CETAK KE LAYAR & FILE)
# ==========================================
def evaluate_model(model, X_test, y_test):
    with open(OUTPUT_REPORT_TXT, "w") as f:
        def log(text):
            print(text)
            f.write(text + "\n")

        y_pred = model.predict(X_test)

        log(f"{'='*20} HASIL EVALUASI: MODEL BALANCED {'='*20}")

        # A. Akurasi
        acc = accuracy_score(y_test, y_pred)
        log(f"Akurasi Total: {acc:.2%}")

        # B. Classification Report (F1, Recall, Precision)
        log("\n[Laporan Klasifikasi]")
        report_str = classification_report(y_test, y_pred)
        log(report_str)

        # C. Sensitivitas & Spesifisitas per Kelas
        cm = confusion_matrix(y_test, y_pred)
        classes = model.classes_

        log("\n[Detail Sensitivitas & Spesifisitas per Kelas]")
        log(f"{'Kelas':<10} | {'Sensitivitas':<15} | {'Spesifisitas':<15}")
        log("-" * 45)

        for i, class_label in enumerate(classes):
            TP = cm[i, i]
            FP = cm[:, i].sum() - TP
            FN = cm[i, :].sum() - TP
            TN = cm.sum() - TP - FP - FN

            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

            log(f"{str(class_label):<10} | {sensitivity:.2%}          | {specificity:.2%}")

        # D. Simpan Gambar Matriks (Visualisasi)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                    xticklabels=classes, yticklabels=classes)
        plt.title(f'Confusion Matrix - MODEL BALANCED\nAkurasi: {acc:.2%}')
        plt.ylabel('Aktual')
        plt.xlabel('Prediksi')
        plt.tight_layout()
        plt.savefig(OUTPUT_MATRIX_IMG)

        log(f"\n[INFO] Gambar matriks disimpan: {OUTPUT_MATRIX_IMG}")
        print(f"[INFO] Laporan teks disimpan: {OUTPUT_REPORT_TXT}")

# ==========================================
# 3. MAIN PROGRAM
# ==========================================
def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # 1. Load Data
    df = load_data(PROCESSED_FILE)
    X = df.drop(columns=['stock_class'])
    y = df['stock_class']

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Filter Fitur
    valid_features = [f for f in SELECTED_FEATURES if f in X_train.columns]
    X_train = X_train[valid_features]
    X_test = X_test[valid_features]

    # 4. Balancing
    X_train_bal, y_train_bal = balance_training_data(X_train, y_train)

    # 5. Training
    print("\n>>> TRAINING MODEL (Random Forest - Balanced) <<<")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train_bal, y_train_bal)

    # 6. Evaluasi & Save Report
    evaluate_model(model, X_test, y_test)

    # 7. Simpan Model
    joblib.dump(model, OUTPUT_MODEL)
    print(f"\nModel Balance berhasil disimpan ke: {OUTPUT_MODEL}")

    # 8. Encoder
    generate_encoder()
    print("\n=== SELESAI ===")

if __name__ == "__main__":
    main()