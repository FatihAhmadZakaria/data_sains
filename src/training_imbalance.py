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

# ==========================================
# 1. KONFIGURASI
# ==========================================
# Fitur Skenario B (Fitur Terpilih)
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

# Fitur Kategori (untuk Encoder)
CATEGORICAL_FEATURES = [
    'kategoriutama',
    'kategorisekunder',
    'marketplacename',
    'tipetransaksi'
]

# Paths
PROCESSED_FILE = "data/processed.csv"
RAW_FILE = "data/data.csv"

# Folder Output
MODEL_DIR = "models"
REPORT_DIR = "reports"

# Nama File Output
OUTPUT_MODEL = f"{MODEL_DIR}/model_imbalance.pkl"
OUTPUT_ENCODER = f"{MODEL_DIR}/encoder_final.pkl"
OUTPUT_REPORT_TXT = f"{REPORT_DIR}/report_imbalance.txt"

# ==========================================
# 2. FUNGSI PERSIAPAN DATA
# ==========================================
def load_and_prepare_target(path):
    print(f"--- Memuat data dari {path} ---")
    df = pd.read_csv(path)
    if 'stock' in df.columns:
        # Manual Binning
        bins = [-float('inf'), 5, 50, float('inf')]
        labels = ['Rendah', 'Sedang', 'Tinggi']
        df['stock_class'] = pd.cut(df['stock'], bins=bins, labels=labels)
        df = df.drop(columns=['stock']) 
    return df

def generate_and_save_encoders():
    print("\n>>> MEMBUAT KAMUS ENCODER (DATA ASLI) <<<")
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
        print(f"   Encoder disimpan: {OUTPUT_ENCODER}")
    except Exception as e:
        print(f"WARNING: Gagal encoder: {e}")

# ==========================================
# 3. FUNGSI EVALUASI (APPEND KE TXT)
# ==========================================
def evaluate_model(model, X_test, y_test, scenario_name, output_image_name):
    with open(OUTPUT_REPORT_TXT, "a") as f:
        
        def log(text):
            print(text)
            f.write(text + "\n")

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        log(f"\n{'='*20} HASIL: {scenario_name} {'='*20}")
        log(f"Akurasi Total: {acc:.2%}")
        
        log("\n[Laporan Klasifikasi]")
        log(classification_report(y_test, y_pred))
        
        # Detail Sensitivitas
        cm = confusion_matrix(y_test, y_pred)
        classes = model.classes_
        log("\n[Detail Sensitivitas & Spesifisitas]")
        log(f"{'Kelas':<10} | {'Sensitivitas':<15} | {'Spesifisitas':<15}")
        log("-" * 45)
        
        for i, class_label in enumerate(classes):
            TP = cm[i, i]
            FP = cm[:, i].sum() - TP
            FN = cm[i, :].sum() - TP
            TN = cm.sum() - TP - FP - FN
            
            sen = TP / (TP + FN) if (TP + FN) > 0 else 0
            spe = TN / (TN + FP) if (TN + FP) > 0 else 0
            log(f"{str(class_label):<10} | {sen:.2%}          | {spe:.2%}")

        # Simpan Gambar Matriks
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes)
        plt.title(f'Confusion Matrix - {scenario_name}\n({output_image_name})')
        plt.ylabel('Aktual')
        plt.xlabel('Prediksi')
        plt.tight_layout()
        
        image_path = f"{REPORT_DIR}/{output_image_name}"
        plt.savefig(image_path)
        plt.close()
        print(f"[INFO] Gambar disimpan: {image_path}")
        
    return acc

# ==========================================
# 4. MAIN PROGRAM
# ==========================================
if __name__ == "__main__":
    # 1. Buat Folder
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    # 2. Reset File Laporan
    open(OUTPUT_REPORT_TXT, 'w').close()

    # 3. Load Data
    df = load_and_prepare_target(PROCESSED_FILE)
    X = df.drop(columns=['stock_class'])
    y = df['stock_class']

    # 4. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # ==========================================
    # SKENARIO A: SEMUA FITUR (IMBALANCE)
    # ==========================================
    print("\n>>> MEMULAI SKENARIO A (Semua Fitur - Imbalance) <<<")

    model_A = RandomForestClassifier(n_estimators=100, random_state=42)
    model_A.fit(X_train, y_train)

    evaluate_model(
        model_A,
        X_test,
        y_test,
        scenario_name="Skenario A (Semua Fitur)",
        output_image_name="matrix_scenarioA_imbalance.png"
    )

    # ==========================================
    # SKENARIO B: FITUR TERPILIH (IMBALANCE)
    # ==========================================
    print("\n>>> MEMULAI SKENARIO B (Fitur Terpilih - Imbalance) <<<")

    # 1. Filter Kolom
    valid_features = [f for f in SELECTED_FEATURES if f in X_train.columns]

    # 2. Siapkan Data (Gunakan X_train ASLI)
    X_train_sel = X_train[valid_features]
    X_test_sel = X_test[valid_features]

    # 3. Training (Gunakan y_train ASLI)
    model_B = RandomForestClassifier(n_estimators=100, random_state=42)
    model_B.fit(X_train_sel, y_train)

    evaluate_model(
        model_B,
        X_test_sel,
        y_test,
        scenario_name="Skenario B (Fitur Terpilih)",
        output_image_name="matrix_scenarioB_imbalance.png"
    )

    # ==========================================
    # SIMPAN HASIL
    # ==========================================
    joblib.dump(model_B, OUTPUT_MODEL)
    print(f"\nModel Final (Skenario B) disimpan ke: {OUTPUT_MODEL}")

    generate_and_save_encoders()
    print(f"Laporan lengkap tersimpan di: {OUTPUT_REPORT_TXT}")