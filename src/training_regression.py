import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. KONFIGURASI
# ==========================================
# Fitur Skenario B (Sama persis dengan sebelumnya)
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

# Paths (Lokasi sama)
PROCESSED_FILE = "data/processed.csv"
RAW_FILE = "data/data.csv" # Opsional jika butuh encoder ulang

# Folder Output
MODEL_DIR = "models"
REPORT_DIR = "reports"

# Nama File Output (Dibedakan agar tidak menimpa)
OUTPUT_MODEL = f"{MODEL_DIR}/model_regression.pkl"
OUTPUT_REPORT_TXT = f"{REPORT_DIR}/report_regression.txt"

# ==========================================
# TAMBAHAN: FUNGSI ENCODER
# ==========================================
def generate_and_save_encoders():
    print("\n>>> MEMBUAT KAMUS ENCODER (REFERENSI) <<<")
    # Kita gunakan RAW_FILE agar mappingnya lengkap dari data asli
    # Sama persis seperti di skrip klasifikasi
    try:
        if os.path.exists(RAW_FILE):
            df_raw = pd.read_csv(RAW_FILE, sep=';') # Sesuaikan separator jika perlu
        else:
            # Fallback jika raw file tidak ada di folder yang sama, pakai processed
            print(f"[INFO] Raw file tidak ketemu, menggunakan {PROCESSED_FILE}")
            df_raw = pd.read_csv(PROCESSED_FILE)

        encoders = {}
        # List kolom kategori yang perlu di-encode
        # Pastikan list ini SAMA dengan yang dipakai saat membuat processed.csv
        CATEGORICAL_FEATURES = [
            'kategoriutama',
            'kategorisekunder',
            'marketplacename',
            'tipetransaksi'
        ]

        for col in CATEGORICAL_FEATURES:
            if col in df_raw.columns:
                # Pastikan format string & handle missing value
                df_raw[col] = df_raw[col].astype(str).fillna("Unknown")
                
                le = LabelEncoder()
                le.fit(df_raw[col])
                encoders[col] = le
        
        # Simpan Encoder
        # Kita beri nama beda dikit biar tau ini pasangan si regresi, 
        # tapi isinya tetap sama.
        encoder_path = f"{MODEL_DIR}/encoder_regression.pkl"
        joblib.dump(encoders, encoder_path)
        print(f"Encoder disimpan: {encoder_path}")
        
    except Exception as e:
        print(f"WARNING: Gagal membuat encoder: {e}")

# ==========================================
# 2. FUNGSI PERSIAPAN DATA
# ==========================================
def load_and_prepare_target(path):
    print(f"--- Memuat data dari {path} ---")
    df = pd.read_csv(path)
    
    # PERUBAHAN UTAMA: Tidak ada Binning / Pengelompokan
    # Kita pastikan target adalah angka
    if 'stock' in df.columns:
        df['stock'] = pd.to_numeric(df['stock'], errors='coerce').fillna(0)
    
    return df

# ==========================================
# 3. FUNGSI EVALUASI REGRESI
# ==========================================
def evaluate_regression(model, X_test, y_test, scenario_name, output_image_name):
    with open(OUTPUT_REPORT_TXT, "a") as f:
        
        def log(text):
            print(text)
            f.write(text + "\n")

        # Prediksi Angka
        y_pred = model.predict(X_test)
        
        # Hitung Metrik Error
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        log(f"\n{'='*20} HASIL REGRESI: {scenario_name} {'='*20}")
        log(f"R² Score (Akurasi Kecocokan): {r2:.4f} (Maks 1.0)")
        log(f"MAE (Rata-rata Meleset)     : {mae:.2f} buku")
        log(f"RMSE (Error Sensitif Outlier): {rmse:.2f}")
        
        # Analisis Tambahan: Bandingkan 5 Data Pertama
        log("\n[Contoh 5 Prediksi Pertama]")
        comparison = pd.DataFrame({'Aktual': y_test.values[:5], 'Prediksi': np.round(y_pred[:5], 1)})
        log(comparison.to_string(index=False))

        # VISUALISASI: Scatter Plot Actual vs Predicted
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='blue')
        
        # Garis Diagonal (Prediksi Sempurna)
        max_val = max(y_test.max(), y_pred.max())
        min_val = min(y_test.min(), y_pred.min())
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')
        
        plt.title(f'Actual vs Predicted Stock - {scenario_name}\n(R²: {r2:.2f})')
        plt.xlabel('Stok Aktual (Gudang)')
        plt.ylabel('Stok Prediksi (Model)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        image_path = f"{REPORT_DIR}/{output_image_name}"
        plt.savefig(image_path)
        plt.close()
        print(f"[INFO] Grafik Scatter disimpan: {image_path}")
        
    return r2

# ==========================================
# 4. MAIN PROGRAM
# ==========================================
if __name__ == "__main__":
    # 1. Buat Folder (jaga-jaga)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    # 2. Reset File Laporan
    open(OUTPUT_REPORT_TXT, 'w').close()

    # 3. Load Data
    df = load_and_prepare_target(PROCESSED_FILE)
    
    # Pastikan target dipisah
    target_col = 'stock'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 4. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ==========================================
    # SKENARIO A: SEMUA FITUR (REGRESI)
    # ==========================================
    print("\n>>> MEMULAI SKENARIO A (Semua Fitur - Regresi) <<<")

    model_A = RandomForestRegressor(n_estimators=100, random_state=42)
    model_A.fit(X_train, y_train)

    evaluate_regression(
        model_A,
        X_test,
        y_test,
        scenario_name="Skenario A (Semua Fitur)",
        output_image_name="scatter_scenarioA_regression.png"
    )

    # ==========================================
    # SKENARIO B: FITUR TERPILIH (REGRESI)
    # ==========================================
    print("\n>>> MEMULAI SKENARIO B (Fitur Terpilih - Regresi) <<<")

    # 1. Filter Kolom
    valid_features = [f for f in SELECTED_FEATURES if f in X_train.columns]

    # 2. Siapkan Data
    X_train_sel = X_train[valid_features]
    X_test_sel = X_test[valid_features]

    # 3. Training
    model_B = RandomForestRegressor(n_estimators=100, random_state=42)
    model_B.fit(X_train_sel, y_train)

    evaluate_regression(
        model_B,
        X_test_sel,
        y_test,
        scenario_name="Skenario B (Fitur Terpilih)",
        output_image_name="scatter_scenarioB_regression.png"
    )

    # ==========================================
    # SIMPAN HASIL
    # ==========================================
    joblib.dump(model_B, OUTPUT_MODEL)
    generate_and_save_encoders()
    print(f"\nModel Regresi Final disimpan ke: {OUTPUT_MODEL}")
    print(f"Laporan & Grafik tersimpan di folder: {REPORT_DIR}")