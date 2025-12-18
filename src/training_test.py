import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==========================================
# 1. KONFIGURASI
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

INPUT_FILE = "data/processed.csv"
OUTPUT_MODEL = "model/test/model_test.pkl"

# ==========================================
# 2. PERSIAPAN DATA (DENGAN PERBAIKAN ERROR)
# ==========================================
def load_and_prepare_target(path):
    print(f"--- Memuat data dari {path} ---")
    df = pd.read_csv(path)

    if 'stock' in df.columns:
        print("Mengubah target 'stock' menjadi kategori (Klasifikasi)...")

        # --- PERBAIKAN ERROR: BINNING STRATEGY ---
        try:
            df['stock_class'] = pd.qcut(df['stock'], q=3, labels=['Rendah', 'Sedang', 'Tinggi'])
            print(">> Berhasil menggunakan pembagian otomatis (qcut).")

        except ValueError:
            print(">> Peringatan: Data didominasi angka 0. Beralih ke pembagian manual.")
            # Logika Manual:
            # Rendah = 0 s/d 5
            # Sedang = 6 s/d 50
            # Tinggi = > 50
            bins = [-float('inf'), 5, 50, float('inf')]
            labels = ['Rendah', 'Sedang', 'Tinggi']
            df['stock_class'] = pd.cut(df['stock'], bins=bins, labels=labels)
            print(">> Berhasil menggunakan pembagian manual (0-5: Rendah, 6-50: Sedang, >50: Tinggi).")

        df = df.drop(columns=['stock']) 
        print(f"Distribusi Kelas:\n{df['stock_class'].value_counts()}")
        
    else:
        print("Peringatan: Kolom 'stock' tidak ditemukan. Pastikan preprocessing benar.")
        
    return df

# ==========================================
# 3. FUNGSI EVALUASI LENGKAP
# ==========================================
def evaluate_model(model, X_test, y_test, scenario_name):
    y_pred = model.predict(X_test)
    
    print(f"\n{'='*20} HASIL: {scenario_name} {'='*20}")
    
    # A. Akurasi
    acc = accuracy_score(y_test, y_pred)
    print(f"Akurasi Total: {acc:.2%}")
    
    # B. F1-Score & Recall
    print("\n[Laporan Klasifikasi]")
    print(classification_report(y_test, y_pred))
    
    # C. Hitung Spesifisitas Manual
    cm = confusion_matrix(y_test, y_pred)
    classes = model.classes_
    
    print("[Detail Sensitivitas & Spesifisitas per Kelas]")
    print(f"{'Kelas':<10} | {'Sensitivitas':<15} | {'Spesifisitas':<15}")
    print("-" * 45)
    
    for i, class_label in enumerate(classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - TP - FP - FN
        
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        
        print(f"{str(class_label):<10} | {sensitivity:.2%}          | {specificity:.2%}")

    # D. Simpan Gambar
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {scenario_name}\nAkurasi: {acc:.2%}')
    plt.ylabel('Aktual')
    plt.xlabel('Prediksi')
    plt.tight_layout()
    
    filename = f"matrix_{scenario_name.replace(' ', '_')}.png"
    plt.savefig(filename)
    print(f"[INFO] Gambar matriks disimpan: {filename}")
    
    return acc

# ==========================================
# 4. EKSEKUSI UTAMA
# ==========================================
if __name__ == "__main__":
    # 1. Load Data
    df = load_and_prepare_target(INPUT_FILE)
    
    # 2. Split Data
    X = df.drop(columns=['stock_class'])
    y = df['stock_class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- SKENARIO A: TANPA SELEKSI ---
    print("\n>>> MEMULAI SKENARIO A (Semua Fitur) <<<")
    model_A = RandomForestClassifier(n_estimators=100, random_state=42)
    model_A.fit(X_train, y_train)
    acc_A = evaluate_model(model_A, X_test, y_test, "Skenario A")
    
    # --- SKENARIO B: FITUR TERPILIH ---
    print("\n>>> MEMULAI SKENARIO B (Fitur Terpilih) <<<")
    valid_features = [f for f in SELECTED_FEATURES if f in X_train.columns]
    missing_features = set(SELECTED_FEATURES) - set(valid_features)
    if missing_features:
        print(f"Warning: Fitur berikut tidak ditemukan di dataset: {missing_features}")
        
    print(f"Fitur yang digunakan ({len(valid_features)} fitur): {valid_features}")
    
    X_train_sel = X_train[valid_features]
    X_test_sel = X_test[valid_features]
    
    model_B = RandomForestClassifier(n_estimators=100, random_state=42)
    model_B.fit(X_train_sel, y_train)
    acc_B = evaluate_model(model_B, X_test_sel, y_test, "Skenario B")
    
    # --- KESIMPULAN ---
    print("\n" + "="*40)
    print("PERBANDINGAN AKHIR")
    print("="*40)
    print(f"Akurasi Skenario A (Full Features) : {acc_A:.2%}")
    print(f"Akurasi Skenario B (Selected)      : {acc_B:.2%}")
    
    if acc_B >= (acc_A - 0.02):
        print("\nKESIMPULAN: Model Skenario B DIPILIH (Efisien).")
        joblib.dump(model_B, OUTPUT_MODEL)
    else:
        print("\nKESIMPULAN: Model Skenario A DIPILIH (Akurasi Tinggi).")
        joblib.dump(model_A, OUTPUT_MODEL)
        
    print(f"Model terbaik disimpan ke: {OUTPUT_MODEL}")