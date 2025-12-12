import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Load Data
df = pd.read_csv("data/data.csv", sep=';') # Pastikan separator sesuai, kadang Excel save as CSV pakai koma (,)
df = df.dropna(axis=1, how='all')
df = df.drop(columns=[col for col in df.columns if col.endswith((".3", ".4"))])

# 2. Pembersihan Data Spesifik (CUSTOM CLEANING)

# A. Bersihkan kolom 'berat' yang ada spasinya (misal '0 26' jadi 0.26 atau 26?)
# Asumsi: Spasi adalah pemisah desimal atau ribuan yang salah format.
# Kode di bawah ini membuang spasi lalu convert ke float.
# Cek dulu tipe datanya, kalau object baru dibersihkan.
if df['berat'].dtype == 'object':
    df['berat'] = df['berat'].astype(str).str.replace(' ', '.', regex=False)
    df['berat'] = pd.to_numeric(df['berat'], errors='coerce') # Ubah error jadi NaN

# B. Perbaiki 'tahunterbit' dari format tanggal (12/7/2018) menjadi Tahun saja (2018)
# Kita anggap formatnya d/m/y atau y-m-d, kita paksa ambil tahunnya.
df['tahunterbit'] = pd.to_datetime(df['tahunterbit'], errors='coerce').dt.year

# 3. Feature Engineering (Tanggal Transaksi)
df['tanggal'] = df['tanggal'].astype(str).str.strip()
df['tanggal'] = pd.to_datetime(df['tanggal'], format="%d/%m/%Y", errors='coerce')

df['tahun_transaksi'] = df['tanggal'].dt.year
df['bulan_transaksi'] = df['tanggal'].dt.month
df['hari_transaksi'] = df['tanggal'].dt.day
df['dayofyear'] = df['tanggal'].dt.dayofyear

# 4. Drop Kolom yang Tidak Relevan / Berisiko Overfit
# judul & penerima terlalu variatif. SKU adalah ID unik.
cols_to_drop = ['tanggal', 'judul', 'penerima', 'sku', 'kategorigabungan']
# Pastikan kolom ada sebelum di-drop
cols_to_drop = [c for c in cols_to_drop if c in df.columns]
df = df.drop(columns=cols_to_drop)

# 5. Handling Missing Values
for col in df.columns:
    if df[col].dtype == "object":
        # Isi kategori kosong dengan 'Unknown' atau modus
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        # Isi angka kosong dengan median
        df[col] = df[col].fillna(df[col].median())

# 6. Encoding (Ubah Kategori jadi Angka)
label_encoders = {}
cat_cols = df.select_dtypes(include=['object']).columns

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str)) # astype(str) jaga-jaga ada campuran tipe
    label_encoders[col] = le

# 7. Define X and y
# Pastikan target 'stock' ada
if 'stock' not in df.columns:
    raise ValueError("Kolom 'stock' tidak ditemukan!")

X = df.drop(columns=['stock'])
y = df['stock']

# 8. Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data siap! Shape X_train: {X_train.shape}")
print(f"Contoh data X:\n{X.head(3)}")

# Simpan data hasil proses (opsional, untuk pengecekan)
df.to_csv("data/processed.csv", index=False)
print("Preprocessing selesai. Siap untuk Random Forest.")