import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

# ==========================================
# KONFIGURASI FILE PATH
# ==========================================
MODEL_NORMAL_PATH = "models/model_imbalance.pkl"    # Model Imbalance / Normal
MODEL_BALANCED_PATH = "models/model_balanced.pkl"   # Model Balanced (SMOTE/RandomUnderSampler)
ENCODER_PATH = "models/encoder_final.pkl"

# Urutan Fitur
FEATURE_ORDER = [
    'lembartotal', 'jumlahcetak', 'hargabuku', 'kategoriutama',
    'servicefee', 'tahunterbit', 'kategorisekunder', 'dayofyear',
    'marketplacename', 'tipetransaksi'
]

# ==========================================
# LOGIKA LOAD ASSETS (DINAMIS)
# ==========================================
def load_assets(model_path):
    try:
        model = joblib.load(model_path)
        encoders = joblib.load(ENCODER_PATH)
        return model, encoders
    except FileNotFoundError:
        st.error(f"Error: File {model_path} atau {ENCODER_PATH} tidak ditemukan.")
        return None, None

def main():
    st.set_page_config(page_title="Prediksi Stok Buku", layout="centered")
    with st.sidebar:
        st.title("Menu Cepat")
        if st.button("Lihat Data Model Balance"):
            st.query_params["mode"] = "balance"
            st.rerun() # Muat ulang aplikasi dengan param baru
        
        if st.button("Lihat Data Model Normal"):
            st.query_params["mode"] = "normal"
            st.rerun()
    # --- 1. DETEKSI ROUTE / MODE ---
    query_params = st.query_params
    current_mode = query_params.get("mode", "normal")

    # --- 2. TENTUKAN MODEL YANG DIPAKAI ---
    if current_mode == "balance":
        active_model_path = MODEL_BALANCED_PATH
        mode_label = "⚖️ Balanced Mode"
        st.toast("Menggunakan Model Balanced (SMOTE)", icon="⚖️")
    else:
        active_model_path = MODEL_NORMAL_PATH
        mode_label = "📊 Normal Mode"

    # Header
    st.title("📚 Sistem Prediksi Stok (AI)")
    st.markdown("---")

    # Load Model sesuai route
    model, encoders = load_assets(active_model_path)
    if not model: return

    # --- FORM INPUT ---
    with st.form("prediction_form"):
        st.subheader("Data Spesifikasi Buku")
        c1, c2 = st.columns(2)
        
        with c1:
            hargabuku = st.number_input("Harga Buku (Rp)", value=50000, step=500)
            lembartotal = st.number_input("Jumlah Halaman", value=100)
            jumlahcetak = st.number_input("Riwayat Jumlah Cetak", value=1000)
            servicefee = st.number_input("Biaya Admin (Rp)", value=0)
            tahunterbit = st.number_input("Tahun Terbit", min_value=1900, max_value=2030, value=2024)
            
        with c2:
            tgl_input = st.date_input("Tanggal Prediksi", datetime.now())
            dayofyear = tgl_input.timetuple().tm_yday 
            
            def get_opt(key): return encoders[key].classes_ if key in encoders else ["Unknown"]

            kategoriutama = st.selectbox("Kategori Utama", get_opt('kategoriutama'))
            kategorisekunder = st.selectbox("Kategori Sekunder", get_opt('kategorisekunder'))
            marketplacename = st.selectbox("Marketplace", get_opt('marketplacename'))
            tipetransaksi = st.selectbox("Tipe Transaksi", get_opt('tipetransaksi'))

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(f"🔍 Analisis ({current_mode.upper()})", type="primary", use_container_width=True)

    # --- PROSES PREDIKSI ---
    if submitted:
        try:
            input_data = {
                'hargabuku': hargabuku, 'lembartotal': lembartotal, 'jumlahcetak': jumlahcetak,
                'servicefee': servicefee, 'tahunterbit': tahunterbit, 'dayofyear': dayofyear
            }
            
            # Transform
            input_data['kategoriutama'] = encoders['kategoriutama'].transform([kategoriutama])[0]
            input_data['kategorisekunder'] = encoders['kategorisekunder'].transform([kategorisekunder])[0]
            input_data['marketplacename'] = encoders['marketplacename'].transform([marketplacename])[0]
            input_data['tipetransaksi'] = encoders['tipetransaksi'].transform([tipetransaksi])[0]

            final_vector = [input_data[f] for f in FEATURE_ORDER]
            
            # Predict menggunakan model yang sedang aktif
            prediction = model.predict([final_vector])[0]
            
            st.success("Analisis Selesai!")
            st.markdown(f"### Hasil: **{prediction}**")
            st.markdown(f"Model yang digunakan: `{active_model_path}`")

            if prediction == "Tinggi":
                st.info("📈 Potensi Laris Manis! Stok > 50 eks.")
            elif prediction == "Sedang":
                st.warning("⚖️ Permintaan Stabil. Stok 6 - 50 eks.")
            else:
                st.error("📉 Perputaran Lambat. Stok 0 - 5 eks.")

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

if __name__ == "__main__":
    main()