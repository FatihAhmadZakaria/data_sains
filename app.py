import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

# ==========================================
# 1. KONFIGURASI FILE PATH
# ==========================================
# Pastikan file-file ini ada di folder 'models/'
MODEL_NORMAL_PATH = "models/model_imbalance.pkl"    # Klasifikasi (Imbalance)
MODEL_BALANCED_PATH = "models/model_balanced.pkl"   # Klasifikasi (Balanced)
MODEL_REGRESSION_PATH = "models/model_regression.pkl" # Regresi (Angka Pasti)
ENCODER_PATH = "models/encoder_final.pkl"           # Encoder (Sama untuk semua)

# Urutan Fitur (Wajib sama persis dengan saat training)
FEATURE_ORDER = [
    'lembartotal', 'jumlahcetak', 'hargabuku', 'kategoriutama',
    'servicefee', 'tahunterbit', 'kategorisekunder', 'dayofyear',
    'marketplacename', 'tipetransaksi'
]

# ==========================================
# 2. LOGIKA LOAD ASSETS
# ==========================================
def load_assets(model_path):
    try:
        model = joblib.load(model_path)
        # Kita pakai encoder yang sama karena mapping datanya konsisten
        encoders = joblib.load(ENCODER_PATH) 
        return model, encoders
    except FileNotFoundError:
        st.error(f"❌ Error: File model tidak ditemukan di path: {model_path}")
        st.error(f"Pastikan file .pkl sudah ada di folder 'models/'")
        return None, None

# ==========================================
# 3. MAIN PROGRAM
# ==========================================
def main():
    st.set_page_config(page_title="Prediksi Stok Buku AI", layout="centered", page_icon="📚")
    
    # --- SIDEBAR NAVIGASI ---
    with st.sidebar:
        st.title("🎛️ Control Panel")
        st.markdown("Pilih mode prediksi:")
        
        # Tombol Navigasi dengan Query Params
        if st.button("📊 Klasifikasi (Normal)", use_container_width=True):
            st.query_params["mode"] = "normal"
            st.rerun()
            
        if st.button("⚖️ Klasifikasi (Balanced)", use_container_width=True):
            st.query_params["mode"] = "balance"
            st.rerun()

        if st.button("📈 Regresi (Angka Pasti)", use_container_width=True):
            st.query_params["mode"] = "regression"
            st.rerun()

        st.markdown("---")
        st.caption("v2.0 - Multi-Model Support")

    # --- LOGIKA PENENTUAN MODE ---
    query_params = st.query_params
    current_mode = query_params.get("mode", "normal") # Default ke normal
    
    # Setup Variabel berdasarkan Mode
    if current_mode == "balance":
        active_model_path = MODEL_BALANCED_PATH
        mode_label = "⚖️ Balanced Mode (Klasifikasi)"
        is_regression = False
        st.toast("Mode Aktif: Balanced Classification", icon="⚖️")
        
    elif current_mode == "regression":
        active_model_path = MODEL_REGRESSION_PATH
        mode_label = "📈 Regression Mode (Prediksi Angka)"
        is_regression = True
        st.toast("Mode Aktif: Regression (Angka)", icon="📈")
        
    else: # Default normal
        active_model_path = MODEL_NORMAL_PATH
        mode_label = "📊 Normal Mode (Klasifikasi)"
        is_regression = False
        st.toast("Mode Aktif: Normal Classification", icon="📊")

    # --- TAMPILAN UTAMA ---
    st.title("📚 AI Stock Predictor")
    st.markdown(f"### Sedang menggunakan: **{mode_label}**")
    st.markdown("---")

    # Load Model
    model, encoders = load_assets(active_model_path)
    if not model: return

    # --- FORM INPUT ---
    with st.form("prediction_form"):
        st.subheader("📝 Input Spesifikasi Buku")
        
        c1, c2 = st.columns(2)
        
        with c1:
            hargabuku = st.number_input("Harga Buku (Rp)", value=50000, step=500)
            lembartotal = st.number_input("Jumlah Halaman", value=100)
            jumlahcetak = st.number_input("Riwayat Jumlah Cetak", value=1000)
            servicefee = st.number_input("Biaya Admin Marketplace (Rp)", value=0)
            tahunterbit = st.number_input("Tahun Terbit", min_value=1900, max_value=2030, value=2024)
            
        with c2:
            tgl_input = st.date_input("Tanggal Prediksi", datetime.now())
            dayofyear = tgl_input.timetuple().tm_yday 
            
            # Helper untuk ambil opsi kategori dari encoder
            def get_opt(key): 
                return encoders[key].classes_ if key in encoders else ["Unknown"]

            kategoriutama = st.selectbox("Kategori Utama", get_opt('kategoriutama'))
            kategorisekunder = st.selectbox("Kategori Sekunder", get_opt('kategorisekunder'))
            marketplacename = st.selectbox("Marketplace", get_opt('marketplacename'))
            tipetransaksi = st.selectbox("Tipe Transaksi", get_opt('tipetransaksi'))

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(f"🔍 Mulai Prediksi ({current_mode.upper()})", type="primary", use_container_width=True)

    # --- EKSEKUSI PREDIKSI ---
    if submitted:
        try:
            # 1. Siapkan Data
            input_data = {
                'hargabuku': hargabuku, 'lembartotal': lembartotal, 'jumlahcetak': jumlahcetak,
                'servicefee': servicefee, 'tahunterbit': tahunterbit, 'dayofyear': dayofyear
            }
            
            # 2. Encoding (Ubah teks jadi angka pakai encoder yg sudah disimpan)
            input_data['kategoriutama'] = encoders['kategoriutama'].transform([kategoriutama])[0]
            input_data['kategorisekunder'] = encoders['kategorisekunder'].transform([kategorisekunder])[0]
            input_data['marketplacename'] = encoders['marketplacename'].transform([marketplacename])[0]
            input_data['tipetransaksi'] = encoders['tipetransaksi'].transform([tipetransaksi])[0]

            # 3. Urutkan vektor sesuai fitur training
            final_vector = [input_data[f] for f in FEATURE_ORDER]
            
            # 4. Prediksi
            prediction = model.predict([final_vector])[0]
            
            st.divider()
            st.subheader("💡 Hasil Analisis AI")

            # --- TAMPILAN HASIL (Cabang Logika) ---
            if is_regression:
                # === HASIL REGRESI (ANGKA) ===
                pred_value = float(prediction)
                st.metric(label="Estimasi Stok Optimal", value=f"{int(pred_value)} Eksemplar")
                
                # Visualisasi sederhana bar progres
                st.progress(min(pred_value / 500, 1.0)) # Asumsi max stok 500 utk visualisasi
                st.caption(f"Prediksi model regresi: {pred_value:.2f}")

            else:
                # === HASIL KLASIFIKASI (LABEL) ===
                # prediction is string: "Tinggi", "Sedang", "Rendah"
                
                col_res, col_info = st.columns([1, 2])
                
                with col_res:
                    if prediction == "Tinggi":
                        st.success(f"## {prediction}")
                    elif prediction == "Sedang":
                        st.warning(f"## {prediction}")
                    else:
                        st.error(f"## {prediction}")
                
                with col_info:
                    if prediction == "Tinggi":
                        st.info("📈 **Fast Moving!** Buku ini diprediksi sangat laris. Disarankan stok > 50 eks.")
                    elif prediction == "Sedang":
                        st.warning("⚖️ **Normal.** Perputaran stok stabil. Disarankan stok 6 - 50 eks.")
                    else:
                        st.error("📉 **Slow Moving.** Hati-hati overstock. Disarankan stok < 5 eks.")

            st.caption(f"Source Model: {active_model_path}")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses data: {e}")
            st.warning("Tips: Pastikan semua file model (.pkl) sudah digenerate ulang dengan fitur yang sama.")

if __name__ == "__main__":
    main()