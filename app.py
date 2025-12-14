import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb # Penting! Pastikan XGBoost di-import

# --- 1. CONFIGURASI & JUDUL ---
st.set_page_config(page_title="Production Fraud Detector", layout="wide")
st.title("🚨 Simulasi Detektor Fraud Produksi")
st.markdown("""
Sistem ini menggunakan model **XGBoost** yang sudah dilatih pada dataset PaySim untuk mendeteksi penipuan dalam transaksi `TRANSFER` dan `CASH_OUT`.
""")

# --- 2. PEMUATAN MODEL (Dibuat Cache) ---
# Gunakan st.cache_resource agar model hanya dimuat sekali saja
@st.cache_resource
def load_model():
    """Memuat model XGBoost dari file joblib."""
    try:
        # PENTING: Pastikan nama file sesuai dengan yang Anda simpan
        model = joblib.load('fraud_detector_xgb_model.joblib')
        return model
    except FileNotFoundError:
        st.error("File model 'fraud_detector_xgb_model.joblib' tidak ditemukan di direktori yang sama.")
        st.stop()
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

# Memuat model
model = load_model()
st.sidebar.success("Model XGBoost berhasil dimuat.")

# --- 3. FUNGSI PREPROCESSING INPUT BARU ---
def preprocess_input(step, type_input, amount, old_bal_orig, new_bal_orig, old_bal_dest, new_bal_dest):
    """
    Melakukan Feature Engineering dan Cleaning data input real-time
    agar sesuai dengan format saat model dilatih di Colab.
    """
    
    # a. Encoding Tipe Transaksi: TRANSFER=0, CASH_OUT=1
    type_val = 0 if type_input == 'TRANSFER' else 1
    
    # b. IMPUTASI: Mengubah saldo 0 yang mencurigakan menjadi -1
    # Ini sesuai dengan langkah 5 di Colab
    if old_bal_dest == 0 and new_bal_dest == 0 and amount != 0:
        old_bal_dest_proc = -1.0
        new_bal_dest_proc = -1.0
    else:
        old_bal_dest_proc = old_bal_dest
        new_bal_dest_proc = new_bal_dest

    # c. FEATURE ENGINEERING: Menghitung Error Balance (KUNCI AKURASI)
    error_orig = new_bal_orig + amount - old_bal_orig
    error_dest = old_bal_dest_proc + amount - new_bal_dest_proc
    
    # d. Membuat DataFrame dengan urutan kolom yang TEPAT
    # Urutan kolom harus sama persis dengan urutan saat training
    data = [[
        step, 
        type_val, 
        amount, 
        old_bal_orig, 
        new_bal_orig, 
        old_bal_dest_proc, # Gunakan nilai yang sudah diproses
        new_bal_dest_proc, # Gunakan nilai yang sudah diproses
        error_orig, 
        error_dest
    ]]
    
    # Kolom harus sesuai dengan urutan X.columns dari Colab:
    columns = [
        'step', 'type', 'amount', 'oldBalanceOrig', 'newBalanceOrig', 
        'oldBalanceDest', 'newBalanceDest', 'errorBalanceOrig', 'errorBalanceDest'
    ]
    
    return pd.DataFrame(data, columns=columns)

# --- 4. ANTARMUKA SIMULATOR PREDIKSI ---

st.header("Simulasi Transaksi Real-Time")
st.markdown("Masukkan detail transaksi untuk dianalisis oleh model.")

with st.form("prediction_form"):
    
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        st.subheader("Detail Transaksi")
        step = st.slider("Time Step (Jam ke-):", min_value=1, max_value=744, value=1)
        type_input = st.selectbox("Tipe Transaksi", ["TRANSFER", "CASH_OUT"])
        amount = st.number_input("Jumlah (Amount)", min_value=0.0, value=50000.0, step=100.0)
        
    with col_input2:
        st.subheader("Detail Saldo Akun")
        # Origin (Pengirim)
        old_bal_orig = st.number_input("Saldo Awal Pengirim (oldBalanceOrig)", min_value=0.0, value=100000.0)
        new_bal_orig = st.number_input("Saldo Akhir Pengirim (newBalanceOrig)", min_value=0.0, value=50000.0)
        st.caption("Pastikan Saldo Akhir - Saldo Awal ≈ -Jumlah")
        
        # Destination (Penerima)
        old_bal_dest = st.number_input("Saldo Awal Penerima (oldBalanceDest)", min_value=0.0, value=0.0)
        new_bal_dest = st.number_input("Saldo Akhir Penerima (newBalanceDest)", min_value=0.0, value=50000.0)
        st.caption("Kasus Fraud Seringkali melibatkan Saldo Awal dan Akhir Destinasi 0.")
        
    submit_btn = st.form_submit_button("ANALISIS FRAUD")

    if submit_btn:
        # Panggil fungsi preprocessing
        input_data_df = preprocess_input(
            step, type_input, amount, old_bal_orig, new_bal_orig, 
            old_bal_dest, new_bal_dest
        )
        
        # 1. PREDIKSI
        with st.spinner('Model sedang memproses...'):
            # Prediksi Probabilitas: nilai antara 0 (Aman) dan 1 (Fraud)
            probability = model.predict_proba(input_data_df)[0][1]
            # Prediksi Binary: 0 atau 1
            prediction = model.predict(input_data_df)[0]
        
        # 2. TAMPILKAN HASIL
        st.divider()
        if prediction == 1:
            st.error(f"❌ FRAUD DITEMUKAN! ({type_input} sebesar Rp {amount:,.0f})")
            st.header(f"Tingkat Risiko: {probability:.2%}")
            st.subheader("Tindakan: Blokir Transaksi dan Lakukan Investigasi.")
            st.markdown(
                f"""
                **Indikator Utama (Error Balance):**
                * Error Origin: `{input_data_df['errorBalanceOrig'].values[0]:.2f}`
                * Error Destination: `{input_data_df['errorBalanceDest'].values[0]:.2f}`
                """
            )
        else:
            st.success(f"✅ Transaksi DIIZINKAN. ({type_input} sebesar Rp {amount:,.0f})")
            st.header(f"Tingkat Risiko: {probability:.2%}")
            st.subheader("Tindakan: Lanjutkan Transaksi.")
            st.info("Risiko dianggap rendah berdasarkan pola fitur Error Balance yang normal.")
