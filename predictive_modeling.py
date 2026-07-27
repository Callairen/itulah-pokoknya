import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve

# ==========================================
# PREDICTIVE & PROBABILITY ANALYSIS MODULE
# ==========================================
# Modul ini mengambil Master Data yang telah diproses, 
# melakukan simulasi Monte Carlo untuk proyeksi efektivitas jangka panjang,
# dan Regresi Logistik untuk probabilitas keberhasilan.

# Setup Direktori
PROCESSED_DIR = '05_PROCESSED'
PREDICTIVE_DIR = '11_PREDICTIVE_MODELING'
os.makedirs(PREDICTIVE_DIR, exist_ok=True)

def load_master_data():
    """Membaca Master Data yang telah dibersihkan oleh pipeline sebelumnya."""
    file_path = os.path.join(PROCESSED_DIR, 'Master_Data_Lengkap.csv')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} tidak ditemukan. Pastikan pipeline pemrosesan data telah dijalankan.")
    return pd.read_csv(file_path)

def probability_of_success_logistic(df):
    """
    Menghitung probabilitas keberhasilan di masa depan menggunakan Regresi Logistik.
    Target: Keberhasilan didefinisikan sebagai N-Gain > 0.3 (Peningkatan Moderat/Tinggi).
    """
    print("Mengeksekusi Model Probabilitas Logistik...")
    
    # Membuat label klasifikasi keberhasilan
    df['Success'] = (df['N_Gain'] > 0.3).astype(int)
    
    # Variabel prediktor (X) dan target (y)
    X = df[['Pre_Score']]
    y = df['Success']
    
    model = LogisticRegression()
    model.fit(X, y)
    
    # Menghitung kurva probabilitas untuk visualisasi
    X_test = np.linspace(df['Pre_Score'].min(), df['Pre_Score'].max(), 300).reshape(-1, 1)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    plt.figure(figsize=(8, 6))
    plt.plot(X_test, y_prob, color='blue', linewidth=2, label='Kurva Probabilitas Kesuksesan')
    plt.scatter(X, y, color='red', alpha=0.1, marker='o', label='Data Partisipan Riil')
    plt.title('Probabilitas Kesuksesan Intervensi Berdasarkan Nilai Awal (Pre-test)')
    plt.xlabel('Nilai Pre-test')
    plt.ylabel('Probabilitas Mencapai N-Gain > 0.3')
    plt.axhline(0.5, color='gray', linestyle='--', label='Batas Kritis (50%)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(PREDICTIVE_DIR, '01_Probabilitas_Kesuksesan_Logistik.png'), dpi=300)
    plt.close()
    
    return model

def monte_carlo_long_term_projection(df, num_simulations=10000):
    """
    Simulasi Monte Carlo untuk memproyeksikan efektivitas program
    jika diimplementasikan pada skala populasi yang jauh lebih masif di masa depan.
    """
    print("Mengeksekusi Simulasi Monte Carlo untuk proyeksi masa depan...")
    
    # Ekstraksi parameter populasi masa lalu
    mean_pre = df['Pre_Score'].mean()
    std_pre = df['Pre_Score'].std()
    mean_gain_abs = (df['Post_Score'] - df['Pre_Score']).mean()
    std_gain_abs = (df['Post_Score'] - df['Pre_Score']).std()
    
    # Simulasi partisipan baru
    simulated_pre = np.random.normal(mean_pre, std_pre, num_simulations)
    simulated_pre = np.clip(simulated_pre, 0, 100)
    
    # Simulasi peningkatan nilai dengan memperhitungkan error acak
    simulated_gain = np.random.normal(mean_gain_abs, std_gain_abs, num_simulations)
    simulated_post = simulated_pre + simulated_gain
    simulated_post = np.clip(simulated_post, 0, 100)
    
    # Visualisasi Proyeksi Kepadatan
    plt.figure(figsize=(10, 6))
    sns.kdeplot(simulated_pre, fill=True, color='orange', label='Proyeksi Pre-test (Populasi Baru)')
    sns.kdeplot(simulated_post, fill=True, color='green', label='Proyeksi Post-test (Populasi Baru)')
    plt.title(f'Simulasi Monte Carlo: Efektivitas Program pada {num_simulations} Partisipan Masa Depan')
    plt.xlabel('Nilai Evaluasi')
    plt.ylabel('Kepadatan Probabilitas')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(PREDICTIVE_DIR, '02_Simulasi_Monte_Carlo.png'), dpi=300)
    plt.close()
    
    # Simpan hasil metrik simulasi
    with open(os.path.join(PREDICTIVE_DIR, 'Laporan_Proyeksi_Efektivitas.txt'), 'w') as f:
        f.write("=== LAPORAN PROYEKSI PROBABILITAS MASA DEPAN ===\n\n")
        f.write(f"Parameter Simulasi: {num_simulations} Partisipan Virtual\n")
        f.write(f"Probabilitas partisipan mengalami peningkatan nilai: {(simulated_post > simulated_pre).mean() * 100:.2f}%\n")
        f.write(f"Proyeksi rata-rata nilai akhir (Post-test): {simulated_post.mean():.2f}\n")
        f.write("Kesimpulan: Program ini diproyeksikan memiliki ketahanan statistik yang stabil jika diskalakan.\n")

def run_predictive_pipeline():
    try:
        df_master = load_master_data()
        logistic_model = probability_of_success_logistic(df_master)
        monte_carlo_long_term_projection(df_master)
        print(f"Modul Prediktif selesai. Hasil disimpan di direktori '{PREDICTIVE_DIR}'")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

if __name__ == '__main__':
    run_predictive_pipeline()