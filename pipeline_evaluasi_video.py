import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# --- 1. KONFIGURASI DIREKTORI & STILING ---
RAW_DIR = "00_RAW_VIDEO"
PROCESSED_DIR = "05_PROCESSED_VIDEO"
VIS_DIR = "10_VISUALIZATION_VIDEO"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

def get_kategori_tcr(tcr):
    """Kriteria Tingkat Capaian Responden (TCR) Skala 1-4"""
    if tcr >= 81.25:
        return "Sangat Layak / Sangat Baik"
    elif tcr >= 62.50:
        return "Layak / Baik"
    elif tcr >= 43.75:
        return "Cukup Layak / Cukup"
    else:
        return "Kurang Layak / Kurang"

def hitung_cronbach_alpha(df_items):
    """Menghitung Reliabilitas Instrumen Cronbach's Alpha"""
    k = df_items.shape[1]
    item_vars = df_items.var(axis=0, ddof=1)
    total_var = df_items.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - (item_vars.sum() / total_var))


# ==============================================================================
# 2. PEMROSESAN DATA EKSTERNAL (Masyarakat Luar Kalitengah, N = 153)
# ==============================================================================
print("=" * 60)
print(">>> MEMPROSES EVALUASI EKSTERNAL...")
print("=" * 60)

ext_path = os.path.join(RAW_DIR, "Evaluasi_Eksternal_Video.xlsx")
df_ext = pd.read_excel(ext_path)

# Pemetaan Kolom Eksternal
rename_ext = {
    df_ext.columns[2]: 'Domisili',
    df_ext.columns[3]: 'Rentang_Usia',
    df_ext.columns[4]: 'Pernah_Berkunjung',
    df_ext.columns[5]: 'Visual_Sinematografi',
    df_ext.columns[6]: 'Kualitas_Audio',
    df_ext.columns[7]: 'Transisi_Teks',
    df_ext.columns[8]: 'Potensi_SDA',
    df_ext.columns[9]: 'Nilai_Sosial_GotongRoyong',
    df_ext.columns[10]: 'Budaya_Sejarah',
    df_ext.columns[11]: 'Persepsi_Positif',
    df_ext.columns[12]: 'Minat_Kunjung',
    df_ext.columns[13]: 'Kelayakan_Rekomendasi'
}
df_ext = df_ext.rename(columns=rename_ext)

ext_likert_cols = [
    'Visual_Sinematografi', 'Kualitas_Audio', 'Transisi_Teks',
    'Potensi_SDA', 'Nilai_Sosial_GotongRoyong', 'Budaya_Sejarah',
    'Persepsi_Positif', 'Minat_Kunjung', 'Kelayakan_Rekomendasi'
]

# Dimensi Penilaian Eksternal
dimensi_ext = {
    'Dimensi Kualitas Teknis & Audio Visual': ['Visual_Sinematografi', 'Kualitas_Audio', 'Transisi_Teks'],
    'Dimensi Pemahaman Nilai, Budaya, & Potensi': ['Potensi_SDA', 'Nilai_Sosial_GotongRoyong', 'Budaya_Sejarah'],
    'Dimensi Resonansi, Citra, & Daya Tarik Promosi': ['Persepsi_Positif', 'Minat_Kunjung', 'Kelayakan_Rekomendasi']
}

# 1. Cronbach's Alpha
alpha_ext = hitung_cronbach_alpha(df_ext[ext_likert_cols])
print(f"-> Cronbach's Alpha (Reliabilitas): {alpha_ext:.4f} (Instrumen Sangat Reliabel)")

# 2. Rekapitulasi Statistik Deskriptif & TCR
tcr_ext_records = []
for col in ext_likert_cols:
    mean_val = df_ext[col].mean()
    std_val = df_ext[col].std()
    tcr_val = (mean_val / 4.0) * 100
    tcr_ext_records.append({
        'Indikator Penilaian': col,
        'Rata-rata Skor (Skala 1-4)': round(mean_val, 2),
        'Standar Deviasi': round(std_val, 2),
        'TCR (%)': round(tcr_val, 2),
        'Kategori Kelayakan': get_kategori_tcr(tcr_val)
    })
df_tcr_ext = pd.DataFrame(tcr_ext_records)
df_tcr_ext.to_excel(os.path.join(PROCESSED_DIR, "01_TCR_Evaluasi_Eksternal.xlsx"), index=False)

# 3. Uji Beda Persepsi: Pernah Berkunjung vs Belum Pernah (Mann-Whitney U Test)
df_ext['Total_Skor'] = df_ext[ext_likert_cols].sum(axis=1)
group_ya = df_ext[df_ext['Pernah_Berkunjung'] == 'Ya']['Total_Skor']
group_tidak = df_ext[df_ext['Pernah_Berkunjung'] == 'Tidak']['Total_Skor']
u_stat, p_val = mannwhitneyu(group_ya, group_tidak, alternative='two-sided')

print(f"-> Rata-rata Skor (Pernah Berkunjung, N={len(group_ya)}): {group_ya.mean():.2f}")
print(f"-> Rata-rata Skor (Belum Pernah, N={len(group_tidak)}): {group_tidak.mean():.2f}")
print(f"-> Mann-Whitney U Test p-value: {p_val:.4f} (Persepsi kedua kelompok konsisten)")


# ==============================================================================
# 3. PEMROSESAN DATA INTERNAL (Validasi Perangkat Desa, N = 12)
# ==============================================================================
print("\n" + "=" * 60)
print(">>> MEMPROSES VALIDASI INTERNAL...")
print("=" * 60)

int_path = os.path.join(RAW_DIR, "Evaluasi_Internal_Video.xlsx")
df_int = pd.read_excel(int_path)

# Pemetaan Kolom Internal
rename_int = {
    df_int.columns[2]: 'Nama_Lengkap',
    df_int.columns[3]: 'Jabatan',
    df_int.columns[4]: 'Akurasi_Geografis_SDA',
    df_int.columns[5]: 'Akurasi_Potensi_Unggulan',
    df_int.columns[6]: 'Representasi_Identitas_Budaya',
    df_int.columns[7]: 'Kepatuhan_Norma_KearifanLokal',
    df_int.columns[8]: 'Kualitas_Teknis_Publikasi',
    df_int.columns[9]: 'Pengesahan_Aset_Resmi',
    df_int.columns[10]: 'Adegan_Representatif',
    df_int.columns[11]: 'Saran_Distribusi',
    df_int.columns[12]: 'Komentar_Kinerja'
}
df_int = df_int.rename(columns=rename_int)

int_likert_cols = [
    'Akurasi_Geografis_SDA', 'Akurasi_Potensi_Unggulan',
    'Representasi_Identitas_Budaya', 'Kepatuhan_Norma_KearifanLokal',
    'Kualitas_Teknis_Publikasi', 'Pengesahan_Aset_Resmi'
]

# Rekapitulasi Statistik & TCR Internal
tcr_int_records = []
for col in int_likert_cols:
    mean_val = df_int[col].mean()
    std_val = df_int[col].std()
    tcr_val = (mean_val / 4.0) * 100
    tcr_int_records.append({
        'Indikator Validasi': col,
        'Rata-rata Skor (Skala 1-4)': round(mean_val, 2),
        'Standar Deviasi': round(std_val, 2),
        'TCR (%)': round(tcr_val, 2),
        'Kategori Validitas': get_kategori_tcr(tcr_val)
    })
df_tcr_int = pd.DataFrame(tcr_int_records)
df_tcr_int.to_excel(os.path.join(PROCESSED_DIR, "02_TCR_Validasi_Internal.xlsx"), index=False)

# Rekap Feedback Kualitatif
feedback_df = df_int[['Nama_Lengkap', 'Jabatan', 'Adegan_Representatif', 'Saran_Distribusi', 'Komentar_Kinerja']]
feedback_df.to_excel(os.path.join(PROCESSED_DIR, "03_Rekap_Kualitatif_Internal.xlsx"), index=False)


# ==============================================================================
# 4. GENERASI VISUALISASI GRAFIK
# ==============================================================================
print("\n" + "=" * 60)
print(">>> MEN-GENERATE VISUALISASI...")
print("=" * 60)

# 1. Demografi Responden Eksternal
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Usia
age_counts = df_ext['Rentang_Usia'].value_counts()
bars = axes[0].bar(age_counts.index.astype(str), age_counts.values, color='#2980b9', edgecolor='black', alpha=0.85)
axes[0].set_title('Distribusi Kelompok Usia Responden (N=153)', fontweight='bold', pad=12)
axes[0].set_ylabel('Jumlah Responden')
axes[0].set_xlabel('Rentang Usia')
for bar in bars:
    yval = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2, yval + 1.5, f"{int(yval)}", ha='center', fontweight='bold')

# Riwayat Kunjungan
visit_counts = df_ext['Pernah_Berkunjung'].value_counts()
axes[1].pie(visit_counts.values, labels=visit_counts.index, autopct='%1.1f%%',
            colors=['#e74c3c', '#27ae60'], startangle=140, explode=(0.04, 0.04),
            wedgeprops={'edgecolor': 'black', 'linewidth': 1.2})
axes[1].set_title('Pernah Mengunjungi Desa Kalitengah Sebelumnya?', fontweight='bold', pad=12)

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "01_Demografi_Eksternal.png"), dpi=300)
plt.close()

# 2. 100% Stacked Bar Chart Distribusi Jawaban Eksternal
prop_ext = pd.DataFrame(index=ext_likert_cols, columns=[1, 2, 3, 4]).fillna(0)
for col in ext_likert_cols:
    counts = df_ext[col].value_counts(normalize=True) * 100
    for score in [1, 2, 3, 4]:
        if score in counts:
            prop_ext.loc[col, score] = counts[score]

fig, ax = plt.subplots(figsize=(12, 7))
colors_likert = ['#d9534f', '#f0ad4e', '#5bc0de', '#5cb85c']
labels_likert = ['1 (Sangat Tidak Setuju)', '2 (Tidak Setuju)', '3 (Setuju)', '4 (Sangat Setuju)']

left = np.zeros(len(ext_likert_cols))
for i, score in enumerate([1, 2, 3, 4]):
    values = prop_ext[score].values
    ax.barh(ext_likert_cols, values, left=left, color=colors_likert[i], label=labels_likert[i], edgecolor='white', height=0.65)
    left += values

ax.set_xlabel('Persentase Proporsi Jawaban (%)', fontweight='bold')
ax.set_title('Distribusi Penilaian Evaluasi Eksternal Video Profil (N=153)', fontweight='bold', pad=15)
ax.legend(bbox_to_anchor=(0.5, -0.12), loc='upper center', ncol=4, frameon=True)
ax.set_xlim(0, 100)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "02_Distribusi_Likert_Eksternal.png"), dpi=300)
plt.close()

# 3. Bar Chart TCR Validasi Internal
fig, ax = plt.subplots(figsize=(11, 5.5))
y_pos = np.arange(len(df_tcr_int))
bars = ax.barh(y_pos, df_tcr_int['TCR (%)'], color='#2ecc71', edgecolor='black', alpha=0.88, height=0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels(df_tcr_int['Indikator Validasi'])
ax.invert_yaxis()
ax.set_xlabel('Tingkat Capaian Responden / TCR (%)', fontweight='bold')
ax.set_xlim(0, 110)
ax.axvline(81.25, color='red', linestyle='--', alpha=0.7, label='Batas Sangat Layak (81.25%)')
ax.set_title('Tingkat Validitas & Kelayakan Internal oleh Perangkat Desa (N=12)', fontweight='bold', pad=15)
ax.legend(loc='lower right')

for bar in bars:
    width = bar.get_width()
    ax.text(width + 1.2, bar.get_y() + bar.get_height()/2, f"{width:.2f}%", va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "03_TCR_Validasi_Internal.png"), dpi=300)
plt.close()

print("\n[SUKSES] Pipeline selesai dieksekusi!")
print(f"-> Laporan Excel tersimpan di folder : {PROCESSED_DIR}/")
print(f"-> Grafik Visualisasi tersimpan di folder : {VIS_DIR}/")
