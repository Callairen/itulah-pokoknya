import pandas as pd
import numpy as np
import os

def run_digikalt_pipeline(file_path):
    print(f"Mencari file: {file_path}")
    if not os.path.exists(file_path):
        print("❌ File Excel tidak ditemukan! Pastikan nama file dan foldernya benar.")
        return

    print("Membaca data Excel...")
    pre_df = pd.read_excel(file_path, sheet_name='01_Pretest')
    post_df = pd.read_excel(file_path, sheet_name='02_Posttest')
    
    # 1. Merge Data
    df = pd.merge(pre_df, post_df, on=['ID', 'Nama', 'Nama_Usaha', 'Jenis_Usaha'], suffixes=('_pre', '_post'))
    
    # 2. Definisi Pilar Konstruk
    constructs = {
        'Literasi Branding & Visual': ['P1', 'P2', 'P3'],
        'Literasi Platform Digital': ['P4', 'P5', 'P6'],
        'Kesiapan Strategis & Adopsi': ['P7', 'P8']
    }
    
    results = []
    
    # 3. Proses Perhitungan
    for construct_name, questions in constructs.items():
        pre_cols = [f"{q}_pre" for q in questions]
        post_cols = [f"{q}_post" for q in questions]
        
        pre_mean = df[pre_cols].mean().mean()
        post_mean = df[post_cols].mean().mean()
        
        gain = post_mean - pre_mean
        denominator = 4 - pre_mean
        n_gain = gain / denominator if denominator != 0 else 0
        
        results.append({
            'Pilar Analisis': construct_name,
            'Skor Pre-Test': round(pre_mean, 2),
            'Skor Post-Test': round(post_mean, 2),
            'Peningkatan (Gain)': round(gain, 2),
            'N-Gain Score': round(n_gain, 2)
        })
        
    df_constructs = pd.DataFrame(results)
    print("\n=== HASIL ANALISIS BERDASARKAN PILAR ===")
    print(df_constructs)
    
    # 4. Simpan ke dalam Excel
    print("\nMenyimpan hasil ke dalam Excel...")
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df_constructs.to_excel(writer, sheet_name='15_Construct_Analysis', index=False)
        
    print("✅ Selesai! Data berhasil ditulis ke sheet '15_Construct_Analysis'.")

if __name__ == "__main__":
    # Sesuaikan dengan nama file Excel aslimu
    NAMA_FILE_EXCEL = 'DIGIKALT Data.xlsx'
    run_digikalt_pipeline(NAMA_FILE_EXCEL)