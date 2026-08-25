import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
import os
import warnings

warnings.filterwarnings('ignore')

class EvaluasiDigikaltAdvanced:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        
        # Setup direktori output
        self.dirs = ['DIGIKALT_ADVANCED_OUTPUT/STATISTICS', 'DIGIKALT_ADVANCED_OUTPUT/VISUALIZATIONS']
        for d in self.dirs:
            os.makedirs(d, exist_ok=True)

    def load_and_clean_data(self):
        print("[1/6] Membaca data Excel...")
        pre_df = pd.read_excel(self.file_path, sheet_name='01_Pretest')
        post_df = pd.read_excel(self.file_path, sheet_name='02_Posttest')
        
        # Standarisasi
        pre_df.rename(columns=lambda x: str(x).strip(), inplace=True)
        post_df.rename(columns=lambda x: str(x).strip(), inplace=True)
        
        pre_cols = {c: f'Pre_{c}' for c in pre_df.columns if c.startswith('P')}
        post_cols = {c: f'Post_{c}' for c in post_df.columns if c.startswith('P')}
        pre_df.rename(columns=pre_cols, inplace=True)
        post_df.rename(columns=post_cols, inplace=True)
        
        self.df = pd.merge(pre_df, post_df[['ID'] + list(post_cols.values())], on='ID', how='inner')
        
        # Skor Total
        self.df['Pre_Total'] = self.df[list(pre_cols.values())].sum(axis=1)
        self.df['Post_Total'] = self.df[list(post_cols.values())].sum(axis=1)
        
        # N-Gain Hake
        max_score = len(pre_cols) * 4
        self.df['Gain_Score'] = self.df['Post_Total'] - self.df['Pre_Total']
        self.df['N_Gain'] = (self.df['Post_Total'] - self.df['Pre_Total']) / (max_score - self.df['Pre_Total'])
        self.df['N_Gain'] = self.df['N_Gain'].replace([np.inf, -np.inf], 0).fillna(0)

    def run_hypothesis_and_stats(self):
        print("[2/6] Mengeksekusi Uji Hipotesis & N-Gain...")
        stat_sw, p_sw = stats.shapiro(self.df['Gain_Score'])
        is_normal = p_sw > 0.05
        
        if is_normal:
            stat, p_val = stats.ttest_rel(self.df['Pre_Total'], self.df['Post_Total'])
            test_used = "Paired T-Test (Parametrik)"
        else:
            stat, p_val = stats.wilcoxon(self.df['Pre_Total'], self.df['Post_Total'])
            test_used = "Wilcoxon Signed-Rank Test (Non-Parametrik)"

        with open('DIGIKALT_ADVANCED_OUTPUT/STATISTICS/01_Uji_Signifikansi.txt', 'w') as f:
            f.write("=== UJI HIPOTESIS PERBEDAAN RATA-RATA ===\n")
            f.write(f"Distribusi Data (Shapiro-Wilk P-Value) : {p_sw:.4f} ({'Normal' if is_normal else 'Tidak Normal'})\n")
            f.write(f"Metode Uji : {test_used}\n")
            f.write(f"P-Value    : {p_val:.5f} ({'Signifikan' if p_val < 0.05 else 'Tidak Signifikan'})\n")
            f.write(f"Mean N-Gain: {self.df['N_Gain'].mean():.3f}\n")

    def plot_violin_and_boxplot(self):
        print("[3/6] Membuat Distribusi Kepadatan (Violin Plot) & Kuartil (Boxplot)...")
        data_melted = pd.DataFrame({
            'Fase': ['Pre-Test']*len(self.df) + ['Post-Test']*len(self.df),
            'Skor': list(self.df['Pre_Total']) + list(self.df['Post_Total'])
        })
        
        plt.figure(figsize=(10, 6))
        # 1. Violin Plot
        sns.violinplot(x='Fase', y='Skor', data=data_melted, inner=None, color=".8", alpha=0.5)
        # 2. Boxplot di dalam Violin
        sns.boxplot(x='Fase', y='Skor', data=data_melted, width=0.2, palette="muted", zorder=10)
        plt.title('Distribusi Kepadatan & Deviasi Kuartil Nilai UMKM', pad=20, fontweight='bold')
        plt.savefig('DIGIKALT_ADVANCED_OUTPUT/VISUALIZATIONS/01_Violin_Boxplot.png', dpi=300)
        plt.close()

    def plot_slopegraph(self):
        print("[4/6] Membuat Trajektori Nilai Individu (Slopegraph)...")
        plt.figure(figsize=(6, 8))
        
        for idx, row in self.df.iterrows():
            plt.plot(['Pre-Test', 'Post-Test'], [row['Pre_Total'], row['Post_Total']], 
                     marker='o', markersize=8, linewidth=2, alpha=0.7)
            # Anotasi ID
            plt.text(1.05, row['Post_Total'], row['ID'], fontsize=9, va='center')
            
        plt.title('Trajektori Perubahan Skor Individu UMKM (Slopegraph)', fontweight='bold')
        plt.ylabel('Total Skor (Maks 32)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('DIGIKALT_ADVANCED_OUTPUT/VISUALIZATIONS/02_Slopegraph.png', dpi=300, bbox_inches='tight')
        plt.close()

    def run_logistic_regression(self):
        print("[5/6] Memodelkan Kurva Probabilitas Klasifikasi (Logistic Regression)...")
        # Definisi "Sukses" = N-Gain >= 0.3
        self.df['Success'] = (self.df['N_Gain'] >= 0.3).astype(int)
        
        X = self.df[['Pre_Total']].values
        y = self.df['Success'].values
        
        if len(np.unique(y)) > 1: # Memastikan ada variasi data sukses dan tidak
            model = LogisticRegression()
            model.fit(X, y)
            
            X_test = np.linspace(X.min() - 2, X.max() + 5, 300).reshape(-1, 1)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            plt.figure(figsize=(8, 5))
            plt.scatter(X, y, color='black', zorder=20, label='Data Aktual UMKM')
            plt.plot(X_test, y_prob, color='red', linewidth=3, label='Kurva Probabilitas Sukses')
            plt.axhline(0.5, color='gray', linestyle='--')
            plt.title('Kurva Probabilitas Keberhasilan Adopsi Digital UMKM\n(Logistic Regression)', fontweight='bold')
            plt.xlabel('Skor Awal (Pre-Test)')
            plt.ylabel('Probabilitas Sukses (N-Gain > 0.3)')
            plt.legend()
            plt.savefig('DIGIKALT_ADVANCED_OUTPUT/VISUALIZATIONS/03_Logistic_Regression.png', dpi=300)
            plt.close()
        else:
            print("  -> Peringatan: Logistic Regression dilewati (Semua UMKM Sukses/Semua Gagal, tidak ada variasi untuk membuat kurva).")

    def run_monte_carlo_simulation(self):
        print("[6/6] Menjalankan Simulasi Stokastik Ekspektasi Masa Depan (Monte Carlo)...")
        # Simulasi 10,000 UMKM di masa depan berdasarkan distibusi saat ini
        mean_gain = self.df['Gain_Score'].mean()
        std_gain = self.df['Gain_Score'].std()
        
        # Generate 10000 random samples from normal distribution
        np.random.seed(42)
        simulated_gains = np.random.normal(mean_gain, std_gain, 10000)
        
        plt.figure(figsize=(9, 5))
        sns.histplot(simulated_gains, bins=50, color='purple', kde=True, alpha=0.5)
        plt.axvline(mean_gain, color='red', linestyle='dashed', linewidth=2, label=f'Ekspektasi Rata-rata: {mean_gain:.2f}')
        plt.title('Simulasi Stokastik Monte Carlo (n=10,000 UMKM)\nPrediksi Peningkatan Poin di Masa Depan', fontweight='bold')
        plt.xlabel('Prediksi Gain Score')
        plt.ylabel('Frekuensi Probabilitas')
        plt.legend()
        plt.savefig('DIGIKALT_ADVANCED_OUTPUT/VISUALIZATIONS/04_Monte_Carlo_Simulation.png', dpi=300)
        plt.close()

    def execute_all(self):
        self.load_and_clean_data()
        self.run_hypothesis_and_stats()
        self.plot_violin_and_boxplot()
        self.plot_slopegraph()
        self.run_logistic_regression()
        self.run_monte_carlo_simulation()
        print("\n🎉 SELURUH 8 PROSES ADVANCED ANALYTICS SELESAI!")

if __name__ == "__main__":
    FILE_EXCEL = "DIGIKALT Data.xlsx" 
    analyzer = EvaluasiDigikaltAdvanced(FILE_EXCEL)
    analyzer.execute_all()