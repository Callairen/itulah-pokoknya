import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings

warnings.filterwarnings('ignore')

class EvaluasiUMKMPipeline:
    def __init__(self, file_path):
        self.file_path = file_path
        self.group = 'UMKM'
        self.pre_sheet = '01_Pretest'
        self.post_sheet = '02_Posttest'
        self.processed_data = {}

        self.dirs = [
            'DIGIKALT_05_PROCESSED',
            'DIGIKALT_08_DESCRIPTIVE',
            'DIGIKALT_09_STATISTICS',
            'DIGIKALT_10_VISUALIZATION'
        ]
        for d in self.dirs:
            os.makedirs(d, exist_ok=True)

    def clean_and_merge(self):
        print("Tahap 1: Membaca dan Menggabungkan Data...")

        df_pre = pd.read_excel(self.file_path, sheet_name=self.pre_sheet)
        df_post = pd.read_excel(self.file_path, sheet_name=self.post_sheet)

        df_pre.rename(columns=lambda x: x.strip(), inplace=True)
        df_post.rename(columns=lambda x: x.strip(), inplace=True)
        df_pre.rename(columns={'Id': 'ID', 'id': 'ID'}, inplace=True)
        df_post.rename(columns={'Id': 'ID', 'id': 'ID'}, inplace=True)

        items_pre = [c for c in df_pre.columns if c.startswith('P') and c[1:].isdigit()]
        items_post = [c for c in df_post.columns if c.startswith('P') and c[1:].isdigit()]

        info_cols = ['ID'] + [c for c in df_pre.columns if c not in items_pre and c != 'ID']
        df_pre_info = df_pre[info_cols].copy()

        rename_pre = {c: f'Pre_{c}' for c in items_pre}
        rename_post = {c: f'Post_{c}' for c in items_post}

        df_pre.rename(columns=rename_pre, inplace=True)
        df_post.rename(columns=rename_post, inplace=True)

        df_merged = pd.merge(df_pre, df_post[['ID'] + list(rename_post.values())],
                             on='ID', how='inner')

        self.processed_data[self.group] = df_merged

    def reverse_coding_and_scoring(self):
        print("Tahap 2: Eksekusi Scoring...")

        reverse_rules = {}

        for group, df in self.processed_data.items():
            if group in reverse_rules:
                for col in reverse_rules[group]:
                    if col in df.columns:
                        df[col] = 5 - df[col]

            pre_cols = sorted([c for c in df.columns if c.startswith('Pre_P')])
            post_cols = sorted([c for c in df.columns if c.startswith('Post_P')])

            df['Pre_Total'] = df[pre_cols].sum(axis=1)
            df['Pre_Mean'] = df[pre_cols].mean(axis=1)
            df['Post_Total'] = df[post_cols].sum(axis=1)
            df['Post_Mean'] = df[post_cols].mean(axis=1)
            df['Gain_Score'] = df['Post_Total'] - df['Pre_Total']

            max_possible_score = len(pre_cols) * 4
            df['N_Gain'] = (df['Post_Total'] - df['Pre_Total']) / (max_possible_score - df['Pre_Total'])
            df['N_Gain'] = df['N_Gain'].replace([np.inf, -np.inf], 0).fillna(0)

            self.processed_data[group] = df

            df.to_excel(f"DIGIKALT_05_PROCESSED/DIGIKALT_PROCESSED_{group}.xlsx", index=False)

    def calculate_reliability(self, df, columns):
        if len(columns) < 2:
            return 0
        df_items = df[columns]
        k = df_items.shape[1]
        var_items = df_items.var(ddof=1).sum()
        var_total = df_items.sum(axis=1).var(ddof=1)
        if var_total == 0:
            return 0
        return (k / (k - 1)) * (1 - (var_items / var_total))

    def run_statistics(self):
        print("Tahap 3: Memproses Analisis Statistik...")

        with open('DIGIKALT_09_STATISTICS/Statistik_Inferensial_UMKM.txt', 'w') as f:
            for group, df in self.processed_data.items():
                pre_cols = sorted([c for c in df.columns if c.startswith('Pre_P')])
                post_cols = sorted([c for c in df.columns if c.startswith('Post_P')])

                alpha_pre = self.calculate_reliability(df, pre_cols)
                alpha_post = self.calculate_reliability(df, post_cols)

                stat_sw, p_sw = stats.shapiro(df['Gain_Score'])
                is_normal = p_sw > 0.05

                if is_normal:
                    stat_test, p_test = stats.ttest_rel(df['Pre_Total'], df['Post_Total'])
                    test_name = "Paired T-Test (Parametrik)"
                else:
                    stat_test, p_test = stats.wilcoxon(df['Pre_Total'], df['Post_Total'])
                    test_name = "Wilcoxon Signed-Rank Test (Non-Parametrik)"

                mean_ngain = df['N_Gain'].mean()
                if mean_ngain >= 0.7:
                    category_ngain = "Tinggi"
                elif mean_ngain >= 0.3:
                    category_ngain = "Sedang"
                else:
                    category_ngain = "Rendah"

                f.write("=========================================\n")
                f.write(f"       ANALISIS KELOMPOK {group.upper()}\n")
                f.write("=========================================\n")
                f.write(f"N (Jumlah Responden) : {len(df)}\n\n")
                f.write("[RELIABILITAS INSTRUMEN]\n")
                f.write(f"- Cronbach's Alpha Pretest  : {alpha_pre:.3f}\n")
                f.write(f"- Cronbach's Alpha Posttest : {alpha_post:.3f}\n\n")
                f.write("[UJI NORMALITAS GAIN SCORE]\n")
                f.write(f"- P-Value Shapiro-Wilk      : {p_sw:.4f} -> {'Distribusi Normal' if is_normal else 'Distribusi Tidak Normal'}\n\n")
                f.write("[UJI HIPOTESIS PRE VS POST]\n")
                f.write(f"- Metode Uji                : {test_name}\n")
                f.write(f"- P-Value (Signifikansi)    : {p_test:.5f} -> {'Signifikan (Terdapat Perbedaan)' if p_test < 0.05 else 'Tidak Signifikan'}\n\n")
                f.write("[EFEKTIVITAS SOSIALISASI]\n")
                f.write(f"- Rata-rata N-Gain          : {mean_ngain:.3f} (Kategori: {category_ngain})\n")
                f.write(f"- Rata-rata Gain Score      : {df['Gain_Score'].mean():.2f}\n")
                f.write(f"- Rata-rata Pre_Total       : {df['Pre_Total'].mean():.2f}\n")
                f.write(f"- Rata-rata Post_Total      : {df['Post_Total'].mean():.2f}\n\n")

                desc_cols = [c for c in df.columns if c.startswith('Pre_P') or c.startswith('Post_P')]
                desc_df = df[desc_cols].describe().T
                desc_df.to_excel(f"DIGIKALT_08_DESCRIPTIVE/Tabel_Deskriptif_{group}.xlsx")

    def generate_visualizations(self):
        print("Tahap 4: Menggambar Visualisasi Data...")
        sns.set_theme(style="whitegrid", palette="muted")

        for group, df in self.processed_data.items():
            plt.figure(figsize=(7, 5))
            means = [df['Pre_Mean'].mean(), df['Post_Mean'].mean()]
            ax = sns.barplot(x=['Pretest', 'Posttest'], y=means, palette=['#e74c3c', '#2ecc71'])
            plt.title(f'Komparasi Rata-rata Skor Item - Kelompok {group}', fontweight='bold')
            plt.ylabel('Skor Rata-rata (Skala 1-4)')
            plt.ylim(0, 4.5)
            for p in ax.patches:
                ax.annotate(format(p.get_height(), '.2f'),
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center',
                            xytext=(0, 9), textcoords='offset points', fontweight='bold')
            plt.savefig(f"DIGIKALT_10_VISUALIZATION/01_Bar_PrePost_{group}.png", dpi=300, bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(8, 5))
            sns.histplot(df['Gain_Score'], bins=8, kde=True, color='#3498db', edgecolor='black')
            plt.title(f'Distribusi Gain Score (Peningkatan) - Kelompok {group}', fontweight='bold')
            plt.xlabel('Poin Peningkatan (Post_Total - Pre_Total)')
            plt.ylabel('Jumlah Responden')
            plt.axvline(df['Gain_Score'].mean(), color='red', linestyle='dashed',
                        linewidth=2, label=f"Mean: {df['Gain_Score'].mean():.2f}")
            plt.legend()
            plt.savefig(f"DIGIKALT_10_VISUALIZATION/02_Hist_Gain_{group}.png", dpi=300, bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(8, 5))
            data_to_plot = pd.DataFrame({
                'Fase': ['Pretest'] * len(df) + ['Posttest'] * len(df),
                'Skor Total': list(df['Pre_Total']) + list(df['Post_Total'])
            })
            sns.boxplot(x='Fase', y='Skor Total', data=data_to_plot, palette=['#ff9999', '#99ff99'])
            sns.stripplot(x='Fase', y='Skor Total', data=data_to_plot, color='black', alpha=0.4, jitter=True)
            plt.title(f'Sebaran Data Skor Total Responden - Kelompok {group}', fontweight='bold')
            plt.savefig(f"DIGIKALT_10_VISUALIZATION/03_Boxplot_Distribusi_{group}.png", dpi=300, bbox_inches='tight')
            plt.close()

    def run_all(self):
        self.clean_and_merge()
        self.reverse_coding_and_scoring()
        self.run_statistics()
        self.generate_visualizations()
        print("\n[SUCCESS] Pipeline DIGIKALT berhasil dieksekusi.")


if __name__ == "__main__":
    FILE_EXCEL = "DIGIKALT Data.xlsx"
    pipeline = EvaluasiUMKMPipeline(FILE_EXCEL)
    pipeline.run_all()
