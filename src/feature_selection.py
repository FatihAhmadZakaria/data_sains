import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import sys

# Konfigurasi Output
OUTPUT_FILE = "./output/hasil_feature_selection.txt"

def log_print(text, file_obj):
    """Mencetak ke layar DAN ke file txt secara bersamaan"""
    print(text)
    file_obj.write(text + "\n")

def load_data(path):
    df = pd.read_csv(path)
    # Drop kolom
    cols_to_drop = ['sku', 'judul', 'penerima', 'kategorigabungan', 'tanggal']
    existing_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_drop)
    return df

def encode_categoricals(df):
    df = df.copy()
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def save_plot(scores_df, title, filename):
    """Membuat dan menyimpan bar chart feature importance"""
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Score", y="Feature", data=scores_df.head(15), palette="viridis")
    plt.title(title)
    plt.xlabel("Score / Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"[INFO] Grafik disimpan sebagai {filename}")

# --- FUNGSI SELEKSI DENGAN SKOR ---

def get_kbest_scores(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    
    selector = SelectKBest(score_func=f_regression, k='all')
    selector.fit(X, y)
    
    scores = pd.DataFrame({
        'Feature': X.columns,
        'Score': selector.scores_
    }).sort_values(by='Score', ascending=False)
    
    return scores

def get_mutual_info_scores(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    
    selector = SelectKBest(score_func=mutual_info_regression, k='all')
    selector.fit(X, y)
    
    scores = pd.DataFrame({
        'Feature': X.columns,
        'Score': selector.scores_
    }).sort_values(by='Score', ascending=False)
    
    return scores

def get_model_based_scores(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    scores = pd.DataFrame({
        'Feature': X.columns,
        'Score': model.feature_importances_
    }).sort_values(by='Score', ascending=False)
    
    return scores

if __name__ == "__main__":
    # Untuk menyimpan hasil txt
    with open(OUTPUT_FILE, "w") as f:
        log_print("=== MULAI ANALISIS SELEKSI FITUR ===", f)
        
        path_file = "./data/processed.csv"
        df = load_data(path_file)
        df = df.fillna(0)
        df_encoded = encode_categoricals(df)
        
        target = "stock"
        
        if target not in df_encoded.columns:
            log_print(f"ERROR: Target '{target}' tidak ditemukan.", f)
        else:
            log_print(f"Target Variabel: {target}\n", f)

            # 1. KBest (Linear)
            log_print("--- 1. KBest Regression (Linear Score) ---", f)
            scores_kbest = get_kbest_scores(df_encoded, target)
            log_print(scores_kbest.head(10).to_string(index=False), f)
            save_plot(scores_kbest, "Feature Importance (KBest F-Regression)", "./output/chart_kbest.png")
            log_print("", f) # Spasi

            # 2. Mutual Info (Non-Linear)
            log_print("--- 2. Mutual Info (Non-Linear Score) ---", f)
            scores_mi = get_mutual_info_scores(df_encoded, target)
            log_print(scores_mi.head(10).to_string(index=False), f)
            save_plot(scores_mi, "Feature Importance (Mutual Info)", "./output/chart_mutual_info.png")
            log_print("", f)

            # 3. Model Based (Random Forest)
            log_print("--- 3. Model-Based (Random Forest Importance) ---", f)
            scores_rf = get_model_based_scores(df_encoded, target)
            log_print(scores_rf.head(10).to_string(index=False), f)
            save_plot(scores_rf, "Feature Importance (Random Forest)", "./output/chart_random_forest.png")
            
        log_print("=== SELESAI ===", f)
        print(f"\nHasil teks tersimpan di: {OUTPUT_FILE}")