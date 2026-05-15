import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    r2_score, mean_absolute_error,
    accuracy_score, confusion_matrix, classification_report,
    silhouette_score
)
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes

st.set_page_config(
    page_title="Prediksi Kerugian Ekonomi Bencana Alam",
    page_icon="🌍",
    layout="wide"
)

st.title("Prediksi Kerugian Ekonomi Berdasarkan Karakteristik Bencana Alam")
st.markdown("**Kelompok 1 - IS411 F / FL**")

# ─── Sidebar ───────────────────────────────────────────────────────────────
st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Section", [
    "Data Understanding",
    "Data Preparation",
    "Statistical Analysis",
    "Modelling & Evaluasi",
    "Classification",
    "Clustering",
    "Neural Network",
    "Kesimpulan"
])

# ─── Load & cache data ──────────────────────────────────────────────────────
@st.cache_data
def load_raw():
    return pd.read_csv("PureDataset1970_2021.csv")

@st.cache_data
def preprocess(data_1970):
    df = data_1970.copy()
    drop_columns = [
        'Dis No','Seq','Glide','Event Name','Location',
        'Admin1 Code','Admin2 Code','Geo Locations','Local Time'
    ]
    df = df.drop(columns=drop_columns, errors='ignore')
    df = df.dropna(subset=["Total Damages ('000 US$)"])
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df["Disaster Duration"] = df["End Year"] - df["Start Year"]

    numeric_cols = df.select_dtypes(include=['int64','float64']).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].mode().empty:
            df[col] = df[col].fillna("Unknown")
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    return df

@st.cache_data
def build_model_data(df):
    X = df.drop("Total Damages ('000 US$)", axis=1)
    y = df["Total Damages ('000 US$)"]
    col_names = X.columns.tolist()

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_df = pd.DataFrame(X_scaled, columns=col_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )

    # Feature importance
    rf_tmp = RandomForestRegressor(random_state=42)
    rf_tmp.fit(X_train, y_train)
    fi = pd.DataFrame({
        "Feature": col_names,
        "Importance": rf_tmp.feature_importances_
    }).sort_values("Importance", ascending=False)

    top_5 = fi['Feature'].head(5).tolist()

    # One-hot encode Disaster Type on df
    data_orig_cat = df.copy()
    # Disaster Type is already label-encoded; re-use top_5 features directly
    X_final = X_df[top_5]
    y_final = y.reset_index(drop=True)

    X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
        X_final, y_final, test_size=0.2, random_state=42
    )
    y_train_log = np.log1p(y_train_new)
    y_test_log = np.log1p(y_test_new)

    return (X_train, X_test, y_train, y_test, fi,
            X_final, y_final, X_train_new, X_test_new, y_train_log, y_test_log)

try:
    data_1970 = load_raw()
except FileNotFoundError:
    st.error("File PureDataset1970_2021.csv tidak ditemukan. Pastikan file ada di folder yang sama dengan app.py.")
    st.stop()

df = preprocess(data_1970)
(X_train, X_test, y_train, y_test, feature_importance,
 X_final, y_final, X_train_new, X_test_new, y_train_log, y_test_log) = build_model_data(df)

damage_col = "Total Damages ('000 US$)"

# ═══════════════════════════════════════════════════════════════════════════
# DATA UNDERSTANDING
# ═══════════════════════════════════════════════════════════════════════════
if menu == "Data Understanding":
    st.header("Data Understanding")

    st.subheader("5 Data Pertama")
    st.dataframe(data_1970.head())

    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Baris", data_1970.shape[0])
    col2.metric("Jumlah Kolom", data_1970.shape[1])
    col3.metric("Missing Values", int(data_1970.isnull().sum().sum()))

    st.subheader("Statistik Deskriptif")
    st.dataframe(data_1970.describe())

    st.subheader("Missing Values per Kolom")
    mv = data_1970.isnull().sum()
    mv = mv[mv > 0].sort_values(ascending=False)
    st.bar_chart(mv)

    st.subheader("Distribusi Jenis Bencana")
    fig, ax = plt.subplots(figsize=(10, 5))
    data_1970['Disaster Type'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title("Distribusi Jenis Bencana")
    ax.set_xlabel("Disaster Type")
    ax.set_ylabel("Jumlah")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Jumlah Bencana per Tahun")
    fig, ax = plt.subplots(figsize=(12, 5))
    data_1970['Year'].value_counts().sort_index().plot(ax=ax)
    ax.set_title("Jumlah Bencana per Tahun")
    ax.set_xlabel("Tahun")
    ax.set_ylabel("Jumlah")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Top 10 Negara dengan Bencana Terbanyak")
    fig, ax = plt.subplots(figsize=(10, 5))
    data_1970['Country'].value_counts().head(10).plot(kind='bar', ax=ax)
    ax.set_title("Top 10 Negara dengan Bencana Terbanyak")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Distribusi Bencana per Kontinen")
    fig, ax = plt.subplots(figsize=(8, 5))
    data_1970['Continent'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title("Distribusi Bencana per Kontinen")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

# ═══════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════
elif menu == "Data Preparation":
    st.header("Data Preparation")

    st.markdown("""
Proses data preparation meliputi:
1. Menghapus kolom yang tidak relevan
2. Menghapus baris dengan missing value pada target
3. Konversi tipe data Latitude dan Longitude
4. Membuat fitur baru Disaster Duration
5. Mengisi missing value numerik dengan median
6. Mengisi missing value kategorikal dengan modus
7. Label Encoding untuk kolom kategorikal
8. Normalisasi dengan MinMaxScaler
9. Split data 80% training dan 20% testing
""")

    col1, col2 = st.columns(2)
    col1.metric("Jumlah data setelah preparation", df.shape[0])
    col2.metric("Jumlah fitur", df.shape[1] - 1)

    st.subheader("Data setelah Preparation (5 baris pertama)")
    st.dataframe(df.head())

    st.subheader("Distribusi Target: Total Damages")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df[damage_col], bins=50, color='steelblue')
    axes[0].set_title("Distribusi Asli")
    axes[0].set_xlabel("Total Damages")
    axes[1].hist(np.log1p(df[damage_col]), bins=50, color='coral')
    axes[1].set_title("Setelah Log Transformation")
    axes[1].set_xlabel("Log(Total Damages)")
    plt.tight_layout()
    st.pyplot(fig)

# ═══════════════════════════════════════════════════════════════════════════
# STATISTICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
elif menu == "Statistical Analysis":
    st.header("Statistical Analysis")

    st.subheader("1. Univariate Analysis")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(df["Total Deaths"], bins=50, ax=axes[0])
    axes[0].set_title("Distribusi Total Deaths")
    sns.histplot(df["Total Affected"], bins=50, ax=axes[1])
    axes[1].set_title("Distribusi Total Affected")
    sns.histplot(df[damage_col], bins=50, ax=axes[2])
    axes[2].set_title("Distribusi Total Damages")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("2. Bivariate Analysis")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(df["Total Deaths"], df[damage_col], alpha=0.3, s=10)
    axes[0].set_title("Total Deaths vs Economic Damage")
    axes[0].set_xlabel("Total Deaths")
    axes[0].set_ylabel("Total Damages")
    axes[1].scatter(df["Total Affected"], df[damage_col], alpha=0.3, s=10)
    axes[1].set_title("Total Affected vs Economic Damage")
    axes[1].set_xlabel("Total Affected")
    axes[1].set_ylabel("Total Damages")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("3. Correlation Matrix")
    numeric_cols = df.select_dtypes(include=["int64", "float64"])
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(numeric_cols.corr(), cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("4. Outlier Analysis")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(x=df[damage_col], ax=axes[0])
    axes[0].set_title("Outlier - Total Damages")
    sns.boxplot(x=df["Total Deaths"], ax=axes[1])
    axes[1].set_title("Outlier - Total Deaths")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("5. Feature Importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feature_importance.head(10), ax=ax)
    ax.set_title("Top 10 Fitur Berpengaruh")
    plt.tight_layout()
    st.pyplot(fig)

# ═══════════════════════════════════════════════════════════════════════════
# MODELLING & EVALUASI
# ═══════════════════════════════════════════════════════════════════════════
elif menu == "Modelling & Evaluasi":
    st.header("Modelling & Evaluasi")

    @st.cache_resource
    def train_rf(X_tr, y_tr):
        from sklearn.model_selection import GridSearchCV
        param_grid = {
            'max_depth': [5, 10, 15],
            'min_samples_split': [10, 20, 30],
            'min_samples_leaf': [5, 10, 15],
            'n_estimators': [100, 200]
        }
        gs = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid, cv=3, scoring='r2', n_jobs=-1
        )
        gs.fit(X_tr, y_tr)
        return gs.best_estimator_, gs.best_params_

    @st.cache_resource
    def train_lr(X_tr, y_tr):
        lr = LinearRegression()
        lr.fit(X_tr, y_tr)
        return lr

    with st.spinner("Training model Random Forest..."):
        best_rf, best_params = train_rf(X_train_new, y_train_log)

    with st.spinner("Training Linear Regression..."):
        lr_model = train_lr(X_train_new, y_train_log)

    # RF evaluation
    y_pred_train_rf = best_rf.predict(X_train_new)
    y_pred_test_rf  = best_rf.predict(X_test_new)
    r2_train_rf = r2_score(y_train_log, y_pred_train_rf)
    r2_test_rf  = r2_score(y_test_log,  y_pred_test_rf)
    mae_rf      = mean_absolute_error(y_test_log, y_pred_test_rf)

    # LR evaluation
    y_pred_train_lr = lr_model.predict(X_train_new)
    y_pred_test_lr  = lr_model.predict(X_test_new)
    r2_train_lr = r2_score(y_train_log, y_pred_train_lr)
    r2_test_lr  = r2_score(y_test_log,  y_pred_test_lr)
    mae_lr      = mean_absolute_error(y_test_log, y_pred_test_lr)

    st.subheader("Best Hyperparameter Random Forest")
    st.json(best_params)

    st.subheader("Hasil Evaluasi")
    eval_df = pd.DataFrame({
        "Model": ["Random Forest (tuned)", "Linear Regression"],
        "R2 Train": [r2_train_rf, r2_train_lr],
        "R2 Test":  [r2_test_rf,  r2_test_lr],
        "MAE Test": [mae_rf,      mae_lr],
        "Gap":      [r2_train_rf - r2_test_rf, r2_train_lr - r2_test_lr]
    })
    st.dataframe(eval_df.style.format({
        "R2 Train": "{:.4f}", "R2 Test": "{:.4f}",
        "MAE Test": "{:.4f}", "Gap": "{:.4f}"
    }))

    st.subheader("Actual vs Predicted - Random Forest")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_test_log, y_pred_test_rf, alpha=0.4, s=15, color='blue')
    mn, mx = float(y_test_log.min()), float(y_test_log.max())
    axes[0].plot([mn, mx], [mn, mx], 'r--', lw=2)
    axes[0].set_title(f"Random Forest  R²={r2_test_rf:.4f}")
    axes[0].set_xlabel("Actual Log(Total Damage)")
    axes[0].set_ylabel("Predicted Log(Total Damage)")

    axes[1].scatter(y_test_log, y_pred_test_lr, alpha=0.4, s=15, color='green')
    axes[1].plot([mn, mx], [mn, mx], 'r--', lw=2)
    axes[1].set_title(f"Linear Regression  R²={r2_test_lr:.4f}")
    axes[1].set_xlabel("Actual Log(Total Damage)")
    axes[1].set_ylabel("Predicted Log(Total Damage)")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Feature Importance - Final Model")
    fi_final = pd.DataFrame({
        'Feature': X_final.columns,
        'Importance': best_rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x='Importance', y='Feature', data=fi_final, ax=ax)
    ax.set_title("Feature Importance - Random Forest Final")
    plt.tight_layout()
    st.pyplot(fig)

# ═══════════════════════════════════════════════════════════════════════════
# CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════
elif menu == "Classification":
    st.header("Classification")
    st.markdown("""
Classification digunakan untuk menentukan apakah suatu bencana menghasilkan kerugian 
ekonomi yang rendah, sedang, atau tinggi berdasarkan karakteristik bencananya.
""")

    @st.cache_data
    def prep_cls(df, damage_col):
        df_cls = df.copy()
        q33 = df_cls[damage_col].quantile(0.33)
        q66 = df_cls[damage_col].quantile(0.66)
        def kategorisasi(val):
            if val <= q33: return 'Low'
            elif val <= q66: return 'Medium'
            else: return 'High'
        df_cls['Damage_Category'] = df_cls[damage_col].apply(kategorisasi)
        return df_cls, q33, q66

    df_cls, q33, q66 = prep_cls(df, damage_col)

    col1, col2, col3 = st.columns(3)
    col1.metric("Low threshold", f"{q33:,.0f}")
    col2.metric("Medium threshold", f"{q66:,.0f}")
    col3.metric("Jumlah data", len(df_cls))

    st.subheader("Distribusi Kategori Kerugian")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    df_cls['Damage_Category'].value_counts().plot(kind='bar', color=colors, ax=ax)
    ax.set_title("Distribusi Kategori Tingkat Kerugian Ekonomi")
    ax.set_xlabel("Kategori")
    ax.set_ylabel("Jumlah")
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)

    @st.cache_data
    def train_cls(df_cls, damage_col):
        feature_cols = [col for col in df_cls.select_dtypes(include=['int64','float64']).columns
                        if col != damage_col and col != 'Damage_Category']
        X_c = df_cls[feature_cols].fillna(df_cls[feature_cols].median())
        y_c = df_cls['Damage_Category']
        le_c = LabelEncoder()
        y_enc = le_c.fit_transform(y_c)
        scaler_c = MinMaxScaler()
        X_sc = scaler_c.fit_transform(X_c)
        X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

        models = {
            'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'Naive Bayes':   GaussianNB()
        }
        results = {}
        for name, m in models.items():
            m.fit(X_tr, y_tr)
            pred = m.predict(X_te)
            acc  = accuracy_score(y_te, pred)
            results[name] = {'pred': pred, 'accuracy': acc, 'report': classification_report(y_te, pred, target_names=le_c.classes_, output_dict=True)}
        return results, y_te, le_c.classes_

    with st.spinner("Training classification models..."):
        results_cls, y_test_cls, class_names = train_cls(df_cls, damage_col)

    st.subheader("Confusion Matrix")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (name, res) in enumerate(results_cls.items()):
        cm = confusion_matrix(y_test_cls, res['pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=class_names, yticklabels=class_names)
        axes[idx].set_title(f"{name}\nAccuracy: {res['accuracy']:.4f}")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")
    plt.suptitle("Confusion Matrix - Perbandingan Model Klasifikasi", fontsize=13)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Perbandingan Akurasi")
    acc_df = pd.DataFrame({
        "Model": list(results_cls.keys()),
        "Accuracy": [r['accuracy'] for r in results_cls.values()]
    })
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(acc_df["Model"], acc_df["Accuracy"], color=['#3498db','#2ecc71','#e74c3c'], width=0.5)
    ax.set_ylim(0, 1.15)
    ax.set_title("Perbandingan Akurasi Model Klasifikasi")
    ax.set_ylabel("Accuracy")
    for bar, val in zip(bars, acc_df["Accuracy"]):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

    best_cls = max(results_cls, key=lambda x: results_cls[x]['accuracy'])
    st.success(f"Model terbaik: **{best_cls}** dengan akurasi **{results_cls[best_cls]['accuracy']:.4f}**")

# ═══════════════════════════════════════════════════════════════════════════
# CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════
elif menu == "Clustering":
    st.header("Clustering")
    st.markdown("""
Clustering digunakan untuk mengelompokkan data berdasarkan kemiripannya tanpa perlu 
tahu lebih dulu kategorinya.
""")

    tab1, tab2, tab3 = st.tabs(["K-Means", "K-Modes", "K-Prototypes"])

    # ── K-Means ──────────────────────────────────────────────────────────
    with tab1:
        st.subheader("K-Means Clustering")

        @st.cache_data
        def run_kmeans(df_cls, damage_col):
            num_features = ['Total Deaths', 'No Injured', 'No Affected',
                            'Total Affected', 'No Homeless', 'Dis Mag Value']
            available = [c for c in num_features if c in df_cls.columns]
            df_cl = df_cls[available].copy()
            for col in df_cl.columns:
                df_cl[col] = pd.to_numeric(df_cl[col], errors='coerce')
                df_cl[col] = df_cl[col].fillna(df_cl[col].median())
            scaler_km = MinMaxScaler()
            X_km = scaler_km.fit_transform(df_cl)

            inertias, silhouettes = [], []
            k_range = range(2, 10)
            for k in k_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(X_km)
                inertias.append(km.inertia_)
                silhouettes.append(silhouette_score(X_km, km.labels_))

            best_k = list(k_range)[silhouettes.index(max(silhouettes))]
            km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            labels = km_final.fit_predict(X_km)
            return X_km, inertias, silhouettes, list(k_range), best_k, labels, df_cl.columns.tolist()

        @st.cache_data
        def prep_cls_for_kmeans(df, damage_col):
            df_cls = df.copy()
            q33 = df_cls[damage_col].quantile(0.33)
            q66 = df_cls[damage_col].quantile(0.66)
            def kat(v):
                if v <= q33: return 'Low'
                elif v <= q66: return 'Medium'
                else: return 'High'
            df_cls['Damage_Category'] = df_cls[damage_col].apply(kat)
            return df_cls

        df_cls_km = prep_cls_for_kmeans(df, damage_col)
        X_km, inertias, silhouettes, k_range, best_k, km_labels, num_feat_used = run_kmeans(df_cls_km, damage_col)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.set_title('Elbow Method')
        ax1.set_xlabel('Jumlah Cluster (K)')
        ax1.set_ylabel('Inertia')
        ax1.grid(True, alpha=0.3)

        ax2.plot(k_range, silhouettes, 'ro-', linewidth=2, markersize=8)
        ax2.axvline(x=best_k, color='green', linestyle='--', label=f'Best K={best_k}')
        ax2.set_title('Silhouette Score')
        ax2.set_xlabel('Jumlah Cluster (K)')
        ax2.set_ylabel('Silhouette Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

        st.write(f"K optimal: **{best_k}**")

        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_km)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sc = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=km_labels, cmap='tab10', alpha=0.5, s=15)
        plt.colorbar(sc, ax=axes[0], label='Cluster')
        axes[0].set_title(f'K-Means (K={best_k}) - PCA')
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')

        df_tmp = df_cls_km.copy()
        df_tmp['KMeans_Cluster'] = km_labels
        dmg_per_cluster = df_tmp.groupby('KMeans_Cluster')[damage_col].mean().sort_values(ascending=False)
        axes[1].bar(dmg_per_cluster.index.astype(str), dmg_per_cluster.values,
                    color=plt.cm.tab10(np.linspace(0, 1, best_k)))
        axes[1].set_title("Rata-rata Kerugian per Cluster")
        axes[1].set_xlabel("Cluster")
        axes[1].set_ylabel("Total Damages")
        plt.tight_layout()
        st.pyplot(fig)

    # ── K-Modes ──────────────────────────────────────────────────────────
    with tab2:
        st.subheader("K-Modes Clustering")

        @st.cache_data
        def run_kmodes(data_1970):
            cat_features = ['Disaster Type', 'Disaster Subgroup', 'Continent', 'Region']
            available = [c for c in cat_features if c in data_1970.columns]
            df_km = data_1970[available].dropna().reset_index(drop=True)

            costs = []
            k_range_km = range(2, 8)
            for k in k_range_km:
                km_m = KModes(n_clusters=k, init='Huang', n_init=5, random_state=42, verbose=0)
                km_m.fit(df_km.values)
                costs.append(km_m.cost_)

            best_k_m = 3
            km_final = KModes(n_clusters=best_k_m, init='Huang', n_init=5, random_state=42, verbose=0)
            labels = km_final.fit_predict(df_km.values)
            df_km['KModes_Cluster'] = labels
            return df_km, costs, list(k_range_km), best_k_m, available

        with st.spinner("Running K-Modes..."):
            df_kmode, costs, k_range_km, best_k_m, cat_feat_used = run_kmodes(data_1970)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(k_range_km, costs, 'go-', linewidth=2, markersize=8)
        ax.set_title('Elbow Method K-Modes')
        ax.set_xlabel('Jumlah Cluster (K)')
        ax.set_ylabel('Cost (Total Dissimilarity)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

        st.write(f"K yang digunakan: **{best_k_m}**")
        st.write("Distribusi cluster:")
        st.dataframe(df_kmode['KModes_Cluster'].value_counts().sort_index().rename("Jumlah"))

        fig, axes = plt.subplots(1, best_k_m, figsize=(18, 5))
        for cid in range(best_k_m):
            cdata = df_kmode[df_kmode['KModes_Cluster'] == cid]
            dc = cdata['Disaster Type'].value_counts().head(8)
            axes[cid].barh(dc.index, dc.values, color=plt.cm.Set2(cid / best_k_m))
            axes[cid].set_title(f'Cluster {cid}\n({len(cdata)} bencana)')
            axes[cid].set_xlabel('Jumlah')
            axes[cid].invert_yaxis()
        plt.suptitle('Distribusi Jenis Bencana per Cluster K-Modes', fontsize=13)
        plt.tight_layout()
        st.pyplot(fig)

    # ── K-Prototypes ─────────────────────────────────────────────────────
    with tab3:
        st.subheader("K-Prototypes Clustering")

        @st.cache_data
        def run_kproto(data_1970):
            num_feat_kp = ['Total Deaths', 'Total Affected']
            cat_feat_kp = ['Disaster Type', 'Continent']
            num_av = [c for c in num_feat_kp if c in data_1970.columns]
            cat_av = [c for c in cat_feat_kp if c in data_1970.columns]

            df_kp = pd.concat([data_1970[num_av], data_1970[cat_av]], axis=1).dropna().reset_index(drop=True)
            scaler_kp = MinMaxScaler()
            df_kp[num_av] = scaler_kp.fit_transform(df_kp[num_av])

            cat_indices = [df_kp.columns.get_loc(c) for c in cat_av]
            kp = KPrototypes(n_clusters=3, init='Huang', n_init=5, random_state=42, verbose=0)
            labels = kp.fit_predict(df_kp.values, categorical=cat_indices)
            df_kp['KProto_Cluster'] = labels
            return df_kp, num_av, cat_av

        with st.spinner("Running K-Prototypes..."):
            df_kproto, num_av, cat_av = run_kproto(data_1970)

        st.write("Distribusi cluster:")
        st.dataframe(df_kproto['KProto_Cluster'].value_counts().sort_index().rename("Jumlah"))

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        sc = axes[0].scatter(df_kproto[num_av[0]], df_kproto[num_av[1]],
                             c=df_kproto['KProto_Cluster'], cmap='tab10', alpha=0.4, s=15)
        plt.colorbar(sc, ax=axes[0], label='Cluster')
        axes[0].set_title('K-Prototypes Clusters')
        axes[0].set_xlabel(f'{num_av[0]} (normalized)')
        axes[0].set_ylabel(f'{num_av[1]} (normalized)')

        cc = df_kproto.groupby(['KProto_Cluster', cat_av[1]]).size().unstack(fill_value=0)
        cc.plot(kind='bar', ax=axes[1], colormap='tab20')
        axes[1].set_title('Distribusi Kontinen per Cluster')
        axes[1].set_xlabel('Cluster')
        axes[1].set_ylabel('Jumlah Bencana')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
        plt.tight_layout()
        st.pyplot(fig)

# ═══════════════════════════════════════════════════════════════════════════
# NEURAL NETWORK
# ═══════════════════════════════════════════════════════════════════════════
elif menu == "Neural Network":
    st.header("Basic Neural Network")
    st.markdown("""
Neural Network yang digunakan adalah MLPRegressor dari scikit-learn dengan arsitektur 
3 hidden layer: 128, 64, dan 32 neuron. Model ini digunakan untuk memprediksi nilai 
kerugian ekonomi akibat bencana.
""")

    @st.cache_resource
    def train_nn(X_tr, y_tr, X_te, y_te):
        scaler_nn = MinMaxScaler()
        X_nn_tr = scaler_nn.fit_transform(X_tr)
        X_nn_te = scaler_nn.transform(X_te)
        y_nn_tr = np.log1p(np.expm1(y_tr))
        y_nn_te = np.log1p(np.expm1(y_te))

        nn = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            verbose=False
        )
        nn.fit(X_nn_tr, y_tr)
        return nn, X_nn_tr, X_nn_te, y_tr, y_te

    with st.spinner("Training Neural Network..."):
        nn_model, X_nn_tr, X_nn_te, y_nn_tr, y_nn_te = train_nn(
            X_train_new, y_train_log, X_test_new, y_test_log
        )

    y_pred_nn_tr = nn_model.predict(X_nn_tr)
    y_pred_nn_te = nn_model.predict(X_nn_te)

    r2_nn_tr  = r2_score(y_nn_tr, y_pred_nn_tr)
    r2_nn_te  = r2_score(y_nn_te, y_pred_nn_te)
    mae_nn    = mean_absolute_error(y_nn_te, y_pred_nn_te)
    gap_nn    = r2_nn_tr - r2_nn_te

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R2 Training", f"{r2_nn_tr:.4f}")
    col2.metric("R2 Testing",  f"{r2_nn_te:.4f}")
    col3.metric("MAE Testing", f"{mae_nn:.4f}")
    col4.metric("Gap", f"{gap_nn:.4f}")

    if gap_nn > 0.15:
        st.warning("Indikasi overfitting")
    elif r2_nn_te < 0.3:
        st.warning("Indikasi underfitting")
    else:
        st.success("Model cukup seimbang")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(nn_model.loss_curve_, color='blue', linewidth=2, label='Training Loss')
    axes[0].set_title('Learning Curve Neural Network')
    axes[0].set_xlabel('Iterasi')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(y_nn_te, y_pred_nn_te, alpha=0.4, color='purple', s=20)
    mn = float(min(y_nn_te.min(), y_pred_nn_te.min()))
    mx = float(max(y_nn_te.max(), y_pred_nn_te.max()))
    axes[1].plot([mn, mx], [mn, mx], 'r--', lw=2)
    axes[1].set_title(f'Actual vs Predicted  R²={r2_nn_te:.4f}')
    axes[1].set_xlabel('Actual')
    axes[1].set_ylabel('Predicted')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # Compare all models
    st.subheader("Perbandingan Semua Model Regresi")

    @st.cache_resource
    def get_rf_lr_scores(X_tr, y_tr, X_te, y_te):
        rf = RandomForestRegressor(max_depth=10, min_samples_leaf=5,
                                    min_samples_split=10, n_estimators=100, random_state=42)
        rf.fit(X_tr, y_tr)
        lr = LinearRegression()
        lr.fit(X_tr, y_tr)
        return (r2_score(y_te, rf.predict(X_te)),
                mean_absolute_error(y_te, rf.predict(X_te)),
                r2_score(y_te, lr.predict(X_te)),
                mean_absolute_error(y_te, lr.predict(X_te)))

    r2_rf, mae_rf_c, r2_lr, mae_lr_c = get_rf_lr_scores(X_train_new, y_train_log, X_test_new, y_test_log)

    comp_df = pd.DataFrame({
        "Model": ["Random Forest (tuned)", "Linear Regression", "Neural Network (MLP)"],
        "R2 Test": [r2_rf, r2_lr, r2_nn_te],
        "MAE Test": [mae_rf_c, mae_lr_c, mae_nn]
    })
    st.dataframe(comp_df.style.format({"R2 Test": "{:.4f}", "MAE Test": "{:.4f}"}))

    fig, ax = plt.subplots(figsize=(9, 5))
    colors_c = ['#27ae60', '#2980b9', '#8e44ad']
    bars = ax.bar(comp_df["Model"], comp_df["R2 Test"], color=colors_c, width=0.5)
    ax.set_title("Perbandingan R2 Test - Semua Model Regresi")
    ax.set_ylabel("R-Squared (Testing)")
    ax.set_ylim(0, max(comp_df["R2 Test"]) * 1.3 + 0.05)
    for bar, val in zip(bars, comp_df["R2 Test"]):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

# ═══════════════════════════════════════════════════════════════════════════
# KESIMPULAN
# ═══════════════════════════════════════════════════════════════════════════
elif menu == "Kesimpulan":
    st.header("Kesimpulan Akhir")

    st.markdown("""
Dataset bencana alam 1970-2021 mencakup lebih dari 15.000 kejadian bencana dari seluruh 
dunia dengan berbagai karakteristik seperti jenis bencana, lokasi, jumlah korban, dan 
kerugian ekonomi.

**Data Understanding dan Preparation**

Dataset memiliki distribusi yang sangat tidak merata pada variabel target Total Damages, 
di mana sebagian besar bencana menghasilkan kerugian kecil tetapi beberapa kejadian besar 
seperti gempa bumi dan badai menyebabkan kerugian yang jauh lebih besar. Banyak kolom 
memiliki missing value yang tinggi, terutama pada kolom kerugian ekonomi dan data bantuan. 
Log-transformation diterapkan pada variabel target untuk menstabilkan distribusi sebelum 
pemodelan.

**Statistical Analysis**

Hasil analisis menunjukkan bahwa variabel seperti Insured Damages, No Injured, No Homeless, 
Total Deaths, dan Total Affected memiliki pengaruh paling besar terhadap prediksi kerugian 
ekonomi. Hal ini menunjukkan bahwa semakin parah dampak fisik dan sosial suatu bencana, 
semakin besar pula kerugian ekonomi yang ditimbulkan.

**Regression Modelling**

Random Forest setelah hyperparameter tuning menghasilkan performa yang lebih baik dengan 
gap training-testing yang lebih kecil dibanding sebelum tuning. Linear Regression menghasilkan 
performa yang lebih rendah karena hubungan antar variabel bencana cenderung tidak linear. 
Neural Network (MLP) mampu menangkap pola yang lebih kompleks namun tetap membutuhkan lebih 
banyak data dan penyesuaian lebih lanjut untuk mencapai hasil yang optimal pada dataset ini.

**Classification**

Model klasifikasi berhasil membagi bencana ke dalam tiga kategori tingkat kerugian ekonomi 
yaitu Low, Medium, dan High. Random Forest menunjukkan akurasi terbaik di antara tiga model 
yang diuji. Hasil ini menunjukkan bahwa karakteristik bencana seperti jenis bencana, jumlah 
korban, dan wilayah kejadian dapat digunakan untuk memperkirakan tingkat keparahan kerugian 
ekonomi yang ditimbulkan.

**Clustering**

K-Means berhasil mengelompokkan bencana berdasarkan intensitas dampaknya secara numerik, 
menghasilkan cluster bencana ringan, sedang, dan berat. K-Modes mengelompokkan bencana 
berdasarkan karakteristik kategorialnya dan menemukan bahwa jenis bencana tertentu cenderung 
terjadi di wilayah tertentu. K-Prototypes memberikan hasil yang paling lengkap karena 
mempertimbangkan kedua jenis data sekaligus dan menghasilkan pengelompokan yang lebih 
mencerminkan kondisi nyata bencana di lapangan.

**Rekomendasi**

Untuk penelitian selanjutnya, disarankan untuk menambahkan data faktor ekonomi wilayah 
seperti GDP per kapita dan kepadatan infrastruktur sebagai fitur tambahan karena kerugian 
ekonomi bencana sangat dipengaruhi oleh kondisi ekonomi daerah yang terdampak. Selain itu, 
penanganan outlier yang lebih baik dan penggunaan model ensemble yang lebih canggih 
berpotensi meningkatkan performa prediksi secara signifikan.
""")
