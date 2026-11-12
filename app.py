import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats
import streamlit as st

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Ames Housing ML & EDA Engine",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# CACHING & MEMUAT DATASET (BAB 2.3: MENAMPILKAN DATA)
# ==============================================================================
@st.cache_data
def load_data():
    if os.path.exists("AmesHousing.csv"):
        return pd.read_csv("AmesHousing.csv")
    elif os.path.exists(os.path.join("data", "AmesHousing.csv")):
        return pd.read_csv(os.path.join("data", "AmesHousing.csv"))
    else:
        st.error("Error: Berkas 'AmesHousing.csv' tidak ditemukan di direktori lokal maupun folder 'data/'.")
        st.stop()

df_raw = load_data()

# ==============================================================================
# SIDEBAR NAVIGASI (BAB 2.7: SIDEBAR)
# ==============================================================================
st.sidebar.title("📌 Navigasi Modul")
st.sidebar.caption("Dashboard Analisis & Pemodelan Ames Housing")

menu = st.sidebar.radio(
    "Pilih Modul / Halaman:",
    [
        "1. Overview & Dataset Preview",
        "2. Diagnostic EDA & Preprocessing (10 Masalah)",
        "3. Reduksi Dimensi (PCA)",
        "4. Komparasi 3 Paradigma ML",
        "5. Simulator Prediksi Harga Real-Time",
    ],
)

# ==============================================================================
# HALAMAN 1: OVERVIEW & DATASET PREVIEW
# ==============================================================================
if menu == "1. Overview & Dataset Preview":
    st.title("🏠 Dashboard Analisis Properti & Machine Learning Ames Housing")
    st.markdown(
        "Aplikasi ini mengimplementasikan **framework Streamlit** untuk mendiagnosis 12 masalah utama *data preprocessing*, "
        "melakukan *feature engineering*, reduksi dimensi, dan membandingkan **3 paradigma algoritma Machine Learning**."
    )
    st.divider()

    # BAB 2.3: METRIC DISPLAY
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Observasi", f"{df_raw.shape[0]:,} baris")
    col2.metric("Total Fitur", f"{df_raw.shape[1]} kolom")
    col3.metric("Rata-rata Harga", f"${df_raw['SalePrice'].mean():,.2f}")
    col4.metric("Median Luas Bangunan", f"{df_raw['Gr Liv Area'].median():,.0f} sqft")

    st.divider()
    st.subheader("📋 Preview Dataset Utama (st.dataframe)")
    st.dataframe(df_raw.head(10), use_container_width=True)

    with st.expander("🔍 Informasi Struktur Data & Fitur Missing Values"):
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.write("**Top 10 Kolom dengan Nilai Hilang (Missing Values):**")
            missing_series = df_raw.isna().sum()
            missing_df = missing_series[missing_series > 0].sort_values(ascending=False).reset_index()
            missing_df.columns = ["Fitur", "Jumlah Missing"]
            st.dataframe(missing_df.head(10), use_container_width=True)
        with col_info2:
            st.write("**Ringkasan Statistik Fitur Numerik Utama:**")
            st.dataframe(
                df_raw[["SalePrice", "Gr Liv Area", "Lot Area", "Overall Qual", "Total Bsmt SF"]].describe().T,
                use_container_width=True,
            )

# ==============================================================================
# HALAMAN 2: DIAGNOSTIC EDA & PREPROCESSING (10 MASALAH DIVERIFIKASI)
# ==============================================================================
elif menu == "2. Diagnostic EDA & Preprocessing (10 Masalah)":
    st.title("🛠️ Diagnosa & Solusi Preprocessing Data (Masalah 1 - 10)")
    st.caption("Eksplorasi mendalam terkait kebocoran data, skewness, imputasi, VIF, dan penanganan outlier.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Leakage & Skewness (M1-M2)",
        "Imputasi & Multikolinearitas (M3-M4)",
        "Target Log & Seleksi Fitur (M5-M6)",
        "Kardinalitas & Outlier Masking (M7-M8)",
        "Multimodal & True Signal (M9-M10)"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: MASALAH 1 & MASALAH 2
    # --------------------------------------------------------------------------
    with tab1:
        st.header("Masalah 1: Kebocoran Data (Data Leakage) saat Preprocessing")
        fitur_m1 = ["Lot Area", "Gr Liv Area", "Total Bsmt SF", "Garage Area", "Lot Frontage"]
        X_m1 = df_raw[fitur_m1].copy()
        y_m1 = df_raw["SalePrice"]

        # Sebelum (Bocor)
        scaler_bocor = StandardScaler()
        X_bocor = scaler_bocor.fit_transform(X_m1.fillna(X_m1.median()))
        X_tr_l, X_ts_l, y_tr_l, y_ts_l = train_test_split(X_bocor, y_m1, test_size=0.2, random_state=42)
        r2_sebelum_m1 = LinearRegression().fit(X_tr_l, y_tr_l).score(X_ts_l, y_ts_l)

        # Sesudah (Isolasi Pipeline)
        X_tr, X_ts, y_tr, y_ts = train_test_split(X_m1, y_m1, test_size=0.2, random_state=42)
        pipe_m1 = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ])
        pipe_m1.fit(X_tr, y_tr)
        r2_sesudah_m1 = pipe_m1.score(X_ts, y_ts)

        m1_col1, m1_col2 = st.columns(2)
        m1_col1.metric("R² Data Uji (Sebelum / Leaked)", f"{r2_sebelum_m1:.4f}")
        m1_col2.metric("R² Data Uji (Sesudah / Isolated Pipeline)", f"{r2_sesudah_m1:.4f}")
        st.info("💡 **Solusi:** `Pipeline` memastikan statistik scaler & imputer hanya dihitung dari data latih (*Train Set*).")

        st.divider()
        st.header("Masalah 2: Penskalaan Fitur Menceng Ekstrem & Evaluasi StandardScaler")
        skew_lot = df_raw["Lot Area"].skew()
        lot_std = StandardScaler().fit_transform(df_raw[["Lot Area"]])
        skew_sebelum_m2 = pd.Series(lot_std.flatten()).skew()
        lot_log = np.log1p(df_raw["Lot Area"])
        skew_sesudah_m2 = pd.Series(lot_log).skew()

        fitur_demo = ["Overall Qual", "Gr Liv Area", "Total Bsmt SF", "Garage Area", "Lot Area"]
        X_demo = df_raw[fitur_demo].fillna(df_raw[fitur_demo].median())
        y_demo = df_raw["SalePrice"]
        X_tr_d, X_ts_d, y_tr_d, y_ts_d = train_test_split(X_demo, y_demo, test_size=0.2, random_state=42)
        r2_knn_raw = KNeighborsRegressor(n_neighbors=5).fit(X_tr_d, y_tr_d).score(X_ts_d, y_ts_d)
        r2_knn_scaled = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsRegressor(n_neighbors=5))]).fit(X_tr_d, y_tr_d).score(X_ts_d, y_ts_d)

        m2_col1, m2_col2 = st.columns(2)
        with m2_col1:
            st.write("**Efek Log Transform pada Skewness Lot Area:**")
            st.write(f"- Skewness Awal: `{skew_lot:.3f}`")
            st.write(f"- Skewness setelah StandardScaler: `{skew_sebelum_m2:.3f}` (Tidak Berubah)")
            st.write(f"- Skewness setelah Log1p Transform: `{skew_sesudah_m2:.3f}` (Normal)")
        with m2_col2:
            st.write("**Revaluasi StandardScaler pada K-NN:**")
            st.write(f"- K-NN Tanpa Scaler (Buta Skala): R² = `{r2_knn_raw:.4f}`")
            st.write(f"- K-NN + StandardScaler (Skala Adil): R² = `{r2_knn_scaled:.4f}`")

    # --------------------------------------------------------------------------
    # TAB 2: MASALAH 3 & MASALAH 4
    # --------------------------------------------------------------------------
    with tab2:
        st.header("Masalah 3: Penghapusan Baris Membabi Buta (Listwise Deletion)")
        df_dropped = df_raw.dropna(subset=["Lot Frontage"])
        mean_target_asli = df_raw["SalePrice"].mean()
        mean_target_dropped = df_dropped["SalePrice"].mean()

        m3_col1, m3_col2 = st.columns(2)
        m3_col1.metric("Sampel Setelah `dropna()`", f"{len(df_dropped)} baris", f"{len(df_dropped)-len(df_raw)} baris")
        m3_col2.metric("Pergeseran Rata-rata SalePrice", f"${mean_target_dropped:,.2f}", f"${mean_target_dropped - mean_target_asli:+,.2f}")
        st.success("✅ **Solusi:** `KNNImputer(n_neighbors=5)` menjaga 100% sampel utuh tanpa bias seleksi.")

        st.divider()
        st.header("Masalah 4: Jebakan Multikolinearitas Tinggi (VIF Ekstrem)")
        def kalkulasi_vif(data):
            vif_list = []
            for col in data.columns:
                X_other = data.drop(columns=[col])
                y_col = data[col]
                r2 = LinearRegression().fit(X_other, y_col).score(X_other, y_col)
                vif = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
                vif_list.append(vif)
            return pd.DataFrame({"Fitur": data.columns, "VIF": np.round(vif_list, 2)})

        vif_sebelum = kalkulasi_vif(df_raw[["Gr Liv Area", "1st Flr SF", "2nd Flr SF", "Total Bsmt SF"]].dropna())
        vif_sesudah = kalkulasi_vif(df_raw[["Gr Liv Area", "Garage Area"]].dropna())

        m4_col1, m4_col2 = st.columns(2)
        with m4_col1:
            st.write("**VIF Sebelum Mitigasi (Multikolinearitas Eksplosif):**")
            st.table(vif_sebelum)
        with m4_col2:
            st.write("**VIF Sesudah Mitigasi (Eliminasi Redundansi):**")
            st.table(vif_sesudah)

    # --------------------------------------------------------------------------
    # TAB 3: MASALAH 5 & MASALAH 6
    # --------------------------------------------------------------------------
    with tab3:
        st.header("Masalah 5: Pelanggaran Asumsi Normalitas Target (Skewed Target)")
        skew_sebelum_m5 = df_raw["SalePrice"].skew()
        kurt_sebelum_m5 = df_raw["SalePrice"].kurt()
        log_saleprice = np.log1p(df_raw["SalePrice"])
        skew_sesudah_m5 = log_saleprice.skew()
        kurt_sesudah_m5 = log_saleprice.kurt()

        m5_col1, m5_col2 = st.columns(2)
        m5_col1.metric("SalePrice Mentah", f"Skew: {skew_sebelum_m5:.3f}", f"Kurt: {kurt_sebelum_m5:.3f}")
        m5_col2.metric("SalePrice Log1p", f"Skew: {skew_sesudah_m5:.3f}", f"Kurt: {kurt_sesudah_m5:.3f}")

        fig_m5, ax_m5 = plt.subplots(1, 2, figsize=(12, 4))
        ax_m5[0].hist(df_raw["SalePrice"], bins=30, color="crimson", alpha=0.7)
        ax_m5[0].set_title("Distribusi Target Mentah (Right-Skewed)")
        ax_m5[1].hist(log_saleprice, bins=30, color="teal", alpha=0.7)
        ax_m5[1].set_title("Distribusi Target Log1p (Gaussian)")
        st.pyplot(fig_m5)

        st.divider()
        st.header("Masalah 6: Bias Asumsi Linearitas (Pearson vs Spearman vs Mutual Info)")
        fitur_m6 = ["Overall Qual", "Gr Liv Area", "Year Built", "Total Bsmt SF", "Lot Area"]
        X_m6 = df_raw[fitur_m6].fillna(df_raw[fitur_m6].median())
        y_target_m6 = df_raw["SalePrice"]

        pearson_scores = X_m6.apply(lambda col: stats.pearsonr(col, y_target_m6)[0]).abs()
        spearman_scores = X_m6.apply(lambda col: stats.spearmanr(col, y_target_m6)[0]).abs()
        mi_scores = pd.Series(mutual_info_regression(X_m6, y_target_m6, random_state=42), index=X_m6.columns)

        komparasi_m6 = pd.DataFrame({
            "Pearson (Linear)": pearson_scores,
            "Spearman (Monotonik)": spearman_scores,
            "Mutual Info (Non-Linear)": mi_scores
        }).round(4)

        st.dataframe(komparasi_m6, use_container_width=True)

    # --------------------------------------------------------------------------
    # TAB 4: MASALAH 7 & MASALAH 8
    # --------------------------------------------------------------------------
    with tab4:
        st.header("Masalah 7: Fitur Identitas Unik Berkardinalitas Tinggi (PID)")
        df_id = df_raw[['PID', 'Gr Liv Area', 'Overall Qual', 'SalePrice']].dropna().copy()
        tree_sebelum = DecisionTreeRegressor(random_state=42).fit(df_id[['PID', 'Gr Liv Area', 'Overall Qual']], df_id['SalePrice'])
        tree_sesudah = DecisionTreeRegressor(random_state=42).fit(df_id[['Gr Liv Area', 'Overall Qual']], df_id['SalePrice'])

        m7_col1, m7_col2 = st.columns(2)
        with m7_col1:
            st.write("**Importance Sebelum Drop `PID` (Tercemar Identitas):**")
            st.dataframe(pd.Series(tree_sebelum.feature_importances_, index=['PID', 'Gr Liv Area', 'Overall Qual']), use_container_width=True)
        with m7_col2:
            st.write("**Importance Sesudah Drop `PID` (Murni Fisik):**")
            st.dataframe(pd.Series(tree_sesudah.feature_importances_, index=['Gr Liv Area', 'Overall Qual']), use_container_width=True)

        st.divider()
        st.header("Masalah 8: Masking Effect pada Deteksi Outlier Parametrik (Z-Score)")
        lot_area_sample = df_raw['Lot Area'].dropna().copy()
        lot_area_masked = pd.concat([lot_area_sample, pd.Series([500000.0])]).reset_index(drop=True)
        z_scores = np.abs(stats.zscore(lot_area_masked))
        outliers_sebelum_m8 = lot_area_masked[z_scores > 3]

        med_m8 = np.median(lot_area_masked)
        mad_m8 = np.median(np.abs(lot_area_masked - med_m8))
        modified_z = 0.6745 * np.abs(lot_area_masked - med_m8) / mad_m8
        outliers_sesudah_m8 = lot_area_masked[modified_z > 3.5]

        m8_col1, m8_col2 = st.columns(2)
        m8_col1.metric("Z-Score Klasik (Ter-masking)", f"{len(outliers_sebelum_m8)} outlier")
        m8_col2.metric("Modified Z-Score (Robust MAD)", f"{len(outliers_sesudah_m8)} outlier")

    # --------------------------------------------------------------------------
    # TAB 5: MASALAH 9 & MASALAH 10
    # --------------------------------------------------------------------------
    with tab5:
        st.header("Masalah 9: Pengabaian Multimodalitas Fitur (Subpopulasi Tersembunyi)")
        mean_global_m9 = df_raw['Gr Liv Area'].mean()
        mean_per_grup_m9 = df_raw.groupby('Bldg Type')['Gr Liv Area'].mean()
        grup_data_m9 = [group['Gr Liv Area'].dropna() for name, group in df_raw.groupby('Bldg Type')]
        f_stat_m9, p_val_m9 = stats.f_oneway(*grup_data_m9)

        st.write(f"**Mean Global `Gr Liv Area`:** `{mean_global_m9:.2f} sqft`")
        st.write("**Mean Berdasarkan `Bldg Type` (ANOVA p-value = {:.5e}):**".format(p_val_m9))
        st.dataframe(mean_per_grup_m9.round(2), use_container_width=True)

        st.divider()
        st.header("Masalah 10: Penanganan Outlier Sejati (Dean De Cock) vs True Signal Rumah Mewah")
        outlier_anomali_mask = (df_raw['Gr Liv Area'] > 4000) & (df_raw['SalePrice'] < 300000)
        df_clean = df_raw[~outlier_anomali_mask].copy().reset_index(drop=True)

        outliers_df_m10 = df_clean[df_clean['Gr Liv Area'] > (df_clean['Gr Liv Area'].quantile(0.75) + 1.5 * (df_clean['Gr Liv Area'].quantile(0.75) - df_clean['Gr Liv Area'].quantile(0.25)))]

        m10_col1, m10_col2, m10_col3 = st.columns(3)
        m10_col1.metric("Anomali Dipangkas (De Cock)", f"{outlier_anomali_mask.sum()} baris")
        m10_col2.metric("Mean Harga Outlier Mewah", f"${outliers_df_m10['SalePrice'].mean():,.2f}")
        m10_col3.metric("Mean Harga Seluruh Populasi", f"${df_clean['SalePrice'].mean():,.2f}")

# ==============================================================================
# HALAMAN 3: REDUKSI DIMENSI BERBASIS PROYEKSI ORTOGONAL (PCA)
# ==============================================================================
elif menu == "3. Reduksi Dimensi (PCA)":
    st.title("📐 Masalah 11: Reduksi Dimensi Proyeksi Ortogonal (PCA)")
    st.markdown("Mengompresi fitur berkorelasi tinggi menjadi komponen utama yang saling **ortogonal (korelasi r ≈ 0)**.")

    outlier_anomali_mask = (df_raw['Gr Liv Area'] > 4000) & (df_raw['SalePrice'] < 300000)
    df_clean = df_raw[~outlier_anomali_mask].copy().reset_index(drop=True)

    fitur_pca_list = ["Gr Liv Area", "Total Bsmt SF", "Garage Area", "Lot Area", "1st Flr SF"]
    df_pca = df_clean[fitur_pca_list].dropna().copy()
    X_train_pca, X_test_pca = train_test_split(df_pca, test_size=0.2, random_state=42)

    pipe_pca = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, random_state=42))
    ])
    X_train_trans = pipe_pca.fit_transform(X_train_pca)
    pca_engine = pipe_pca.named_steps["pca"]
    ind_var = pca_engine.explained_variance_ratio_
    cum_var = np.cumsum(ind_var)

    col_pca1, col_pca2 = st.columns(2)
    with col_pca1:
        st.metric("Komponen Terpilih (Cutoff ≥ 95%)", f"{pca_engine.n_components_} Components")
        st.metric("Total Kumulatif Varians Terjaga", f"{cum_var[-1] * 100:.2f}%")
    with col_pca2:
        st.write("**Explained Variance Ratio per Komponen:**")
        st.dataframe(pd.DataFrame({"PC": [f"PC{i+1}" for i in range(len(ind_var))], "Variance Ratio": ind_var.round(4)}))

    # BAB 2.4: VISUALISASI SCREE PLOT
    st.subheader("📊 Scree Plot Kompresi PCA")
    fig_scree, ax_scree = plt.subplots(figsize=(8, 4))
    pc_range = range(1, len(ind_var) + 1)
    ax_scree.bar(pc_range, ind_var, alpha=0.5, label='Individual Var', color='navy')
    ax_scree.step(pc_range, cum_var, where='mid', label='Cumulative Var', color='crimson', linewidth=2)
    ax_scree.axhline(y=0.95, color='black', linestyle='--', label='95% Cutoff Threshold')
    ax_scree.set_xlabel('Indeks Komponen Utama (PC)')
    ax_scree.set_ylabel('Explained Variance Ratio')
    ax_scree.legend()
    ax_scree.grid(axis='y', linestyle=':')
    st.pyplot(fig_scree)

# ==============================================================================
# HALAMAN 4: KOMPARASI 3 PARADIGMA ML
# ==============================================================================
elif menu == "4. Komparasi 3 Paradigma ML":
    st.title("🏆 Masalah 12: Komparasi 3 Paradigma Algoritma Supervised Learning")
    st.markdown("Evaluasi performa model Parametrik Linear vs Berbasis Jarak (K-NN) vs Berbasis Pohon (Decision Tree & Random Forest).")

    outlier_anomali_mask = (df_raw['Gr Liv Area'] > 4000) & (df_raw['SalePrice'] < 300000)
    df_clean = df_raw[~outlier_anomali_mask].copy().reset_index(drop=True)

    fitur_numerik = ["Overall Qual", "Gr Liv Area", "Total Bsmt SF", "Garage Area", "1st Flr SF", "Year Built", "Lot Area"]
    X = df_clean[fitur_numerik]
    y = df_clean["SalePrice"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    fitur_skewed = ["Lot Area", "Gr Liv Area", "Total Bsmt SF"]
    fitur_reguler = [f for f in fitur_numerik if f not in fitur_skewed]

    preprocessor_linear_knn = ColumnTransformer(
        transformers=[
            ("skew_transformer", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("yeo_johnson", PowerTransformer(method="yeo-johnson"))
            ]), fitur_skewed),
            ("reguler_scaler", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), fitur_reguler)
        ]
    )
    preprocessor_tree = SimpleImputer(strategy="median")

    # Construct Models
    model_linear = TransformedTargetRegressor(
        regressor=Pipeline([("prep", preprocessor_linear_knn), ("reg", LinearRegression())]),
        func=np.log1p, inverse_func=np.expm1
    )
    model_knn = TransformedTargetRegressor(
        regressor=Pipeline([("prep", preprocessor_linear_knn), ("reg", KNeighborsRegressor(n_neighbors=5))]),
        func=np.log1p, inverse_func=np.expm1
    )
    model_tree = Pipeline([("prep", preprocessor_tree), ("reg", DecisionTreeRegressor(max_depth=10, random_state=42))])
    model_rf = Pipeline([("prep", preprocessor_tree), ("reg", RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1))])

    daftar_paradigma = {
        "1. Parametrik Linear (Linear Regression)": model_linear,
        "2. Berbasis Jarak (K-Nearest Neighbors)": model_knn,
        "3. Berbasis Pohon (Decision Tree)": model_tree,
        "4. Tree Ensemble (Random Forest)": model_rf
    }

    hasil_eval = []
    for nama, mdl in daftar_paradigma.items():
        mdl.fit(X_train, y_train)
        y_tr_pred = mdl.predict(X_train)
        y_ts_pred = mdl.predict(X_test)
        hasil_eval.append({
            "Paradigma / Algoritma": nama,
            "R2 Train": round(r2_score(y_train, y_tr_pred), 4),
            "R2 Test": round(r2_score(y_test, y_ts_pred), 4),
            "RMSE ($)": round(root_mean_squared_error(y_test, y_ts_pred), 2),
            "MAE ($)": round(mean_absolute_error(y_test, y_ts_pred), 2)
        })

    df_hasil_eval = pd.DataFrame(hasil_eval)

    # BAB 2.3: MENAMPILKAN TABEL EVALUASI
    st.subheader("📊 Tabel Ringkasan Performa Model")
    st.dataframe(df_hasil_eval, use_container_width=True)

    # BAB 2.4: MENAMPILKAN GRAFIK MATPLOTLIB/BAR CHART
    st.subheader("📈 Visualisasi Perbandingan Akurasi (R²) & Galat (RMSE)")
    fig_eval, axes_eval = plt.subplots(1, 2, figsize=(14, 5))

    x_pos = np.arange(len(df_hasil_eval))
    bar_w = 0.35
    axes_eval[0].bar(x_pos - bar_w/2, df_hasil_eval['R2 Train'], width=bar_w, label='Train R²', color='#2980b9', alpha=0.85)
    axes_eval[0].bar(x_pos + bar_w/2, df_hasil_eval['R2 Test'], width=bar_w, label='Test R²', color='#27ae60', alpha=0.85)
    axes_eval[0].set_xticks(x_pos)
    axes_eval[0].set_xticklabels(['Linear Reg', 'K-NN', 'Decision Tree', 'Random Forest'], rotation=15, fontweight='bold')
    axes_eval[0].set_ylabel('R² Score')
    axes_eval[0].set_title('Komparasi Akurasi (R²: Train vs Test)', fontweight='bold')
    axes_eval[0].set_ylim(0.70, 1.02)
    axes_eval[0].legend(loc='lower right')
    axes_eval[0].grid(axis='y', linestyle=':')

    axes_eval[1].bar(['Linear Reg', 'K-NN', 'Decision Tree', 'Random Forest'], df_hasil_eval['RMSE ($)'], color=['#e74c3c', '#f39c12', '#9b59b6', '#1abc9c'], alpha=0.85)
    axes_eval[1].set_ylabel('RMSE ($)')
    axes_eval[1].set_title('Komparasi Galat Rata-rata (RMSE $)', fontweight='bold')
    axes_eval[1].grid(axis='y', linestyle=':')

    st.pyplot(fig_eval)

# ==============================================================================
# HALAMAN 5: SIMULATOR PREDIKSI HARGA REAL-TIME (DENGAN TAMPILAN BARU IDR & USD)
# ==============================================================================
elif menu == "5. Simulator Prediksi Harga Real-Time":
    st.title("⚡ Simulator Valuasi Real-Time Harga Rumah Ames")
    st.markdown("Masukkan karakteristik rumah untuk mendapatkan estimasi harga seketika dari model terlatih.")

    # Train Model Ensemble Random Forest untuk Backend Simulator
    outlier_anomali_mask = (df_raw['Gr Liv Area'] > 4000) & (df_raw['SalePrice'] < 300000)
    df_clean = df_raw[~outlier_anomali_mask].copy().reset_index(drop=True)
    fitur_numerik = ["Overall Qual", "Gr Liv Area", "Total Bsmt SF", "Garage Area", "1st Flr SF", "Year Built", "Lot Area"]
    X = df_clean[fitur_numerik]
    y = df_clean["SalePrice"]

    model_rf_final = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("reg", RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1))
    ])
    model_rf_final.fit(X, y)

    # BAB 2.8: CONTROL FLOW MENGGUNAKAN FORM
    with st.form(key="form_simulasi_harga"):
        st.subheader("📋 Input Spesifikasi Properti")
        col_in1, col_in2 = st.columns(2)

        with col_in1:
            overall_qual = st.slider("Skala Kualitas Keseluruhan (Overall Qual 1-10):", 1, 10, 7)
            gr_liv_area = st.number_input("Luas Area Tinggal Atas Tanah (Gr Liv Area sqft):", min_value=300, max_value=5000, value=1500, step=50)
            total_bsmt_sf = st.number_input("Luas Basement (Total Bsmt SF sqft):", min_value=0, max_value=3000, value=900, step=50)
            first_flr_sf = st.number_input("Luas Lantai Pertama (1st Flr SF sqft):", min_value=300, max_value=3000, value=1000, step=50)

        with col_in2:
            garage_area = st.number_input("Luas Garasi (Garage Area sqft):", min_value=0, max_value=1500, value=480, step=20)
            year_built = st.slider("Tahun Konstruksi Bangunan (Year Built):", 1872, 2010, 2003)
            lot_area = st.number_input("Luas Tanah (Lot Area sqft):", min_value=1000, max_value=100000, value=9500, step=100)

        # BAB 2.8: FORM SUBMIT BUTTON
        submit_simulasi = st.form_submit_button(label="🚀 Jalankan Estimasi Harga Properti")

    if submit_simulasi:
        data_input = pd.DataFrame([{
            "Overall Qual": overall_qual,
            "Gr Liv Area": gr_liv_area,
            "Total Bsmt SF": total_bsmt_sf,
            "Garage Area": garage_area,
            "1st Flr SF": first_flr_sf,
            "Year Built": year_built,
            "Lot Area": lot_area
        }])

        # Perhitungan Prediksi USD dan Konversi IDR
        prediksi_usd = model_rf_final.predict(data_input)[0]
        KURS_USD_TO_IDR = 15800  # Estimasi Kurs 1 USD = Rp 15.800
        prediksi_idr = prediksi_usd * KURS_USD_TO_IDR

        st.balloons()
        st.subheader("💰 Hasil Estimasi Valuasi Properti")

        # Tampilan Metrik Berdampingan
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric(
                label="Estimasi Harga (Rupiah)",
                value=f"Rp {prediksi_idr:,.0f}".replace(",", "."),
                delta=f"≈ {prediksi_idr / 1e9:.2f} Miliar IDR"
            )
        with col_res2:
            st.metric(
                label="Estimasi Harga (US Dollar)",
                value=f"${prediksi_usd:,.2f}"
            )

        st.divider()

        # Parameter Ringkasan Input
        st.subheader("📌 Ringkasan Spesifikasi Input:")
        st.dataframe(data_input, use_container_width=True)