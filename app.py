import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

# Scikit-learn Tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# FP-Growth Tools
from mlxtend.frequent_patterns import fpgrowth, association_rules

# 1. Dashboard Configuration & Beautiful Header
st.set_page_config(page_title="Spotify Mining Dashboard", layout="wide")
st.markdown("<h1 style='text-align: center; color: #1DB954;'>🎵 Spotify Music Listening Behavior Data Mining Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>UCSY IS-212 End-to-End Data Mining Project</p>", unsafe_allow_html=True)
st.divider()

# 2. Automated Dataset Loader
@st.cache_data
def load_data():
    csv_candidates = glob.glob("*.csv")
    if not csv_candidates:
        raise FileNotFoundError("CSV dataset not found in the repository root directory.")
    df = pd.read_csv(csv_candidates[0])
    
    # Preprocessing & Feature Engineering matching Colab
    if 'log_stream_count' not in df.columns and 'stream_count' in df.columns:
        df['log_stream_count'] = np.log1p(df['stream_count'])
    if 'artist_track_count' not in df.columns and 'artist_name' in df.columns:
        df['artist_track_count'] = df['artist_name'].map(df['artist_name'].value_counts())
    if 'popularity_category' not in df.columns and 'popularity' in df.columns:
        df['popularity_category'] = pd.qcut(df['popularity'], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# 3. Four Dedicated Mining Pages (Tabs)
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Frequent Pattern Mining (FP-Growth)",
    "2. Clustering (K-Means)",
    "3. Classification Benchmark",
    "4. Numeric Prediction (Regression)"
])

# ==============================================================================
# PAGE 1: FREQUENT PATTERN MINING (FP-GROWTH)
# ==============================================================================
with tab1:
    st.subheader("🛒 Frequent Pattern Mining (FP-Growth)")
    st.write("Select audio attributes to discover co-occurring patterns and clean association rules without raw labels.")
    
    available_num = [c for c in ['danceability', 'energy', 'speechiness', 'acousticness', 'valence', 'tempo', 'popularity'] if c in df.columns]
    
    # Left and Right Selector Boxes
    col_left, col_right = st.columns(2)
    with col_left:
        left_features = st.multiselect("Left Side Attributes (Subset A):", available_num, default=available_num[:2])
    with col_right:
        right_features = st.multiselect("Right Side Attributes (Subset B):", available_num, default=available_num[2:4] if len(available_num) >= 4 else available_num[:1])
    
    selected_mining_attrs = list(set(left_features + right_features))
    
    # Sliders for Support and Confidence
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        min_sup = st.slider("Minimum Support (minsup)", min_value=0.01, max_value=0.5, value=0.05, step=0.01)
    with col_s2:
        min_conf = st.slider("Minimum Confidence (minconf)", min_value=0.1, max_value=1.0, value=0.3, step=0.05)
        
    if st.button("🔍 Mine Association Rules", type="primary"):
        if len(selected_mining_attrs) < 2:
            st.warning("Please select at least 2 distinct attributes across the two boxes.")
        else:
            # Binarization using median split (Clean format without raw label strings)
            df_bin = pd.DataFrame()
            for col in selected_mining_attrs:
                df_bin[f"High_{col}"] = df[col] >= df[col].median()
                
            freq_itemsets = fpgrowth(df_bin, min_support=min_sup, use_colnames=True)
            
            if not freq_itemsets.empty:
                rules = association_rules(freq_itemsets, metric="confidence", min_threshold=min_conf)
                if not rules.empty:
                    # Clean presentation table
                    clean_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
                    clean_rules['antecedents'] = clean_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                    clean_rules['consequents'] = clean_rules['consequents'].apply(lambda x: ', '.join(list(x)))
                    clean_rules = clean_rules.sort_values(by='lift', ascending=False).reset_index(drop=True)
                    
                    st.success(f"Discovered {len(clean_rules)} valid association rules.")
                    st.dataframe(clean_rules.style.format({'support': '{:.4f}', 'confidence': '{:.4f}', 'lift': '{:.4f}'}), use_container_width=True)
                else:
                    st.info("No rules satisfied the minimum confidence threshold.")
            else:
                st.info("No frequent itemsets found matching the support threshold.")

# ==============================================================================
# PAGE 2: CLUSTERING (K-MEANS)
# ==============================================================================
with tab2:
    st.subheader("🎯 Cluster Analysis (K-Means)")
    st.write("Group tracks based on natural audio similarity profiles.")
    
    clust_options = [c for c in ['danceability', 'energy', 'valence', 'popularity', 'stream_count'] if c in df.columns]
    c_col1, c_col2 = st.columns([2, 1])
    with c_col1:
        chosen_clust_features = st.multiselect("Select Feature Dimensions:", clust_options, default=clust_options[:3])
    with c_col2:
        k_clusters = st.slider("Number of Clusters (k):", min_value=2, max_value=8, value=3)
        
    if len(chosen_clust_features) >= 2:
        df_k = df[chosen_clust_features].dropna()
        scaled_k = StandardScaler().fit_transform(df_k)
        
        km = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        df_k['Cluster'] = km.fit_predict(scaled_k)
        
        c_res1, c_res2 = st.columns([1, 1])
        with c_res1:
            st.markdown("**Cluster Size Distribution**")
            st.dataframe(df_k['Cluster'].value_counts().rename("Track Count"), use_container_width=True)
        with c_res2:
            st.markdown("**Bivariate Scatter Visualization**")
            fig, ax = plt.subplots(figsize=(6, 4))
            scatter = ax.scatter(df_k[chosen_clust_features[0]], df_k[chosen_clust_features[1]], c=df_k['Cluster'], cmap='viridis', alpha=0.5)
            ax.set_xlabel(chosen_clust_features[0])
            ax.set_ylabel(chosen_clust_features[1])
            plt.colorbar(scatter, label='Cluster ID')
            st.pyplot(fig)
    else:
        st.warning("Please choose at least 2 features to render the clustering model.")

# ==============================================================================
# PAGE 3: CLASSIFICATION (4 MODELS BENCHMARK)
# ==============================================================================
with tab3:
    st.subheader("🏆 Predictive Mining: Model Comparison")
    st.write("Benchmark Decision Tree, Naïve Bayes, KNN, and Random Forest.")
    
    # 1. Target Selector
    target_option = st.selectbox("1. Choose Target Label:", ['popularity_category', 'genre'])
    
    # 2. 4 Features Selector
    pred_pool = [c for c in ['danceability', 'energy', 'valence', 'stream_count', 'log_stream_count', 'artist_track_count', 'popularity'] if c in df.columns and c != target_option]
    selected_4_features = st.multiselect("2. Select 4 Predictor Features:", pred_pool, default=pred_pool[:4])
    
    if st.button("🚀 Run 4 Classifier Benchmark", type="primary"):
        if len(selected_4_features) != 4:
            st.error("Please ensure exactly 4 predictor features are selected.")
        else:
            df_clf = df.dropna(subset=selected_4_features + [target_option]).copy()
            if target_option == 'genre':
                top_genres = df_clf['genre'].value_counts().nlargest(8).index
                df_clf = df_clf[df_clf['genre'].isin(top_genres)]
                
            X = df_clf[selected_4_features]
            y = LabelEncoder().fit_transform(df_clf[target_option].astype(str))
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            classifiers = {
                "Decision Tree": DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42),
                "Naïve Bayes": GaussianNB(),
                "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
            }
            
            eval_metrics = []
            for name, clf in classifiers.items():
                clf.fit(X_train, y_train)
                preds = clf.predict(X_test)
                acc = accuracy_score(y_test, preds)
                prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average='weighted', zero_division=0)
                eval_metrics.append({
                    "Model": name,
                    "Accuracy (%)": round(acc * 100, 2),
                    "Error Rate (%)": round((1.0 - acc) * 100, 2),
                    "Precision": round(prec, 4),
                    "Recall": round(rec, 4),
                    "F1-Score": round(f1, 4)
                })
                
            benchmark_df = pd.DataFrame(eval_metrics).sort_values(by="Accuracy (%)", ascending=False)
            
            # Display Table and Chart Side-by-Side
            col_t, col_c = st.columns([1.1, 0.9])
            with col_t:
                st.markdown("**Evaluation Metric Summary**")
                st.dataframe(benchmark_df, use_container_width=True)
            with col_c:
                st.markdown("**Accuracy Comparison Chart**")
                fig, ax = plt.subplots(figsize=(6, 4))
                bars = ax.bar(benchmark_df["Model"], benchmark_df["Accuracy (%)"], color=['#1DB954', '#4682B4', '#FFA500', '#9370DB'])
                ax.set_ylim(0, 105)
                plt.xticks(rotation=15)
                for b in bars:
                    h = b.get_height()
                    ax.text(b.get_x() + b.get_width()/2.0, h + 1.5, f"{h:.2f}%", ha='center', va='bottom', fontweight='bold')
                st.pyplot(fig)

# ==============================================================================
# PAGE 4: MULTIPLE LINEAR REGRESSION
# ==============================================================================
with tab4:
    st.subheader("📈 Numeric Prediction: Multiple Linear Regression")
    st.write("Predict Popularity scores based on track features.")
    
    reg_candidates = [c for c in ['danceability', 'energy', 'valence', 'speechiness', 'acousticness', 'artist_track_count', 'log_stream_count'] if c in df.columns]
    selected_reg_inputs = st.multiselect("Select Independent Predictors for Popularity:", reg_candidates, default=reg_candidates[:3])
    
    if st.button("🚀 Train Linear Regression", type="primary"):
        if len(selected_reg_inputs) < 1:
            st.error("Please select at least 1 predictor variable.")
        else:
            df_reg = df.dropna(subset=selected_reg_inputs + ['popularity']).copy()
            X_r = df_reg[selected_reg_inputs]
            y_r = df_reg['popularity']
            
            X_tr, X_te, y_tr, y_te = train_test_split(X_r, y_r, test_size=0.2, random_state=42)
            
            lr = LinearRegression()
            lr.fit(X_tr, y_tr)
            y_pred = lr.predict(X_te)
            
            mae = mean_absolute_error(y_te, y_pred)
            rmse = np.sqrt(mean_squared_error(y_te, y_pred))
            r2 = r2_score(y_te, y_pred)
            
            r_col1, r_col2 = st.columns([1, 1])
            with r_col1:
                st.markdown("**Linear Formula Parameters**")
                st.write(f"- **Intercept ($b$):** `{lr.intercept_:.4f}`")
                for col_name, coef in zip(selected_reg_inputs, lr.coef_):
                    st.write(f"- Weight for **{col_name}** ($w$): `{coef:.4f}`")
                st.divider()
                st.markdown("**Evaluation Metrics**")
                st.write(f"- **MAE:** `{mae:.4f}`")
                st.write(f"- **RMSE:** `{rmse:.4f}`")
                st.write(f"- **$R^2$ Score:** `{r2:.4f}`")
            with r_col2:
                st.markdown("**Actual vs. Predicted Scatter Plot**")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(y_te, y_pred, alpha=0.3, color='#1DB954')
                ax.plot([y_te.min(), y_te.max()], [y_te.min(), y_te.max()], 'r--', lw=2)
                ax.set_xlabel("Actual Popularity")
                ax.set_ylabel("Predicted Popularity")
                st.pyplot(fig)
