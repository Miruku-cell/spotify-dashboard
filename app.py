import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

# Scikit-learn Tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# FP-Growth
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Page Configuration
st.set_page_config(page_title="Spotify Data Mining Dashboard", layout="wide")
st.title("🎵 Spotify End-to-End Data Mining Project Dashboard")

# 1. Cache Data Loading & Dynamic Feature Engineering
@st.cache_data
def load_data():
    csv_files = glob.glob("*.csv")
    if not csv_files:
        raise FileNotFoundError("CSV dataset not found in the repository root directory.")
    df = pd.read_csv(csv_files[0])
    
    # Feature Transformations
    if 'log_stream_count' not in df.columns and 'stream_count' in df.columns:
        df['log_stream_count'] = np.log1p(df['stream_count'])
    if 'artist_track_count' not in df.columns and 'artist_name' in df.columns:
        df['artist_track_count'] = df['artist_name'].map(df['artist_name'].value_counts())
    if 'popularity_category' not in df.columns and 'popularity' in df.columns:
        df['popularity_category'] = pd.qcut(
            df['popularity'], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop'
        )
    return df

try:
    df = load_data()
    st.sidebar.success(f"Dataset successfully loaded: {len(df):,} total records.")
except Exception as e:
    st.error(f"Data loading failed: {e}")
    st.stop()

# Multi-Tab Layout for All 4 Coursework Tasks
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Association (FP-Growth)", 
    "2. Clustering (K-Means)", 
    "3. Classification (4 Classifiers)", 
    "4. Numeric Prediction (Regression)"
])

# ==============================================================================
# TAB 1: FP-GROWTH ASSOCIATION RULES
# ==============================================================================
with tab1:
    st.header("🛒 Frequent Pattern Mining (FP-Growth)")
    st.markdown("Discover co-occurrence patterns and strong association rules across binarized audio attributes.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        min_sup = st.slider("Minimum Support Threshold (minsup)", min_value=0.01, max_value=0.5, value=0.05, step=0.01)
    with col_b:
        min_conf = st.slider("Minimum Confidence Threshold (minconf)", min_value=0.1, max_value=1.0, value=0.3, step=0.05)
        
    candidate_num = [c for c in ['danceability', 'energy', 'speechiness', 'acousticness', 'popularity'] if c in df.columns]
    
    # Median split binarization
    df_bin = pd.DataFrame()
    for c in candidate_num:
        med = df[c].median()
        df_bin[f"High_{c}"] = df[c] >= med
        
    frequent_itemsets = fpgrowth(df_bin, min_support=min_sup, use_colnames=True)
    
    if not frequent_itemsets.empty:
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)
        if not rules.empty:
            rules_disp = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by='lift', ascending=False)
            rules_disp['antecedents'] = rules_disp['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules_disp['consequents'] = rules_disp['consequents'].apply(lambda x: ', '.join(list(x)))
            st.dataframe(rules_disp.head(20), use_container_width=True)
        else:
            st.warning("No association rules satisfied the minimum confidence threshold.")
    else:
        st.warning("No frequent itemsets satisfied the minimum support threshold.")

# ==============================================================================
# TAB 2: K-MEANS CLUSTERING
# ==============================================================================
with tab2:
    st.header("🎯 Cluster Analysis (K-Means)")
    st.markdown("Partition track audio profiles into natural clusters based on Euclidean distance similarity.")
    
    cluster_features = [c for c in ['danceability', 'energy', 'valence', 'popularity', 'stream_count'] if c in df.columns]
    selected_cluster_features = st.multiselect("Select Feature Dimensions", cluster_features, default=cluster_features[:3])
    k_val = st.slider("Number of Clusters (k)", min_value=2, max_value=8, value=3)
    
    if len(selected_cluster_features) >= 2:
        df_clust = df[selected_cluster_features].dropna()
        scaler = StandardScaler()
        scaled_clust = scaler.fit_transform(df_clust)
        
        kmeans = KMeans(n_clusters=k_val, random_state=42, n_init=10)
        df_clust['Cluster'] = kmeans.fit_predict(scaled_clust)
        
        col_c1, col_c2 = st.columns([1, 1])
        with col_c1:
            st.subheader("Cluster Distribution")
            st.write(df_clust['Cluster'].value_counts().rename("Track Count"))
        with col_c2:
            st.subheader("Bivariate Cluster Scatter Plot")
            fig, ax = plt.subplots(figsize=(6, 4))
            scatter = ax.scatter(
                df_clust[selected_cluster_features[0]], 
                df_clust[selected_cluster_features[1]], 
                c=df_clust['Cluster'], cmap='viridis', alpha=0.6
            )
            ax.set_xlabel(selected_cluster_features[0])
            ax.set_ylabel(selected_cluster_features[1])
            plt.colorbar(scatter, label='Cluster ID')
            st.pyplot(fig)
    else:
        st.error("Please select at least 2 features to perform cluster analysis.")

# ==============================================================================
# TAB 3: CLASSIFICATION (4 MODELS)
# ==============================================================================
with tab3:
    st.header("🏆 Predictive Mining: Classifier Benchmark (4 Models)")
    st.markdown("Compare Decision Tree, Naïve Bayes, KNN, and Random Forest on selected target classes.")
    
    target_var = st.selectbox("Target Label", ['popularity_category', 'genre'])
    candidate_features = ['danceability', 'popularity', 'stream_count', 'log_stream_count', 'artist_track_count']
    avail_f = [c for c in candidate_features if c in df.columns and c != target_var]
    selected_f = st.multiselect("Select Predictor Features", avail_f, default=avail_f[:4])
    
    if st.button("🚀 Train & Evaluate Classifiers", type="primary"):
        if len(selected_f) < 1:
            st.error("Please select at least 1 predictor feature.")
        else:
            df_eval = df.dropna(subset=selected_f + [target_var]).copy()
            if target_var == 'genre':
                top_genres = df_eval['genre'].value_counts().nlargest(8).index
                df_eval = df_eval[df_eval['genre'].isin(top_genres)]

            X = df_eval[selected_f]
            y = LabelEncoder().fit_transform(df_eval[target_var].astype(str))

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            models = {
                "Decision Tree (Gini)": DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42),
                "Naïve Bayes (Gaussian)": GaussianNB(),
                "K-Nearest Neighbors (k=5)": KNeighborsClassifier(n_neighbors=5),
                "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=100, random_state=42)
            }

            records = []
            for name, m in models.items():
                m.fit(X_train, y_train)
                preds = m.predict(X_test)
                acc = accuracy_score(y_test, preds)
                err = 1.0 - acc
                prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average='weighted', zero_division=0)
                records.append({
                    "Classifier Model": name,
                    "Accuracy (%)": round(acc * 100, 2),
                    "Error Rate (%)": round(err * 100, 2),
                    "Precision": round(prec, 4),
                    "Recall": round(rec, 4),
                    "F1-Score": round(f1, 4)
                })

            res_df = pd.DataFrame(records).sort_values(by="Accuracy (%)", ascending=False)
            
            c1, c2 = st.columns([1.1, 0.9])
            with c1:
                st.subheader("📋 Performance Evaluation Table")
                st.dataframe(res_df, use_container_width=True)
            with c2:
                st.subheader("📊 Accuracy Comparison Chart")
                fig, ax = plt.subplots(figsize=(6, 4))
                bars = ax.bar(res_df["Classifier Model"], res_df["Accuracy (%)"], color=['#1DB954', '#4682B4', '#FFA500', '#9370DB'])
                ax.set_ylim(0, 105)
                plt.xticks(rotation=20)
                for b in bars:
                    h = b.get_height()
                    ax.text(b.get_x() + b.get_width()/2.0, h + 1.5, f"{h:.2f}%", ha='center', va='bottom', fontweight='bold')
                st.pyplot(fig)

# ==============================================================================
# TAB 4: MULTIPLE LINEAR REGRESSION
# ==============================================================================
with tab4:
    st.header("📈 Numeric Prediction: Multiple Linear Regression")
    st.markdown("Estimate continuous target responses based on multidimensional audio attributes.")
    
    num_target = 'log_stream_count' if 'log_stream_count' in df.columns else 'popularity'
    st.info(f"Target Response Variable: **{num_target}**")
    
    reg_features = [c for c in ['danceability', 'energy', 'valence', 'artist_track_count'] if c in df.columns]
    selected_reg_features = st.multiselect("Select Independent Explanatory Variables", reg_features, default=reg_features[:3])
    
    if st.button("🚀 Train Linear Regression Model"):
        df_reg = df.dropna(subset=selected_reg_features + [num_target])
        X_reg = df_reg[selected_reg_features]
        y_reg = df_reg[num_target]
        
        X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
        
        reg_model = LinearRegression()
        reg_model.fit(X_tr, y_tr)
        y_pred = reg_model.predict(X_te)
        
        mae = mean_absolute_error(y_te, y_pred)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        r2 = r2_score(y_te, y_pred)
        
        rc1, rc2 = st.columns([1, 1])
        with rc1:
            st.subheader("Model Parameters")
            st.write(f"**Intercept ($b$):** {reg_model.intercept_:.4f}")
            for feat, coef in zip(selected_reg_features, reg_model.coef_):
                st.write(f"- Coefficient for **{feat}** ($w$): {coef:.4f}")
            st.markdown("---")
            st.subheader("Error & Goodness-of-Fit Metrics")
            st.write(f"- **Mean Absolute Error (MAE):** {mae:.4f}")
            st.write(f"- **Root Mean Squared Error (RMSE):** {rmse:.4f}")
            st.write(f"- **Coefficient of Determination ($R^2$):** {r2:.4f}")
            
        with rc2:
            st.subheader("Observed vs. Predicted Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(y_te, y_pred, alpha=0.3, color='#1DB954')
            ax.plot([y_te.min(), y_te.max()], [y_te.min(), y_te.max()], 'r--', lw=2)
            ax.set_xlabel("Observed Ground Truth")
            ax.set_ylabel("Predicted Value")
            st.pyplot(fig)
