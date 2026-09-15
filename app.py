import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

# Scikit-learn Tools
# Scikit-learn Tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix, 
    classification_report,
    roc_curve, 
    auc
)
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix, 
    classification_report,
    roc_curve, 
    auc
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# FP-Growth Tools
from mlxtend.frequent_patterns import fpgrowth, association_rules

# 1. Page Configuration & Header
st.set_page_config(page_title="Spotify Mining Dashboard", layout="wide")
st.markdown("<h1 style='text-align: center; color: #1DB954;'>🎵 Spotify Music Listening Behavior Data Mining Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>UCSY IS-212 End-to-End Data Mining Project</p>", unsafe_allow_html=True)
st.divider()

# 2. Strict Dataset Loader (Only Allowed Clean Columns)
ALLOWED_COLS = [
    'track_name', 'artist_name', 'album_name', 'genre', 'popularity', 
    'danceability', 'stream_count', 'label', 'popularity_category', 
    'loudness_category', 'log_stream_count', 'artist_track_count'
]

@st.cache_data
def load_data():
    csv_candidates = glob.glob("*.csv")
    if not csv_candidates:
        raise FileNotFoundError("CSV dataset not found in the repository root directory.")
    df = pd.read_csv(csv_candidates[0])
    
    # Feature Engineering (ensure columns exist if raw CSV lacks them)
    if 'log_stream_count' not in df.columns and 'stream_count' in df.columns:
        df['log_stream_count'] = np.log1p(df['stream_count'])
    if 'artist_track_count' not in df.columns and 'artist_name' in df.columns:
        df['artist_track_count'] = df['artist_name'].map(df['artist_name'].value_counts())
    if 'popularity_category' not in df.columns and 'popularity' in df.columns:
        df['popularity_category'] = pd.qcut(df['popularity'], q=3, labels=['Low_Popularity', 'Medium_Popularity', 'High_Popularity'], duplicates='drop')
    if 'loudness_category' not in df.columns and 'loudness' in df.columns:
        df['loudness_category'] = pd.qcut(df['loudness'], q=3, labels=['Low_Loudness', 'Medium_Loudness', 'High_Loudness'], duplicates='drop')
    if 'label' not in df.columns:
        df['label'] = 'Major_Label'
        
    # Strictly filter only the cleaned dataset columns
    valid_cols = [c for c in ALLOWED_COLS if c in df.columns]
    df = df[valid_cols].dropna().copy()
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
    "4. Model Evaluation (Random Forest)"
])

# ==============================================================================
# PAGE 1: FP-GROWTH (12 Specific Association Pairs)
# ==============================================================================
with tab1:
    st.subheader("🛒 Frequent Pattern Mining (FP-Growth)")
    st.write("Mine targeted association rules based on the 12 pairwise attribute combinations.")
    
    PAIR_OPTIONS = {
        "1. genre ⇒ popularity_category": ('genre', 'popularity_category'),
        "2. popularity_category ⇒ genre": ('popularity_category', 'genre'),
        "3. genre ⇒ loudness_category": ('genre', 'loudness_category'),
        "4. loudness_category ⇒ genre": ('loudness_category', 'genre'),
        "5. genre ⇒ label": ('genre', 'label'),
        "6. label ⇒ genre": ('label', 'genre'),
        "7. loudness_category ⇒ popularity_category": ('loudness_category', 'popularity_category'),
        "8. popularity_category ⇒ loudness_category": ('popularity_category', 'loudness_category'),
        "9. label ⇒ popularity_category": ('label', 'popularity_category'),
        "10. popularity_category ⇒ label": ('popularity_category', 'label'),
        "11. label ⇒ loudness_category": ('label', 'loudness_category'),
        "12. loudness_category ⇒ label": ('loudness_category', 'label'),
    }
    
    # Left Box and Right Box Selectors
    col_left, col_right = st.columns(2)
    with col_left:
        selected_left_attr = st.selectbox("Left Attribute (Antecedent X):", ['genre', 'popularity_category', 'loudness_category', 'label'], index=0)
    with col_right:
        right_candidates = [c for c in ['genre', 'popularity_category', 'loudness_category', 'label'] if c != selected_left_attr]
        selected_right_attr = st.selectbox("Right Attribute (Consequent Y):", right_candidates, index=0)
    
    st.info(f"Selected Pair Rule Target: **{selected_left_attr} ⇒ {selected_right_attr}**")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        min_sup = st.slider("Minimum Support Threshold (minsup):", min_value=0.005, max_value=0.5, value=0.02, step=0.005)
    with col_s2:
        min_conf = st.slider("Minimum Confidence Threshold (minconf):", min_value=0.05, max_value=1.0, value=0.10, step=0.05)
        
    if st.button("🔍 Mine Association Rules", type="primary"):
        # One-hot encode only the selected pair attributes for clean results
        df_sub = df[[selected_left_attr, selected_right_attr]].astype(str)
        df_encoded = pd.get_dummies(df_sub)
        
        freq_itemsets = fpgrowth(df_encoded, min_support=min_sup, use_colnames=True)
        
        if not freq_itemsets.empty:
            rules = association_rules(freq_itemsets, metric="confidence", min_threshold=min_conf)
            if not rules.empty:
                # Filter specifically matching selected Antecedent => Consequent direction
                rules['ant_str'] = rules['antecedents'].apply(lambda x: list(x)[0])
                rules['con_str'] = rules['consequents'].apply(lambda x: list(x)[0])
                
                filtered_rules = rules[
                    rules['ant_str'].str.startswith(selected_left_attr) & 
                    rules['con_str'].str.startswith(selected_right_attr)
                ].copy()
                
                if not filtered_rules.empty:
                    # Clean presentation formatting
                    filtered_rules['antecedents'] = filtered_rules['ant_str'].str.replace(f"{selected_left_attr}_", "")
                    filtered_rules['consequents'] = filtered_rules['con_str'].str.replace(f"{selected_right_attr}_", "")
                    
                    display_tbl = filtered_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by='lift', ascending=False).reset_index(drop=True)
                    st.success(f"Found {len(display_tbl)} clean association rule(s).")
                    st.dataframe(display_tbl.style.format({'support': '{:.4f}', 'confidence': '{:.4f}', 'lift': '{:.4f}'}), use_container_width=True)
                else:
                    st.warning(f"No rules found matching '{selected_left_attr} ⇒ {selected_right_attr}' under current thresholds.")
            else:
                st.warning("No rules exceeded the confidence threshold.")
        else:
            st.warning("No frequent itemsets found matching the support threshold.")

# ==============================================================================
# PAGE 2: CLUSTERING (K-MEANS)
# ==============================================================================
with tab2:
    st.subheader("🎯 Cluster Analysis (K-Means)")
    st.write("Segment tracks into distinct natural clusters based on numeric attributes.")
    
    numeric_for_clust = [c for c in ['danceability', 'popularity', 'stream_count', 'artist_track_count', 'log_stream_count'] if c in df.columns]
    c1, c2 = st.columns([2, 1])
    with c1:
        chosen_k_features = st.multiselect("Clustering Features:", numeric_for_clust, default=['danceability', 'popularity'])
    with c2:
        k_val = st.slider("Number of Clusters (k):", 2, 8, 3)
        
    if len(chosen_k_features) >= 2:
        X_clust = df[chosen_k_features]
        scaled_X = StandardScaler().fit_transform(X_clust)
        
        km = KMeans(n_clusters=k_val, random_state=42, n_init=10)
        clusters = km.fit_predict(scaled_X)
        
        c_res1, c_res2 = st.columns([1, 1])
        with c_res1:
            st.markdown("**Cluster Size Distribution**")
            st.dataframe(pd.Series(clusters).value_counts().rename("Track Count"), use_container_width=True)
        with c_res2:
            st.markdown("**Bivariate Scatter Visualization**")
            fig, ax = plt.subplots(figsize=(6, 4))
            scatter = ax.scatter(df[chosen_k_features[0]], df[chosen_k_features[1]], c=clusters, cmap='viridis', alpha=0.5)
            ax.set_xlabel(chosen_k_features[0])
            ax.set_ylabel(chosen_k_features[1])
            plt.colorbar(scatter, label='Cluster ID')
            st.pyplot(fig)
    else:
        st.warning("Please select at least 2 features for K-Means.")

# ==============================================================================
# PAGE 3: CLASSIFICATION (4 MODELS BENCHMARK)
# ==============================================================================
with tab3:
    st.subheader("🏆 Predictive Mining: Classifier Benchmark")
    
    # 1. Target Selector
    target_var = st.selectbox("1. Select Target Class:", ['genre', 'popularity_category'])
    
    # 2. 4 Predictor Features Selector
    available_pred = [c for c in ['danceability', 'popularity', 'stream_count', 'log_stream_count', 'artist_track_count'] if c != target_var and c in df.columns]
    selected_4 = st.multiselect("2. Select 4 Predictor Features:", available_pred, default=available_pred[:4])
    
    if st.button("🚀 Train & Compare 4 Classifiers", type="primary"):
        if len(selected_4) != 4:
            st.error("Please ensure exactly 4 predictor features are selected.")
        else:
            df_eval = df.dropna(subset=selected_4 + [target_var]).copy()
            if target_var == 'genre':
                top_genres = df_eval['genre'].value_counts().nlargest(6).index
                df_eval = df_eval[df_eval['genre'].isin(top_genres)]
                
            X = df_eval[selected_4]
            y = LabelEncoder().fit_transform(df_eval[target_var].astype(str))
            
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            
            models = {
                "Decision Tree": DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42),
                "Naïve Bayes": GaussianNB(),
                "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
            }
            
            records = []
            for name, m in models.items():
                m.fit(X_tr, y_tr)
                preds = m.predict(X_te)
                acc = accuracy_score(y_te, preds)
                prec, rec, f1, _ = precision_recall_fscore_support(y_te, preds, average='weighted', zero_division=0)
                records.append({
                    "Model": name,
                    "Accuracy (%)": round(acc * 100, 2),
                    "Error Rate (%)": round((1.0 - acc) * 100, 2),
                    "Precision": round(prec, 4),
                    "Recall": round(rec, 4),
                    "F1-Score": round(f1, 4)
                })
                
            res_df = pd.DataFrame(records).sort_values(by="Accuracy (%)", ascending=False)
            
            col_t, col_c = st.columns([1.1, 0.9])
            with col_t:
                st.markdown("**Evaluation Metric Summary**")
                st.dataframe(res_df, use_container_width=True)
            with col_c:
                st.markdown("**Accuracy Comparison Chart**")
                fig, ax = plt.subplots(figsize=(6, 4))
                bars = ax.bar(res_df["Model"], res_df["Accuracy (%)"], color=['#1DB954', '#4682B4', '#FFA500', '#9370DB'])
                ax.set_ylim(0, 105)
                plt.xticks(rotation=15)
                for b in bars:
                    h = b.get_height()
                    ax.text(b.get_x() + b.get_width()/2.0, h + 1.5, f"{h:.2f}%", ha='center', va='bottom', fontweight='bold')
                st.pyplot(fig)

# ==============================================================================
# PAGE 4: IN-DEPTH EVALUATION (RANDOM FOREST)
# ==============================================================================
with tab4:
    st.subheader("📊 In-Depth Model Evaluation (Random Forest)")
    st.write("Comprehensive performance diagnostics: Confusion Matrix, Feature Importance, and Multiclass ROC Curves.")
    
    # Clean Features (Data leakage မဖြစ်အောင် popularity ကို ဖယ်ထုတ်ထားသည်)
    rf_features = ['danceability', 'log_stream_count', 'artist_track_count']
    target_rf = 'popularity_category'
    
    if st.button("🚀 Run Comprehensive Evaluation", type="primary"):
        df_rf = df.dropna(subset=rf_features + [target_rf]).copy()
        
        X_eval = df_rf[rf_features]
        le_rf = LabelEncoder()
        y_eval = le_rf.fit_transform(df_rf[target_rf].astype(str))
        class_labels = [str(c) for c in le_rf.classes_]
        
        X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
            X_eval, y_eval, test_size=0.2, random_state=42, stratify=y_eval
        )
        
        rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        rf.fit(X_train_rf, y_train_rf)
        
        y_pred_rf = rf.predict(X_test_rf)
        y_proba_rf = rf.predict_proba(X_test_rf)
        
        acc_rf = accuracy_score(y_test_rf, y_pred_rf)
        err_rf = 1.0 - acc_rf
        p_rf, r_rf, f1_rf, _ = precision_recall_fscore_support(y_test_rf, y_pred_rf, average='weighted', zero_division=0)
        
        # Summary Metrics Row
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{acc_rf * 100:.2f}%")
        m2.metric("Error Rate", f"{err_rf * 100:.2f}%")
        m3.metric("Precision", f"{p_rf:.4f}")
        m4.metric("Recall", f"{r_rf:.4f}")
        m5.metric("F1-Score", f"{f1_rf:.4f}")
        
        st.divider()
        
        # 3 Diagnostic Plots
        p_col1, p_col2, p_col3 = st.columns(3)
        
        with p_col1:
            st.markdown("**Confusion Matrix Heatmap**")
            fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.8))
            cm_data = confusion_matrix(y_test_rf, y_pred_rf)
            sns.heatmap(cm_data, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                        xticklabels=class_labels, yticklabels=class_labels)
            ax_cm.set_xlabel("Predicted Label")
            ax_cm.set_ylabel("True Ground Truth")
            st.pyplot(fig_cm)
            
        with p_col2:
            st.markdown("**Feature Importance Ranking**")
            fig_fi, ax_fi = plt.subplots(figsize=(4.5, 3.8))
            fi_series = pd.Series(rf.feature_importances_, index=rf_features).sort_values(ascending=True)
            fi_series.plot(kind='barh', ax=ax_fi, color='#1DB954', edgecolor='black')
            ax_fi.set_xlabel("Gini Importance Score")
            st.pyplot(fig_fi)
            
        with p_col3:
            st.markdown("**Multiclass ROC Curve Analysis**")
            fig_roc, ax_roc = plt.subplots(figsize=(4.5, 3.8))
            y_bin = label_binarize(y_test_rf, classes=range(len(class_labels)))
            curve_colors = ['#1DB954', '#4682B4', '#FFA500', '#9370DB']
            
            for idx in range(len(class_labels)):
                fpr, tpr, _ = roc_curve(y_bin[:, idx], y_proba_rf[:, idx])
                roc_auc_val = auc(fpr, tpr)
                ax_roc.plot(fpr, tpr, lw=2, color=curve_colors[idx % len(curve_colors)],
                            label=f'{class_labels[idx]} (AUC = {roc_auc_val:.2f})')
                
            ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.2)
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel("FPR")
            ax_roc.set_ylabel("TPR")
            ax_roc.legend(loc="lower right", fontsize='small')
            st.pyplot(fig_roc)
