import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Spotify Mining Dashboard", layout="wide")
st.title("🎵 Spotify Predictive Mining Dashboard")

@st.cache_data
def load_data():
    csv_candidates = glob.glob("*.csv")
    if not csv_candidates:
        raise FileNotFoundError("CSV dataset not found in repository.")
    df = pd.read_csv(csv_candidates[0])
    
    if 'log_stream_count' not in df.columns and 'stream_count' in df.columns:
        df['log_stream_count'] = np.log1p(df['stream_count'])
    if 'artist_track_count' not in df.columns and 'artist_name' in df.columns:
        df['artist_track_count'] = df['artist_name'].map(df['artist_name'].value_counts())
    if 'popularity_category' not in df.columns and 'popularity' in df.columns:
        df['popularity_category'] = pd.qcut(df['popularity'], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
    return df

try:
    df = load_data()
    st.sidebar.header("🎛️ Settings")
    target_var = st.sidebar.selectbox("1. Choose Target Label", ['popularity_category', 'genre'])
    
    candidate_features = ['danceability', 'popularity', 'stream_count', 'log_stream_count', 'artist_track_count']
    available_features = [c for c in candidate_features if c in df.columns and c != target_var]
    
    selected_features = st.sidebar.multiselect("2. Choose Predictor Features", available_features, default=available_features[:4])

    if st.sidebar.button("🚀 Train 4 Models & Compare", type="primary"):
        if len(selected_features) < 1:
            st.error("Please select at least 1 feature.")
        else:
            df_eval = df.dropna(subset=selected_features + [target_var]).copy()
            if target_var == 'genre':
                top_genres = df_eval['genre'].value_counts().nlargest(8).index
                df_eval = df_eval[df_eval['genre'].isin(top_genres)]

            X = df_eval[selected_features]
            y = LabelEncoder().fit_transform(df_eval[target_var].astype(str))

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            models = {
                "Decision Tree (Gini)": DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42),
                "Naïve Bayes": GaussianNB(),
                "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
            }

            records = []
            for name, m in models.items():
                m.fit(X_train, y_train)
                preds = m.predict(X_test)
                acc = accuracy_score(y_test, preds)
                prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average='weighted', zero_division=0)
                records.append({
                    "Classifier Model": name,
                    "Accuracy (%)": round(acc * 100, 2),
                    "Precision": round(prec, 4),
                    "Recall": round(rec, 4),
                    "F1-Score": round(f1, 4)
                })

            res_df = pd.DataFrame(records).sort_values(by="Accuracy (%)", ascending=False)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("📋 Evaluation Table")
                st.dataframe(res_df, use_container_width=True)
            with col2:
                st.subheader("📊 Accuracy Chart")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(res_df["Classifier Model"], res_df["Accuracy (%)"], color=['#1DB954', '#4682B4', '#FFA500', '#9370DB'])
                ax.set_ylim(0, 105)
                plt.xticks(rotation=20)
                st.pyplot(fig)
except Exception as e:
    st.error(f"Error loading data: {e}")
