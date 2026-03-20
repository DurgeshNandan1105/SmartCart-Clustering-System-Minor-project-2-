import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Title
st.title("🛒 SmartCart Customer Segmentation (Clustering)")

# Upload dataset
file = st.file_uploader("Upload Customer Dataset (CSV)", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    # ===============================
    # SHOW DATA
    # ===============================
    st.subheader("📊 Raw Data")
    st.dataframe(df.head())

    # ===============================
    # DATA CLEANING (same as your notebook)
    # ===============================
    st.subheader("🧹 Data Cleaning")

    df = df.dropna()

    # Drop unnecessary columns (adjust if needed)
    drop_cols = ["ID", "Dt_Customer"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    st.write("After cleaning:", df.shape)

    # ===============================
    # ENCODING (for categorical columns)
    # ===============================
    df_encoded = pd.get_dummies(df, drop_first=True)

    # ===============================
    # SCALING
    # ===============================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)

    # ===============================
    # ELBOW METHOD
    # ===============================
    st.subheader("📉 Elbow Method")

    inertia = []
    K = range(1, 11)

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    fig, ax = plt.subplots()
    ax.plot(K, inertia, marker='o')
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method")

    st.pyplot(fig)

    # ===============================
    # SELECT K
    # ===============================
    st.subheader("🎯 Choose Number of Clusters")

    k = st.slider("Select K", 2, 10, 3)

    # ===============================
    # KMEANS MODEL
    # ===============================
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    df["Cluster"] = clusters

    # ===============================
    # SHOW RESULT
    # ===============================
    st.subheader("📊 Clustered Data")
    st.dataframe(df.head())

    # ===============================
    # VISUALIZATION (2D)
    # ===============================
    st.subheader("📈 Cluster Visualization")

    numeric_cols = df.select_dtypes(include=['number']).columns

    col1 = st.selectbox("X-axis", numeric_cols)
    col2 = st.selectbox("Y-axis", numeric_cols)

    fig2, ax2 = plt.subplots()

    sns.scatterplot(
        x=df[col1],
        y=df[col2],
        hue=df["Cluster"],
        palette="Set2",
        ax=ax2
    )

    ax2.set_title("Customer Segments")
    st.pyplot(fig2)

    # ===============================
    # CLUSTER COUNT
    # ===============================
    st.subheader("📌 Cluster Distribution")
    st.write(df["Cluster"].value_counts())

else:
    st.info("👆 Upload your dataset to start")
