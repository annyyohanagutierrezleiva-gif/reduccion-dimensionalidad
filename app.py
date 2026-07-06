import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

st.title("PCA + K-Means + SVM sobre MNIST - Jeimy Palma")

@st.cache_data
def cargar_datos():
    df = pd.read_csv("train_muestra.csv")  # debes subir un train.csv reducido al repo
    X = df.drop(columns=["label"]) / 255.0
    y = df["label"]
    return X, y

X, y = cargar_datos()

n_components = st.slider("Número de componentes PCA", 2, 100, 2)

pca = PCA(n_components=n_components, random_state=42)
X_pca = pca.fit_transform(X)

st.write(f"Varianza explicada: {pca.explained_variance_ratio_.sum():.2%}")

# Visualización 2D (siempre con las 2 primeras componentes)
fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap="tab10", alpha=0.5, s=5)
legend = ax.legend(*scatter.legend_elements(), title="Dígito")
ax.add_artist(legend)
st.pyplot(fig)

# K-Means
k = st.slider("Número de clusters (K-Means)", 2, 15, 10)
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)

fig2, ax2 = plt.subplots()
ax2.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap="tab10", alpha=0.5, s=5)
ax2.set_title("Clusters K-Means")
st.pyplot(fig2)

# SVM
if st.button("Entrenar SVM"):
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.3, random_state=42, stratify=y
    )
    svm = SVC(kernel="rbf", C=1, gamma="scale")
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.text(classification_report(y_test, y_pred))

    fig3, ax3 = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax3)
    st.pyplot(fig3)
