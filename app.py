from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

DATA_PATH = Path(__file__).parent / "train_muestra.csv"

st.title(" Reducción de Dimensionalidad y Clasificación con PCA, K-Means y SVM")
st.write("Anny Gutierrez - 20211930078")

st.write(
    "Esta aplicacion aplica reduccion de dimensionalidad (PCA), agrupamiento "
    "(K-Means) y clasificacion (SVM) sobre una muestra del dataset MNIST "
    "(digitos escritos a mano)."
)


@st.cache_data
def cargar_datos():
    if not DATA_PATH.exists():
        st.error(
            f"No se encontro el archivo '{DATA_PATH.name}'. Sube una muestra "
            "reducida de train.csv al repositorio, junto a app.py."
        )
        st.stop()

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["label"]) / 255.0
    y = df["label"]
    return X, y


X, y = cargar_datos()
st.write("Numero de imagenes cargadas:", len(X))

# ---------- PCA ----------
st.header("1. Reduccion de dimensionalidad (PCA)")

n_components = st.slider("Numero de componentes PCA", 2, 100, 2)

pca = PCA(n_components=n_components, random_state=42)
X_pca = pca.fit_transform(X)

st.write(f"Varianza explicada: {pca.explained_variance_ratio_.sum():.2%}")

# Visualizacion 2D (siempre con las 2 primeras componentes)
fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="tab10", alpha=0.5, s=5)
legend = ax.legend(*scatter.legend_elements(), title="Digito")
ax.add_artist(legend)
ax.set_title("Proyeccion PCA (primeras 2 componentes)")
st.pyplot(fig)

# ---------- K-Means ----------
st.header("2. Agrupamiento (K-Means)")

k = st.slider("Numero de clusters (K-Means)", 2, 15, 10)

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)

fig2, ax2 = plt.subplots()
ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="tab10", alpha=0.5, s=5)
ax2.set_title("Clusters K-Means")
st.pyplot(fig2)

# ---------- SVM ----------
st.header("3. Clasificacion (SVM)")

if st.button("Entrenar SVM"):
    with st.spinner("Entrenando el modelo..."):
        X_train, X_test, y_train, y_test = train_test_split(
            X_pca, y, test_size=0.3, random_state=42, stratify=y
        )

        svm = SVC(kernel="rbf", C=1, gamma="scale")
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)

    st.write("Accuracy:", round(accuracy_score(y_test, y_pred), 4))

    st.subheader("Reporte de clasificacion")
    st.text(classification_report(y_test, y_pred))

    st.subheader("Matriz de confusion")
    fig3, ax3 = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax3)
    st.pyplot(fig3)
else:
    st.write("Presiona el boton para entrenar el modelo SVM con los datos reducidos por PCA.")
