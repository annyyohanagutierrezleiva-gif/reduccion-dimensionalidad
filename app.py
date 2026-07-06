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

st.set_page_config(page_title=" Reducción de Dimensionalidad y Clasificación con PCA, K-Means y SVM", layout="wide")


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


# ---------- Barra lateral: controles ----------
with st.sidebar:
    st.header("PCA + K-Means + SVM")
    st.caption("Anny Gutierrez - 20211930078")
    st.divider()

    n_components = st.slider("Componentes PCA", 2, 100, 2)
    k = st.slider("Clusters (K-Means)", 2, 15, 10)
    st.divider()
    entrenar = st.button("Entrenar SVM", use_container_width=True)

X, y = cargar_datos()

# ---------- Encabezado ----------
st.title("Clasificacion de digitos MNIST")
st.write(
    "Reduccion de dimensionalidad (PCA), agrupamiento (K-Means) y "
    "clasificacion (SVM) sobre una muestra del dataset MNIST. Ajusta los "
    "parametros desde la barra lateral."
)

col_a, col_b = st.columns(2)
col_a.metric("Imagenes cargadas", len(X))

pca = PCA(n_components=n_components, random_state=42)
X_pca = pca.fit_transform(X)
col_b.metric("Varianza explicada (PCA)", f"{pca.explained_variance_ratio_.sum():.2%}")

st.divider()

# ---------- Pestañas ----------
tab_pca, tab_kmeans, tab_svm = st.tabs(["PCA", "K-Means", "SVM"])

with tab_pca:
    st.subheader("Proyeccion PCA (primeras 2 componentes)")
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="tab10", alpha=0.5, s=5)
    legend = ax.legend(*scatter.legend_elements(), title="Digito")
    ax.add_artist(legend)
    st.pyplot(fig)

with tab_kmeans:
    st.subheader(f"Clusters K-Means (k={k})")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_pca)

    fig2, ax2 = plt.subplots()
    ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="tab10", alpha=0.5, s=5)
    st.pyplot(fig2)

with tab_svm:
    st.subheader("Clasificacion con SVM")

    if entrenar:
        with st.spinner("Entrenando el modelo..."):
            X_train, X_test, y_train, y_test = train_test_split(
                X_pca, y, test_size=0.3, random_state=42, stratify=y
            )

            svm = SVC(kernel="rbf", C=1, gamma="scale")
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)

        st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")

        col_rep, col_mat = st.columns(2)

        with col_rep:
            st.caption("Reporte de clasificacion")
            st.text(classification_report(y_test, y_pred))

        with col_mat:
            st.caption("Matriz de confusion")
            fig3, ax3 = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax3)
            st.pyplot(fig3)
    else:
        st.info("Presiona 'Entrenar SVM' en la barra lateral para ver los resultados.")
