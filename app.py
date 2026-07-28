from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import streamlit as st

from src.analyseData import build_feature_table, extraction_features
from src.modeling import evaluate_models, prepare_matrix, predict_label, train_final_models
from src.history import add_diagnostic, get_history


st.set_page_config(page_title="Analyse feuille de maïs", page_icon="🌽", layout="wide")


@st.cache_data(show_spinner=False)
def load_dataset(dataset_root: str):
    dataframe = build_feature_table(dataset_root)
    return dataframe


def train_models(dataframe):
    X, y, _ = prepare_matrix(dataframe)
    return train_final_models(X, y), evaluate_models(X, y)


def save_upload_to_tempfile(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(uploaded_file.getbuffer())
        return Path(handle.name)


def save_image_to_dataset(file_bytes: bytes, filename: str, label: int) -> Path:
    """Save uploaded image to the appropriate dataset folder (saines or malades)."""
    label_folder = "saines" if label == 0 else "malades"
    target_folder = Path("dataset") / label_folder
    target_folder.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename to avoid conflicts
    suffix = Path(filename).suffix or ".jpg"
    unique_name = f"upload_{uuid.uuid4().hex[:8]}{suffix}"
    target_path = target_folder / unique_name
    
    with open(target_path, "wb") as f:
        f.write(file_bytes)
    
    return target_path


# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Diagnostic", "Galerie"])

dataset_root = st.sidebar.text_input("Dossier dataset", value="dataset")
st.sidebar.caption("L'entraînement se lance depuis cette page sur les images présentes dans le dossier dataset.")

dataframe = load_dataset(dataset_root)
if dataframe.empty:
    st.warning("Aucune image n'a été trouvée dans le dataset.")
    st.stop()

if "trained_models" not in st.session_state:
    st.session_state.trained_models = None
if "model_reports" not in st.session_state:
    st.session_state.model_reports = None

train_clicked = st.sidebar.button("Entraîner le modèle", type="primary")
if train_clicked or st.session_state.trained_models is None:
    with st.spinner("Entraînement et comparaison des modèles en cours..."):
        models, reports = train_models(dataframe)
        st.session_state.trained_models = models
        st.session_state.model_reports = reports

models = st.session_state.trained_models
reports = st.session_state.model_reports

if models is None or reports is None:
    st.info("Cliquez sur 'Entraîner le modèle' dans la barre latérale pour lancer l'apprentissage.")
    st.stop()

# Page: Diagnostic
if page == "Diagnostic":
    st.title("Classification d'images de feuilles de maïs")
    st.write(
        "Pipeline complet: extraction de caractéristiques, arbre Max-Minority, comparaison "
        "avec les modèles standards et prédiction interactive sur une image chargée."
    )

    st.success("Modèles entraînés et prêts pour la prédiction.")

    left_column, right_column = st.columns([1.1, 0.9])

    with left_column:
        st.subheader("Tableau de features")
        st.dataframe(dataframe, use_container_width=True)

    with right_column:
        st.subheader("Comparaison des modèles")
        for report in reports:
            st.metric(report.name, f"{report.accuracy:.2%}")
            st.caption(
                f"Precision: {report.precision:.2%} | Recall: {report.recall:.2%} | F1: {report.f1:.2%}"
            )
            st.text(f"Matrice: {report.confusion.tolist()}")

    st.divider()
    st.subheader("Prédire une nouvelle image")
    uploaded_file = st.file_uploader("Charger une image", type=["png", "jpg", "jpeg", "webp", "bmp"])

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        temp_path = save_upload_to_tempfile(uploaded_file)
        try:
            features = extraction_features(temp_path, label=None)
            feature_vector = [
                features["pct_rouille"],
                features["rugosite"],
                features["saturation_moyenne"],
                features["indice_chlorose"],
                features["valeur_moyenne"],
                features["surface_feuille"],
            ]

            st.image(uploaded_file, caption="Image chargée", use_container_width=True)

            prediction_labels = {0: "Feuille saine", 1: "Feuille malade"}
            for name, model in models.items():
                prediction, probability = predict_label(model, feature_vector)
                label = prediction_labels.get(prediction, str(prediction))
                if probability is None:
                    st.success(f"{name}: {label}")
                else:
                    st.success(f"{name}: {label} (confiance {probability:.2%})")

            chosen_model_name = st.selectbox("Choisir un modèle pour la décision finale", list(models.keys()))
            chosen_prediction, chosen_probability = predict_label(models[chosen_model_name], feature_vector)
            chosen_label = prediction_labels.get(chosen_prediction, str(chosen_prediction))
            
            # Save image to dataset and add to history
            saved_path = save_image_to_dataset(file_bytes, uploaded_file.name, chosen_prediction)
            add_diagnostic(
                image_name=uploaded_file.name,
                prediction=chosen_label,
                confidence=chosen_probability,
                model_name=chosen_model_name,
                image_path=str(saved_path),
            )
            
            st.info(
                f"Décision retenue avec {chosen_model_name}: {chosen_label}"
                + (
                    f" (confiance {chosen_probability:.2%})"
                    if chosen_probability is not None
                    else ""
                )
            )
        finally:
            temp_path.unlink(missing_ok=True)

# Page: Galerie
elif page == "Galerie":
    st.title("Galerie des diagnostics")
    
    history = get_history()
    
    if not history:
        st.info("Aucun diagnostic effectué pour le moment. Upload une image depuis la page 'Diagnostic' pour commencer.")
    else:
        # Statistics
        saine_count = sum(1 for item in history if item["prediction"] == "Feuille saine")
        malade_count = sum(1 for item in history if item["prediction"] == "Feuille malade")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Feuilles saines", saine_count)
        with col2:
            st.metric("Feuilles malades", malade_count)
        with col3:
            st.metric("Total diagnostics", len(history))
        
        st.divider()
        
        # Display gallery
        for item in reversed(history):  # Show newest first
            with st.container():
                cols = st.columns([1, 3])
                with cols[0]:
                    if Path(item["image_path"]).exists():
                        st.image(str(item["image_path"]), use_container_width=True)
                    else:
                        st.write("Image non disponible")
                with cols[1]:
                    st.markdown(f"**{item['image_name']}**")
                    st.write(f"Verdict: {item['prediction']}")
                    st.write(f"Modèle: {item['model_name']}")
                    if item['confidence'] is not None:
                        st.write(f"Confiance: {item['confidence']:.2%}")
                    st.write(f"Date: {item['timestamp']}")
                st.divider()