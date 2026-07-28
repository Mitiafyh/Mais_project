from __future__ import annotations

import base64
import io
import shutil
import sys
import tempfile
import uuid
from functools import lru_cache
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analyseData import build_feature_table, extraction_features
from src.modeling import evaluate_models, prepare_matrix, predict_label, train_final_models
from src.history import add_diagnostic, get_history


# Create static folder for history images
STATIC_FOLDER = PROJECT_ROOT / "static"
HISTORY_FOLDER = STATIC_FOLDER / "history"
STATIC_FOLDER.mkdir(parents=True, exist_ok=True)
HISTORY_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder=str(STATIC_FOLDER))
app.config["DATASET_ROOT"] = PROJECT_ROOT / "dataset"


def image_to_data_uri(uploaded_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(uploaded_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def save_image_to_history(file_bytes: bytes, filename: str) -> Path:
    """Save uploaded image to the history folder for gallery display."""
    suffix = Path(filename).suffix or ".jpg"
    unique_name = f"upload_{uuid.uuid4().hex[:8]}{suffix}"
    target_path = HISTORY_FOLDER / unique_name
    
    with open(target_path, "wb") as f:
        f.write(file_bytes)
    
    return target_path


def save_image_to_dataset(file_bytes: bytes, filename: str, label: int) -> Path:
    """Save uploaded image to the appropriate dataset folder (saines or malades)."""
    label_folder = "saines" if label == 0 else "malades"
    target_folder = app.config["DATASET_ROOT"] / label_folder
    target_folder.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename to avoid conflicts
    suffix = Path(filename).suffix or ".jpg"
    unique_name = f"upload_{uuid.uuid4().hex[:8]}{suffix}"
    target_path = target_folder / unique_name
    
    with open(target_path, "wb") as f:
        f.write(file_bytes)
    
    return target_path


@lru_cache(maxsize=1)
def train_and_score_models(dataset_root: str):
    dataframe = build_feature_table(dataset_root)
    if dataframe.empty:
        return None

    X, y, _ = prepare_matrix(dataframe)
    reports = evaluate_models(X, y)
    models = train_final_models(X, y)
    best_report = max(reports, key=lambda report: report.accuracy)
    best_model = models[best_report.name]

    return {
        "dataframe": dataframe,
        "reports": reports,
        "models": models,
        "best_report": best_report,
        "best_model_name": best_report.name,
        "best_model": best_model,
        "dataset_size": len(dataframe),
    }


def clear_training_cache() -> None:
    train_and_score_models.cache_clear()


@app.route("/", methods=["GET", "POST"])
def index():
    dataset_root = str(app.config["DATASET_ROOT"])
    message = None
    uploaded_image = None
    prediction = None
    confidence = None

    action = request.form.get("action", "")
    if request.method == "POST" and action == "retrain":
        clear_training_cache()
        message = "Modèles réentraînés à partir du dataset local."

    training_state = train_and_score_models(dataset_root)
    if training_state is None:
        return render_template(
            "index.html",
            dataset_root=dataset_root,
            message="Aucune image exploitable n'a été trouvée dans dataset/saines et dataset/malades.",
            training_state=None,
            uploaded_image=None,
            prediction=None,
            confidence=None,
        )

    if request.method == "POST" and "leaf_image" in request.files:
        file = request.files["leaf_image"]
        if file and file.filename:
            file_bytes = file.read()
            mime_type = file.mimetype or "image/jpeg"
            uploaded_image = image_to_data_uri(file_bytes, mime_type)

            suffix = Path(file.filename).suffix or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(file_bytes)
                temp_path = Path(temp_file.name)

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

                selected_model = training_state["best_model"]
                selected_model_name = training_state["best_model_name"]
                raw_prediction, confidence = predict_label(selected_model, feature_vector)
                prediction = {
                    0: "Feuille saine",
                    1: "Feuille malade",
                }.get(raw_prediction, str(raw_prediction))
                
                # Save image to dataset and add to history
                saved_path = save_image_to_dataset(file_bytes, file.filename, raw_prediction)
                history_path = save_image_to_history(file_bytes, file.filename)
                add_diagnostic(
                    image_name=file.filename,
                    prediction=prediction,
                    confidence=confidence,
                    model_name=selected_model_name,
                    image_path=f"/static/history/{history_path.name}",
                )
                
                message = (
                    f"Image analysée avec le modèle retenu: {selected_model_name}."
                )
            finally:
                temp_path.unlink(missing_ok=True)

    return render_template(
        "index.html",
        dataset_root=dataset_root,
        message=message,
        training_state=training_state,
        uploaded_image=uploaded_image,
        prediction=prediction,
        confidence=confidence,
    )


@app.route("/gallery")
def gallery():
    history = get_history()
    
    # Calculate statistics
    saine_count = sum(1 for item in history if item["prediction"] == "Feuille saine")
    malade_count = sum(1 for item in history if item["prediction"] == "Feuille malade")
    
    stats = {
        "saine": saine_count,
        "malade": malade_count,
        "total": len(history),
    }
    
    return render_template("gallery.html", history=history, stats=stats)


if __name__ == "__main__":
    app.run(debug=True)