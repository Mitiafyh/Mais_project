from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analyseData import build_feature_table, save_feature_table
from src.modeling import evaluate_models, prepare_matrix, train_final_models


def main() -> None:
    dataset_root = PROJECT_ROOT / "dataset"
    output_csv = PROJECT_ROOT / "features_mais.csv"

    dataframe = build_feature_table(dataset_root)
    if dataframe.empty:
        print("Aucune image exploitable trouvée dans le dataset.")
        return

    save_feature_table(dataframe, output_csv)
    X, y, columns = prepare_matrix(dataframe)

    print("Colonnes retenues:", ", ".join(columns))
    print("\nAperçu du tableau de features:")
    print(dataframe.head().to_string(index=False))

    print("\nComparaison des modèles:")
    reports = evaluate_models(X, y)
    for report in reports:
        print(
            f"- {report.name}: accuracy={report.accuracy:.3f}, precision={report.precision:.3f}, "
            f"recall={report.recall:.3f}, f1={report.f1:.3f}"
        )
        print(f"  matrice de confusion: {report.confusion.tolist()}")

    trained_models = train_final_models(X, y)
    print("\nModèles entraînés:", ", ".join(trained_models.keys()))
    print(f"Tableau de features sauvegardé dans: {output_csv}")


if __name__ == "__main__":
    main()