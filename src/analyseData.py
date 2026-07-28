from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def load_image(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Impossible de lire l'image: {image_path}")
    return image


def makeImageToHSV(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def maskImage(img: np.ndarray, lower_bound1, upper_bound1, lower_bound2, upper_bound2):
    mask1 = cv2.inRange(img, lower_bound1, upper_bound1)
    mask2 = cv2.inRange(img, lower_bound2, upper_bound2)
    return cv2.bitwise_or(mask1, mask2)


def calculPCT_ROUILLE_X1(mask: np.ndarray) -> float:
    total_pixels = mask.size
    if total_pixels == 0:
        return 0.0
    rouille_pixels = cv2.countNonZero(mask)
    return (rouille_pixels / total_pixels) * 100


def calcule_X2(image: np.ndarray, leaf_mask: np.ndarray | None = None) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    if leaf_mask is not None and np.any(leaf_mask == 0):
        values = magnitude[leaf_mask == 0]
    else:
        values = magnitude.ravel()
    return float(np.var(values)) if values.size else 0.0


def calcule_X3(image_hsv: np.ndarray, leaf_mask: np.ndarray | None = None) -> float:
    saturation = image_hsv[:, :, 1]
    if leaf_mask is not None and np.any(leaf_mask == 0):
        values = saturation[leaf_mask == 0]
    else:
        values = saturation.ravel()
    return float(np.mean(values)) if values.size else 0.0


def isolerFeuilleCentrale(img: np.ndarray) -> np.ndarray | None:
    blurred = cv2.GaussianBlur(img, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 35, 35])
    upper_green = np.array([90, 255, 255])
    mask_leaf = cv2.inRange(hsv, lower_green, upper_green)

    kernel_large = np.ones((25, 25), np.uint8)
    mask_leaf = cv2.morphologyEx(mask_leaf, cv2.MORPH_CLOSE, kernel_large)
    mask_leaf = cv2.morphologyEx(mask_leaf, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask_leaf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    result = np.ones_like(mask_leaf) * 255
    cv2.drawContours(result, [largest_contour], -1, 0, thickness=cv2.FILLED)
    return cv2.medianBlur(result, 5)


def rust_mask_from_hsv(image_hsv: np.ndarray) -> np.ndarray:
    lower1 = np.array([0, 50, 35])
    upper1 = np.array([35, 255, 255])
    lower2 = np.array([160, 50, 35])
    upper2 = np.array([179, 255, 255])
    return maskImage(image_hsv, lower1, upper1, lower2, upper2)


def extraction_features(image_path: str | Path, label: int | None = None) -> dict[str, float | int | str | None]:
    image = load_image(image_path)
    image_hsv = makeImageToHSV(image)
    leaf_mask = isolerFeuilleCentrale(image)
    if leaf_mask is None:
        leaf_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    rust_mask = rust_mask_from_hsv(image_hsv)
    rust_on_leaf = cv2.bitwise_and(rust_mask, rust_mask, mask=(leaf_mask == 0).astype(np.uint8) * 255)
    leaf_area = int(np.count_nonzero(leaf_mask == 0)) or image.shape[0] * image.shape[1]

    pct_rouille = calculPCT_ROUILLE_X1(rust_on_leaf)
    rugosite = calcule_X2(image, leaf_mask)
    saturation_moyenne = calcule_X3(image_hsv, leaf_mask)

    leaf_pixels = image_hsv[leaf_mask == 0] if np.any(leaf_mask == 0) else image_hsv.reshape(-1, 3)
    indice_chlorose = float(np.mean(leaf_pixels[:, 0])) if leaf_pixels.size else 0.0
    valeur_moyenne = float(np.mean(leaf_pixels[:, 2])) if leaf_pixels.size else 0.0

    return {
        "ID_Image": Path(image_path).name,
        "pct_rouille": float(pct_rouille),
        "rugosite": float(rugosite),
        "saturation_moyenne": float(saturation_moyenne),
        "indice_chlorose": indice_chlorose,
        "valeur_moyenne": valeur_moyenne,
        "surface_feuille": float(leaf_area / (image.shape[0] * image.shape[1])),
        "label_malade": int(label) if label is not None else None,
    }


def list_image_files(folder: str | Path) -> list[Path]:
    root = Path(folder)
    if not root.exists():
        return []
    return [path for path in sorted(root.iterdir()) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]


def build_feature_table(dataset_root: str | Path) -> pd.DataFrame:
    dataset_root = Path(dataset_root)
    class_map = {"saines": 0, "malades": 1}
    rows: list[dict[str, float | int | str | None]] = []

    for class_name, label in class_map.items():
        for image_path in list_image_files(dataset_root / class_name):
            try:
                rows.append(extraction_features(image_path, label))
            except ValueError:
                continue

    columns = [
        "ID_Image",
        "pct_rouille",
        "rugosite",
        "saturation_moyenne",
        "indice_chlorose",
        "valeur_moyenne",
        "surface_feuille",
        "label_malade",
    ]
    return pd.DataFrame(rows, columns=columns)


def save_feature_table(dataframe: pd.DataFrame, output_path: str | Path) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)



    