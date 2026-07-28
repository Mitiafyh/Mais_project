# Explication du projet - TP Classification Maïs

## Contexte
Ce projet vise à diagnostiquer automatiquement la Rouille Polysora sur les feuilles de maïs à Madagascar. Le système classe les feuilles en deux catégories : Saine (0) ou Malade (1).

---

## ETAPE 1 : Feature Engineering

### Objectif
Transformer les images en descripteurs numériques exploitables par les algorithmes de ML.

### Pourquoi ces features ?

**1. Pourcentage de rouille (pct_rouille)**
- **Justification** : La rouille Polysora apparaît sous forme de pustules orangées/brunâtres
- **Implémentation** : `src/analyseData.py` lignes 79-84
  - Conversion RGB vers HSV (`cv2.cvtColor`)
  - Masque couleur pour isoler rouge/orange (teintes 0-35 et 160-179)
  - Calcul du ratio pixels rouille / pixels totaux

**2. Rugosité (rugosite)**
- **Justification** : Les pustules créent des irrégularités sur la surface
- **Implémentation** : `src/analyseData.py` lignes 37-46
  - Filtre Sobel pour détecter les gradients d'intensité
  - Variance des gradients comme mesure de texture

**3. Variables supplémentaires**
- `saturation_moyenne` : Détecte la coloration des taches
- `indice_chlorose` : Teinte dominante de la feuille (indicateur de chlorose)
- `valeur_moyenne` : Luminosité moyenne
- `surface_feuille` : Proportion de la surface utile

### Code clé
```python
# src/analyseData.py
def build_feature_table(dataset_root):
    # Parcourt dataset/saines et dataset/malades
    # Extrait 6 features par image
    # Retourne DataFrame avec colonnes: ID_Image, pct_rouille, rugosite, ...
```

---

## ETAPE 2 : Algorithme Max-Minority

### Objectif
Implémenter un arbre de décision avec une métrique de pureté personnalisée.

### Pourquoi Max-Minority ?
- Mesure la pureté par la proportion de la classe majoritaire
- Plus simple que Gini/Entropie
- Adapté aux jeux de données déséquilibrés

### Formule mathématique
```
P(t) = max(n_c) / N
P_split = (|G|/N) × P(G) + (|D|/N) × P(D)
```

### Code clé
```python
# src/max_minority_tree.py
def max_minority_purity(y):
    return max(counts) / len(y)

def _best_split(self, X, y):
    # Teste tous les seuils possibles
    # Retourne le seuil maximisant la pureté pondérée
```

---

## ETAPE 3 : Modèles et Comparaison

### Objectif
Comparer 3 modèles :
1. Arbre Max-Minority (maison)
2. Decision Tree sklearn (Gini)
3. Random Forest sklearn

### Pourquoi Leave-One-Out ?
- Dataset petit → maximiser l'utilisation des données
- Chaque image testée une fois

### Code clé
```python
# src/modeling.py
def evaluate_models(X, y):
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(X):
        # Entraîne et prédit pour chaque fold
```

---

## ETAPE 4 : Application Web

### Objectif
Interface web pour les techniciens agricoles.

### Architecture

**Streamlit (app.py)**
- Sidebar : navigation et contrôles
- Page Diagnostic : upload + prédiction
- Page Galerie : historique des diagnostics

**Flask (server.py)**
- Route `/` : entraînement + diagnostic
- Route `/gallery` : galerie
- Templates HTML dans `templates/`

### Code clé
```python
# server.py
@app.route("/", methods=["GET", "POST"])
def index():
    # Gère l'entraînement et le diagnostic
    
@app.route("/gallery")
def gallery():
    # Affiche l'historique
```

---

## Démarrage

```bash
# Streamlit
streamlit run app.py

# Flask
python server.py