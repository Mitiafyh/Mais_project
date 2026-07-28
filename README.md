# ANALYSE-DE-DONNEES
Analyse: classification d'images de feuilles de maïs

## Objectif
Construire un pipeline complet pour classifier des feuilles de maïs saines et malades à partir d'images.

## Phases
1. Préparation et feature engineering
	- segmentation de la feuille centrale
	- extraction de caractéristiques numériques: rouille, rugosité, saturation, teintes, surface utile
2. Développement de l'algorithme Max-Minority
	- implémentation d'un arbre de décision personnalisé basé sur la pureté pondérée des noeuds
3. Modélisation et analyse comparative
	- comparaison du modèle maison avec `DecisionTreeClassifier` et `RandomForestClassifier`
4. Application web Streamlit
	- interface interactive pour tester une image et visualiser les prédictions

## Lancement
Génération du tableau de données et comparaison des modèles:

```bash
venv/bin/python src/main.py
```

Lancer l'application web HTML:

```bash
venv/bin/python server.py
```

Lancer l'ancienne version Streamlit si besoin:

```bash
venv/bin/streamlit run app.py
```

## Arborescence utile
- [src/analyseData.py](src/analyseData.py) contient l'extraction de features
- [src/max_minority_tree.py](src/max_minority_tree.py) contient l'arbre Max-Minority
- [src/modeling.py](src/modeling.py) contient l'entraînement et la comparaison
- [server.py](server.py) contient l'interface HTML principale