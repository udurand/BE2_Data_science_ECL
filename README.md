# BE2_Data_science_ECL : Recherche d'information dans la littérature scientifique

**MOD 7.2 – Introduction à la science des données**
BE séances 4, 5, 6 – Décembre 2025 / Janvier 2026

Enseignants : Julien Velcin (CM, BE), Erwan Versmée (BE)

---

## Description

Ce projet consiste à construire un **moteur de recherche d’articles scientifiques** : étant donnée une publication (requête), le système doit **retrouver les articles sémantiquement les plus proches**.

Pour chaque requête, on dispose d’environ **30 candidats** :
- **5 articles pertinents** (citations de l’article requête : exemples positifs)
- **~25 articles non pertinents** (exemples négatifs)

L’objectif est de produire un score d’appariement **S(q, c)** permettant de classer les candidats, et de **remonter les 5 pertinents**.

Approches mises en œuvre :
1. **Représentations creuses (sparse)** : CountVectorizer + TF-IDF, similarité cosinus.
2. **Représentations denses (dense)** : embeddings Sentence-Transformers (SBERT) + cosinus.
3. **Approche structurelle (graph)** : construction du graphe de citations (NetworkX) et **embeddings enrichis par les voisins** (moyenne pondérée).

Évaluation sur `valid.tsv` avec :
- **AUC** (métrique Kaggle / leaderboard)
- **P@5 / R@5 / F1@5**
- Courbes ROC globales (comparaison sparse vs dense vs graph)

Le script génère aussi les fichiers de **soumission Kaggle**.

---

## Contenu du dépôt

| Fichier/Dossier | Description |
|-----------------|-------------|
| `BE1_data_science_ECL_2025.py` | Script principal : chargement, représentations, moteurs, évaluations, graph, et soumissions. |
| `requirements.txt` | Dépendances Python pour exécuter le projet. |
| `README.md` | Ce fichier. |
| `sample_submission.csv` | Modèle de soumission Kaggle (fourni par l’enseignant / la compétition). |
| `sample_submission_dense.csv` | Soumission générée (dense). |
| `sample_submission_graph.csv` | Soumission générée (graph-enhanced). |

⚠️ Les fichiers de données (`corpus.jsonl`, `queries.jsonl`, `valid.tsv`, `test_final.tsv`) ne sont pas inclus dans le dépôt.

---

## Installation

### 1. Cloner le dépôt
```bash
git clone https://github.com/udurand/BE2_Data_science_ECL.git
cd BE2_Data_science_ECL
```

### 2. Créer un environnement Python

```bash
python -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## Utilisation

### 1. Placer les fichiers de données dans un dossier local (ex : `data/`)

* `corpus.jsonl`
* `queries.jsonl`
* `valid.tsv`
* `sample_submission.csv`
* (optionnel) `test_final.tsv`

### 2. Modifier les chemins au début du script si besoin

```python
BASE_PATH = r"..."
CORPUS_PATH = os.path.join(BASE_PATH, "corpus.jsonl")
QUERIES_PATH = os.path.join(BASE_PATH, "queries.jsonl")
QRELS_PATH = os.path.join(BASE_PATH, "valid.tsv")
SAMPLE_PATH = os.path.join(BASE_PATH, "sample_submission.csv")
```

### 3. Lancer le script

```bash
python BE2_data_science_ECL_2025.py
```

Le script :

* charge les données,
* construit les représentations sparse/dense,
* évalue sur `valid.tsv`,
* construit et analyse le graphe,
* calcule une variante **dense+graph**,
* exporte les fichiers `sample_submission_dense.csv` et `sample_submission_graph.csv`.

---

## Structure du code

| Section   | Contenu                                                                                          |
|-----------|--------------------------------------------------------------------------------------------------|
| Section 2 | Exploration des données + vectorisation sparse (titres / résumés / TF-IDF fusion).               |
| Section 3 | Similarité cosinus + moteur sparse (requête libre + baseline).                                   |
| Section 4 | Embeddings Sentence-Transformers + sauvegarde / rechargement.                                    |
| Section 5 | Ranking des candidats + métriques (P@5/R@5/F1@5 + AUC) + soumission dense.                       |
| Section 6 | Intermède exploratoire LDA + visualisations (optionnel).                                         |
| Section 7 | Graphe de citations : stats, matrices A/A²/A³, centralités (PageRank, Katz, etc.).               |
| Section 8 | Embeddings enrichis par le graphe + comparaison dense vs graph + ROC globale + soumission graph. |

---

## Notes techniques

* Les embeddings SBERT sont calculés **une seule fois** puis sauvegardés (pickle) pour accélérer les exécutions suivantes.
* La partie **LDA / UMAP / pyLDAvis** est exploratoire et peut être coûteuse (en temps + dépendances).
  Elle peut être désactivée en commentant l’appel correspondant dans le script.

---

## Auteurs

* Groupe : THEO FOREVER
* Membres : Julien Durand, Laurène Cristol, Théo Florence
