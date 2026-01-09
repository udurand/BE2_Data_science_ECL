# BE2_Data_science_ECL : Recherche d'information dans la littérature scientifique

**MOD 7.2 – Introduction à la science des données**
BE séances 4, 5, 6 – Décembre 2025 / Janvier 2026
**Enseignants** : Julien Velcin (CM, BE), Erwan Versmée (BE)
**Groupe** : THEO FOREVER
**Auteurs** : Julien Durand, Laurène Cristol, Théo Florence

---

## Description du projet

Ce projet implémente un **moteur de recherche d’articles scientifiques** capable de retrouver, pour une publication donnée (requête), les **5 articles sémantiquement les plus proches** parmi un ensemble de 30 candidats (5 pertinents + ~25 non pertinents).
L’objectif est de produire un **score d’appariement S(q, c)** pour classer les candidats et maximiser la pertinence des résultats.

### Approches mises en œuvre :
1. **Représentations creuses (sparse)** :
   - `CountVectorizer` et `TF-IDF` sur les titres et résumés.
   - Similarité cosinus pour le ranking.
2. **Représentations denses (dense)** :
   - Embeddings via [Sentence-Transformers](https://www.sbert.net/) (modèles `all-MiniLM-L6-v2` et `all-mpnet-base-v2`).
   - Similarité cosinus dans l’espace dense.
3. **Approche structurelle (graph)** :
   - Construction du **graphe de citations** avec [NetworkX](https://networkx.org/).
   - Enrichissement des embeddings par la moyenne pondérée des voisins (prédécesseurs + successeurs).
   - Combinaison des approches dense et graphique pour améliorer les performances.

### Évaluation :
- **Métriques** : AUC (leaderboard Kaggle), Précision@5, Rappel@5, F1@5.
- **Visualisations** : Courbes ROC, distributions de degrés, centralités (PageRank, Betweenness, etc.).
- **Soumissions** : Génération de fichiers CSV pour [Kaggle](https://www.kaggle.com/) (`sample_submission_dense.csv`, `sample_submission_graph.csv`, `submission_graph_all-mpnet-base-v2.csv`).

---

## Contenu du dépôt

| Fichier/Dossier                     | Description                                                                                     |
|-------------------------------------|-------------------------------------------------------------------------------------------------|
| `BE2_data_science_ECL_2025.py`      | Script principal : chargement des données, représentations, moteurs, évaluations, et soumissions. |
| `requirements.txt`                  | Liste des dépendances Python (voir [Installation](#installation)).                              |
| `README.md`                         | Ce fichier.                                                                                     |
| `sample_submission.csv`             | Modèle de soumission Kaggle (fournis par l’enseignant).                                         |
| `sample_submission_dense.csv`       | Soumission générée par l’approche dense (SBERT).                                                |
| `sample_submission_graph.csv`       | Soumission générée par l’approche graph-enhanced.                                               |
| `submission_graph_all-mpnet-base-v2.csv` | Soumission optimisée avec le modèle `all-mpnet-base-v2` + graphe.                          |
| `Stop-words-en.txt`                 | Liste de stopwords pour le prétraitement LDA (Section 6).                                      |

⚠️ **Fichiers non inclus** (à placer dans le dossier racine) :
- Données d’entrée : `corpus.jsonl`, `queries.jsonl`, `valid.tsv`, `test_final.tsv`.
- Cache : `corpus_embeddings.pkl` et `corpus_embeddings__all-mpnet-base-v2.pkl` (générés automatiquement).

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
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```
> ⚠️ **Note** :
> - L’installation de `faiss-cpu` (pour l’indexation vectorielle) peut nécessiter des dépendances système (ex: `g++` sur Linux).
> - Pour `pyLDAvis`, voir [la documentation officielle](https://pyldavis.readthedocs.io/en/latest/readme.html#installation) si des erreurs surviennent.

---

## Utilisation

### 1. Préparer les données
Placez les fichiers suivants dans le dossier racine du projet :
- `corpus.jsonl`
- `queries.jsonl`
- `valid.tsv`
- `sample_submission.csv` (pour générer les soumissions)
- *(Optionnel)* `test_final.tsv` (pour évaluer sur le jeu de test final).

### 2. Configurer les chemins
Modifiez les variables en début de script si vos fichiers ne sont pas dans le dossier par défaut :
```python
BASE_PATH = r"C:\Chemin\Vers\Vos\Données"  # À adapter selon votre système
CORPUS_PATH = os.path.join(BASE_PATH, "corpus.jsonl")
# ... (autres chemins)
```

### 3. Lancer le script
```bash
python BE2_data_science_ECL_2025.py
```
> ⏳ **Temps d’exécution** :
> - La **première exécution** peut prendre **plusieurs minutes** (voire heures selon votre machine), surtout pour :
>   - Le calcul des embeddings SBERT (`corpus_embeddings.pkl` et `corpus_embeddings__all-mpnet-base-v2.pkl`).
>   - L’entraînement du modèle LDA (Section 6, optionnelle).
> - Les exécutions suivantes seront plus rapides grâce au cache (`corpus_embeddings.pkl` et `corpus_embeddings__all-mpnet-base-v2.pkl`).

### 4. Résultats attendus
Le script :
1. Affiche des **statistiques** sur les données (Section 2).
2. Évalue les approches **sparse**, **dense**, et **graph-enhanced** (Sections 3–8).
3. Génère les fichiers de soumission Kaggle :
   - `sample_submission_dense.csv`
   - `sample_submission_graph.csv`
   - `submission_graph_all-mpnet-base-v2.csv` (meilleure performance attendue).

---

## Structure du code

| Section | Contenu                                                                                          |
|---------|--------------------------------------------------------------------------------------------------|
| 2       | Exploration des données + vectorisation sparse (TF-IDF).                                         |
| 3       | Moteur de recherche sparse + évaluation baseline.                                                |
| 4       | Embeddings SBERT + sauvegarde/rechargement.                                                     |
| 5       | Ranking des candidats + métriques (P@5/R@5/F1@5/AUC) + soumission dense.                         |
| 6       | *(Optionnel)* Analyse thématique avec LDA/UMAP/pyLDAvis.                                         |
| 7       | Construction du graphe de citations + calcul de centralités.                                    |
| 8       | Embeddings enrichis par le graphe + comparaison dense vs graph + ROC globale.                    |
| 9       | Expérimentations (variantes TF-IDF, modèles SBERT).                                             |
| 10      | Génération de la soumission optimisée (`all-mpnet-base-v2` + graphe).                           |

---

## Notes techniques

### Optimisations :
- **Cache des embeddings** : Les représentations SBERT sont sauvegardées dans `corpus_embeddings.pkl` pour éviter de les recalculer.
- **FAISS** : Utilisé pour une recherche vectorielle rapide (démo en Section 4).
- **Parallelisation** : Les calculs de similarité cosinus sont vectorisés avec `numpy`/`scipy`.

### Désactiver des sections :
- Pour **ignorer la partie LDA** , commentez l’appel à `run_lda_topic_modeling(corpus)` dans le `main()`.
- Pour **éviter de recalculer les embeddings**, assurez-vous que `corpus_embeddings.pkl` et `corpus_embeddings__all-mpnet-base-v2.pkl` existent déjà.

---


## Auteurs et licence
- **Auteurs** : [Julien Durand](https://github.com/udurand), Laurène Cristol, Théo Florence.


