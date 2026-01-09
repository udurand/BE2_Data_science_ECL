# ============================================================================
#  MOD 7.2 — Recherche d'information dans la littérature scientifique
#  Centrale Lyon — Introduction à la science des données
#
#
#  Objectif :
#  Implémenter et comparer plusieurs moteurs de recherche d’articles scientifiques :
#   - représentations creuses (TF / TF-IDF),
#   - représentations denses (Sentence-Transformers),
#   - représentations enrichies par la structure du graphe de citations.
#
#  Auteurs : Groupe Théo forever, Julien Durand, Laurène Cristol, Théo Florence
#  Date    : Janvier 2026
# ============================================================================

import os
import json
import pickle
import warnings
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import networkx as nx
import scipy.sparse as sparse
from scipy.sparse import find

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, roc_curve, auc
)

# Sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Missing library: {e}")
    print("Please install with: pip install sentence-transformers scikit-learn networkx")
    raise

import re

# --- STEMMING (NLTK) ---
try:
    from nltk.stem.snowball import SnowballStemmer
except ImportError as e:
    print(f"Missing library: {e}")
    print("Please install with: pip install nltk")
    raise

# --- FAISS ---
try:
    import faiss
except ImportError as e:
    print(f"Missing library: {e}")
    print("Missing library faiss. Install with: pip install faiss-cpu")
    raise

warnings.filterwarnings("ignore")
np.random.seed(42)


# ============================================================
#  PRINTS STANDARDISÉS
# ============================================================

def print_section(title: str):
    """Affiche un titre de grande section (lisible dans Spyder)."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_subsection(title: str):
    """Affiche un titre de sous-section."""
    print("\n" + "-" * 60)
    print(f"> {title}")
    print("-" * 60)


def print_info(msg: str):
    """Message d'information."""
    print(f"[INFO] {msg}")


def print_ok(msg: str):
    """Message de succès / fin d'étape."""
    print(f"[ OK ] {msg}")


def print_warn(msg: str):
    """Message d'avertissement (non bloquant)."""
    print(f"[WARN] {msg}")


def print_err(msg: str):
    """Message d'erreur (potentiellement bloquant)."""
    print(f"[ERR ] {msg}")


# ============================================================
#  CONFIG — chemins fichiers
# ============================================================

BASE_PATH   = r"C:\Chemin\Vers\Vos\Données"
CORPUS_PATH = os.path.join(BASE_PATH, "corpus.jsonl")
QUERIES_PATH = os.path.join(BASE_PATH, "queries.jsonl")
QRELS_PATH  = os.path.join(BASE_PATH, "valid.tsv")

EMBEDDINGS_FILE = os.path.join(BASE_PATH, "corpus_embeddings.pkl")

SAMPLE_PATH = os.path.join(BASE_PATH, "sample_submission.csv")
OUTPUT_SPARSE_SUB = os.path.join(BASE_PATH, "sample_submission_dense.csv")
OUTPUT_GRAPH_SUB  = os.path.join(BASE_PATH, "sample_submission_graph.csv")

# LDA (section 6)
LDA_DIR = BASE_PATH
STOPWORDS_PATH = os.path.join(LDA_DIR, "Stop-words-en.txt")


# ============================================================
#  UTILS — affichage vecteurs / tables
# ============================================================

def print_feats(v, features, top_n=30):
    """
    Retourne un DataFrame (mot, valeur) pour les top_n termes d'un vecteur sparse.
    Utile pour interpréter les features d'un document vectorisé.
    """
    _, ids, values = find(v)
    feats = [(ids[i], values[i], features[ids[i]]) for i in range(len(list(ids)))]
    top_feats = sorted(feats, key=lambda x: x[1], reverse=True)[0:top_n]
    return pd.DataFrame({
        "word":  [t[2] for t in top_feats],
        "value": [t[1] for t in top_feats]
    })


def display_side_by_side(dfs: list, captions: list):
    """
    Affiche des tableaux pandas côte à côte (style notebook).
    Remarque : dans Spyder, l'affichage peut dépendre du backend IPython.
    """
    from IPython.core.display import HTML
    from IPython.display import display

    output = ""
    combined = dict(zip(captions, dfs))
    for caption, df in combined.items():
        output += df.style.set_table_attributes(
            "style='display:inline'"
        ).set_caption(caption)._repr_html_()
        output += "&emsp;"
    display(HTML(output))



# ============================================================
#  DATA LOADING
# ============================================================

def load_corpus(file_path):
    """
    Charge corpus.jsonl.
    - Chaque ligne : un JSON contenant au moins la clé "_id".
    - Retour : dict {doc_id: doc_dict}
    """
    corpus = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = doc
    return corpus


def load_queries(file_path):
    """
    Charge queries.jsonl.
    - Chaque ligne : un JSON contenant au moins la clé "_id".
    - Retour : dict {query_id: query_dict}
    """
    queries = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            query = json.loads(line)
            queries[query["_id"]] = query
    return queries


def load_qrels(file_path):
    """
    Charge valid.tsv (qrels).
    Format attendu :
        query-id \\t corpus-id \\t score
    La première ligne est un header → ignorée.
    Retour :
        dict {qid: {docid: int(score)}}
    """
    qrels = {}
    with open(file_path, "r", encoding="utf-8") as f:
        next(f)  # Skip header
        for line in f:
            qid, docid, score = line.strip().split("\t")
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][docid] = int(score)
    return qrels


# ============================================================
#  SECTION 2 — Stats + Vectorisation (sparse)
# ============================================================

def compute_basic_stats(corpus, queries, qrels_valid):
    """
    Calcule et affiche des stats simples  :
    - tailles corpus / queries
    - nb paires query-document dans valid.tsv
    - proportion de pertinents / non-pertinents par requête
    """
    print_subsection("Statistiques du dataset (corpus/queries/qrels)")
    print_info(f"Taille du corpus : {len(corpus)} documents")
    print_info(f"Nombre de requêtes : {len(queries)}")

    num_pairs = sum(len(docs) for docs in qrels_valid.values())
    print_info(f"Nombre de paires requête/document : {num_pairs}")

    relevance_stats = {}
    for qid, docs in qrels_valid.items():
        total = len(docs)
        pos = sum(1 for s in docs.values() if s > 0)
        neg = total - pos
        relevance_stats[qid] = {
            "positif": pos,
            "negatif": neg,
            "total": total,
            "proportion_positif": pos / total if total else 0.0,
            "proportion_negatif": neg / total if total else 0.0
        }

    relevance_df = pd.DataFrame(relevance_stats).T
    print_info("Aperçu proportions pertinents (5 premières requêtes) :")
    print(relevance_df.head(5))

    mean_pos = relevance_df["proportion_positif"].mean()
    mean_neg = relevance_df["proportion_negatif"].mean()
    print_ok(f"Moyenne proportion pertinents : {mean_pos:.3f}")
    print_ok(f"Moyenne proportion non-pertinents : {mean_neg:.3f}")

    return relevance_df


def show_one_query_example(corpus, queries, qrels_valid):
    """
    Affiche une requête exemple + quelques candidats pertinents/non pertinents.
    """
    print_subsection("Exemple de requête (qrels_valid) + candidats + / -")
    first_qid = next(iter(qrels_valid.keys()))
    print_info(f"Query ID : {first_qid}")
    print_info(f"Query object (brut) : {queries[first_qid]}")

    pos_docs = [doc_id for doc_id, score in qrels_valid[first_qid].items() if score > 0]
    neg_docs = [doc_id for doc_id, score in qrels_valid[first_qid].items() if score == 0]

    print_info(f"Exemples positifs (IDs) : {pos_docs[:3]}")
    print_info(f"Exemples négatifs (IDs) : {neg_docs[:3]}")


def build_sparse_representations(corpus):
    """
    Construit 3 représentations creuses :
    1) Titres (CountVectorizer)
    2) Textes/résumés (CountVectorizer)
    3) Title+Text fusionnés (TF-IDF)
    """
    print_subsection("Vectorisation sparse : titres / résumés / fusion TF-IDF")

    titles = [doc["title"] for doc in corpus.values()]
    texts  = [doc["text"] for doc in corpus.values()]

    # --- TITRES ---
    vectorizer_titles = CountVectorizer()
    X_titles = vectorizer_titles.fit_transform(titles)
    features_titles = vectorizer_titles.get_feature_names_out()
    print_ok(f"Matrice titres : {X_titles.shape} (docs x termes)")

    # --- TEXTES ---
    vectorizer_texts = CountVectorizer()
    X_texts = vectorizer_texts.fit_transform(texts)
    features_texts = vectorizer_texts.get_feature_names_out()
    print_ok(f"Matrice résumés : {X_texts.shape} (docs x termes)")

    # --- MERGED TF-IDF ---
    docs_merged = [(doc["title"] + " " + doc["text"]) for doc in corpus.values()]
    vectorizer_merged = TfidfVectorizer()
    X_merged = vectorizer_merged.fit_transform(docs_merged)
    features_merged = vectorizer_merged.get_feature_names_out()
    print_ok(f"Matrice fusionnée TF-IDF : {X_merged.shape} (docs x termes)")

    return (
        titles, texts,
        vectorizer_titles, X_titles, features_titles,
        vectorizer_texts, X_texts, features_texts,
        docs_merged, vectorizer_merged, X_merged, features_merged
    )


def show_sparse_examples(corpus, X_titles, features_titles, X_texts, features_texts):
    """
    Permet d’expliquer concrètement une représentation creuse :
    - un doc i = un vecteur très sparse de mots (vocabulaire global)
    - on affiche les termes les plus présents (ou plus forts) par doc
    - on regarde aussi les mots les plus fréquents globalement
    """
    doc_ids = list(corpus.keys())
    vectors_to_show = [0, 1, 2, 3, 4]

    # --- Exemples vecteurs TITRES ---
    print("\n=== Quelques vecteurs documents (TITRES) ===")
    dfs, captions = [], []
    for i in vectors_to_show:
        df = print_feats(X_titles[i], features_titles, top_n=10)
        dfs.append(df)
        captions.append(f"Document {i}: {corpus[doc_ids[i]]['title']}")
    display_side_by_side(dfs, captions)
    print_info("Exemples de vecteurs (titres) : top termes (affichage console)")
    for cap, df in zip(captions, dfs):
        print("\n" + "-" * 80)
        print(cap)
        print(df.to_string(index=False))


    # --- Distribution mots (TITRES) ---
    word_counts = X_titles.sum(axis=0)
    word_counts_array = word_counts.A1
    df_counts = pd.DataFrame({"word": features_titles, "count": word_counts_array})
    df_counts_sorted = df_counts.sort_values(by="count", ascending=False).head(20)


    plt.figure(figsize=(12, 6))
    plt.bar(df_counts_sorted["word"], df_counts_sorted["count"])
    plt.xticks(rotation=45)
    plt.title("Distribution des 20 mots les plus fréquents (titres)")
    plt.show()

    # --- Exemples vecteurs TEXTS ---
    print("\n=== Quelques vecteurs documents (RÉSUMÉS / TEXTS) ===")
    dfs_text, captions_text = [], []
    for i in vectors_to_show:
        df = print_feats(X_texts[i], features_texts, top_n=10)
        dfs_text.append(df)
        captions_text.append(f"Résumé Doc {i}: {corpus[doc_ids[i]]['title']}")
    display_side_by_side(dfs_text, captions_text)
    print_info("Exemples de vecteurs (résumés) : top termes (affichage console)")
    for cap, df in zip(captions_text, dfs_text):
        print("\n" + "-" * 80)
        print(cap)
        print(df.to_string(index=False))

    # --- Distribution mots (TEXTS) ---
    word_counts_text = X_texts.sum(axis=0)
    word_counts_array_text = word_counts_text.A1
    df_counts_text = pd.DataFrame({"word": features_texts, "count": word_counts_array_text})
    df_counts_sorted_text = df_counts_text.sort_values(by="count", ascending=False).head(20)


    plt.figure(figsize=(12, 6))
    plt.bar(df_counts_sorted_text["word"], df_counts_sorted_text["count"])
    plt.xticks(rotation=45)
    plt.title("Distribution des 20 mots les plus fréquents (résumés)")
    plt.show()

# ============================================================
#  SECTION 3 — Cosine + moteur sparse
# ============================================================

def test_cosine_on_pairs(titles, texts, X_titles, X_texts):
    """
    Test de similarité cosinus sur quelques paires (titres puis résumés)
    """
    print_subsection("Tests cosine_similarity sur quelques paires (titres / résumés)")
    pairs = [(0, 3), (1, 7), (2, 5)]

    print_info("Similarités cosinus sur TITRES")
    for a, b in pairs:
        print("\n" + "-" * 40)
        print_info(f"Doc {a} TITLE : {titles[a]}")
        print_info(f"Doc {b} TITLE : {titles[b]}")
        sim = cosine_similarity(X_titles[a], X_titles[b])[0][0]
        print_ok(f"Cosine(TITLE) = {sim:.4f}")

    print_info("Similarités cosinus sur RÉSUMÉS")
    for a, b in pairs:
        print("\n" + "-" * 40)
        print_info(f"Doc {a} TEXT (début) : {texts[a][:200]} ...")
        print_info(f"Doc {b} TEXT (début) : {texts[b][:200]} ...")
        sim_text = cosine_similarity(X_texts[a], X_texts[b])[0][0]
        print_ok(f"Cosine(TEXT) = {sim_text:.4f}")


def search_sparse(query, vectorizer, X, corpus, top_k=10, field="merged"):
    """
    Petit moteur de recherche sparse :
    - vectorize(query)
    - cosine_similarity(query_vec, X)
    - top_k documents (tri décroissant)
    """
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, X)[0]
    top_idx = np.argsort(scores)[::-1][:top_k]

    results = []
    ids = list(corpus.keys())
    for idx in top_idx:
        doc = corpus[ids[idx]]

        if field == "title":
            content = doc["title"]
        elif field == "text":
            content = doc["text"]
        else:
            content = doc["title"] + " " + doc["text"]

        preview = content[:300] + "..." if len(content) > 300 else content
        results.append({
            "doc_id": ids[idx],
            "score": scores[idx],
            "title": doc["title"],
            "preview": preview
        })
    return results


def demo_sparse_search(corpus, vectorizer_titles, X_titles, vectorizer_texts, X_texts, vectorizer_merged, X_merged):
    """
    Démo rapide du moteur sparse sur une requête texte libre,
    sur 3 espaces :
    - fusion (TF-IDF)
    - titres
    - résumés
    """
    print_subsection("Démo moteur sparse (requête libre)")
    query = "machine learning medical diagnosis"
    print_info(f"Query de test : {query}")

    print_info("Résultats sur TITRE+TEXTE (TF-IDF)")
    results = search_sparse(query, vectorizer_merged, X_merged, corpus, top_k=10, field="merged")
    for r in results:
        print(f"[{r['score']:.4f}] {r['title']}\nPreview: {r['preview']}\n")

    print_info("Résultats sur TITRE seulement")
    results = search_sparse(query, vectorizer_titles, X_titles, corpus, top_k=10, field="title")
    for r in results:
        print(f"[{r['score']:.4f}] {r['title']}\nPreview: {r['preview']}\n")

    print_info("Résultats sur RÉSUMÉ seulement")
    results = search_sparse(query, vectorizer_texts, X_texts, corpus, top_k=10, field="text")
    for r in results:
        print(f"[{r['score']:.4f}] {r['title']}\nPreview: {r['preview']}\n")


def evaluate_sparse_auc(corpus, queries, qrels_valid, vectorizer_merged, X_merged):
    """
    Évalue la solution sparse via l'AUC moyenne sur valid.tsv :
    - on ne compare que parmi les candidats fournis par valid.tsv
    - score = cosine(query_vec, doc_vec)
    - AUC calculée par requête puis moyenne
    """
    print_subsection("Évaluation sparse : AUC moyenne sur valid.tsv")

    all_doc_ids = list(corpus.keys())
    auc_list = []

    for qid, cand_map in qrels_valid.items():
        if qid not in queries:
            continue

        query_text = queries[qid]["text"]
        candidate_ids = [d for d in cand_map.keys() if d in corpus]
        if len(candidate_ids) < 2:
            continue

        cand_idx = [all_doc_ids.index(d) for d in candidate_ids]
        cand_vectors = X_merged[cand_idx]
        q_vec = vectorizer_merged.transform([query_text])

        scores = cosine_similarity(q_vec, cand_vectors)[0]
        labels = [int(cand_map[d]) for d in candidate_ids]

        if len(set(labels)) > 1:
            auc_list.append(roc_auc_score(labels, scores))

    if len(auc_list) > 0:
        print_ok(f"AUC moyen sur valid.tsv = {np.mean(auc_list):.4f}")
    else:
        print_warn("Impossible de calculer l'AUC (pas assez de labels variés)")


# ============================================================
#  SECTION 4 — Dense embeddings + sauvegarde
# ============================================================

def build_or_load_embeddings(corpus, embeddings_file):
    """
    Construit (ou recharge) les embeddings Sentence-Transformers :
    - modèle : all-MiniLM-L6-v2
    - texte encodé : title + text
    - sauvegarde : pickle (pour éviter de recalculer)
    """
    print_subsection("Construction/chargement embeddings (Sentence-Transformers)")

    model_name = "all-MiniLM-L6-v2"
    print_info(f"Modèle : {model_name}")
    model = SentenceTransformer(model_name)

    docs_merged = [doc["title"] + " " + doc["text"] for doc in corpus.values()]
    print_info(f"Nb documents à encoder : {len(docs_merged)}")

    if os.path.exists(embeddings_file):
        print_info("Embeddings déjà présents → chargement depuis disque")
        with open(embeddings_file, "rb") as f:
            corpus_embeddings = pickle.load(f)
        print_ok("Embeddings chargés")
    else:
        print_info("Embeddings absents → encodage (peut prendre du temps)")
        corpus_embeddings = model.encode(docs_merged, show_progress_bar=True, convert_to_numpy=True)
        with open(embeddings_file, "wb") as f:
            pickle.dump(corpus_embeddings, f)
        print_ok(f"Embeddings sauvegardés : {embeddings_file}")

    print_ok(f"Shape embeddings : {corpus_embeddings.shape}")
    return model, corpus_embeddings


def search_embeddings(query, model, embeddings, corpus, top_k=10):
    """
    Moteur dense basé embeddings :
    - embed(query)
    - cosine_similarity(query_emb, docs_emb)
    - top_k
    """
    q_vec = model.encode([query], convert_to_numpy=True)
    scores = (embeddings @ q_vec.T).squeeze() / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_vec)
    )
    top_idx = np.argsort(scores)[::-1][:top_k]

    results = []
    doc_ids = list(corpus.keys())
    for idx in top_idx:
        results.append({
            "doc_id": doc_ids[idx],
            "score": scores[idx],
            "title": corpus[doc_ids[idx]]["title"],
            "preview": corpus[doc_ids[idx]]["text"][:300] + "..."
        })
    return results


def demo_dense_search(corpus, model, corpus_embeddings):
    """
    Démo dense embeddings sur une requête libre.
    """
    print_subsection("Démo moteur dense embeddings (requête libre)")

    query = "machine learning medical diagnosis"
    print_info(f"Query de test : {query}")

    results = search_embeddings(query, model, corpus_embeddings, corpus, top_k=10)
    for r in results:
        print(f"[{r['score']:.4f}] {r['title']}\nPreview: {r['preview']}\n")


# ============================================================
#  SECTION 5 — Dense search among candidates + métriques
# ============================================================

def dense_search_among_candidates(
    query_text, model, corpus_embeddings, corpus, candidate_ids, all_doc_ids, top_k_return=25
):
    """
    Classement des candidats d'une requête, en mode dense.
    Important : on ne compare que parmi candidate_ids (≈25 docs fournis pour la query).
    """
    cand_ids_in_corpus = []
    cand_embeddings = []

    for did in candidate_ids:
        if did in corpus:
            idx = all_doc_ids.index(did)
            cand_embeddings.append(corpus_embeddings[idx])
            cand_ids_in_corpus.append(did)
        else:
            warnings.warn(f"Candidate {did} absent du corpus -> ignored")

    if len(cand_ids_in_corpus) == 0:
        return []

    cand_embeddings = np.vstack(cand_embeddings)
    q_vec = model.encode([query_text], convert_to_numpy=True)
    scores = cosine_similarity(q_vec, cand_embeddings)[0]

    order = np.argsort(scores)[::-1][:top_k_return]

    results = []
    for i in order:
        did = cand_ids_in_corpus[i]
        results.append({
            "doc_id": did,
            "score": float(scores[i]),
            "title": corpus[did]["title"],
            "preview": corpus[did]["text"][:300] + "..."
        })
    return results


def evaluate_dense_all_queries(corpus, queries, qrels_valid, model, corpus_embeddings):
    """
    Évaluation dense :
    - Affiche quelques exemples top-5 (inspection manuelle)
    - Calcule P@5 / R@5 / F1@5 (macro + micro)
    - Calcule AUC par requête puis moyenne (macro AUC)
    """
    print_subsection("Évaluation dense sur valid.tsv (P@5/R@5/F1@5 + AUC)")

    TOP_K_RETURN = 25
    TOP_K_CHECK = 5
    PRINT_EXAMPLES = 5

    all_doc_ids = list(corpus.keys())

    # --- Exemples (inspection) ---
    print_info("Exemples : top-5 pour les 5 premières requêtes")
    example_qids = list(qrels_valid.keys())[:PRINT_EXAMPLES]

    for qid in example_qids:
        if qid not in queries:
            print_warn(f"QID {qid} absent de queries -> skip")
            continue

        query_text = queries[qid].get("text", "")
        candidate_ids = list(qrels_valid[qid].keys())

        results = dense_search_among_candidates(
            query_text, model, corpus_embeddings, corpus, candidate_ids, all_doc_ids, top_k_return=TOP_K_RETURN
        )

        print("\n" + "=" * 70)
        print_info(f"Query id: {qid}")
        print_info(f"Query text: {query_text}")
        print_info(f"Top-{TOP_K_CHECK} predictions (doc_id | score | label)")
        for r in results[:TOP_K_CHECK]:
            label = qrels_valid[qid].get(r["doc_id"], 0)
            print(f" - {r['doc_id']} | {r['score']:.4f} | label={label} | title: {r['title']}")
        print("=" * 70)

    # --- Évaluation globale ---
    per_query_metrics = []
    sum_TP = 0
    sum_retrieved = 0
    sum_positives = 0

    for qid, cand_map in qrels_valid.items():
        if qid not in queries:
            continue

        query_text = queries[qid].get("text", "")
        candidate_ids = list(cand_map.keys())

        results = dense_search_among_candidates(
            query_text, model, corpus_embeddings, corpus, candidate_ids, all_doc_ids, top_k_return=TOP_K_RETURN
        )
        if len(results) == 0:
            continue

        topk_results = results[:TOP_K_CHECK]
        topk_doc_ids = [r["doc_id"] for r in topk_results]

        # AUC : aligner labels/scores sur les candidats réellement présents
        cand_ids_in_corpus = [did for did in candidate_ids if did in corpus]
        if len(cand_ids_in_corpus) == 0:
            continue

        cand_embeddings = np.vstack([corpus_embeddings[all_doc_ids.index(d)] for d in cand_ids_in_corpus])
        cand_scores = cosine_similarity(model.encode([query_text], convert_to_numpy=True), cand_embeddings)[0]
        cand_labels = [int(cand_map.get(did, 0)) for did in cand_ids_in_corpus]

        # AUC par requête si possible
        try:
            auc_q = roc_auc_score(cand_labels, cand_scores) if len(set(cand_labels)) > 1 else None
        except Exception:
            auc_q = None

        # P@5/R@5/F1@5
        tp = sum(1 for did in topk_doc_ids if int(cand_map.get(did, 0)) == 1)
        retrieved = len(topk_doc_ids)
        positives_total = sum(int(v) for v in cand_map.values())

        prec_q = tp / retrieved if retrieved > 0 else 0.0
        rec_q = tp / positives_total if positives_total > 0 else 0.0
        f1_q = (2 * prec_q * rec_q / (prec_q + rec_q)) if (prec_q + rec_q) > 0 else 0.0

        per_query_metrics.append({
            "qid": qid,
            "precision_at_k": prec_q,
            "recall_at_k": rec_q,
            "f1_at_k": f1_q,
            "auc": auc_q,
            "tp": tp,
            "retrieved": retrieved,
            "positives_total": positives_total
        })

        sum_TP += tp
        sum_retrieved += retrieved
        sum_positives += positives_total

    if len(per_query_metrics) == 0:
        raise RuntimeError("Aucune requête évaluée : vérifier la cohérence queries/qrels_valid.")

    macro_precision = np.mean([m["precision_at_k"] for m in per_query_metrics])
    macro_recall = np.mean([m["recall_at_k"] for m in per_query_metrics])
    macro_f1 = np.mean([m["f1_at_k"] for m in per_query_metrics])

    if any(m["auc"] is not None for m in per_query_metrics):
        macro_auc = np.mean([m["auc"] for m in per_query_metrics if m["auc"] is not None])
    else:
        macro_auc = None

    micro_precision = sum_TP / sum_retrieved if sum_retrieved > 0 else 0.0
    micro_recall = sum_TP / sum_positives if sum_positives > 0 else 0.0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)) if (micro_precision + micro_recall) > 0 else 0.0

    print_info("=== Résultats globaux (valid.tsv) ===")
    print_ok(f"Nombre de requêtes évaluées : {len(per_query_metrics)}")
    print_ok(f"Macro P@5 : {macro_precision:.4f}")
    print_ok(f"Macro R@5 : {macro_recall:.4f}")
    print_ok(f"Macro F1@5: {macro_f1:.4f}")
    print_ok(f"Micro P@5 : {micro_precision:.4f}")
    print_ok(f"Micro R@5 : {micro_recall:.4f}")
    print_ok(f"Micro F1@5: {micro_f1:.4f}")

    if macro_auc is not None:
        print_ok(f"Macro AUC (moyenne des AUC par requête): {macro_auc:.4f}")
    else:
        print_warn("Macro AUC non disponible (aucune AUC calculable)")


# ============================================================
#  SECTION 5 — Génération submission dense (ordre préservé)
# ============================================================

def generate_submission_dense(corpus, queries, model, corpus_embeddings, sample_path, output_file):
    """
    Génère un fichier de soumission Kaggle (dense) :
    - lit sample_submission.csv
    - calcule un score cosine pour chaque paire (query-id, corpus-id)
    - IMPORTANT : l'ordre du fichier est conservé (groupby sort=False + index)
    """
    print_subsection("Génération submission dense (sample_submission.csv -> score)")

    print_info(f"Lecture : {sample_path}")
    df = pd.read_csv(sample_path, sep=",", dtype=str)

    doc_index = {doc_id: i for i, doc_id in enumerate(corpus.keys())}
    df["score"] = np.nan

    print_info("Calcul scores (ordre préservé)")
    for qid, group in df.groupby("query-id", sort=False):
        rows_idx = group.index

        if qid not in queries:
            df.loc[rows_idx, "score"] = 0.0
            continue

        q_vec = model.encode([queries[qid]["text"]], convert_to_numpy=True)

        scores = []
        for docid in group["corpus-id"].tolist():
            i = doc_index.get(docid, None)
            if i is None:
                scores.append(0.0)
            else:
                doc_vec = corpus_embeddings[i].reshape(1, -1)
                sim = cosine_similarity(q_vec, doc_vec)[0, 0]
                scores.append(float(sim))

        df.loc[rows_idx, "score"] = scores

    df["score"] = df["score"].fillna(0.0).astype(float)
    df.to_csv(output_file, sep=",", index=False)

    print_ok(f"Fichier exporté : {output_file}")
    print_info("Score = similarité cosinus dense (alignée ligne à ligne)")


# ============================================================
#  SECTION 6 — LDA + UMAP + PyLDAvis (optionnel)
# ============================================================

def run_lda_topic_modeling(corpus):
    """
    Intermède exploratoire (LDA) :
    - nettoyage + stopwords
    - apprentissage LDA (ou chargement si déjà entraîné)
    - export topic dominant par doc
    - UMAP pour visualiser
    - PyLDAvis pour exploration interactive
    """
    import warnings
    warnings.filterwarnings("ignore")

    from gensim import corpora, matutils
    from gensim.utils import simple_preprocess
    from gensim.models.ldamodel import LdaModel
    from umap.umap_ import UMAP
    import umap.plot
    import pyLDAvis
    import pyLDAvis.gensim_models as gensimvis
    from IPython.display import HTML, display

    print_subsection("Analyse thématique LDA (optionnel)")

    if not os.path.exists(STOPWORDS_PATH):
        print_err("Stopwords introuvables")
        raise FileNotFoundError(" Le fichier Stop-words-en.txt est introuvable !")

    docs = [doc["title"] + " " + doc["text"] for doc in corpus.values()]
    ndocs = len(docs)
    print_info(f"Documents : {ndocs}")

    def sent_to_words(sentences):
        for s in sentences:
            yield simple_preprocess(str(s), deacc=True)

    data_words = list(sent_to_words(docs))

    with open(STOPWORDS_PATH) as f:
        stopwords = [w.strip() for w in f.readlines()]

    def remove_stopwords(texts):
        return [[w for w in doc if w not in stopwords] for doc in texts]

    data_words_nostops = remove_stopwords(data_words)

    dico = corpora.Dictionary(data_words_nostops)
    dico.filter_extremes(no_below=10, no_above=0.5)

    corpus_bow = [dico.doc2bow(text) for text in data_words_nostops]
    print_info(f"Vocab retenu : {len(dico)} tokens")

    ntopics = 30
    model_path = os.path.join(LDA_DIR, f"lda_model_{ntopics}.mdl")

    if os.path.exists(model_path):
        print_info(f"Chargement modèle LDA : {model_path}")
        ldamodel = LdaModel.load(model_path)
        print_ok("Modèle LDA chargé")
    else:
        print_info(f"Entraînement LDA ({ntopics} topics)")
        ldamodel = LdaModel(corpus=corpus_bow, num_topics=ntopics, id2word=dico, passes=30, random_state=42)
        ldamodel.save(model_path)
        print_ok(f"Modèle LDA sauvegardé : {model_path}")

    N_TOPICS_TO_PRINT = min(8, ntopics)
    doc_vec = np.zeros((ndocs, ntopics))
    topic_importance = doc_vec.sum(axis=0)
    top_topics = np.argsort(topic_importance)[::-1][:N_TOPICS_TO_PRINT]
    
    print_info(f"Topics dominants (top {N_TOPICS_TO_PRINT} sur {ntopics})")
    for t in top_topics:
        words = ldamodel.show_topic(t, topn=10)
        words_str = " + ".join([f"{w}({p:.3f})" for w, p in words])
        print(f"Topic {t}: {words_str}")
    
    for i, bow in enumerate(corpus_bow):
        for topic, weight in ldamodel.get_document_topics(bow):
            doc_vec[i, topic] = weight

    dominant_topic = np.argmax(doc_vec, axis=1)

    df_topics = pd.DataFrame({
        "doc_id": list(corpus.keys()),
        "topic_dominant": dominant_topic
    })
    out_csv = os.path.join(LDA_DIR, "topics_by_document.csv")
    df_topics.to_csv(out_csv, index=False)
    print_ok(f"Export : {out_csv}")

    print_info("UMAP clustering")
    mapper = UMAP(metric="cosine", n_neighbors=15, min_dist=0.1).fit(doc_vec)
    umap.plot.points(mapper, labels=dominant_topic)

    plt.title("Projection UMAP des documents selon LDA Topics")
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.hist(dominant_topic, bins=ntopics)
    plt.title("Répartition des documents par topic dominant")
    plt.xlabel("Topic")
    plt.ylabel("Documents")
    plt.show()

    print_info("PyLDAvis (export HTML pour Spyder)")
    corpus_csc = matutils.corpus2csc(corpus_bow)
    lda_vis = gensimvis.prepare(ldamodel, corpus_csc, dico)
    
    # 1) Sauvegarde dans un fichier HTML (le plus fiable sous Spyder)
    out_html = os.path.join(LDA_DIR, "pyldavis_lda.html")
    pyLDAvis.save_html(lda_vis, out_html)
    print_ok(f"PyLDAvis sauvegardé : {out_html}")
    
    # 2) Ouverture automatique dans le navigateur (optionnel)
    try:
        import webbrowser
        webbrowser.open("file:///" + out_html.replace("\\", "/"))
        print_ok("PyLDAvis ouvert dans le navigateur")
    except Exception as e:
        print_warn(f"Impossible d'ouvrir automatiquement le navigateur : {e}")


# ============================================================
#  SECTION 7 — Graphe citations + stats + centralités
# ============================================================

def build_citation_graph(corpus):
    """
    Construit le graphe orienté de citations :
    - arc doc -> ref si doc cite ref
    - arc citer -> doc si doc est cité par citer
    """
    print_subsection("Construction du graphe de citations (NetworkX)")

    edges = set()
    missing_refs = 0
    missing_citers = 0

    for doc_id, doc in corpus.items():
        for ref in doc.get("metadata", {}).get("references", []):
            if ref in corpus and ref != doc_id:
                edges.add((doc_id, ref))
            else:
                missing_refs += 1

        for citer in doc.get("metadata", {}).get("cited_by", []):
            if citer in corpus and citer != doc_id:
                edges.add((citer, doc_id))
            else:
                missing_citers += 1

    G = nx.DiGraph()
    G.add_nodes_from(corpus.keys())
    G.add_edges_from(edges)

    print_ok(f"Nœuds : {G.number_of_nodes()}")
    print_ok(f"Arcs  : {G.number_of_edges()}")
    print_info(f"Densité : {nx.density(G):.10f}")
    print_warn(f"Références manquantes : {missing_refs} | Citers manquants : {missing_citers}")

    return G


def analyze_and_plot_graph(G, corpus, top_n_subgraph=500):
    """
    Analyse + visualisation :
    - sous-graphe des nœuds les plus connectés
    - distributions degrés entrants/sortants
    - visualisation matrice A, A², A³
    - centralités (PageRank, Betweenness, Katz, Degree)
    """
    print_subsection("Analyse du graphe (degrés, matrice, centralités)")

    print_info(f"Visualisation sur les {top_n_subgraph} nœuds avec plus haut degré")
    top_deg = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:top_n_subgraph]
    sub_nodes = [n for n, _ in top_deg]
    G_sub = G.subgraph(sub_nodes)

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G_sub, k=0.25, iterations=40)

    nx.draw_networkx_nodes(G_sub, pos, node_size=40, node_color="skyblue")
    nx.draw_networkx_edges(G_sub, pos, width=0.3, alpha=0.4, arrows=False)
    nx.draw_networkx_labels(G_sub, pos, font_size=5)
    plt.title(f"Sous-graphe des {top_n_subgraph} articles les plus connectés")
    plt.show()

    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.hist(in_deg.values(), bins=60, color="steelblue")
    plt.title("Degrés entrants (citations reçues)")
    plt.xlabel("Nombre de citations")

    plt.subplot(122)
    plt.hist(out_deg.values(), bins=60, color="orange")
    plt.title("Degrés sortants (références faites)")
    plt.xlabel("Nombre de références")
    plt.show()

    print_ok(f"Moyenne IN : {np.mean(list(in_deg.values())):.4f} | Variance IN : {np.var(list(in_deg.values())):.4f}")
    print_ok(f"Moyenne OUT: {np.mean(list(out_deg.values())):.4f} | Variance OUT: {np.var(list(out_deg.values())):.4f}")

    print_info("Matrice d'adjacence (sous-graphe)")
    A = nx.to_scipy_sparse_array(G_sub, dtype=int)

    plt.spy(A, markersize=2)
    plt.title(f"Matrice A (liens directs) — {top_n_subgraph} nœuds")
    plt.show()

    A2 = A @ A
    A3 = A @ A2

    plt.figure(figsize=(14, 5))
    plt.subplot(121)
    plt.imshow(A2.todense(), cmap="binary", interpolation="none")
    plt.title("A² — citations indirectes à 2 sauts")
    plt.colorbar()

    plt.subplot(122)
    plt.imshow(A3.todense(), cmap="binary", interpolation="none")
    plt.title("A³ — citations à 3 sauts")
    plt.colorbar()
    plt.show()

    print_info("Calcul centralités (PageRank / Betweenness / Katz / Degree)")
    centralities = {
        "PageRank": nx.pagerank(G, alpha=0.85),
        "Betweenness": nx.betweenness_centrality(G, k=300, seed=42),
        "Katz": nx.katz_centrality_numpy(G, alpha=0.005),
        "Degree": dict(G.degree())
    }

    pos_full = nx.spring_layout(G_sub, seed=42)

    plt.figure(figsize=(16, 10))
    for i, (name, values) in enumerate(centralities.items()):
        plt.subplot(2, 2, i + 1)
        nx.draw(
            G_sub, pos=pos_full, node_size=30,
            node_color=[values[n] for n in G_sub], cmap="viridis",
            with_labels=False
        )
        plt.title(name)
    plt.tight_layout()
    plt.show()

    for name, values in centralities.items():
        print("\n" + "-" * 40)
        print_info(f"Top 10 {name}")
        top = sorted(values.items(), key=lambda x: x[1], reverse=True)[:10]
        for doc, score in top:
            print(f" {doc} ({score:.4f}) — {corpus[doc]['title'][:90]}")

    print_ok("Graphe construit et analysé")
    return G_sub


# ============================================================
#  SECTION 8 — Graph-enhanced embeddings + éval + ROC
# ============================================================

def build_graph_enhanced_embeddings(corpus, corpus_embeddings, G, alpha=0.70, beta=0.30):
    """
    Construit des embeddings "graph-aware" :
    - pour chaque doc : alpha * embedding(doc) + beta * mean(embedding(voisins))
    - voisins = prédécesseurs + successeurs dans le graphe
    """
    from tqdm import tqdm

    print_subsection("Graph-enhanced embeddings (moyenne des voisins)")

    node_to_idx = {doc_id: i for i, doc_id in enumerate(corpus.keys())}
    idx_to_node = list(corpus.keys())

    corpus_size = len(corpus_embeddings)
    emb_dim = corpus_embeddings.shape[1]
    enhanced_embeddings = np.zeros_like(corpus_embeddings)

    print_info(f"alpha={alpha} | beta={beta}")
    print_info(f"Shape embeddings : {corpus_size} docs x {emb_dim} dims")

    for i, doc_id in tqdm(enumerate(idx_to_node), total=corpus_size):
        base_vec = corpus_embeddings[i]

        neigh = list(G.predecessors(doc_id)) + list(G.successors(doc_id))
        neigh = [n for n in neigh if n in node_to_idx]

        if len(neigh) == 0:
            enhanced_embeddings[i] = base_vec
            continue

        neigh_vecs = np.array([corpus_embeddings[node_to_idx[n]] for n in neigh])
        mean_vec = np.mean(neigh_vecs, axis=0)

        enhanced_embeddings[i] = alpha * base_vec + beta * mean_vec

    print_ok("Graph-Enhanced Embeddings générés")
    return enhanced_embeddings, node_to_idx


def evaluate_dense_vs_graph(corpus, queries, qrels_valid, model, corpus_embeddings, enhanced_embeddings, node_to_idx):
    """
    Compare dense vs graph+ :
    - encode_query_graph : même logique que pour les docs (ajout des voisins de la query)
    - scores_old : cos(query_emb, doc_emb)
    - scores_new : cos(query_graph_emb, doc_graph_emb)
    - P@5 et AUC moyenne
    """
    print_subsection("Comparaison Dense vs Graph+ (P@5 / F1@5 / AUC)")

    TOP_K_CHECK = 5

    def encode_query_graph(qid):
        q = model.encode(queries[qid]["text"], convert_to_numpy=True)

        refs = queries[qid].get("metadata", {}).get("references", [])
        cites = queries[qid].get("metadata", {}).get("cited_by", [])
        neigh = refs + cites

        neigh_vecs = []
        for n in neigh:
            if n in node_to_idx:
                neigh_vecs.append(enhanced_embeddings[node_to_idx[n]])

        if len(neigh_vecs) == 0:
            return q
        return 0.7 * q + 0.3 * np.mean(neigh_vecs, axis=0)

    def compare_search(qid, candidate_ids):
        q_text = queries[qid]["text"]
        cand_valid = [d for d in candidate_ids if d in corpus]

        emb_old = np.vstack([corpus_embeddings[node_to_idx[d]] for d in cand_valid])
        q_old = model.encode([q_text], convert_to_numpy=True)
        scores_old = cosine_similarity(q_old, emb_old)[0]

        emb_new = np.vstack([enhanced_embeddings[node_to_idx[d]] for d in cand_valid])
        q_new = encode_query_graph(qid).reshape(1, -1)
        scores_new = cosine_similarity(q_new, emb_new)[0]

        labels = np.array([qrels_valid[qid].get(d, 0) for d in cand_valid])

        auc_old = roc_auc_score(labels, scores_old) if len(set(labels)) > 1 else None
        auc_new = roc_auc_score(labels, scores_new) if len(set(labels)) > 1 else None

        def metrics(scores):
            order = np.argsort(scores)[::-1][:TOP_K_CHECK]
            top_docs = [cand_valid[i] for i in order]
            tp = sum(qrels_valid[qid].get(d, 0) == 1 for d in top_docs)
            prec = tp / TOP_K_CHECK
            rec = tp / 5
            f1v = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
            return prec, rec, f1v

        P_old, R_old, F_old = metrics(scores_old)
        P_new, R_new, F_new = metrics(scores_new)

        return auc_old, auc_new, P_old, P_new, F_old, F_new, encode_query_graph

    auc_o, auc_n = [], []
    P_o, P_n = [], []
    F_o, F_n = [], []

    encode_query_graph_ref = None

    for qid, cands in qrels_valid.items():
        if qid not in queries:
            continue

        auc_old, auc_new, P_old, P_new, F_old, F_new, enc_fun = compare_search(qid, list(cands.keys()))
        encode_query_graph_ref = enc_fun

        if auc_old is not None and auc_new is not None:
            auc_o.append(auc_old)
            auc_n.append(auc_new)

        P_o.append(P_old)
        P_n.append(P_new)
        F_o.append(F_old)
        F_n.append(F_new)

    print_info("Résumé comparaison")
    print_ok(f"Nombre de requêtes évaluées : {len(P_o)}")
    print_ok(f"AUC moyen dense  : {np.mean(auc_o) if len(auc_o)>0 else np.nan}")
    print_ok(f"AUC moyen graph+ : {np.mean(auc_n) if len(auc_n)>0 else np.nan}")
    print_ok(f"P@5 dense        : {np.mean(P_o)}")
    print_ok(f"P@5 graph+       : {np.mean(P_n)}")
    print_ok(f"F1@5 dense       : {np.mean(F_o)}")
    print_ok(f"F1@5 graph+      : {np.mean(F_n)}")

    return encode_query_graph_ref


def plot_global_roc(
    corpus, queries, qrels_valid,
    model, corpus_embeddings, enhanced_embeddings, node_to_idx, encode_query_graph,
    vectorizer_merged, X_merged, all_doc_ids
):
    """
    Trace une ROC globale agrégée sur toutes les requêtes :
    - concaténation des scores et labels
    - calcule AUC globale Sparse TF-IDF vs Dense vs Graph+
    """
    print_subsection("ROC globale (Sparse TF-IDF vs Dense vs Graph+)")

    all_scores_sparse, all_labels_sparse = [], []
    all_scores_dense,  all_labels_dense  = [], []
    all_scores_graph,  all_labels_graph  = [], []

    for qid, cand_map in qrels_valid.items():
        if qid not in queries:
            continue

        q_text = queries[qid]["text"]

        # on garde uniquement les candidats présents dans le corpus
        candidate_ids = [d for d in cand_map.keys() if d in corpus]
        if len(candidate_ids) == 0:
            continue

        labels = np.array([cand_map.get(d, 0) for d in candidate_ids])

        # ---------- SPARSE TF-IDF ----------
        q_vec_sparse = vectorizer_merged.transform([q_text])
        cand_idx = [all_doc_ids.index(d) for d in candidate_ids]
        X_cand = X_merged[cand_idx]
        scores_sparse = cosine_similarity(q_vec_sparse, X_cand)[0]
        all_scores_sparse.extend(scores_sparse)
        all_labels_sparse.extend(labels)

        # ---------- DENSE ----------
        emb_old = np.vstack([corpus_embeddings[node_to_idx[d]] for d in candidate_ids])
        q_old = model.encode([q_text], convert_to_numpy=True)
        scores_old = cosine_similarity(q_old, emb_old)[0]
        all_scores_dense.extend(scores_old)
        all_labels_dense.extend(labels)

        # ---------- GRAPH+ ----------
        emb_new = np.vstack([enhanced_embeddings[node_to_idx[d]] for d in candidate_ids])
        q_new = encode_query_graph(qid).reshape(1, -1)
        scores_new = cosine_similarity(q_new, emb_new)[0]
        all_scores_graph.extend(scores_new)
        all_labels_graph.extend(labels)

    # ROC
    fpr_sparse, tpr_sparse, _ = roc_curve(all_labels_sparse, all_scores_sparse)
    fpr_dense,  tpr_dense,  _ = roc_curve(all_labels_dense,  all_scores_dense)
    fpr_graph,  tpr_graph,  _ = roc_curve(all_labels_graph,  all_scores_graph)

    auc_sparse = auc(fpr_sparse, tpr_sparse)
    auc_dense  = auc(fpr_dense,  tpr_dense)
    auc_graph  = auc(fpr_graph,  tpr_graph)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr_sparse, tpr_sparse, label=f"Sparse TF-IDF (AUC={auc_sparse:.4f})", lw=2)
    plt.plot(fpr_dense,  tpr_dense,  label=f"Dense (AUC={auc_dense:.4f})", lw=2)
    plt.plot(fpr_graph,  tpr_graph,  label=f"Graph+ (AUC={auc_graph:.4f})", lw=2)

    plt.plot([0, 1], [0, 1], "--", color="grey", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Globale — Sparse TF-IDF vs Dense vs Dense+Graph")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    print_ok(f"AUC globale Sparse TF-IDF : {auc_sparse:.4f}")
    print_ok(f"AUC globale Dense : {auc_dense:.4f}")
    print_ok(f"AUC globale Graph : {auc_graph:.4f}")



def generate_submission_graph(corpus, queries, enhanced_embeddings, sample_path, output_file, encode_query_graph):
    """
    Génère un fichier de soumission Kaggle (Graph-enhanced) :
    - score = cosine(query_graph, doc_graph)
    - ordre préservé
    """
    print_subsection("Génération submission Graph+ (sample_submission.csv -> score)")

    print_info(f"Lecture : {sample_path}")
    df = pd.read_csv(sample_path, sep=",", dtype=str)

    doc_index = {doc_id: i for i, doc_id in enumerate(corpus.keys())}
    df["score"] = 0.0

    print_info("Calcul scores (ordre préservé)")
    for qid, group_df in df.groupby("query-id", sort=False):
        if qid not in queries:
            df.loc[group_df.index, "score"] = 0.0
            continue

        q_vec = encode_query_graph(qid).reshape(1, -1)

        candidates = group_df["corpus-id"].tolist()
        scores = []

        for docid in candidates:
            idx = doc_index.get(docid)
            if idx is None:
                scores.append(-999.0)
            else:
                emb = enhanced_embeddings[idx].reshape(1, -1)
                scores.append(float(cosine_similarity(q_vec, emb)[0, 0]))

        df.loc[group_df.index, "score"] = scores

    df.to_csv(output_file, sep=",", index=False)

    print_ok(f"Fichier exporté : {output_file}")
    print_info("Score = similarité cosinus sur embeddings graph-enhanced")
    
    

# ============================================================
#  SECTION 9 — EXPÉRIMENTATIONS (VARIANTES SPARSE + DENSE)
#  + Ajout variantes STEMMING côté SPARSE
#
#  IMPORTANT (dataset) :
#  - Query text = queries[qid]["text"]
#  - Corpus doc text = title + text
#  - Évaluation : uniquement parmi les ~25 candidats (valid.tsv)
#  - Dense : embeddings cachés en local (MiniLM réutilise EMBEDDINGS_FILE)
# ============================================================


def _build_doc_index_fast(corpus: dict) -> dict:
    """Index doc_id -> position (évite les all_doc_ids.index(...) O(N))."""
    return {doc_id: i for i, doc_id in enumerate(corpus.keys())}


def _evaluate_from_scorefn(qrels_valid: dict, score_fn, k: int = 5):
    """
    Calcule Macro P@k / R@k / F1@k + Macro AUC.
    score_fn(qid, candidate_ids) -> (scores, labels) alignés.
    """
    p_list, r_list, f_list = [], [], []
    auc_list = []
    n_q = 0
    n_auc = 0

    for qid, cand_map in qrels_valid.items():
        candidate_ids = list(cand_map.keys())
        scores, labels = score_fn(qid, candidate_ids)

        if scores is None or labels is None or len(labels) == 0:
            continue

        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels, dtype=int)

        order = np.argsort(scores)[::-1]
        topk = order[: min(k, len(order))]

        tp = int(np.sum(labels[topk] == 1))
        retrieved = len(topk)
        positives_total = int(np.sum(labels))

        prec = tp / retrieved if retrieved else 0.0
        rec = tp / positives_total if positives_total else 0.0
        f1v = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

        p_list.append(prec)
        r_list.append(rec)
        f_list.append(f1v)

        if len(set(labels.tolist())) > 1:
            try:
                auc_list.append(roc_auc_score(labels, scores))
                n_auc += 1
            except Exception:
                pass

        n_q += 1

    macro_p = float(np.mean(p_list)) if p_list else 0.0
    macro_r = float(np.mean(r_list)) if r_list else 0.0
    macro_f = float(np.mean(f_list)) if f_list else 0.0
    macro_auc = float(np.mean(auc_list)) if auc_list else None

    return macro_p, macro_r, macro_f, macro_auc, n_q, n_auc


# ============================================================
#  STEMMING UTILITIES
# ============================================================

_stemmer = SnowballStemmer("english")


def _simple_word_tokenize(text: str) -> list[str]:
    """Tokenizer simple et robuste."""
    return re.findall(r"\b\w+\b", (text or "").lower())


def _stem_tokenizer(text: str) -> list[str]:
    """Tokenize puis stemming Snowball."""
    toks = _simple_word_tokenize(text)
    return [_stemmer.stem(t) for t in toks]


# ----------------------------
# SPARSE variants (TF-IDF)
# ----------------------------
def _make_sparse_variant_scorefn(corpus, queries, qrels_valid, tfidf_vectorizer: TfidfVectorizer):
    doc_index = _build_doc_index_fast(corpus)
    docs_corpus = [(d.get("title", "") + " " + d.get("text", "")).strip() for d in corpus.values()]
    X_corpus = tfidf_vectorizer.fit_transform(docs_corpus)

    def score_fn(qid: str, candidate_ids: list[str]):
        if qid not in queries:
            return None, None

        q_text = (queries[qid].get("text", "") or "").strip()
        if not q_text:
            return None, None

        cand_ids = [d for d in candidate_ids if d in doc_index]
        if not cand_ids:
            return None, None

        cand_idx = [doc_index[d] for d in cand_ids]
        X_cand = X_corpus[cand_idx]

        q_vec = tfidf_vectorizer.transform([q_text])
        scores = cosine_similarity(q_vec, X_cand)[0]
        labels = [int(qrels_valid[qid].get(d, 0)) for d in cand_ids]
        return scores, labels

    return score_fn


def run_experiments_sparse_variants(corpus, queries, qrels_valid) -> pd.DataFrame:
    print_subsection("EXPÉRIMENTATIONS — Variantes SPARSE (TF-IDF + Stemming)")

    variants = [
        ("tfidf_baseline", TfidfVectorizer()),
        ("tfidf_stopwords", TfidfVectorizer(stop_words="english")),
        ("tfidf_min2_max90", TfidfVectorizer(min_df=2, max_df=0.90)),
        ("tfidf_bigrams", TfidfVectorizer(ngram_range=(1, 2))),
        ("tfidf_stop_bigrams", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
        ("tfidf_sublinear_tf", TfidfVectorizer(sublinear_tf=True)),
        ("tfidf_stem", TfidfVectorizer(tokenizer=_stem_tokenizer, token_pattern=None, lowercase=False)),
        ("tfidf_stop_stem",
         TfidfVectorizer(tokenizer=_stem_tokenizer, token_pattern=None, lowercase=False, stop_words="english")),
    ]

    rows = []
    for name, vec in variants:
        print_info(f"[SPARSE] Variant: {name}")
        score_fn = _make_sparse_variant_scorefn(corpus, queries, qrels_valid, vec)
        mp, mr, mf, mauc, n_q, n_auc = _evaluate_from_scorefn(qrels_valid, score_fn, k=5)

        rows.append({
            "family": "sparse",
            "variant": name,
            "macro_P@5": mp,
            "macro_R@5": mr,
            "macro_F1@5": mf,
            "macro_AUC": (mauc if mauc is not None else np.nan),
            "n_queries": n_q,
            "n_queries_auc": n_auc
        })

    df = pd.DataFrame(rows).sort_values("macro_AUC", ascending=False)
    return df

        
# ----------------------------
# DENSE variants (cache embeddings par modèle)
# ----------------------------
def _emb_cache_path(base_path: str, model_name: str) -> str:
    safe = model_name.replace("/", "_")
    return os.path.join(base_path, f"corpus_embeddings__{safe}.pkl")


def _build_or_load_embeddings_for_model(corpus: dict, model_name: str, base_path: str) -> np.ndarray:
    cache_path = _emb_cache_path(base_path, model_name)

    if model_name == "all-MiniLM-L6-v2" and os.path.exists(EMBEDDINGS_FILE):
        print_info(f"[DENSE] Réutilisation cache existant: {EMBEDDINGS_FILE}")
        with open(EMBEDDINGS_FILE, "rb") as f:
            emb = pickle.load(f)
        print_ok(f"[DENSE] Embeddings chargés (MiniLM): shape={emb.shape}")
        return emb

    if os.path.exists(cache_path):
        print_info(f"[DENSE] Chargement embeddings cache: {cache_path}")
        with open(cache_path, "rb") as f:
            emb = pickle.load(f)
        print_ok(f"[DENSE] Embeddings chargés: shape={emb.shape}")
        return emb

    print_info(f"[DENSE] Encodage corpus avec: {model_name} (1 seule fois puis cache)")
    model = SentenceTransformer(model_name)
    docs = [(d.get("title", "") + " " + d.get("text", "")).strip() for d in corpus.values()]
    emb = model.encode(docs, show_progress_bar=True, convert_to_numpy=True)

    with open(cache_path, "wb") as f:
        pickle.dump(emb, f)
    print_ok(f"[DENSE] Cache écrit: {cache_path} | shape={emb.shape}")
    return emb


def _make_dense_variant_scorefn(corpus, queries, qrels_valid, model_name: str, corpus_embeddings: np.ndarray):
    doc_index = _build_doc_index_fast(corpus)
    model = SentenceTransformer(model_name)

    def score_fn(qid: str, candidate_ids: list[str]):
        if qid not in queries:
            return None, None

        q_text = (queries[qid].get("text", "") or "").strip()
        if not q_text:
            return None, None

        cand_ids = [d for d in candidate_ids if d in doc_index]
        if not cand_ids:
            return None, None

        cand_idx = [doc_index[d] for d in cand_ids]
        cand_emb = corpus_embeddings[cand_idx]

        q_vec = model.encode([q_text], convert_to_numpy=True)
        scores = cosine_similarity(q_vec, cand_emb)[0]
        labels = [int(qrels_valid[qid].get(d, 0)) for d in cand_ids]
        return scores, labels

    return score_fn


def run_experiments_dense_variants(corpus, queries, qrels_valid, base_path: str) -> pd.DataFrame:
    print_subsection("EXPÉRIMENTATIONS — Variantes DENSE (Sentence-Transformers + cache local)")

    model_names = [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
    ]

    rows = []
    for mn in model_names:
        print_info(f"[DENSE] Variant: {mn}")
        emb = _build_or_load_embeddings_for_model(corpus, mn, base_path=base_path)
        score_fn = _make_dense_variant_scorefn(corpus, queries, qrels_valid, mn, emb)

        mp, mr, mf, mauc, n_q, n_auc = _evaluate_from_scorefn(qrels_valid, score_fn, k=5)

        rows.append({
            "family": "dense",
            "variant": mn,
            "macro_P@5": mp,
            "macro_R@5": mr,
            "macro_F1@5": mf,
            "macro_AUC": (mauc if mauc is not None else np.nan),
            "n_queries": n_q,
            "n_queries_auc": n_auc
        })

    df = pd.DataFrame(rows).sort_values("macro_AUC", ascending=False)
    return df


def run_full_experimental_section(corpus, queries, qrels_valid, base_path: str) -> pd.DataFrame:


    df_sparse = run_experiments_sparse_variants(corpus, queries, qrels_valid)
    print("\n--- RÉCAP SPARSE (trié par AUC) ---")
    print(df_sparse.to_string(index=False))

    df_dense = run_experiments_dense_variants(corpus, queries, qrels_valid, base_path=base_path)
    print("\n--- RÉCAP DENSE (trié par AUC) ---")
    print(df_dense.to_string(index=False))

    df_all = pd.concat([df_sparse, df_dense], ignore_index=True)
    df_all = df_all.sort_values(["family", "macro_AUC"], ascending=[True, False])

    out_csv = os.path.join(base_path, "variant_results.csv")
    df_all.to_csv(out_csv, index=False)
    print_ok(f"Export résultats variantes : {out_csv}")

    print("\n--- Meilleur par famille (macro_AUC) ---")
    print(df_all.groupby("family", as_index=False).head(1).to_string(index=False))

    return df_all



# ============================================================
#  FAISS — Index dense pour recherche rapide sur tout le corpus
# ============================================================

def build_faiss_index(corpus_embeddings: np.ndarray) -> faiss.Index:
    """
    Construit un index FAISS exact (IndexFlatIP) pour similarité cosinus.
    """
    print_subsection("Construction index FAISS (IndexFlatIP)")

    emb = corpus_embeddings.astype("float32")

    # Normalisation L2 pour que le produit scalaire = cosinus
    faiss.normalize_L2(emb)

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    print_ok(f"Index FAISS construit : {index.ntotal} vecteurs, dim={dim}")
    return index

def search_faiss(query_text: str, model, index: faiss.Index, corpus: dict, top_k: int = 10):
    """
    Recherche top_k documents dans tout le corpus via FAISS.
    """
    q_vec = model.encode([query_text], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_vec)

    scores, indices = index.search(q_vec, top_k)

    doc_ids = list(corpus.keys())
    results = []
    for rank, idx in enumerate(indices[0]):
        results.append({
            "rank": rank + 1,
            "doc_id": doc_ids[idx],
            "score": float(scores[0][rank]),
            "title": corpus[doc_ids[idx]]["title"]
        })
    return results

def demo_faiss_search(corpus, model, corpus_embeddings):
    """
    Démo simple de recherche FAISS sur tout le corpus.
    """
    print_subsection("Démo FAISS — recherche dense sur tout le corpus")

    index = build_faiss_index(corpus_embeddings)

    query = "machine learning medical diagnosis"
    print_info(f"Query FAISS : {query}")

    results = search_faiss(query, model, index, corpus, top_k=10)
    for r in results:
        print(f"{r['rank']:>2}. [{r['score']:.4f}] {r['title']}")

# ============================================================
#  SECTION 10 — Submission Graph+ avec all-mpnet-base-v2
#  Objectif : générer "submission_graph_all-mpnet-base-v2.csv" 
# ============================================================

def generate_submission_graph_with_model(
    corpus, queries, G, qrels_valid,
    model_name: str,
    base_path: str,
    sample_path: str,
    output_file: str,
    alpha: float = 0.70,
    beta: float = 0.30,
):
    """
    Pipeline complet Graph+ pour un modèle sentence-transformers donné :
    1) load/compute + cache embeddings corpus (title+text) pour model_name
    2) graph-enhanced embeddings sur le corpus
    3) encode_query_graph(qid) (query = title-only, + voisins via metadata refs/cited_by)
    4) génération du CSV submission (ordre conservé via groupby(sort=False))

    Sortie : output_file (csv)
    """
    
    # 1) Embeddings corpus (cache par modèle)
    print_subsection("1) Embeddings corpus (cache par modèle)")
    corpus_embeddings = _build_or_load_embeddings_for_model(corpus, model_name, base_path=base_path)

    # 2) Graph-enhanced embeddings (moyenne voisins)
    print_subsection("2) Graph-enhanced embeddings")
    enhanced_embeddings, node_to_idx = build_graph_enhanced_embeddings(
        corpus, corpus_embeddings, G, alpha=alpha, beta=beta
    )

    # 3) Encodeur query_graph spécifique (query = queries[qid]["text"])
    print_subsection("3) Encodage query_graph (title-only + voisins)")
    model = SentenceTransformer(model_name)

    def encode_query_graph_mpnet(qid: str) -> np.ndarray:
        q_text = (queries[qid].get("text", "") or "").strip()
        q = model.encode(q_text, convert_to_numpy=True)

        refs = queries[qid].get("metadata", {}).get("references", [])
        cites = queries[qid].get("metadata", {}).get("cited_by", [])
        neigh = refs + cites

        neigh_vecs = []
        for n in neigh:
            if n in node_to_idx:
                neigh_vecs.append(enhanced_embeddings[node_to_idx[n]])

        if len(neigh_vecs) == 0:
            return q
        return alpha * q + beta * np.mean(neigh_vecs, axis=0)

    # 4) Génération CSV submission Graph+
    print_subsection("4) Génération submission CSV (ordre préservé)")
    generate_submission_graph(
        corpus=corpus,
        queries=queries,
        enhanced_embeddings=enhanced_embeddings,
        sample_path=sample_path,
        output_file=output_file,
        encode_query_graph=encode_query_graph_mpnet
    )

    print_ok(f"Submission Graph+ ({model_name}) générée : {output_file}")



# ============================================================
#  MAIN
# ============================================================

def main():
    # --------------------
    # Load data
    # --------------------
    print_section("INITIALISATION & CHARGEMENT DES DONNÉES")

    print_subsection("Chargement des fichiers")
    print_info("Lecture du corpus.jsonl")
    corpus = load_corpus(CORPUS_PATH)
    print_ok(f"{len(corpus)} documents chargés")

    print_info("Lecture du queries.jsonl")
    queries = load_queries(QUERIES_PATH)
    print_ok(f"{len(queries)} requêtes chargées")

    print_info("Lecture du valid.tsv (qrels)")
    qrels_valid = load_qrels(QRELS_PATH)
    print_ok(f"{len(qrels_valid)} requêtes annotées (valid)")

    # Debug (désactivé par défaut)
    # print_subsection("DEBUG — premières lignes (désactivé)")
    # first_corpus_id, first_corpus_doc = next(iter(corpus.items()))
    # print("=== Première ligne du corpus ===")
    # print(f"{first_corpus_id}: {first_corpus_doc}")
    #
    # first_query_id, first_query_text = next(iter(queries.items()))
    # print("\n=== Première ligne des queries ===")
    # print(f"{first_query_id}: {first_query_text}")
    #
    # first_qrel_id, first_qrel_docs = next(iter(qrels_valid.items()))
    # print("\n=== Première ligne des qrels ===")
    # print(f"{first_qrel_id}: {first_qrel_docs}")

    # --------------------
    # Section 2: Stats + Sparse
    # --------------------
    print_section("SECTION 2 — EXPLORATION & REPRÉSENTATIONS SPARSE")

    compute_basic_stats(corpus, queries, qrels_valid)
    show_one_query_example(corpus, queries, qrels_valid)

    (
        titles, texts,
        vectorizer_titles, X_titles, features_titles,
        vectorizer_texts, X_texts, features_texts,
        docs_merged, vectorizer_merged, X_merged, features_merged
    ) = build_sparse_representations(corpus)

    show_sparse_examples(corpus, X_titles, features_titles, X_texts, features_texts)

    # --------------------
    # Section 3: Cosine + moteur sparse
    # --------------------
    print_section("SECTION 3 — SIMILARITÉ COSINUS & MOTEUR SPARSE")

    test_cosine_on_pairs(titles, texts, X_titles, X_texts)
    demo_sparse_search(corpus, vectorizer_titles, X_titles, vectorizer_texts, X_texts, vectorizer_merged, X_merged)
    evaluate_sparse_auc(corpus, queries, qrels_valid, vectorizer_merged, X_merged)

    # --------------------
    # Section 4: Dense embeddings
    # --------------------
    print_section("SECTION 4 — REPRÉSENTATIONS DENSES (EMBEDDINGS)")
    model, corpus_embeddings = build_or_load_embeddings(corpus, EMBEDDINGS_FILE)
    demo_dense_search(corpus, model, corpus_embeddings)
    
    # --- AJOUT FAISS ---
    demo_faiss_search(corpus, model, corpus_embeddings)

    # --------------------
    # Section 5: Dense evaluation
    # --------------------
    print_section("SECTION 5 — MOTEUR DENSE & ÉVALUATION")

    evaluate_dense_all_queries(corpus, queries, qrels_valid, model, corpus_embeddings)

    # --------------------
    # Section 5: Submission dense
    # --------------------
    print_section("SECTION 5 — GÉNÉRATION SOUMISSION DENSE")

    generate_submission_dense(corpus, queries, model, corpus_embeddings, SAMPLE_PATH, OUTPUT_SPARSE_SUB)

    # --------------------
    # Section 6: LDA 
    # --------------------
    print_section("SECTION 6 — LDA ")


    run_lda_topic_modeling(corpus)


    # --------------------
    # Section 7: Graphe citations
    # --------------------
    print_section("SECTION 7 — GRAPHE DE CITATIONS")

    G = build_citation_graph(corpus)
    _ = analyze_and_plot_graph(G, corpus, top_n_subgraph=500)

    # --------------------
    # Section 8: Graph-enhanced embeddings + éval + ROC + submission
    # --------------------
    print_section("SECTION 8 — EMBEDDINGS ENRICHIS PAR LE GRAPHE")

    enhanced_embeddings, node_to_idx = build_graph_enhanced_embeddings(
        corpus, corpus_embeddings, G, alpha=0.70, beta=0.30
    )

    encode_query_graph = evaluate_dense_vs_graph(
        corpus, queries, qrels_valid, model, corpus_embeddings, enhanced_embeddings, node_to_idx
    )
    
    all_doc_ids = list(corpus.keys())
    plot_global_roc(
        corpus, queries, qrels_valid,
        model, corpus_embeddings, enhanced_embeddings, node_to_idx, encode_query_graph,
        vectorizer_merged, X_merged, all_doc_ids
    )
    
    print_section("SECTION 8 — GÉNÉRATION SOUMISSION GRAPH+")
    generate_submission_graph(
        corpus, queries, enhanced_embeddings, SAMPLE_PATH, OUTPUT_GRAPH_SUB, encode_query_graph
    )
    

# --------------------
# Section 9: EXPÉRIMENTATIONS (VARIANTES SPARSE + DENSE)
# --------------------

    print_section("SECTION 9: EXPÉRIMENTATIONS (VARIANTES SPARSE + DENSE)")
    run_full_experimental_section(corpus, queries, qrels_valid, BASE_PATH)

  
# --------------------
# SECTION 10 — Submission Graph+ avec all-mpnet-base-v2
# --------------------  

    print_section("SECTION 10: Submission Graph+ avec all-mpnet-base-v2")
    
    # --- Génère submission_graph_all-mpnet-base-v2.csv ---
    output_mpnet = os.path.join(BASE_PATH, "submission_graph_all-mpnet-base-v2.csv")
    generate_submission_graph_with_model(
        corpus=corpus,
        queries=queries,
        G=G,
        qrels_valid=qrels_valid,     # (pas utilisé pour la submission, mais gardé si tu veux logger plus tard)
        model_name="all-mpnet-base-v2",
        base_path=BASE_PATH,
        sample_path=SAMPLE_PATH,
        output_file=output_mpnet,
        alpha=0.70,
        beta=0.30
    )


    print_section("FIN DU SCRIPT")
    print_ok("Exécution terminée.")
        
if __name__ == "__main__":
    main()

