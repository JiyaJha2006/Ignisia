import json
import re
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from text_normalizer import build_embedding_text, clean_ocr_artifacts


QUESTION_ONE_MARKERS = r"(?:Q\.?\s*1|Que\s*1|Ans?:?\s*1|Answer\s*1|प्रश्न\s*1|उत्तर\s*1)"
QUESTION_TWO_MARKERS = r"(?:Q\.?\s*2|Que\s*2|Ans?:?\s*2|Answer\s*2|प्रश्न\s*2|उत्तर\s*2|9\.?2\)?|0?2\]|92\])"


def extract_q1(text):
    match = re.search(
        rf"{QUESTION_ONE_MARKERS}.*?(?={QUESTION_TWO_MARKERS}|Any: 2|2>|2\)|$)",
        str(text),
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(0).strip()

    fallback_text = re.sub(
        rf"(?:{QUESTION_TWO_MARKERS}|Any: 2|2>|2\)).*",
        "",
        str(text),
        flags=re.IGNORECASE | re.DOTALL,
    )
    return fallback_text.strip()


def extract_q2(text):
    match = re.search(
        rf"(?:{QUESTION_TWO_MARKERS}|Any: ?2|2>|2\)).*",
        str(text),
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        cleaned_text = re.sub(
            r"(Teacher's Signature|AMAR KRISH)",
            "",
            match.group(0),
            flags=re.IGNORECASE,
        )
        return cleaned_text.strip()
    return "No Answer"


def _safe_normalize_series(series):
    return series.fillna("").apply(clean_ocr_artifacts).apply(build_embedding_text)


def _normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def _reassign_outliers(cluster_ids, embeddings, similarity_threshold=0.58):
    cluster_ids = np.array(cluster_ids, dtype=int)
    normalized_embeddings = _normalize_embeddings(np.asarray(embeddings))
    unique_clusters = [cluster_id for cluster_id in sorted(set(cluster_ids.tolist())) if cluster_id != -1]
    if not unique_clusters:
        return cluster_ids

    centroids = {}
    for cluster_id in unique_clusters:
        cluster_vectors = normalized_embeddings[cluster_ids == cluster_id]
        if len(cluster_vectors) == 0:
            continue
        centroid = cluster_vectors.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm == 0:
            continue
        centroids[cluster_id] = centroid / centroid_norm

    if not centroids:
        return cluster_ids

    for index, cluster_id in enumerate(cluster_ids):
        if cluster_id != -1:
            continue
        similarities = {
            candidate_id: float(np.dot(normalized_embeddings[index], centroid))
            for candidate_id, centroid in centroids.items()
        }
        if not similarities:
            continue
        best_cluster, best_similarity = max(similarities.items(), key=lambda item: item[1])
        if best_similarity >= similarity_threshold:
            cluster_ids[index] = best_cluster

    return cluster_ids


def cluster_answers(
    results_json_path,
    output_csv_path,
    output_json_path,
    include_flagged=True,
):
    results_json_path = Path(results_json_path)
    output_csv_path = Path(output_csv_path)
    output_json_path = Path(output_json_path)

    print("--- Loading JSON OCR Data ---")
    with open(results_json_path, "r", encoding="utf-8") as file:
        ocr_data = json.load(file)

    df = pd.DataFrame(ocr_data)
    if df.empty:
        raise ValueError("OCR results are empty. Cannot cluster answers.")

    clean_df = df.copy() if include_flagged else df[df["flagged"] == False].copy()

    clean_df["Q1_Answer"] = clean_df["full_text"].apply(extract_q1)
    clean_df["Q2_Answer"] = clean_df["full_text"].apply(extract_q2)
    clean_df["Q1_Embedding_Text"] = _safe_normalize_series(clean_df["Q1_Answer"])
    clean_df["Q2_Embedding_Text"] = _safe_normalize_series(clean_df["Q2_Answer"])

    print("--- Loading the SentenceTransformer Model ---")
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2,
        min_samples=1,
        metric="euclidean",
        cluster_selection_epsilon=0.05,
    )

    print("\n--- PASS 1: Clustering Question 1 ---")
    q1_embeddings = model.encode(clean_df["Q1_Embedding_Text"].tolist())
    clean_df["Q1_Cluster_ID"] = _reassign_outliers(clusterer.fit_predict(q1_embeddings), q1_embeddings)

    print("--- PASS 2: Clustering Question 2 ---")
    q2_embeddings = model.encode(clean_df["Q2_Embedding_Text"].tolist())
    clean_df["Q2_Cluster_ID"] = _reassign_outliers(clusterer.fit_predict(q2_embeddings), q2_embeddings)

    print("\n=== FINAL CLUSTERING RESULTS ===")
    print(clean_df[["student_id", "Q1_Cluster_ID", "Q2_Cluster_ID"]])

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    export_df = clean_df.drop(columns=["full_text"], errors="ignore")
    export_df.to_csv(output_csv_path, index=False)
    export_df.to_json(output_json_path, orient="records", indent=2)

    print(f"Saved clustered CSV: {output_csv_path}")
    print(f"Saved clustered JSON: {output_json_path}")

    return export_df


if __name__ == "__main__":
    backend_dir = Path(__file__).resolve().parent
    project_root = backend_dir.parent
    cluster_answers(
        results_json_path=project_root / "ocr_output" / "results.json",
        output_csv_path=backend_dir / "final_clustered_grades.csv",
        output_json_path=backend_dir / "final_clustered_grades.json",
    )
