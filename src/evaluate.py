"""
evaluate.py
Simple retrieval evaluation: Recall@k against a hand-labeled Q&A set.

Expected format for data/eval/qa_pairs.json:
[
  {
    "question": "How do I reset the network switch to factory defaults?",
    "expected_source": "network_manual.pdf",
    "expected_page": 12
  },
  ...
]
Add "expected_modality": "image" for questions whose answer depends on a diagram,
otherwise it defaults to "text".
"""

import json
import pandas as pd
from src.retrieval import retrieve


def recall_at_k(qa_pairs: list[dict], text_index, image_index, text_metadata,
                 image_metadata, k: int = 3) -> pd.DataFrame:
    rows = []
    for qa in qa_pairs:
        query = qa["question"]
        expected_source = qa["expected_source"]
        expected_page = qa.get("expected_page")
        modality = qa.get("expected_modality", "text")

        retrieved_texts, retrieved_images, _ = retrieve(
            query, text_index, image_index, text_metadata, image_metadata, k=k
        )
        candidates = retrieved_texts if modality == "text" else retrieved_images

        hit = any(
            c["source"] == expected_source and
            (expected_page is None or c["page"] == expected_page)
            for c in candidates
        )
        rows.append({
            "question": query,
            "modality": modality,
            "expected_source": expected_source,
            "expected_page": expected_page,
            "hit_at_k": hit
        })

    df = pd.DataFrame(rows)
    return df


def summarize(df: pd.DataFrame, k: int) -> None:
    overall = df["hit_at_k"].mean()
    print(f"Overall Recall@{k}: {overall:.2%}")
    for modality, group in df.groupby("modality"):
        print(f"  {modality} Recall@{k}: {group['hit_at_k'].mean():.2%}  (n={len(group)})")


if __name__ == "__main__":
    from src.indexing import load_indexes

    with open("../data/eval/qa_pairs.json") as f:
        qa_pairs = json.load(f)

    text_index, image_index, text_metadata, image_metadata = load_indexes("../indexes")
    df = recall_at_k(qa_pairs, text_index, image_index, text_metadata, image_metadata, k=3)
    summarize(df, k=3)
    df.to_csv("../results/retrieval_metrics.csv", index=False)
