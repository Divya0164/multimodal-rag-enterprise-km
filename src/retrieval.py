"""
retrieval.py
Given a query, embeds it into both the text and image embedding spaces
and retrieves top-k nearest neighbors from each FAISS index.
"""

import numpy as np
from src.embeddings import embed_text, embed_text_for_image_search


def retrieve(query: str, text_index, image_index, text_metadata, image_metadata, k: int = 3):
    """
    Returns:
        retrieved_texts: list of text_metadata dicts (top-k text matches)
        retrieved_images: list of image_metadata dicts (top-k image matches)
        scores: dict with "text_scores" and "image_scores" (cosine similarity)
    """
    retrieved_texts, text_scores = [], []
    if text_index.ntotal > 0:
        q_text_vec = embed_text(query, task_type="retrieval_query")
        q_text_vec = q_text_vec / np.linalg.norm(q_text_vec)
        sims, ids = text_index.search(np.array([q_text_vec], dtype="float32"), min(k, text_index.ntotal))
        for score, idx in zip(sims[0], ids[0]):
            if idx != -1:
                retrieved_texts.append(text_metadata[idx])
                text_scores.append(float(score))

    retrieved_images, image_scores = [], []
    if image_index.ntotal > 0:
        q_img_vec = embed_text_for_image_search(query)
        sims, ids = image_index.search(np.array([q_img_vec], dtype="float32"), min(k, image_index.ntotal))
        for score, idx in zip(sims[0], ids[0]):
            if idx != -1:
                retrieved_images.append(image_metadata[idx])
                image_scores.append(float(score))

    return retrieved_texts, retrieved_images, {"text_scores": text_scores, "image_scores": image_scores}
