"""
indexing.py
Builds, saves, and loads two FAISS indexes (text, image) plus a shared
metadata store so vector positions can be mapped back to source/page/content.
"""

import os
import pickle
import numpy as np
import faiss

TEXT_DIM = 3072   # Gemini text-embedding-001
IMAGE_DIM = 512  # CLIP ViT-B/32


def build_index(vectors: np.ndarray, dim: int) -> faiss.Index:
    """Build a flat inner-product FAISS index (vectors must already be L2-normalized)."""
    index = faiss.IndexFlatIP(dim)
    if len(vectors) > 0:
        index.add(vectors)
    return index


def save_indexes(text_index, image_index, text_metadata, image_metadata, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    faiss.write_index(text_index, os.path.join(out_dir, "text_index.faiss"))
    faiss.write_index(image_index, os.path.join(out_dir, "image_index.faiss"))
    with open(os.path.join(out_dir, "metadata.pkl"), "wb") as f:
        pickle.dump({"text_metadata": text_metadata, "image_metadata": image_metadata}, f)
    print(f"Saved indexes and metadata to {out_dir}")


def load_indexes(out_dir: str):
    text_index = faiss.read_index(os.path.join(out_dir, "text_index.faiss"))
    image_index = faiss.read_index(os.path.join(out_dir, "image_index.faiss"))
    with open(os.path.join(out_dir, "metadata.pkl"), "rb") as f:
        meta = pickle.load(f)
    return text_index, image_index, meta["text_metadata"], meta["image_metadata"]


def build_and_save_all(text_chunks, text_vectors, images, image_vectors, out_dir: str):
    """Convenience wrapper: build both indexes and save everything in one call."""
    text_index = build_index(text_vectors, TEXT_DIM)
    image_index = build_index(image_vectors, IMAGE_DIM)
    save_indexes(text_index, image_index, text_chunks, images, out_dir)
    return text_index, image_index
