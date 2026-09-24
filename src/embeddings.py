"""
embeddings.py
Text embeddings via Gemini text-embedding-004, image embeddings via CLIP.
Two different embedding spaces are kept deliberately separate (see retrieval.py
for how queries get embedded into both).
"""

import os
import time
import numpy as np
import google.generativeai as genai
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

# --- setup ---
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

_clip_model = None
_clip_processor = None


def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        print("Loading CLIP model (first call only)...")
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _clip_model, _clip_processor


def embed_text(text: str, task_type: str = "retrieval_document") -> np.ndarray:
    """
    Embed a text string with Gemini text-embedding-004.
    task_type: "retrieval_document" for corpus chunks, "retrieval_query" for user queries.
    """
    result = genai.embed_content(
    model="models/gemini-embedding-001",   # was: models/text-embedding-004
    content=text,
    task_type=task_type
)
    return np.array(result["embedding"], dtype="float32")


def embed_image(image_path: str) -> np.ndarray:
    """Embed an image file with CLIP's image encoder."""
    model, processor = _load_clip()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    emb = emb.pooler_output.cpu().squeeze().numpy().astype("float32")
    return emb / np.linalg.norm(emb)  # normalize for cosine similarity via inner product


def embed_text_for_image_search(query: str) -> np.ndarray:
    """
    Embed a text query with CLIP's text encoder, so it lands in the SAME space
    as embed_image() output. This is what makes text -> image retrieval work.
    """
    model, processor = _load_clip()
    inputs = processor(text=[query], return_tensors="pt", padding=True)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
    emb = emb.squeeze().numpy().astype("float32")
    return emb / np.linalg.norm(emb)


def embed_all_text_chunks(text_chunks: list[dict]) -> np.ndarray:
    """Embed a list of text chunk dicts (from ingestion.py), return an (N, dim) array."""
    vectors = []
    for i, chunk in enumerate(text_chunks):
        vec = embed_text(chunk["text"], task_type="retrieval_document")
        vectors.append(vec / np.linalg.norm(vec))
        if (i + 1) % 10 == 0:
            print(f"  embedded {i + 1}/{len(text_chunks)} text chunks")
        time.sleep(4)  
    return np.array(vectors, dtype="float32")


def embed_all_images(images: list[dict]) -> np.ndarray:
    """Embed a list of image dicts (from ingestion.py), return an (N, dim) array."""
    vectors = []
    for i, img in enumerate(images):
        vec = embed_image(img["path"])
        vectors.append(vec)
        if (i + 1) % 10 == 0:
            print(f"  embedded {i + 1}/{len(images)} images")
    return np.array(vectors, dtype="float32")
