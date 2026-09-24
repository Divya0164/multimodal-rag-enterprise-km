"""
generation.py
Feeds retrieved text chunks and images to Gemini for a grounded, cited answer.
"""

import os
import google.generativeai as genai

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

SYSTEM_INSTRUCTIONS = """You are an enterprise technical knowledge assistant.
Answer the user's question using ONLY the provided context (text excerpts and images below).
Cite the source document and page number for every claim, like: (source.pdf, p.3).
If the provided context does not contain enough information to answer, say so explicitly
instead of guessing or using outside knowledge."""


def generate_answer(query: str, retrieved_texts: list[dict], retrieved_images: list[dict],
                     model_name: str = "gemini-2.0-flash") -> str:
    """
    retrieved_texts: list of {"text", "source", "page", ...} from retrieval.py
    retrieved_images: list of {"path", "source", "page", ...} from retrieval.py
    """
    model = genai.GenerativeModel(model_name, system_instruction=SYSTEM_INSTRUCTIONS)

    content = [f"Question: {query}\n\nText context:\n"]
    for t in retrieved_texts:
        content.append(f"[{t['source']}, page {t['page']}]\n{t['text']}\n")

    if retrieved_images:
        content.append("\nImage context (see attached images, in order):\n")
        for img in retrieved_images:
            content.append(f"[{img['source']}, page {img['page']}]")
            uploaded = genai.upload_file(img["path"])
            content.append(uploaded)

    response = model.generate_content(content)
    return response.text
