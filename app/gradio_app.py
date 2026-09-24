"""
gradio_app.py
Simple demo UI: type a question, see retrieved chunks/images and the final
generated answer. Run with: python app/gradio_app.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from src.indexing import load_indexes
from src.retrieval import retrieve
from src.generation import generate_answer

INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "indexes")

print("Loading indexes...")
text_index, image_index, text_metadata, image_metadata = load_indexes(INDEX_DIR)
print(f"Loaded {text_index.ntotal} text vectors and {image_index.ntotal} image vectors.")


def rag_pipeline(query: str):
    if not query.strip():
        return "Please enter a question.", []

    retrieved_texts, retrieved_images, _ = retrieve(
        query, text_index, image_index, text_metadata, image_metadata, k=3
    )
    answer = generate_answer(query, retrieved_texts, retrieved_images)
    image_paths = [img["path"] for img in retrieved_images]
    return answer, image_paths


demo = gr.Interface(
    fn=rag_pipeline,
    inputs=gr.Textbox(label="Ask a question about your technical documents",
                       placeholder="e.g. How do I reset the switch to factory defaults?"),
    outputs=[
        gr.Textbox(label="Answer", lines=8),
        gr.Gallery(label="Retrieved images", columns=3)
    ],
    title="Multimodal RAG — Enterprise Technical Knowledge Management",
    description="Retrieves relevant text and images from the indexed document corpus, "
                 "then generates a cited answer using Gemini."
)

if __name__ == "__main__":
    demo.launch(share=True)
