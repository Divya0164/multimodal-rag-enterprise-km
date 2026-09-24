"""
ingestion.py
Parses PDFs into separate text chunks and extracted images, with metadata
linking each piece back to its source document and page.
"""

import os
import json
import fitz  # PyMuPDF


def parse_pdf(pdf_path: str, image_out_dir: str) -> tuple[list[dict], list[dict]]:
    """
    Parse a single PDF into text chunks (per page) and extracted images.

    Returns:
        text_chunks: list of {"text": str, "page": int, "source": str, "chunk_id": str}
        images: list of {"path": str, "page": int, "source": str, "image_id": str}
    """
    os.makedirs(image_out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    source_name = os.path.basename(pdf_path)

    text_chunks = []
    images = []

    for page_num, page in enumerate(doc):
        # --- text ---
        text = page.get_text().strip()
        if text:
            text_chunks.append({
                "text": text,
                "page": page_num,
                "source": source_name,
                "chunk_id": f"{source_name}_p{page_num}_text"
            })

        # --- images ---
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]
            img_filename = f"{source_name}_p{page_num}_img{img_index}.{ext}"
            img_path = os.path.join(image_out_dir, img_filename)
            with open(img_path, "wb") as f:
                f.write(image_bytes)
            images.append({
                "path": img_path,
                "page": page_num,
                "source": source_name,
                "image_id": f"{source_name}_p{page_num}_img{img_index}"
            })

    doc.close()
    return text_chunks, images


def chunk_text(text_chunks: list[dict], max_chars: int = 1500) -> list[dict]:
    """
    Split long page-level text chunks further if they exceed max_chars.
    Keeps splits on paragraph boundaries where possible so tables/sections
    don't get cut mid-content.
    """
    refined = []
    for chunk in text_chunks:
        text = chunk["text"]
        if len(text) <= max_chars:
            refined.append(chunk)
            continue

        paragraphs = text.split("\n\n")
        buffer = ""
        part = 0
        for para in paragraphs:
            if len(buffer) + len(para) > max_chars and buffer:
                refined.append({
                    **chunk,
                    "text": buffer.strip(),
                    "chunk_id": f"{chunk['chunk_id']}_part{part}"
                })
                part += 1
                buffer = para
            else:
                buffer += "\n\n" + para
        if buffer.strip():
            refined.append({
                **chunk,
                "text": buffer.strip(),
                "chunk_id": f"{chunk['chunk_id']}_part{part}"
            })
    return refined


def parse_corpus(raw_dir: str, processed_dir: str) -> tuple[list[dict], list[dict]]:
    """
    Parse every PDF in raw_dir, save results to processed_dir as JSON,
    and return the combined text_chunks and images lists.
    """
    image_out_dir = os.path.join(processed_dir, "images")
    text_out_path = os.path.join(processed_dir, "text_chunks", "chunks.json")

    all_text_chunks = []
    all_images = []

    pdf_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDFs found in {raw_dir}. Add source documents and re-run.")
        return [], []

    for fname in pdf_files:
        pdf_path = os.path.join(raw_dir, fname)
        print(f"Parsing {fname}...")
        text_chunks, images = parse_pdf(pdf_path, image_out_dir)
        text_chunks = chunk_text(text_chunks)
        all_text_chunks.extend(text_chunks)
        all_images.extend(images)

    os.makedirs(os.path.dirname(text_out_path), exist_ok=True)
    with open(text_out_path, "w") as f:
        json.dump({"text_chunks": all_text_chunks, "images": all_images}, f, indent=2)

    print(f"Parsed {len(all_text_chunks)} text chunks and {len(all_images)} images "
          f"from {len(pdf_files)} document(s).")
    return all_text_chunks, all_images


if __name__ == "__main__":
    parse_corpus(raw_dir="../data/raw", processed_dir="../data/processed")
