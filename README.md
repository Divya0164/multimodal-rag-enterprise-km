# Multimodal RAG for Enterprise Technical Knowledge Management

## Objective
Build a Retrieval-Augmented Generation (RAG) system that can answer natural-language
questions over enterprise technical documents (manuals, SOPs, architecture docs) — retrieving
and reasoning over **both text and images** (diagrams, screenshots, tables) rather than text alone.

## Why this matters
Enterprise technical knowledge rarely lives in plain text. Architecture diagrams, network
topology screenshots, and scanned tables carry information that text-only RAG systems ignore
or lose during parsing. This project demonstrates a retrieval pipeline that indexes and
retrieves both modalities, then feeds them jointly to a multimodal LLM for grounded, cited answers.

## Architecture (summary)
1. **Ingestion** — parse PDFs into text chunks and extracted images (`src/ingestion.py`)
2. **Embedding** — text embedded with Gemini `text-embedding-004`, images embedded with CLIP (`src/embeddings.py`)
3. **Indexing** — two separate FAISS indexes (text, image) with a shared metadata store (`src/indexing.py`)
4. **Retrieval** — query embedded into both spaces, top-k retrieved from each (`src/retrieval.py`)
5. **Generation** — retrieved text + images passed to Gemini 2.0 Flash for a grounded, cited answer (`src/generation.py`)
6. **Evaluation** — Recall@k on retrieval, manual faithfulness scoring on generation (`src/evaluate.py`)

See `docs/architecture_diagram.png` for the visual and `docs/final_report.md` for full write-up.

## Project structure
```
multimodal-rag-enterprise-km/
├── data/           # raw docs, processed chunks/images, eval Q&A set
├── notebooks/       # one notebook per pipeline stage (weeks 1-5)
├── src/              # reusable pipeline code, imported by notebooks and app
├── indexes/           # saved FAISS indexes + metadata (rebuildable)
├── app/                # Gradio demo
├── docs/                # architecture diagram, weekly progress log, final report
└── results/              # metrics + sample outputs for the report
```

## Setup
```bash
git clone <your-repo-url>
cd multimodal-rag-enterprise-km
pip install -r requirements.txt
cp .env.example .env        # then add your GEMINI_API_KEY
```
Get a free Gemini API key at https://ai.google.dev.

## Running the demo
```bash
python app/gradio_app.py
```
This launches a local (and public share) link where you can query the indexed corpus.

## Rebuilding the pipeline from scratch
1. Place source PDFs in `data/raw/`
2. Run `notebooks/01_ingestion.ipynb` — parses text + images into `data/processed/`
3. Run `notebooks/02_embeddings.ipynb` — builds and saves FAISS indexes into `indexes/`
4. Run `notebooks/03_retrieval_test.ipynb` and `04_generation_test.ipynb` to sanity-check
5. Run `notebooks/05_evaluation.ipynb` against `data/eval/qa_pairs.json` to produce `results/retrieval_metrics.csv`

## Status
See `docs/weekly_progress.md` for the current stage and mentor feedback log.

## Author
<your name> — HCL Internship Project, <cohort/dates>
