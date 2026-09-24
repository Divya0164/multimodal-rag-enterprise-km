# Multimodal RAG for Enterprise Technical Knowledge Management
### Final Report

**Author:**
**HCL Cohort / Dates:**
**Mentor:**

---

## 1. Problem Statement
<!-- What gap does this fill? Why do text-only RAG systems fall short for enterprise technical docs? -->

## 2. Objectives
<!-- 3-4 bullet points, concrete and measurable where possible -->

## 3. Corpus
<!-- What documents did you use? How many? What types (manuals, SOPs, architecture docs)? Any preprocessing challenges? -->

## 4. System Architecture
<!-- Paste docs/architecture_diagram.png here and walk through each stage -->

### 4.1 Ingestion
### 4.2 Embedding
### 4.3 Indexing
### 4.4 Retrieval
### 4.5 Generation

## 5. Design Decisions & Trade-offs
<!-- Important one: why explicit chunking + FAISS retrieval instead of just dumping
     full PDFs into Gemini's long context window? What does the explicit retrieval
     step buy you (explainability, cost control, precision) vs. the simpler approach? -->

## 6. Evaluation
### 6.1 Retrieval metrics (Recall@k)
<!-- Insert results/retrieval_metrics.csv summary + a chart -->

### 6.2 Generation faithfulness
<!-- Summary of manual spot-checks: how often were citations correct, how often
     did it correctly say "insufficient context" instead of hallucinating -->

## 7. Demo
<!-- Screenshots from results/sample_outputs/, link to Gradio share link if still live -->

## 8. Limitations
<!-- Corpus size, CLIP's limits on diagram understanding, cost/rate limits, etc. -->

## 9. Future Work
<!-- e.g. re-ranking, OCR on diagram text, larger corpus, fine-tuned embeddings -->

## 10. Conclusion

## Appendix
- A. Full eval Q&A set (`data/eval/qa_pairs.json`)
- B. Weekly progress log (`docs/weekly_progress.md`)
- C. Setup instructions (`README.md`)
