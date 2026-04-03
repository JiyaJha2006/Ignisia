# ExamIQ — AI-Powered Exam Grading System

**Project by Ignisia**

---

## The Problem

Faculty members spend hundreds of hours manually grading subjective exam answers. A professor marking 300 papers reads the same correct explanation 60 times, the same partial-credit answer 40 times, and the same conceptual error 30 times — never realising the repetition because papers are graded sequentially in random order.

The cognitive fatigue this creates leads to grading inconsistency: the same answer receives different marks depending on whether the professor reads it at 9 AM or 11 PM.

**What is needed is not an AI that replaces the professor's judgment, but one that organises the work** — so the professor grades each distinct answer type once and applies that decision to all matching papers simultaneously.

---

## How It Works

The system processes scanned handwritten answer sheets through a five-stage pipeline:

### Phase 1 — OCR Ingestion
Scanned exam sheets (PNGs or PDF) are digitised using Tesseract OCR (free, offline) or Google Cloud Vision (higher accuracy for messy handwriting). Each student's two-page answer sheet is merged into a single text record. Low-confidence pages are flagged automatically for manual review.

### Phase 2 — Multilingual Embedding
Every answer is converted into a semantic vector using a multilingual sentence embedding model (LaBSE or `paraphrase-multilingual-MiniLM-L12-v2`). This maps answers from English and Hindi/Hinglish into the same vector space, so a correct Hindi answer and its correct English equivalent land near each other geometrically.

### Phase 3 — Semantic Clustering
Answers are grouped by meaning using DBSCAN (or K-Means). Students who wrote the same correct reasoning end up in the same cluster regardless of different phrasing or handwriting. A hybrid rule-based detector identifies edge cases — answers where the formula is correct but the arithmetic is wrong — and places them in a separate partial-credit cluster.

### Phase 4 — Rubric Highlighting + LLM Summaries
The professor uploads a rubric (keywords and expected concepts). A keyword matcher highlights rubric terms in every answer. An LLM (Claude Haiku) generates a one-sentence description of each cluster so the professor can understand a group of 20 answers without opening a single one.

### Phase 5 — Grader's Dashboard
A web dashboard presents all clusters to the professor. They read 3–5 representative answers per cluster, assign a mark, and click "Apply to all X answers in this cluster." One decision — grades 20+ students simultaneously.

---

## Deliverables

| Deliverable | Status |
|---|---|
| OCR ingestion pipeline (ZIP of PNGs / folder) | ✅ Complete |
| Semantic clustering engine | 🔄 In progress |
| Rubric keyword highlighting module | 📋 Planned |
| Edge-case cluster (formula correct, calc wrong) | 📋 Planned |
| Multilingual clustering (English + Hindi) | 📋 Planned |
| Cost and efficiency log | ✅ Built into OCR pipeline |
| Grader's Dashboard (React + FastAPI) | 📋 Planned |

---

## Tech Stack

| Component | Tool |
|---|---|
| OCR | Tesseract 5 (free/offline) → Google Cloud Vision (production) |
| Image preprocessing | OpenCV + Pillow |
| Sentence embeddings | LaBSE / paraphrase-multilingual-MiniLM |
| Clustering | DBSCAN / K-Means (scikit-learn) |
| Edge-case detection | Regex + SymPy arithmetic check |
| Rubric matching | spaCy PhraseMatcher + rapidfuzz |
| LLM summaries | Claude Haiku |
| Backend | FastAPI + SQLite |
| Frontend | React + Tailwind CSS |

---

## Project Structure

```
ignisiia/
│
├── ocr_pipeline.py          # Phase 1 — OCR ingestion
├── embed_cluster.py         # Phase 2+3 — Embedding + clustering (coming)
├── rubric_highlight.py      # Phase 4 — Rubric matching (coming)
├── dashboard/               # Phase 5 — React frontend (coming)
│   ├── src/
│   └── package.json
│
├── raw images/
│   └── full dataset/
│       └── archive-2/       # Your exam sheet PNGs (001_Sheet1.png ...)
│
└── ocr_output/              # Generated after running OCR pipeline
    ├── student_001.json
    ├── student_002.json
    ├── ...
    └── all_answers.json     # Master file — all students + cost log
```

---

## Running the OCR Pipeline

### Requirements

```bash
# Install system dependency
brew install tesseract tesseract-lang       # macOS
sudo apt install tesseract-ocr             # Ubuntu

# Install Python packages
pip install pytesseract pymupdf opencv-python-headless Pillow tqdm
```

### Run

```bash
# From your ignisiia folder:
python3 ocr_pipeline.py --folder "raw images/full dataset/archive-2"

# Or with a zip file:
python3 ocr_pipeline.py --zip exam_sheets.zip

# Inspect results after a run:
python3 ocr_pipeline.py --inspect

# If answers are in a single narrow column (try if confidence is low):
python3 ocr_pipeline.py --folder "raw images/full dataset/archive-2" --psm 4
```

### Output

Each student produces a JSON record:

```json
{
  "student_id": "001",
  "full_text": "Encapsulation is the process of wrapping data and methods...",
  "page_texts": {
    "1": "Encapsulation is the process of wrapping data...",
    "2": "For example in Python, private variables use double underscore..."
  },
  "avg_confidence": 74.2,
  "total_word_count": 58,
  "flagged": false,
  "processing_ms": 2100
}
```

**Flagged students** (confidence below 60) are listed at the end of the run and should be checked manually — their OCR text may be unreliable.

---

## Cost Estimates (per 100 students / 200 pages)

| Component | Tool | Cost |
|---|---|---|
| OCR | Tesseract | $0.00 |
| OCR | Google Cloud Vision | ~$1.50 |
| Embeddings | LaBSE (local) | $0.00 |
| Cluster summaries | Claude Haiku | ~$0.003 |
| **Total (Tesseract path)** | | **$0.003** |
| **Total (Google Vision path)** | | **~$1.503** |

---

## Expected Outcome

A professor with 100 papers and 5 distinct answer types should be able to complete grading in **under 30 minutes** — compared to 4–6 hours of sequential reading — with more consistent marks across the cohort.

---

## Team

Ignisia — Building AI tools for education.
