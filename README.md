# Lab Report Extractor — mera.health Submission

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install google-genai pdfplumber opencv-python pillow pdf2image

export GEMINI_API_KEY=your_key_here
```

Get a free Gemini API key at https://aistudio.google.com (no credit card required).

## Usage

```bash
python extractor.py report.pdf
python extractor.py scan.jpg
python extractor.py report1.pdf report2.jpg   # batch
```

Output:

```json
{
  "file": "report.pdf",
  "total_tests": 20,
  "results": [
    {
      "test_name": "Hemoglobin",
      "value": 13.5,
      "unit": "g/dL",
      "reference_range": "13.0-17.0",
      "confidence": "high",
      "flags": []
    },
    {
      "test_name": "Hemoglobin",
      "value": 145,
      "unit": "g/dL",
      "reference_range": "13.0-17.0",
      "confidence": "low",
      "flags": ["out_of_bounds(5.0-25.0)"]
    }
  ]
}
```

---

## 1. Problem Decomposition

The pipeline has four stages, each with a single responsibility:

**Stage 0 — Image Preprocessing (OpenCV, no API call)**
Only runs for image inputs (JPG, PNG, scanned PDFs). Applies deskewing via minAreaRect, denoising via fastNlMeansDenoising, and CLAHE contrast enhancement. A rotated or low-contrast scan meaningfully degrades vision model accuracy. This is 20 lines of OpenCV that costs nothing and runs in under a second.

**Stage 1 — Vision Read (Gemini 2.5 Flash, 1 API call per image page)**
Only runs for image/scanned inputs. The prompt asks the model to read what is on the page in a flat line format (`TEST | VALUE | UNIT | REF`), not to produce JSON. Mixing "read the image" and "structure the data" in one prompt makes both tasks worse — the model starts guessing structure around uncertain readings.

**Stage 1 is skipped entirely for digital PDFs.** `pdfplumber` checks at the start whether the PDF has extractable text. If yes, text is pulled directly — no vision call, no image conversion, no cost. The sterling_accuris 19-page PDF goes from 38 API calls to 1.

**Stage 2 — Structured Extraction (Gemini 2.5 Flash, 1 API call)**
Takes Stage 1's flat text (or pdfplumber's extracted text) and normalizes it into `{test_name, value, unit, reference_range}` JSON. This is a pure text task — faster, cheaper, and easier to debug than a combined vision+extraction call. For digital multi-page PDFs, all pages are concatenated and sent in a single call.

**Stage 3 — Validation (rule-based Python, no API call)**
A dictionary of 34 physiological bounds covers common tests. Every result is checked: is the value numeric? Is the value within the known physiological range? Is the unit recognized? Results get a `confidence` field (`high`/`low`) and a `flags` list. No LLM involved — a Python dict lookup is free, deterministic, and fast.

---

## 2. Trust Calibration

Every extracted result carries `confidence` and `flags`.

Flags are raised for:
- `missing_value` — value could not be parsed as a number
- `out_of_bounds(lo-hi)` — value falls outside the physiological range for that test
- `unrecognized_unit(x)` — unit string is not in the known vocabulary

Example — the assignment's own test case:
```json
{ "test_name": "Hemoglobin", "value": 145, "unit": "g/dL",
  "confidence": "low", "flags": ["out_of_bounds(5.0-25.0)"] }
```

The bounds cover 34 common tests (CBC, LFT, KFT, thyroid, lipids). For tests not in the bounds dict, no bounds check is applied — the flag is not raised, because a false flag is worse than no flag for uncommon tests.

I trust Stage 2 output more than Stage 1 output. Stage 1 is doing OCR and reading simultaneously and is the more likely point of digit misreads. The two-stage split means if Stage 2 fails to parse JSON, Stage 2 can be retried on the same Stage 1 text without another expensive vision call.

---

## 3. Cost Awareness

**Per report (image/scanned path):**
| Call | Input tokens | Output tokens |
|---|---|---|
| Stage 1 (vision) | ~1,500 | ~300 |
| Stage 2 (text) | ~600 | ~400 |
| **Total** | **~2,100** | **~700** |

Gemini 2.5 Flash pricing: $0.075/MTok input, $0.30/MTok output
→ ~$0.0002 per image page (~0.02 cents)
→ 3-page report: **~$0.0006 per report**
→ 10,000 reports/day: **~$6/day**

**Per report (digital PDF path — the optimized path):**
The entire document (all pages combined) goes through one Stage 2 call. Stage 1 is skipped.
→ ~$0.0001 per document regardless of page count
→ 10,000 reports/day: **~$1/day**

This is workable at scale. If cost became a concern, Stage 2 could be routed to a smaller/cheaper model since it receives clean text with no ambiguity.

**Free tier note:** Gemini 2.5 Flash free tier is capped at 20 requests/day. This is fine for testing and development, not for production (needs a paid key — cost math above applies).

---

## 4. What I Chose NOT To Do

**No separate OCR step (Tesseract, AWS Textract):** Gemini vision handles OCR internally. Adding Tesseract means an extra system dependency, extra latency, and the output still needs an LLM for extraction. Net result: more complexity, no clear accuracy gain for a prototype. Textract would be the right production call — it understands table layout and columns, which lab reports heavily use — but it's out of scope here.

**No ICR for handwritten text:** ICR (Intelligent Character Recognition) is designed for fully handwritten documents. In Indian lab reports, handwriting is almost always limited to annotations — the test names, values, and units are printed. Fully handwritten lab reports exist (old/rural labs) but are the minority. Flagged as a known limitation; in production, these would route to a human review queue.

**No retry/fallback logic:** Out of scope for 4–6 hours. Production would add exponential backoff on 429s and a dead-letter queue for failed extractions.

**No API server or database:** A well-structured script is the right answer for a prototype. FastAPI + Postgres solves problems we don't have yet.

**No unit conversion layer:** Indian labs mix mg/dL and mmol/L for the same test. Normalizing across conventions is a week of work in production and not what the prototype needs to demonstrate.

**No multi-language handling beyond Gemini's native capability:** Gemini 2.5 Flash reads Hindi/Tamil headers well without explicit translation. Flagged as a production concern for accuracy on less common regional languages.

---

## 5. Failure Modes

**Worst case 1 — Handwritten/scanned prescriptions (not lab reports)**
The scanned_report.jpg sample was a handwritten doctor's prescription, not a structured lab report. The pipeline correctly returned a single ESR value it found mentioned in the text and skipped everything else as non-lab data. This is the right behavior — but it reveals that the `NO_LAB_DATA` detection relies on the model making a correct judgment about what constitutes lab test data. A prescription with "check CBC" written on it might cause the model to hallucinate CBC values. With more time: add a document classifier as a pre-step to reject non-lab-report inputs before they hit Stage 1.

**Worst case 2 — Multi-page PDFs where tests span pages**
The current pipeline processes each image page independently. If a test name appears on the bottom of page N and its value appears on the top of page N+1 (rare but possible in some lab formats), the extraction misses it silently — no error, just a missing test. For digital PDFs this is not an issue since all text is combined into one call. For scanned multi-page PDFs it is a real gap. With more time: include the last 3 extracted lines of the previous page as context at the top of the next page's prompt.

---

## Prompts and Rationale

**Stage 1 prompt** uses a flat `TEST | VALUE | UNIT | REF` line format, not JSON. Reason: forcing JSON output in a vision task causes the model to hallucinate structure around uncertain readings — it fills in missing fields rather than leaving them blank. The flat format keeps the model in "transcription mode." Stage 2 handles structuring on clean text where it can be deliberate.

The prompt includes three few-shot examples (Hemoglobin, Glucose, TSH). Zero-shot works but few-shot consistently improves format compliance — the model knows exactly what a correct output line looks like.

**Stage 2 prompt** includes explicit normalization rules for unit variants (`g/dl` → `g/dL`, `thou/cumm` → `10^3/µL`) and instructs the model to skip rows where value is not a number. Handling normalization in the prompt is cheaper and simpler than a post-processing normalization layer. The `Output ONLY the JSON array` instruction prevents markdown wrapping, which is the most common reason for JSON parse failures in LLM outputs (we also strip fences defensively).

---

## What I'd Do Differently in Production

| Component | Prototype | Production |
|---|---|---|
| Digital PDFs | pdfplumber + 1 LLM call | Same — already optimal |
| Scanned PDFs | Gemini vision | AWS Textract (layout + table-aware) |
| Handwritten | Gemini vision (weak) | ICR service + human review queue |
| Extraction LLM | Gemini 2.5 Flash | Gemini Flash or equivalent — route cheapest model that handles the task |
| Validation | Rule-based Python (34 tests) | Same + unit conversion layer + per-lab reference range storage |
| Multi-page scanned | Independent page processing | Previous page context passed forward |
| Scale | Script | Job queue (SQS/Redis) + worker pool, no Kubernetes |
| Trust | confidence + flags | Same + confidence score (0.0–1.0) + human review queue for confidence < 0.7 |
| Audit | None | Every extraction stored with source page, model version, confidence — required for medical data compliance |
