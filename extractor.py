"""
Lab Report Extractor
Pipeline: file → preprocess → Stage1 (vision read) → Stage2 (structured extract) → validate
"""

import os
import sys
import json
import re
import pathlib
import tempfile

import time
import cv2
import numpy as np
import pdfplumber
from PIL import Image
from pdf2image import convert_from_path
from google import genai
from google.genai import types

# ── Gemini client ──────────────────────────────────────────────────────────────
API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = "gemini-2.5-flash"
_client = None

def get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=API_KEY)
    return _client

# ── Physiological bounds (test_name_lower → (min, max)) ───────────────────────
BOUNDS = {
    "hemoglobin":          (5.0,   25.0),
    "hb":                  (5.0,   25.0),
    "rbc":                 (1.0,   8.0),
    "wbc":                 (1000,  50000),
    "platelet":            (10,    900000),  # bounds cover both /µL and 10^3/µL
    "platelets":           (10,    900000),
    "glucose":             (20,    700),
    "fasting glucose":     (20,    700),
    "creatinine":          (0.2,   20.0),
    "urea":                (5,     300),
    "sodium":              (100,   180),
    "potassium":           (2.0,   9.0),
    "chloride":            (70,    130),
    "bilirubin":           (0.1,   30.0),
    "alt":                 (1,     5000),
    "ast":                 (1,     5000),
    "alkaline phosphatase":(10,    2000),
    "albumin":             (1.0,   6.0),
    "tsh":                 (0.001, 100.0),
    "t3":                  (0.5,   10.0),
    "t4":                  (1.0,   25.0),
    "hba1c":               (3.0,   20.0),
    "cholesterol":         (50,    500),
    "triglycerides":       (20,    2000),
    "hdl":                 (5,     150),
    "ldl":                 (10,    300),
    "mcv":                 (50,    130),
    "mch":                 (10,    60),
    "mchc":                (20,    45),
    "neutrophils":         (0,     100),
    "lymphocytes":         (0,     100),
    "eosinophils":         (0,     100),
    "monocytes":           (0,     100),
    "basophils":           (0,     100),
}

KNOWN_UNITS = {
    "g/dl", "g/l", "mg/dl", "mg/l", "mmol/l", "umol/l", "u/l", "iu/l",
    "meq/l", "meq/l", "%", "cells/cumm", "cells/µl", "10^3/µl", "10^6/µl",
    "fl", "pg", "µiu/ml", "uiu/ml", "ng/dl", "ng/ml", "µg/dl", "miu/ml",
    "mm/hr", "sec", "ratio", "units", "thousands/µl", "millions/µl",
    "lakh/cumm", "/cumm", "mil/cumm", "thou/cumm", "/µl", "/µl", "/ul",
}


# ── Stage 0: image preprocessing ──────────────────────────────────────────────
def preprocess_image(img_path: str) -> str:
    """Deskew, denoise, enhance contrast. Returns path to cleaned temp image."""
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # denoise
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # deskew via minAreaRect on thresholded blobs
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) > 100:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) > 0.5:
            (h, w) = gray.shape
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)

    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    cv2.imwrite(tmp.name, gray)
    return tmp.name


# ── File → pages: each page is either {"text": str} or {"image": path} ────────
def file_to_pages(filepath: str) -> list[dict]:
    ext = pathlib.Path(filepath).suffix.lower()
    if ext == ".pdf":
        with pdfplumber.open(filepath) as pdf:
            pages = pdf.pages[:10]  # cap at 10 for free tier
            all_text = "".join(p.extract_text() or "" for p in pages)
            if len(all_text.strip()) > 100:
                # digital PDF — extract text per page, skip vision entirely
                print("  Detected digital PDF — using text extraction (no vision call)")
                result = []
                for p in pages:
                    t = p.extract_text() or ""
                    if t.strip():
                        result.append({"text": t})
                return result
        # scanned PDF — convert pages to images
        print("  Detected scanned PDF — using vision")
        img_pages = convert_from_path(filepath, dpi=200)
        result = []
        for i, page in enumerate(img_pages[:10]):
            tmp = tempfile.NamedTemporaryFile(suffix=f"_p{i}.png", delete=False)
            page.save(tmp.name, "PNG")
            result.append({"image": tmp.name})
        return result
    else:
        # image file — preprocess then vision
        return [{"image": preprocess_image(filepath)}]


# ── Gemini call helper ─────────────────────────────────────────────────────────
def gemini(prompt: str, image_path: str = None) -> str:
    parts = []
    if image_path:
        with open(image_path, "rb") as f:
            data = f.read()
        mime = "image/png" if image_path.endswith(".png") else "image/jpeg"
        parts.append(types.Part.from_bytes(data=data, mime_type=mime))
    parts.append(prompt)
    # free tier: 5 RPM — small buffer between calls
    time.sleep(2)
    response = get_client().models.generate_content(model=MODEL, contents=parts)
    return response.text.strip()


# ── Stage 1: Vision read ───────────────────────────────────────────────────────
STAGE1_PROMPT = """You are reading a medical lab report image.
Extract ALL lab test data visible on this page. For every test you see, output one line:
TEST: <test name> | VALUE: <numeric value> | UNIT: <unit> | REF: <reference range>

Rules:
- Include every test, even if unit or reference range is missing (leave blank).
- Copy values exactly as printed. Do not interpret or convert.
- If multiple values appear (e.g. male/female ranges), include both in REF.
- Ignore patient name, age, doctor name, lab name, dates.
- If this page has no lab test data, output: NO_LAB_DATA

Examples:
TEST: Hemoglobin | VALUE: 13.5 | UNIT: g/dL | REF: 13.0-17.0
TEST: Glucose Fasting | VALUE: 92 | UNIT: mg/dL | REF: 70-100
TEST: TSH | VALUE: 2.45 | UNIT: µIU/mL | REF: 0.4-4.0
"""

def stage1_read(image_path: str) -> str:
    return gemini(STAGE1_PROMPT, image_path)


# ── Stage 2: Structured extraction ────────────────────────────────────────────
STAGE2_PROMPT = """You are given raw text extracted from a lab report. Convert it into structured JSON.

Output a JSON array where each element is:
{{"test_name": "...", "value": <number>, "unit": "...", "reference_range": "..."}}

Rules:
- test_name: use standard medical name (e.g. "Hemoglobin", "Fasting Glucose", "TSH")
- value: must be a number (float or int). If not parseable as number, skip that row.
- unit: normalize common variants (g/dl and g/dL → g/dL, thou/cumm → 10^3/µL)
- reference_range: string as-is from source (e.g. "13.0-17.0" or "70-100 mg/dL")
- If a field is missing, use null.
- Output ONLY the JSON array, no explanation.

Raw text:
{raw}
"""

def stage2_extract(raw_text: str) -> list[dict]:
    prompt = STAGE2_PROMPT.format(raw=raw_text)
    response = gemini(prompt)
    # strip markdown fences if present
    response = re.sub(r"```(?:json)?", "", response).strip().rstrip("`").strip()
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # try extracting first [...] block
        m = re.search(r"\[.*\]", response, re.DOTALL)
        if m:
            return json.loads(m.group())
        return []


# ── Stage 3: Validation ────────────────────────────────────────────────────────
def validate(results: list[dict]) -> list[dict]:
    for r in results:
        flags = []
        val = r.get("value")
        name = (r.get("test_name") or "").lower().strip()
        unit = (r.get("unit") or "").lower().strip()

        if val is None:
            flags.append("missing_value")
        else:
            # physiological bounds check
            for key, (lo, hi) in BOUNDS.items():
                if key in name:
                    if not (lo <= float(val) <= hi):
                        flags.append(f"out_of_bounds({lo}-{hi})")
                    break

        # unit check
        if unit and unit not in KNOWN_UNITS:
            flags.append(f"unrecognized_unit({unit})")

        r["confidence"] = "low" if flags else "high"
        r["flags"] = flags
    return results


# ── Main pipeline ──────────────────────────────────────────────────────────────
def extract(filepath: str) -> list[dict]:
    print(f"\n→ Processing: {filepath}")
    pages = file_to_pages(filepath)
    print(f"  Pages: {len(pages)}")

    # For digital PDFs: combine all page text into one Stage 2 call (saves quota)
    text_pages = [p for p in pages if "text" in p]
    image_pages = [p for p in pages if "image" in p]

    all_results = []

    if text_pages:
        combined = "\n\n--- PAGE BREAK ---\n\n".join(p["text"] for p in text_pages)
        print(f"  [Stage 2] Extracting all {len(text_pages)} text pages in one call...")
        results = stage2_extract(combined)
        all_results.extend(validate(results))

    for i, page in enumerate(image_pages):
        print(f"  [Stage 1] Reading image page {i+1} (vision)...")
        raw = stage1_read(page["image"])
        if "NO_LAB_DATA" in raw:
            print(f"  No lab data on image page {i+1}, skipping.")
            continue
        print(f"  [Stage 2] Extracting image page {i+1}...")
        results = stage2_extract(raw)
        all_results.extend(validate(results))

    return all_results


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not API_KEY:
        print("Error: set GEMINI_API_KEY environment variable")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Usage: python extractor.py <report_file> [report_file ...]")
        sys.exit(1)

    for filepath in sys.argv[1:]:
        results = extract(filepath)
        output = {
            "file": filepath,
            "total_tests": len(results),
            "results": results
        }
        print(json.dumps(output, indent=2))
