"""
FAST HYBRID OCR PIPELINE (PARALLEL + SMART OCR)
"""

import os
import cv2
import numpy as np
import pytesseract
import zipfile
import json
import argparse
import re
from pathlib import Path
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import easyocr

# --- CONFIG ---
PSM = 4
CONF_THRESH = 55
OUTPUT_DIR = Path("ocr_output")
MAX_WORKERS = 6  # adjust based on CPU

reader = None
reader_lock = threading.Lock()

# --- INIT EASYOCR SAFELY ---
def get_reader():
    global reader
    if reader is None:
        with reader_lock:
            if reader is None:
                print("🔄 Initializing EasyOCR (one-time)...")
                reader = easyocr.Reader(['en'], gpu=False)
    return reader

# --- PARSE FILENAMES ---
def parse_filenames(names):
    patterns = [
        r"^(\w+)[_\-]sheet(\d+)\.png",
        r".*?[_\-\s]?(\w+)[_\-\s]p(\d+)\.png",
        r".*?[_\-\s](\w+)[_\-\s]page(\d+)\.png",
        r"^(\w+)[_\-](\d)\.png",
        r".*?(\d+)pg(\d+)\.png",
        r"^(\d+)-(\d+)\.png",
    ]
    grouped = defaultdict(dict)

    for name in names:
        if not name.lower().endswith(".png"):
            continue
        basename = Path(name).name

        for pat in patterns:
            m = re.match(pat, basename, re.IGNORECASE)
            if m:
                grouped[m.group(1)][int(m.group(2))] = name
                break

    return dict(grouped)

# --- PREPROCESS ---
def preprocess(pil_img):
    img = np.array(pil_img.convert("L"))

    # upscale
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # fast threshold
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary

# --- TESSERACT ---
def tesseract_ocr(img, psm):
    pil = Image.fromarray(img)

    data = pytesseract.image_to_data(
        pil,
        config=f"--oem 3 --psm {psm}",
        output_type=pytesseract.Output.DICT
    )

    words, confs = [], []
    for w, c in zip(data["text"], data["conf"]):
        c = int(c)
        if w.strip() and c > 40:
            words.append(w)
            confs.append(c)

    return {
        "text": " ".join(words),
        "confidence": sum(confs)/len(confs) if confs else 0
    }

# --- EASYOCR ---
def easyocr_ocr(img):
    reader = get_reader()
    results = reader.readtext(img)

    words, confs = [], []
    for _, text, conf in results:
        words.append(text)
        confs.append(conf * 100)

    return {
        "text": " ".join(words),
        "confidence": sum(confs)/len(confs) if confs else 0
    }

# --- SMART HYBRID ---
def hybrid_ocr(img):
    t1 = tesseract_ocr(img, 4)
    t2 = tesseract_ocr(img, 6)

    best_t = t1 if t1["confidence"] > t2["confidence"] else t2

    # 🔥 ONLY fallback if really bad
    if best_t["confidence"] > 60:
        return best_t

    e = easyocr_ocr(img)
    return e if e["confidence"] > best_t["confidence"] else best_t

# --- PROCESS ONE STUDENT ---
def process_student(student_id, pages, read_file):
    page_results = {}

    for p in sorted(pages):
        img = Image.open(BytesIO(read_file(pages[p])))
        clean = preprocess(img)
        page_results[p] = hybrid_ocr(clean)

    full_text = " ".join(page_results[p]["text"] for p in page_results)
    confs = [page_results[p]["confidence"] for p in page_results if page_results[p]["confidence"] > 0]
    avg_conf = sum(confs)/len(confs) if confs else 0

    return {
        "student_id": student_id,
        "full_text": full_text,
        "avg_confidence": round(avg_conf,1),
        "flagged": avg_conf < CONF_THRESH
    }

# --- MAIN ---
def run_pipeline(source):
    print("🚀 FAST OCR PIPELINE\n")

    if os.path.isdir(source):
        all_names = [str(p) for p in Path(source).rglob("*.png")]
        read_file = lambda x: open(x, "rb").read()
    else:
        zf = zipfile.ZipFile(source)
        all_names = zf.namelist()
        read_file = lambda x: zf.read(x)

    grouped = parse_filenames(all_names)

    results = []
    flagged = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(process_student, sid, pages, read_file)
            for sid, pages in grouped.items()
        ]

        for f in tqdm(as_completed(futures), total=len(futures)):
            res = f.result()
            results.append(res)
            if res["flagged"]:
                flagged += 1

    avg = sum(r["avg_confidence"] for r in results)/len(results)

    print("\n" + "─"*40)
    print(f"Avg confidence: {round(avg,1)}")
    print(f"Flagged: {flagged}")
    print("─"*40)

# --- ENTRY ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder")
    parser.add_argument("--zip")
    args = parser.parse_args()

    if args.folder:
        run_pipeline(args.folder)
    elif args.zip:
        run_pipeline(args.zip)
