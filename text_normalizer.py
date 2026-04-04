import re
import unicodedata


DEVANAGARI_RANGE = re.compile(r"[\u0900-\u097F]")
LATIN_RANGE = re.compile(r"[A-Za-z]")


def normalize_multilingual_text(text):
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = text.replace("\x0c", " ")
    text = text.replace("|", " ")
    text = text.replace("॥", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def clean_ocr_artifacts(text):
    text = normalize_multilingual_text(text)
    text = re.sub(r"(Teacher'?s Signature|AMAR KRISH)", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"([A-Za-z])\1{3,}", r"\1", text)
    text = re.sub(r"([०-९0-9])\s+([०-९0-9])", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_script_mix(text):
    text = str(text or "")
    has_devanagari = bool(DEVANAGARI_RANGE.search(text))
    has_latin = bool(LATIN_RANGE.search(text))

    if has_devanagari and has_latin:
        return "mixed"
    if has_devanagari:
        return "devanagari"
    if has_latin:
        return "latin"
    return "unknown"


def build_embedding_text(text):
    text = clean_ocr_artifacts(text)
    text = re.sub(r"[^\w\s\u0900-\u097F%+=*/()-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
