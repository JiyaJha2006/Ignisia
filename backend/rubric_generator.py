import re

def generate_rubric(teacher_data):

    # -----------------------------
    # SAFE EXTRACTION
    # -----------------------------
    model_answer = ""
    required = ""

    if isinstance(teacher_data, dict):
        model_answer = teacher_data.get("model_answer", "") or ""
        required = teacher_data.get("required", "") or ""
    else:
        model_answer = str(teacher_data)

    text = (model_answer + " " + required).lower()

    # -----------------------------
    # AUTO DETECT TYPE
    # -----------------------------
    if "=" in text or any(op in text for op in ["+", "-", "*", "/", "^"]):
        q_type = "math"
    elif len(text.split()) > 25:
        q_type = "language"
    else:
        q_type = "theory"

    # -----------------------------
    # REQUIRED ELEMENTS
    # -----------------------------
    parts = re.split(r'\d+\.\s*', required)
    required_elements = [p.strip() for p in parts if p.strip()]

    # fallback if empty
    if not required_elements:
        words = re.findall(r'[a-zA-Z]+', text)
        required_elements = list(set(words))[:3]

    return {
        "type": q_type,
        "required_elements": required_elements,
        "model_answer": model_answer,
        "equation": model_answer if q_type == "math" else None
    }
