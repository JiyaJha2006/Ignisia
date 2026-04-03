import re

def detect_question_type(model_answer):
    text = model_answer.lower()

    # math detection
    if "=" in text or any(op in text for op in ["+", "-", "*", "/", "^"]):
        return "math"

    # long descriptive → language
    if len(text.split()) > 25:
        return "language"

    return "theory"

def extract_required_keywords(required_text):
    # Split numbered points
    parts = re.split(r'\d+\.\s*', required_text)

    keywords = []

    for part in parts:
        words = re.findall(r'[a-zA-Z]+', part.lower())

        # Keep important words only
        filtered = [w for w in words if len(w) > 4]

        keywords.extend(filtered[:2])  # take 1–2 per point

    return list(set(keywords))


def generate_rubric(teacher_data):
    model_answer = teacher_data["model_answer"]
    required = teacher_data["required"]

    q_type = detect_question_type(model_answer)

    import re
    parts = re.split(r'\d+\.\s*', required)
    required_elements = [p.strip() for p in parts if p.strip()]

    return {
        "type": q_type,
        "required_elements": required_elements,
        "model_answer": model_answer,
        "equation": model_answer if q_type == "math" else None
    }
