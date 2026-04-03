import re

def extract_math_expressions(text):
    pattern = r'[a-zA-Z]+\s*=\s*[a-zA-Z0-9\*\+\-/\^\.]+'
    return re.findall(pattern, text)

def extract_units(text):
    units_pattern = r'\b(m/s|kg|N|J|cm|m)\b'
    return re.findall(units_pattern, text)