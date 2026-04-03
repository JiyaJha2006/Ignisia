from grading.scoring_engine import grade_cluster
from grading.regex_parser import extract_math_expressions

test_data = {
    "cluster_id": 1,
    "answers": [
        {
            "student_id": "101",
            "raw_text": "momentum is conserved and p = m*v"
        }
    ],
    "rubric": {
        "keywords": ["momentum", "conserved"],
        "optional_keywords": ["collision"],
        "keyword_weight": 0.5,
        "equation": "m*v",
        "math_weight": 0.3
    }
}

# 🔍 Debug: check what math is being extracted
for ans in test_data["answers"]:
    print("Extracted math:", extract_math_expressions(ans["raw_text"]))

# ✅ Run grading
result = grade_cluster(test_data)

print("\nFinal Result:")
print(result)