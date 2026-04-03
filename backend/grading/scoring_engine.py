from grading.regex_parser import extract_math_expressions
from grading.math_validator import validate_equation
from grading.keyword_matcher import keyword_score
from collections import Counter

def grade_cluster(cluster_payload):
    results = []
    rubric = cluster_payload["rubric"]

    for ans in cluster_payload["answers"]:
        student_text = ans["raw_text"]

        # Keyword scoring
        kw_score = keyword_score(student_text, rubric["keywords"])
        optional_score = keyword_score(
            student_text,
            rubric.get("optional_keywords", [])
        )

        # Math detection
        student_math = extract_math_expressions(student_text)

        math_correct = False
        if rubric.get("equation") and student_math:
            math_correct = any(
                validate_equation(expr, rubric["equation"])
                for expr in student_math
            )

        # Final score
        final_score = (
            kw_score * rubric["keyword_weight"] +
            optional_score * 0.2 +
            (1 if math_correct else 0) * rubric["math_weight"]
        )

        # Feedback (FIXED: no duplicates)
        feedback = set()

        missing_keywords = [
            kw for kw in rubric["keywords"]
            if kw.lower() not in student_text.lower()
        ]

        if missing_keywords:
            feedback.add(f"Missing keywords: {', '.join(missing_keywords)}")

        if rubric.get("equation") and not math_correct:
            feedback.add("Incorrect or missing formula")

        if not feedback:
            feedback.add("Perfect answer")

        # Confidence
        confidence = round((kw_score + (1 if math_correct else 0)) / 2, 2)

        results.append({
            "student_id": ans["student_id"],
            "score": round(final_score, 2),
            "math_correct": math_correct,
            "keyword_score": round(kw_score, 2),
            "confidence": confidence,
            "feedback": list(feedback)  # convert set → list
        })

    # ✅ ADD CLUSTER ANALYTICS
    avg_score = sum(r["score"] for r in results) / len(results)

    # ✅ TOP ISSUES (for frontend insights)
    all_feedback = []
    for r in results:
        all_feedback.extend(r["feedback"])

    top_issues = [
        issue for issue, _ in Counter(all_feedback).most_common(2)
    ]

    return {
        "cluster_id": cluster_payload["cluster_id"],
        "cluster_size": len(results),
        "avg_score": round(avg_score, 2),
        "top_issues": top_issues,
        "results": results
    }