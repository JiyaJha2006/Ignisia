from csv_loader import load_csv
from cluster_processor import group_by_cluster
from grading.scoring_engine import grade_cluster

# Load CSV
data = load_csv("answers.csv")

# Group into clusters
clusters = group_by_cluster(data)

final_output = []

# Process each cluster
for cluster_id, answers in clusters.items():

    cluster_payload = {
        "cluster_id": cluster_id,
        "answers": answers,
        "rubric": {
            "keywords": ["momentum", "conserved"],
            "optional_keywords": ["collision"],
            "keyword_weight": 0.5,
            "equation": "m*v",
            "math_weight": 0.3
        }
    }

    result = grade_cluster(cluster_payload)
    final_output.append(result)

import json
print(json.dumps(final_output, indent=2))