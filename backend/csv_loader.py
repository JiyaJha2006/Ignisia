import csv

def load_csv(filepath):
    data = []

    with open(filepath, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        for row in reader:
            data.append({
                "student_id": row["student_id"],
                "cluster_id": int(row.get("cluster_id", 1)),
                "raw_text": row["raw_text"]
            })

    return data