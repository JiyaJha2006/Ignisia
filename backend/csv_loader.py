import csv

def load_csv(filepath):
    data = []

    with open(filepath, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        for row in reader:
            data.append({
                "student_id": str(row["ID"]),
                "cluster_id": int(row["Cluster_ID"]),
                "raw_text": row["Student Answer"],
                "reference_answer": row["Q1_Answer"]   # 🔥 ADD THIS
            })

    return data
