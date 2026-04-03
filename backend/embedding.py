import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import hdbscan

print("--- Loading JSON OCR Data ---")
 
with open('ocr_output/ocr_log.json', 'r', encoding='utf-8') as file:
    ocr_data = json.load(file)
 
 
df = pd.DataFrame(ocr_data['answers'])
 
bad_scans_df = df[df['flagged'] == True]
clean_df = df[df['flagged'] == False].copy()

print(f"Total students loaded: {len(df)}")
print(f"Clean answers ready for ML clustering: {len(clean_df)}")
print(f"Bad handwriting flagged for manual review: {len(bad_scans_df)}\n")

print(df.head())
print("\n")

 
print("--- Loading the SentenceTransformer Model ---")
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
print("Model Ready!\n")

 
print("--- Generating Semantic Vectors (Embeddings) ---")
answers_list = clean_df['full_text'].tolist()
embeddings = model.encode(answers_list)
 
clean_df['embeddings'] = list(embeddings)

print(f"Success! We generated embeddings for the {len(df)} answers in your CSV.")
print(f"The matrix shape is: {embeddings.shape}\n")

print("\n--- Sorting Answers into Clusters ---")
# Build groups of at least size 2
clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, metric='euclidean')
cluster_labels = clusterer.fit_predict(list(clean_df['embeddings']))
clean_df['Cluster_ID'] = cluster_labels
print("\n--- FINAL CLUSTERING RESULTS ---")
print(clean_df[['student_id', 'full_text', 'Cluster_ID']])
# -------------------------------------------------------------
# STEP 5: EXPORT TO THE UI DASHBOARD
# -------------------------------------------------------------
print("\n--- Exporting final JSON for Grader Dashboard ---")
# Drop the massive math vectors (frontend doesn't need them)
final_export_df = clean_df.drop(columns=['embeddings'])
final_export_df.to_json('ocr_output/final_clustered_grades.json', orient='records', indent=4)
print("Saved successfully to 'ocr_output/final_clustered_grades.json'!")

