import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import hdbscan
import re

print("--- Loading JSON OCR Data ---")
 
with open('ocr_output/results.json', 'r', encoding='utf-8') as file:
    ocr_data = json.load(file)
 
df = pd.DataFrame(ocr_data)
bad_scans_df = df[df['flagged'] == True]
clean_df = df[df['flagged'] == False].copy()

clean_df=df.copy()
print(clean_df)
def extract_q1(text):
    
    match = re.search(r'(?:Q\.? ?1|Que 1|Ans?:? 1).*?(?=Q\.? ?2|Que 2|Ans?:? 2|Any: 2|9\.2|2>|2\)|$)', str(text), re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(0).strip()
    
     
    fallback_text = re.sub(r'(?:Q\.? ?2|Que 2|Ans?:? 2|Any: 2|9\.2|2>|2\)).*', '', str(text), flags=re.IGNORECASE | re.DOTALL)
    return fallback_text.strip()

def extract_q2(text):
    
    match = re.search(r'(?:Q\.? ?2|Que 2|Ans?:? ?2|Any: ?2|9\.?2\)?|2>|2\)|0?2\]|92\]).*', str(text), re.IGNORECASE | re.DOTALL)
    if match:
        # Strip out the teacher signatures
        clean_text = re.sub(r'(Teacher\'s Signature|AMAR KRISH)', '', match.group(0), flags=re.IGNORECASE)
        return clean_text.strip()
    return "No Answer"
 
clean_df['Q1_Answer'] = clean_df['full_text'].apply(extract_q1)
clean_df['Q2_Answer'] = clean_df['full_text'].apply(extract_q2)

 
print("--- Loading the SentenceTransformer Model ---")
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# HDBSCAN Strictness dialed to Maximum (epsilon=0.1) for maximum fragmentation!
clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, metric='euclidean', cluster_selection_epsilon=0.05)

 
print("\n--- PASS 1: Clustering Question 1 ---")
q1_embeddings = model.encode(clean_df['Q1_Answer'].tolist())
clean_df['Q1_Cluster_ID'] = clusterer.fit_predict(q1_embeddings)

 
print("--- PASS 2: Clustering Question 2 ---")
q2_embeddings = model.encode(clean_df['Q2_Answer'].tolist())
clean_df['Q2_Cluster_ID'] = clusterer.fit_predict(q2_embeddings)

 
print("\n=== FINAL CLUSTERING RESULTS ===")
print("Notice the strict dynamic fragmentation:")
print(clean_df[['student_id', 'Q1_Cluster_ID', 'Q2_Cluster_ID']]) 
print("\n--- Exporting final JSON for Grader Dashboard ---")
final_export_df = clean_df.drop(columns=['full_text']) 
final_export_df.to_json('ocr_output/final_clustered_grades.json', orient='records', indent=4)
print("Saved successfully to 'ocr_output/final_clustered_grades.json'!")
