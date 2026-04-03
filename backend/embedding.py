import pandas as pd
from sentence_transformers import SentenceTransformer
import hdbscan
import re
 
print("--- Loading Your CSV Data ---")
df = pd.read_csv(r"C:\Users\garim\Downloads\Mock_Dataset_Physics_Momentum_50.csv")

 
clean_df = df.copy()

print(f"Total students loaded from CSV: {len(clean_df)}\n")

 
clean_df.to_csv('ocr_output/clean_data.csv', index=False)
 
def extract_q1(text):
     
    match = re.search(r'(?:Q1|Que 1|Ans 1).*?(?=Q2|Que 2|Ans 2|$)', str(text), re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(0).strip()
    return str(text)

 
clean_df['Q1_Answer'] = clean_df['Student Answer'].apply(extract_q1)

 
 
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
print("Model Ready!\n")
 
 
answers_list = clean_df['Q1_Answer'].tolist()
embeddings = model.encode(answers_list)
 
clean_df['embeddings'] = list(embeddings)

print(f"Success! We generated embeddings for the {len(clean_df)} answers.")
print(f"The matrix shape is: {embeddings.shape}\n")

 
clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, metric='euclidean', cluster_selection_epsilon=0.5)
cluster_labels = clusterer.fit_predict(list(clean_df['embeddings']))

clean_df['Cluster_ID'] = cluster_labels

print("\n--- FINAL CLUSTERING RESULTS ---")
print(clean_df[['ID', 'Q1_Answer', 'Cluster_ID']])
 
 
print("\n--- Exporting final files for Grader Dashboard ---")
 
final_export_df = clean_df.drop(columns=['embeddings'])

 
final_export_df.to_json('ocr_output/final_clustered_grades.json', orient='records', indent=4)
final_export_df.to_csv('ocr_output/final_clustered_grades.csv', index=False)

print("Saved successfully to both JSON and CSV inside 'ocr_output/'!")
