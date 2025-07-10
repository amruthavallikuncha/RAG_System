# # Import TensorFlow and hub
# import tensorflow_hub as hub
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.neighbors import NearestNeighbors
# import numpy as np

# model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
# model = hub.load(model_url)

# def embed(texts):
#     return model(texts)

# def recommend(data, input_issue):
#     # Combine relevant text fields into one for embedding
#     issue_text = data['Issue Description'] + " " + data['Resolution']

#     # Embed issue descriptions
#     embeddings = embed(issue_text)

#     # Initialize Nearest Neighbors model
#     knn_model = NearestNeighbors(n_neighbors=len(data))
#     knn_model.fit(embeddings)
    
#     # Embed the input issue description
#     input_emb = embed([input_issue])

#     # Find nearest neighbors (similar issues) to the input
#     neighbors = knn_model.kneighbors(input_emb, return_distance=False)[0]

#     recommended_issues = []
#     for neighbor in neighbors[0:5]:  # Adjust to get the top 5 recommendations
#         issue_id = data.iloc[neighbor]['ID']
#         create_date = data.iloc[neighbor]['Create Date']
#         description = data.iloc[neighbor]['Issue Description']
#         resolution = data.iloc[neighbor]['Resolution']
#         category = data.iloc[neighbor]['Category']
#         similarity_score = cosine_similarity([input_emb[0]], [embeddings[neighbor]])[0][0]
#         recommended_issues.append({'Issue ID': issue_id, 'Create Date': create_date, 'Category': category, 'Description': description, 'Resolution': resolution, 'Relevancy Score': similarity_score})

#     return recommended_issues

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import numpy as np

model_name = "all-MiniLM-L6-v2"  # You can choose a different model from Sentence Transformers
model = SentenceTransformer(model_name)

def embed(texts):
    return model.encode(texts)

def recommend(data, input_issue):
    # Combine relevant text fields into one for embedding
    issue_text = data['Issue Description'] + " " + data['Resolution']

    # Embed issue descriptions
    embeddings = embed(issue_text)

    # Initialize Nearest Neighbors model
    knn_model = NearestNeighbors(n_neighbors=len(data))
    knn_model.fit(embeddings)
    
    # Embed the input issue description
    input_emb = embed([input_issue])

    # Find nearest neighbors (similar issues) to the input
    neighbors = knn_model.kneighbors(input_emb, return_distance=False)[0]

    recommended_issues = []
    for neighbor in neighbors[0:5]:  # Adjust to get the top 5 recommendations
        issue_id = data.iloc[neighbor]['ID']
        create_date = data.iloc[neighbor]['Create Date']
        description = data.iloc[neighbor]['Issue Description']
        resolution = data.iloc[neighbor]['Resolution']
        category = data.iloc[neighbor]['Category']
        similarity_score = cosine_similarity([input_emb[0]], [embeddings[neighbor]])[0][0]
        recommended_issues.append({'Issue ID': issue_id, 'Create Date': create_date, 'Category': category, 'Description': description, 'Resolution': resolution, 'Relevancy Score': similarity_score})

    return recommended_issues
