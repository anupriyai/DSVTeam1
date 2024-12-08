from data import *
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans 
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np

reason_problem_df = full_data[full_data["Category"] == "Language Understanding"].dropna()

# Using SentenceTransformer & tfidf to turn text into vectors
# pip install sentence-transformers 
def cluster_acc(df, index):
    corpus = [df["GPT4"].iloc[index], df["Gemini"].iloc[index], df["Claude3.5"].iloc[index], df["Llama"].iloc[0]] 
    # "235 multiplied by 47 equals 11048.", "The sky is blue." , reason_problem_df["Llama"].iloc[0],
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(corpus)

    # KMeans clustering for tfidf vectors
    tfidf = KMeans(n_clusters = 2, random_state = 42)
    tfidf.fit(X_tfidf)
    labels_tfidf = tfidf.labels_
    print(labels_tfidf)
    
    # If outlier exists, assign its score & others scores else only assign its score 
    labels_tfidf_series = pd.Series(labels_tfidf) 
    if len(labels_tfidf_series.value_counts().loc[labels_tfidf_series.value_counts() == 1]) == 0:
        outlier_cluster = None
        acc_score = 1
        acc_score_outlier = None
    else:
        outlier_cluster = labels_tfidf_series.value_counts().loc[labels_tfidf_series.value_counts() == 1].index[0]
        # Calculate distance between centroids of clusters
        tfidf_centroids = tfidf.cluster_centers_
        tfidf_distances = euclidean_distances(tfidf_centroids)
        print (tfidf_distances)
        # Use outlier distance to calculate its accuracy score
        acc_score = 1
        outlier_distance = max(tfidf_distances.flatten())
        if outlier_distance >= 2:
            acc_score_outlier = 0
        elif ((outlier_distance > 0) & (outlier_distance < 1)):
            acc_score_outlier = 1 - outlier_distance
        else: 
            acc_score_outlier = (2 - outlier_distance) / 2

    return acc_score_outlier, acc_score, X_tfidf, labels_tfidf

# for i, cluster in enumerate(labels_tfidf):
#     print(f"TF-IDF: Response {i+1} from Cluster {cluster}")


acc_score_outlier, acc_score, X_tfidf, labels_tfidf = cluster_acc(reason_problem_df, 0)
print(acc_score_outlier, acc_score)


def plot_cluster(X_tfidf, labels_tfidf):
    # Reduce dimensions for plotting
    pca = PCA(n_components=2)
    Xtfidf_reduced = pca.fit_transform(X_tfidf.toarray())

    jitter = 0.01
    Xtfidf_reduced += jitter * np.random.randn(*Xtfidf_reduced.shape)

    # Plot clusters
    plt.scatter(Xtfidf_reduced[:, 0], Xtfidf_reduced[:, 1], c=labels_tfidf)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Clustering of LLM Responses (TFIDF)")
    plt.show()

plot_cluster(X_tfidf, labels_tfidf)
# # Reduce dimensions for plotting
# pca = PCA(n_components=2)
# Xtfidf_reduced = pca.fit_transform(X_tfidf.toarray())

# # Plot clusters
# plt.scatter(Xtfidf_reduced[:, 0], Xtfidf_reduced[:, 1], c=labels_tfidf)
# plt.xlabel("PCA Component 1")
# plt.ylabel("PCA Component 2")
# plt.title("Clustering of LLM Responses (TFIDF)")
# plt.show()

