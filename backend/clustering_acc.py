from data import *
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans 
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np

# Using tfidf to turn text into vectors
# pip install sentence-transformers 
def cluster_acc_df(df, index):
    corpus = [df["GPT4"].iloc[index], df["Gemini"].iloc[index], df["Claude3.5"].iloc[index], "The sky is blue."] 
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
        #outlier_cluster = None
        acc_score = 1
        acc_score_outlier = None
    else:
        #outlier_cluster = labels_tfidf_series.value_counts().loc[labels_tfidf_series.value_counts() == 1].index[0]
        # Calculate distance between centroids of clusters
        tfidf_centroids = tfidf.cluster_centers_
        tfidf_distances = euclidean_distances(tfidf_centroids)
        #print(tfidf_distances)

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

def cluster_acc(responses):
    # responses: list of diff llm's responses (strings) 

    if all(len(response.split()) == 1 for response in responses):
        return [0] * 4
    
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(responses).toarray()
    normalized_X_tfidf = X_tfidf / np.linalg.norm(X_tfidf, axis=1, keepdims=True)

    # KMeans clustering for tfidf vectors
    tfidf = KMeans(n_clusters = 2, random_state = 42)
    tfidf.fit(normalized_X_tfidf)
    labels_tfidf = tfidf.labels_
    #print(labels_tfidf)
    
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
        #print(tfidf_distances)

        # Use outlier distance to calculate its accuracy score
        acc_score = 1
        outlier_distance = max(tfidf_distances.flatten())
        if outlier_distance >= 2:
            acc_score_outlier = 0
        elif ((outlier_distance > 0) & (outlier_distance < 1)):
            acc_score_outlier = 1 - outlier_distance
        else: 
            acc_score_outlier = (2 - outlier_distance) / 2
    
    if outlier_cluster == None:
        cluster_acc_scores = [acc_score] * 4
    else:
        cluster_acc_scores = []
        for i in labels_tfidf:
            if i != outlier_cluster:
                cluster_acc_scores.append(acc_score)
        cluster_acc_scores.append(acc_score_outlier)
  
    return cluster_acc_scores

#reason_problem_df = full_data[full_data["Category"] == "Reasoning and Problem-Solving"].dropna()
# print(cluster_acc([reason_problem_df["GPT4"].iloc[0], reason_problem_df["Gemini"].iloc[0], reason_problem_df["Claude3.5"].iloc[0], "blue sky"]))
# for i, cluster in enumerate(labels_tfidf):
#     print(f"TF-IDF: Response {i+1} from Cluster {cluster}")


# acc_score_outlier, acc_score, X_tfidf, labels_tfidf = cluster_acc_df(reason_problem_df, 0)
# # print(acc_score_outlier, acc_score)
# practice = reason_problem_df[["Prompt", "GPT4", "Gemini", "Claude3.5"]].iloc[[0]]
# practice["Outlier"] = "The sky is blue."

# print(practice)


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

# plot_cluster(X_tfidf, labels_tfidf)
# # Reduce dimensions for plotting
# pca = PCA(n_components=2)
# Xtfidf_reduced = pca.fit_transform(X_tfidf.toarray())

# # Plot clusters
# plt.scatter(Xtfidf_reduced[:, 0], Xtfidf_reduced[:, 1], c=labels_tfidf)
# plt.xlabel("PCA Component 1")
# plt.ylabel("PCA Component 2")
# plt.title("Clustering of LLM Responses (TFIDF)")
# plt.show()

