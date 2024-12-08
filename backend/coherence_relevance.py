# Relevance (semantic similarity & word matching), Coherence (perplexity, semantic similarity b/w consecutive sentences)

# RELEVANCE (to the prompt asked): pip install sentence-transformers scikit-learn, pip install rouge-score
# Cosine similarity: checks for similarity in meaning
# Return a value b/w -1 and 1
    # 1: the sentences are identical in meaning
    # 0: no similarity
    # -1: sentences are opposites
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from accuracy import *
from data import *

def st_semantic_similarity_score(reference_text, output_text):
    # reference_text: list of strings to compare output_text to
    # output_text: list of strings 
    sentence_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    reference_embeddings = sentence_transformer.encode(reference_text)
    output_embeddings = sentence_transformer.encode(output_text)

    similarity_scores = cosine_similarity(reference_embeddings, output_embeddings)
    similarity_score = [similarity_scores[i][i] for i in range(len(similarity_scores))]   

    return similarity_score

# Combined with RougeSU1: word matching 
from rouge_score import rouge_scorer
def word_match_rouge(reference_text, output_text):
    # reference_text: list of strings to compare output_text to
    # output_text: list of strings
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    rougeScore = []
    for i in range(len(reference_text)):
        rouge_score = scorer.score(reference_text[i], output_text[i])
        fmeasure = rouge_score["rouge1"].fmeasure
        rougeScore.append(fmeasure) 
    return rougeScore

# precision: __% of the output matches input/actual
# recall: __% of the input/actual was found in the output

def relevance_score(similarity_score, rouge_score):
    return 0.3 * np.array(rouge_score) + 0.7 * np.array(similarity_score) # care more about meaning than matching    


# COHERENCE: pip install transformers, pip install -U evaluate
# perplexity + cosine similarity between consecutive sentences
from evaluate import load
def perplexity_score(text):
    # text: list of strings
    perplexity = load("perplexity", module_type = "metric")
    perplexity_dict = perplexity.compute(predictions = text, model_id='gpt2')
    perplexity_values = perplexity_dict['perplexities'] 
    normalized_perplexity = 1 - (1 / (np.array(perplexity_values)))

    return normalized_perplexity

# similarity between consecutive sentences
def calculate_consec_similarity(text):
    # text: string
    sentence_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    sentences = text.split(".")
    scores = []
    for i in range(len(sentences) - 1):
        curr = sentences[i]
        next = sentences[i + 1]
        encode_curr = sentence_transformer.encode([curr])
        encode_next = sentence_transformer.encode([next])
        score = cosine_similarity(encode_curr, encode_next)[0][0]
        scores.append(score)
    return np.mean(scores)

def total_coherence(consec_similarity, coherence_values):
    return 0.7 * coherence_values + 0.3 * consec_similarity

# TESTING
# data = {
#     'prompt': [
#         "If every apple is a fruit and some fruits are red, are all apples red?",
#         "What is the area of a rectangle with a length of 10 meters and a width of 4 meters?", 
#         "If John is taller than Sarah and Sarah is taller than Mike, who is the shortest?"
#     ],
#     'gpt4': [
#         "No, not all apples are necessarily red. While every apple is a fruit and some fruits are red, this does not imply that all apples must be red. Apples can come in other colors, such as green or yellow.",
#         "The area of a rectangle is calculated using the formula: Area=Length×Width. Substitute the given values: Area=10meters×4meters=40square meters. Thus, the area of the rectangle is 40 square meters.",
#         "Sports are awesome! I love playing basketball and soccer. I also enjoy watching football and baseball. Go team!"
#     ]
# }

# df = pd.DataFrame(data)
# copy_data = df.copy()

# similarity_score = st_semantic_similarity_score(copy_data["prompt"].tolist(), copy_data["gpt4"].tolist())
# rougeScore = word_match_rouge(copy_data["prompt"].tolist(), copy_data["gpt4"].tolist())
# relevant_score = relevance_score(similarity_score, rougeScore)
# copy_data["relevance"] = relevant_score

# normalized_perplexity = perplexity_score(copy_data["gpt4"].tolist())
# split_gpt4 = copy_data["gpt4"].apply(calculate_consec_similarity)

# copy_data["coherence"] = total_coherence(split_gpt4, normalized_perplexity)

# acc = accuracymetric(copy_data["gpt4"].tolist(), "backend/simplewiki-20240720-pages-articles-multistream.xml")
# copy_data["accuracy"] = acc
# print(copy_data[['relevance', 'coherence']])
