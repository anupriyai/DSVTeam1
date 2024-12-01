# Model: CustomScore = θ1(x1) + θ2(x2) +...+ θn(xn)
#   where x's are relevance, coherence, accuracy, creativity, ethics (bias, toxis, etc.)

# Focus on relevance (using semantic similarity), coherence (using perplexity)

# RELEVANCE: pip install sentence-transformers scikit-learn, pip install rouge-score
# documentation: https://sbert.net/
# Cosine similarity: checks for similarity in meaning
# Return a value b/w -1 and 1
    # 1: the sentences are identical in meaning
    # 0: no similarity
    # -1: sentences are opposites
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# input = ["To find the area A of a rectangle, we use the formula: A = lengthxwidth Given that the length is 10 meters and the width is 4 meters, we can substitute these values into the formula: A = 10m x 4m Now, we perform the multiplication: A = 40 m^2 Thus, the area of the rectangle is 40 m^2"]
# output = ["The area of a rectangle is calculated using the formula: Area=Length×Width Substitute the given values: Area=10meters×4meters=40square meters Thus, the area of the rectangle is 40 square meters."]
# input = ["If two statements contradict each other, can they both be true? Why or why not?"]
# output = ["No, two contradictory statements cannot both be true. A contradiction occurs when one statement directly negates the other. By definition, if one is true, the other must be false. This fundamental principle of logic, often referred to as the law of non-contradiction, is essential for consistent and rational thinking."]

data = {
    'prompt': [
        "If every apple is a fruit and some fruits are red, are all apples red?",
        "What is the area of a rectangle with a length of 10 meters and a width of 4 meters?", 
        "If John is taller than Sarah and Sarah is taller than Mike, who is the shortest?"
    ],
    'gpt4': [
        "No, not all apples are necessarily red. While every apple is a fruit and some fruits are red, this does not imply that all apples must be red. Apples can come in other colors, such as green or yellow.",
        "The area of a rectangle is calculated using the formula: Area=Length×Width. Substitute the given values: Area=10meters×4meters=40square meters. Thus, the area of the rectangle is 40 square meters.",
        "Sports are awesome! I love playing basketball and soccer. I also enjoy watching football and baseball. Go team!"
    ]
}

df = pd.DataFrame(data)
copy_data = df.copy()

sentence_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
input_embeddings = sentence_transformer.encode(copy_data["prompt"].tolist())
output_embeddings = sentence_transformer.encode(copy_data["gpt4"].tolist())

#print(embeddings)

similarity_scores = cosine_similarity(input_embeddings, output_embeddings)
similarity_score = [similarity_scores[i][i] for i in range(len(similarity_scores))]   

# Combined with RougeSU4: word matching 
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
rougeScore = []
for i in range(len(copy_data)):
    rouge_score = scorer.score(copy_data["prompt"][i], copy_data["gpt4"][i])
    fmeasure = rouge_score["rouge1"].fmeasure
    rougeScore.append(fmeasure) 

# precision: __% of the output matches input/actual
# recall: __% of the input/actual was found in the output

relevant_score = 0.3 * np.array(rougeScore) + 0.7 * np.array(similarity_score) # care more about meaning than matching
copy_data["relevance"] = relevant_score

# COHERENCE: pip install transformers, pip install -U evaluate
# https://huggingface.co/docs/transformers/en/perplexity
# https://huggingface.co/spaces/evaluate-metric/perplexity
# Perplexity: lower the score the better

# from transformers import GPT2LMHeadModel, GPT2Tokenizer

# model_id = "gpt2"
# model = GPT2LMHeadModel.from_pretrained(model_id)
# tokenizer = GPT2Tokenizer.from_pretrained(model_id)

from evaluate import load
perplexity = load("perplexity", module_type = "metric")
coherence_dict = perplexity.compute(predictions = copy_data["gpt4"].tolist(), model_id='gpt2')
coherence_values = coherence_dict['perplexities'] 

# normalize perplexity bc its range is diff from cosine similarity
normalized_coherence = 1 - (1 / (np.array(coherence_values)))

# similarity between consecutive sentences
split_gpt4 = copy_data["gpt4"].str.split(pat = ".")
def calculate_similarity(text):
    scores = []
    for sentence in range(len(text) - 1):
        curr = text[sentence]
        next = text[sentence + 1]
        encode_curr = sentence_transformer.encode([curr])
        encode_next = sentence_transformer.encode([next])
        score = cosine_similarity(encode_curr, encode_next)[0][0]
        scores.append(score)
    return np.mean(scores)

split_gpt4 = split_gpt4.apply(calculate_similarity)

copy_data["coherence"] = 0.7 * normalized_coherence + 0.3 * split_gpt4
print(copy_data)
# def custom_score(relevance, coherence, weight_relevance = 0.5, weight_coherence = 0.5):
#     
#     score = (relevance * weight_relevance) + (normalized_coherence * weight_coherence)
#     return score

# #print(custom_score(relevance[0][0], coherence_value)) # 0.054113088526403896

