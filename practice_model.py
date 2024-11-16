# Model: CustomScore = θ1(x1) + θ2(x2) +...+ θn(xn)
#   where x's are relevance, coherence, accuracy, creativity, ethics (bias, toxis, etc.)

# Focus on relevance (using semantic similarity), coherence (using perplexity)

# RELEVANCE: pip install sentence-transformers scikit-learn
# documentation: https://sbert.net/
# The cosine similarity return a value b/w -1 and 1
    # 1: the sentences are identical in meaning
    # 0: no similarity
    # -1: sentences are opposites
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

input = ["What if these shoes don't fit?"]
output = ["We offer a 30-day full refund at no extra cost."]
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
input_embeddings = model.encode(input)
output_embeddings = model.encode(output)

#print(embeddings)

relevance = cosine_similarity([input_embeddings[0]], [output_embeddings[0]])
# print(relevance) # relevance = [[0.05094387]]

# COHERENCE: pip install transformers, pip install -U evaluate
# https://huggingface.co/docs/transformers/en/perplexity
# https://huggingface.co/spaces/evaluate-metric/perplexity
# lower the score the better

# from transformers import GPT2LMHeadModel, GPT2Tokenizer

# model_id = "gpt2"
# model = GPT2LMHeadModel.from_pretrained(model_id)
# tokenizer = GPT2Tokenizer.from_pretrained(model_id)

from evaluate import load
perplexity = load("perplexity", module_type = "metric")
coherence_dict = perplexity.compute(predictions = output, model_id='gpt2')
coherence_value = coherence_dict['perplexities'][0] # coherence_dict = {'perplexities': [16.4573974609375], 'mean_perplexity': 16.4573974609375}

#print(coherence_value)

def custom_score(relevance, coherence, weight_relevance = 0.5, weight_coherence = 0.5):
    # normalize perplexity bc its range is diff from cosine similarity
    normalized_coherence = 1 / (1 + coherence)
    score = (relevance * weight_relevance) + (normalized_coherence * weight_coherence)
    return score

print(custom_score(relevance[0][0], coherence_value)) # 0.054113088526403896

