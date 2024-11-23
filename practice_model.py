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


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

input = ["To find the area A of a rectangle, we use the formula: A = lengthxwidth Given that the length is 10 meters and the width is 4 meters, we can substitute these values into the formula: A = 10m x 4m Now, we perform the multiplication: A = 40 m^2 Thus, the area of the rectangle is 40 m^2"]
output = ["The area of a rectangle is calculated using the formula: Area=Length×Width Substitute the given values: Area=10meters×4meters=40square meters Thus, the area of the rectangle is 40 square meters."]
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
input_embeddings = model.encode(input)
output_embeddings = model.encode(output)

#print(embeddings)

similarity_score = cosine_similarity([input_embeddings[0]], [output_embeddings[0]])[0][0]
#relevance_ST = model.similarity(input_embeddings, output_embeddings) 
print(similarity_score) # similarity_score = 0.89554524

# Combined with RougeSU4: word matching, but this version cares less about the order 
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

rouge_score = scorer.score(input[0], output[0])
fmeasure = rouge_score["rouge1"].fmeasure
print(fmeasure) # {'rouge1': Score(precision=0.78125, recall=0.43103448275862066, fmeasure=0.5555555555555556)}

# precision: 78% of the output matches input/actual
# recall: 43% of the input/actual was found in the output

relevant_score = 0.3 * fmeasure + 0.7 * similarity_score # care more about meaning and matching
print(relevant_score) # 0.79354835 - good score; closer to 1 the better

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

#print(custom_score(relevance[0][0], coherence_value)) # 0.054113088526403896

