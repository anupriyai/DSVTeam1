from transformers import AutoTokenizer, AutoModelForSequenceClassification
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from torch.nn.functional import softmax
import nltk
import torch

# Download nltk resources, a lot of shit sorry (around 2 gigabytes of shit)
#nltk.download("popular")
#nltk.download("stopwords")
#nltk.download('punkt_tab')
#nltk.download('wordnet')
#nltk.download('omw-1.4')

# ---- THIS SHIT IS JUST FOR TESTING PURPOSES ONLY. USING THE ACTUAL WIKIPEDIA DATASET IMO IS OVERKILL. 
       # ILL (chATGPT) WILL ADD THE PARSER FROM WIKIPEDIA LATER


# Define corpus and claim
evidence_corpus = [
    "The Eiffel Tower is located in Paris, France.",
    "Berlin is the capital city of Germany.",
    "The Empire State Building is in New York City.",
    "The Great Wall of China is a historical structure in China."
]
claim = "The Eiffel Tower is in Berlin."


# ----


# Preprocess corpus
stop_words = set(stopwords.words("english"))
tokenized_corpus = [
    [word for word in word_tokenize(doc.lower()) if word.isalnum() and word not in stop_words]
    for doc in evidence_corpus
]

# Initialize BM25
bm25 = BM25Okapi(tokenized_corpus)
tokenized_claim = [word for word in word_tokenize(claim.lower()) if word.isalnum() and word not in stop_words]

# Retrieve top evidence
top_k = 1
scores = bm25.get_scores(tokenized_claim)
top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
evidence = evidence_corpus[top_indices[0]]

# Load RoBERTa 
tokenizer = AutoTokenizer.from_pretrained("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")
model = AutoModelForSequenceClassification.from_pretrained("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")

# Tokenize inputs for RoBERTa
inputs = tokenizer(f"Claim: {claim} Evidence: {evidence}", return_tensors="pt", truncation=True)

# Perform inference
outputs = model(**inputs)
logits = outputs.logits
probs = softmax(logits, dim=1)
class_idx = torch.argmax(probs).item()
labels = ["Supported", "Refuted", "Not Enough Info"]

# Output results
print(f"/n /n /n /n ------------------RESULTS------------------")
print(f"Top Evidence: {evidence}")
print(f"Prediction: {labels[class_idx]} (Probabilities: {probs.tolist()})")
print(f"------------------RESULTS------------------")
