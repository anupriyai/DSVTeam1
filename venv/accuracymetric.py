from transformers import AutoTokenizer, AutoModelForSequenceClassification
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from torch.nn.functional import softmax
import nltk
import torch
import mwparserfromhell
import os
from wikipediapipeline import *


# FOR THIS CODE TO WORK UNCOMMENT AND DOWNLOAD WHATS BELOW --------------------------------------------------
# Download nltk resources, a lot of shit sorry (around 2 gigabytes of shit)
#nltk.download("popular")
#nltk.download("stopwords")
#nltk.download('punkt_tab')
#nltk.download('wordnet')
#nltk.download('omw-1.4')

def accuracymetric(claim = "The Eiffel Tower is in Berlin.", evidence_corpus = "venv\enwiki-20241120-pages-articles-multistream1.xml-p1p41242\enwiki-20241120-pages-articles-multistream1.xml-p1p41242"):
    # Define corpus and claim
    # Defaults to parts of a wikipedia dump ~256 M/b. I'm working on a laptop

    # ---- parsing Wikipedia files to file

    def parsetolist(file_name=evidence_corpus):

        shittycorpus = []
        with open(file_name, "r", encoding="utf-8") as file:
            for line in file:
                # Detect and parse articles
                if "<text" in line:
                    wikicode = mwparserfromhell.parse(line)
                    shittycorpus.append(wikicode.strip_code())

        return shittycorpus

    evidence_corpus = parsetolist(evidence_corpus)

    # Preprocess corpus, tokenize corpus
    stop_words = set(stopwords.words("english"))
    tokenized_corpus = [
        [word for word in word_tokenize(doc.lower()) if word.isalnum() and word not in stop_words] for doc in evidence_corpus
    ]

    # Initialize BM25, tokenize claim
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_claim = [word for word in word_tokenize(claim.lower()) if word.isalnum() and word not in stop_words]

    # Retrieve top evidence from corpus
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
    probs = softmax(outputs.logits, dim=1)
    class_idx = torch.argmax(probs).item()
    labels = ["Supported", "Refuted", "Not Enough Info"]

    # Output results
    print(f"------------------RESULTS------------------")
    print(f"Top Evidence: {evidence}")
    print(f"Prediction: {labels[class_idx]} \n (Probability of Accurate: {probs.tolist()[0][0]})")
    print(f"------------------RESULTS------------------")
