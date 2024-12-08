from transformers import AutoTokenizer, AutoModelForSequenceClassification
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from torch.nn.functional import softmax
import nltk
import torch
import mwparserfromhell
import os
import pandas as pd
import numpy as np
import nltk


# FOR THIS CODE TO WORK UNCOMMENT AND DOWNLOAD WHATS BELOW --------------------------------------------------
# Download nltk resources, a lot of shit sorry (around 2 gigabytes of shit)
# nltk.download("popular")
# nltk.download("stopwords")
# nltk.download('punkt_tab')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

def accuracymetric(claims, dump_files):
    # claims: list of strings
    def parsetolist(file_name):
        with open(file_name, "r", encoding="utf-8") as file:
            for line in file:
                if "<text" in line:
                    wikicode = mwparserfromhell.parse(line)
                    yield wikicode.strip_code()
    # Parse the Wikipedia dump
    evidence_corpus = list(parsetolist(dump_files))
    # Preprocess corpus and initialize BM25
    stop_words = set(stopwords.words("english"))
    tokenized_corpus = [
        [word for word in word_tokenize(doc.lower()) if word.isalnum() and word not in stop_words]
        for doc in evidence_corpus
    ]
    bm25 = BM25Okapi(tokenized_corpus)
    # Tokenize claims
    tokenized_claims_list = [
        [word for word in word_tokenize(claim.lower()) if word.isalnum() and word not in stop_words]
        for claim in claims
    ]
    # Retrieve top evidence
    evidence_list = []
    for tokenized_claim in tokenized_claims_list:
        scores = bm25.get_scores(tokenized_claim)
        top_index = np.argmax(scores)
        evidence_list.append(evidence_corpus[top_index])
    # Load RoBERTa model
    tokenizer = AutoTokenizer.from_pretrained("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")
    model = AutoModelForSequenceClassification.from_pretrained("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")
    # Generate probabilities
    results = []
    for i, claim in enumerate(claims):
        inputs = tokenizer(f"Claim: {claim} Evidence: {evidence_list[i]}", return_tensors="pt", truncation=True)
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1)
        labels = ["Supported", "Refuted", "Not Enough Info"]
        class_idx = torch.argmax(probs).item()
        # print(f"Claim: {claim}")
        # print(f"Top Evidence: {evidence_list[i]}")
        # print(f"Prediction: {labels[class_idx]} (Probability: {probs.tolist()[0][class_idx]:.4f})\n")
        results.append(probs.tolist()[0][class_idx])
    return results 

# df = accuracymetric(["Neil Armstrong was the first person to walk on the Moon. He stepped onto the lunar surface on July 20, 1969, as part of the Apollo 11 mission.", "The main cause of the Great Depression in 1929 was the stock market crash.", "Albert Einstein is considered the father of modern physics."], "simplewiki-20240720-pages-articles-multistream.xml")
# print(df)
