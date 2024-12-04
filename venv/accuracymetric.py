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

import nltk


# FOR THIS CODE TO WORK UNCOMMENT AND DOWNLOAD WHATS BELOW --------------------------------------------------
# Download nltk resources, a lot of shit sorry (around 2 gigabytes of shit)
#nltk.download("popular")
#nltk.download("stopwords")
#nltk.download('punkt_tab')
#nltk.download('wordnet')
#nltk.download('omw-1.4')

def accuracymetric(claims = ["The Eiffel Tower is in Berlin."], dump_files = "venv\enwiki-20241120-pages-articles-multistream1.xml-p1p41242\enwiki-20241120-pages-articles-multistream1.xml-p1p41242"):
    # Define corpus and claim
    # Defaults to parts of a wikipedia dump ~256 M/b. I'm working on a laptop

    # ---- parsing Wikipedia files to file

    def parsetolist(file_name=dump_files):

        shittycorpus = []
        with open(file_name, "r", encoding="utf-8") as file:
            for line in file:
                # Detect and parse articles
                if "<text" in line:
                    wikicode = mwparserfromhell.parse(line)
                    shittycorpus.append(wikicode.strip_code())

        return shittycorpus
    
    # ---- parsing Wikipedia files to file

    evidence_corpus = parsetolist(dump_files)


    # ---- Preprocess corpus, tokenize corpus from dumpfiles and initialize BM25
    stop_words = set(stopwords.words("english"))
    tokenized_corpus = [
        [word for word in word_tokenize(doc.lower()) if word.isalnum() and word not in stop_words] for doc in evidence_corpus
    ]
    bm25 = BM25Okapi(tokenized_corpus) # Tokenize claim and initialize BM25, 
      
    # ---- Preprocess corpus, tokenize corpus


    # ---- Load RoBERTa 
    tokenizer = AutoTokenizer.from_pretrained("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")
    model = AutoModelForSequenceClassification.from_pretrained("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")

    # ---- Load RoBERTa 


    # ---- tokenizes all the claims
    tokenized_claims_list = [[word for word in word_tokenize(l.lower()) if word.isalnum() and word not in stop_words] for l in claims]

    # ---- tokenizes all the claims


    # ---- Retrieve top evidence from corpus for the claims
    top_k = 1
    scores = []
    evidence_list = []

    for _ in range(len(tokenized_claims_list)):
        scores.append(bm25.get_scores(tokenized_claims_list)) # get scores for every tokenized claim

        
    for score in scores:
        top_indices = sorted(range(len(score)), key=lambda i: score[i], reverse=True)[:top_k]
        evidence_list.append(evidence_corpus[top_indices[0]])

    # ---- Retrieve top evidence from corpus for the claims


    # ---- Generating probabiilties

    result = {}

    for i in range(len(tokenized_claims_list)):
        # Tokenize inputs for RoBERTa
        inputs = tokenizer(f"Claim: {claims[i]} Evidence: {evidence_list[i]}", return_tensors="pt", truncation=True)

        # Perform inference
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1)
        class_idx = torch.argmax(probs).item()
        labels = ["Supported", "Refuted", "Not Enough Info"]

        # Printing results
        print(f"------------------RESULTS------------------")
        print(f"Top Evidence: {evidence_list[i]}")
        print(f"Prediction: {labels[class_idx]} \n (Probability of Accurate: {probs.tolist()[0][0]})")
        print(f"------------------RESULTS------------------")

        result[claims[i]] = probs.tolist()[0][0]

    return pd.Dataframe(result)    



