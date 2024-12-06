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
nltk.download("popular")
nltk.download("stopwords")
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

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


        # Retrieve top evidence from corpus for the claims
    scores = []
    evidence_list = []

    for tokenized_claim in tokenized_claims_list:
        scores.append(bm25.get_scores(tokenized_claim))

    for score in scores:
        max_index = score.index(max(score))
        evidence_list.append(evidence_corpus[max_index])

    # Generate probabilities and predictions
    result = {}

    for i in range(len(claims)):
        # Tokenize inputs for RoBERTa
        inputs = tokenizer(f"Claim: {claims[i]} Evidence: {evidence_list[i]}", return_tensors="pt", truncation=True, max_length=512)

        # Perform inference
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1)
        class_idx = torch.argmax(probs).item()
        labels = ["Supported", "Refuted", "Not Enough Info"]

        predicted_label = labels[class_idx]
        predicted_prob = probs[0][class_idx].item()

        # Print results
        print(f"------------------RESULTS------------------")
        print(f"Top Evidence: {evidence_list[i]}")
        print(f"Prediction: {predicted_label} \n (Probability of Accurate: {predicted_prob})")
        print(f"------------------RESULTS------------------")

        result[claims[i]] = predicted_prob


    return pd.Dataframe(result)    



