from flask import Flask, request, jsonify
from flask_cors import CORS
from coherence_relevance import *
from clustering_acc import *
from accuracy import *
from creativity_metric import *
from bias import *
from data import *

app = Flask(__name__)
CORS(app)

relevance_score = 0

def calculate_relevance_score(data):
    prompt = [data["prompt"]] * 4
    responses = list(data["responses"].values())
    ss_score = st_semantic_similarity_score(prompt, responses)
    rouge_score = word_match_rouge(prompt, responses)
    return total_relevance_score(ss_score, rouge_score).tolist()
    # print("RELEVANCE SCORE", relevance_score)

def calculate_coherence_score(data):
    responses = list(data["responses"].values())
    perplex_score = perplexity_score(responses)
    consec_sim_score = [calculate_consec_similarity(i) for i in responses]
    return total_coherence(consec_sim_score, perplex_score).tolist()

def calculate_cluster_acc_score(data):
    responses = list(data["responses"].values())
    return cluster_acc(responses)

def calculate_acc_score(data):
    responses = list(data["responses"].values())
    return accuracymetric(responses, "backend/simplewiki-20240720-pages-articles-multistream.xml")

def calculate_creativity_score(data):
    # placeholder
    responses = list(data["responses"].values())
    test_strings = responses

def calculate_bias_score(data):
    # add perspecitive score + calculate total
    responses = list(data["responses"].values())
    ethics = [ethics_score(i) for i in responses]
    return ethics

def custom_score(cluster, acc, coher, rele, creat, bias, weight):
    return np.array(cluster).astype(float) * weight[0] + np.array(acc).astype(float) * weight[1] + np.array(coher).astype(float) * weight[2] + np.array(rele).astype(float) * weight[3] + np.array(creat).astype(float) * weight[4] + np.array(bias).astype(float) * weight[5]

def calculate_all_scores(data):
    cluster_acc_scores = calculate_cluster_acc_score(data)
    acc_scores = calculate_acc_score(data)
    coherence_scores = calculate_coherence_score(data)
    relevance_scores = calculate_relevance_score(data)
    creativity_scores = calculate_creativity_score(data)
    bias_scores =  calculate_bias_score(data)

    my_dict = {}
    models = ["GPT-4o", "Gemini", "Claude 3.5 Sonnet", "Llama"]
    # need to change when changed to select only 1 category
    category = data["categories"][0]

    if category == "Factual Knowledge":
        weight = [0.2, 0.5, 0, 0.3, 0, 0]
        custom = custom_score(cluster_acc_scores, acc_scores, coherence_scores, relevance_scores, creativity_scores, bias_scores, weight)

        for i, model in enumerate(models):
            my_dict[model] = {
                "cluster_acc": cluster_acc_scores[i],
                "accuracy": acc_scores[i],
                "coherence": coherence_scores[i],
                "relevance": relevance_scores[i],
                "creativity": "Not Considered",
                "bias": "Not Considered",
                "custom": custom[i]
            }           
    elif category == "Reasoning and Problem-Solving":
        weight = [0.5, 0, 0.2, 0.2, 0.1, 0]
        custom = custom_score(cluster_acc_scores, acc_scores, coherence_scores, relevance_scores, creativity_scores, bias_scores, weight)

        for i, model in enumerate(models):
            my_dict[model] = {
                "cluster_acc": cluster_acc_scores[i],
                "accuracy": "Not Considered",
                "coherence": coherence_scores[i],
                "relevance": relevance_scores[i],
                "creativity": creativity_scores[i],
                "bias": "Not Considered",
                "custom": custom[i]
            }  
    elif category == "Creative Writing":
        weight = [0, 0, 0.2, 0.15, 0.6, 0.05]
        custom = custom_score(cluster_acc_scores, acc_scores, coherence_scores, relevance_scores, creativity_scores, bias_scores, weight)

        for i, model in enumerate(models):
            my_dict[model] = {
                "cluster_acc": "Not Considered",
                "accuracy": "Not Considered",
                "coherence": coherence_scores[i],
                "relevance": relevance_scores[i],
                "creativity": creativity_scores[i],
                "bias": bias_scores[i],
                "custom": custom[i]
            }   
    elif category == "Language Understanding": # actually for Ethics
        weight = [0, 0, 0.1, 0.1, 0, 0.8]
        custom = custom_score(cluster_acc_scores, acc_scores, coherence_scores, relevance_scores, creativity_scores, bias_scores, weight)

        for i, model in enumerate(models):
            my_dict[model] = {
                "cluster_acc": "Not Considered",
                "accuracy": "Not Considered",
                "coherence": coherence_scores[i],
                "relevance": relevance_scores[i],
                "creativity": "Not Considered",
                "bias": bias_scores[i],
                "custom": custom[i]
            }

    return my_dict

def calculate_preset_scores(data):
    category = data["categories"][0]
    prompt = data["prompt"]
    filt_resp = full_data[(full_data["Category"] == category) & (full_data["Prompt"] == prompt)][["GPT4", "Gemini", "Claude3.5", "Llama"]].values[0]
    if len(filt_resp) == 4:
        data["responses"] = {"GPT-4o": filt_resp[0], "Gemini": filt_resp[1], "Claude 3.5 Sonnet": filt_resp[2], "Llama": filt_resp[3]}
        return calculate_all_scores(data)
    else:
        print("ERROR: Not enough responses")
        models = ["GPT-4o", "Gemini", "Claude 3.5 Sonnet", "Llama"]
        my_dict = {}
        for model in models:
            my_dict[model] = {
                "cluster_acc": [0] * 4,
                "accuracy": [0] * 4,
                "coherence": [0] * 4,
                "relevance": [0] * 4,
                "creativity": [0] * 4,
                "bias": [0] * 4,
                "custom": [0] * 4
            }  
        return my_dict

@app.route("/api/accuracy", methods=["POST"])
def receive_data():
    data = request.json
    print("DATA HERE", data)

    # determines calculations for presets vs custom
    if data.get("responses" , 0) == 0: # checks if there are responses in data
        scores = calculate_preset_scores(data)
    else:
        scores = calculate_all_scores(data)
    print("SCORES HERE", scores)
    return jsonify({
        "message": scores
        })

if __name__ == "__main__":
    app.run(debug=True, port=8080)
    