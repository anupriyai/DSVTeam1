from flask import Flask, request, jsonify
from flask_cors import CORS
from coherence_relevance import *
from clustering_acc import *
from accuracy import *
from creativity_metric import *
from bias import *

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
    return [0] * 4
    # accuracymetric(responses, "backend/simplewiki-20240720-pages-articles-multistream.xml")

def calculate_creativity_score(data):
    # placeholder
    return [1] * 4

def calculate_bias_score(data):
    # add perspecitive score + calculate total
    responses = list(data["responses"].values())
    ethics = [ethics_score(i) for i in responses]
    return ethics

def custom_score(data, weight):
    cluster_acc_scores = [0] * 4
    acc_scores = calculate_acc_score(data)
    coherence_scores = calculate_coherence_score(data)
    relevance_scores = calculate_relevance_score(data)
    creativity_scores = [0] * 4
    bias_scores = [0] * 4

    score = score = (
        weight[0] * np.array(cluster_acc_scores)
        + weight[1] * np.array(acc_scores)
        + weight[2] * np.array(coherence_scores)
        + weight[3] * np.array(relevance_scores)
        + weight[4] * np.array(creativity_scores)
        + weight[5] * np.array(bias_scores)
    )
    
    return score.tolist()

def calculate_all_scores(data):
    cluster_acc_scores = calculate_cluster_acc_score(data)
    acc_scores = calculate_acc_score(data)
    coherence_scores = calculate_coherence_score(data)
    relevance_scores = calculate_relevance_score(data)
    creativity_scores = calculate_creativity_score(data)
    bias_scores =  calculate_bias_score(data)

    # print("Cluster Accuracy Scores:", cluster_acc_scores)
    # print("Accuracy Scores:", acc_scores)
    # print("Coherence Scores:", coherence_scores)
    # print("Relevance Scores:", relevance_scores)
    # print("Creativity Scores:", creativity_scores)
    # print("Bias Scores:", bias_scores)

    my_dict = {}
    models = ["GPT-4o", "Gemini", "Claude 3.5 Sonnet", "Llama"]
    # need to change when changed to select only 1 category
    category = data["categories"][0]

    #print("CATEGORY:", category)

    #SOMETHINGS WRONG WITH CUSTOM SCORE
    if category == "Factual Knowledge":
        # weight = [0.2, 0.5, 0, 0.3, 0, 0]
        # custom = custom_score(data, weight)

        for i, model in enumerate(models):
            my_dict[model] = {
                "cluster_acc": cluster_acc_scores[i],
                "accuracy": acc_scores[i],
                "coherence": coherence_scores[i],
                "relevance": relevance_scores[i],
                "creativity": "Not Considered",
                "bias": "Not Considered",
                # "custom": custom[i]
            }           
    elif category == "Reasoning and Problem-Solving":
        # weight = [0.5, 0, 0.2, 0.2, 0.1, 0]
        # custom = custom_score(data, weight)

        for i, model in enumerate(models):
            my_dict[model] = {
                "cluster_acc": cluster_acc_scores[i],
                "accuracy": "Not Considered",
                "coherence": coherence_scores[i],
                "relevance": relevance_scores[i],
                "creativity": creativity_scores[i],
                "bias": "Not Considered",
                # "custom": custom[i]
            }  
    elif category == "Creative Writing":
        # weight = [0, 0, 0.2, 0.15, 0.6, 0.05]
        # custom = custom_score(data, weight)

        for i, model in enumerate(models):
            my_dict[model] = {
                "cluster_acc": "Not Considered",
                "accuracy": "Not Considered",
                "coherence": coherence_scores[i],
                "relevance": relevance_scores[i],
                "creativity": creativity_scores[i],
                "bias": bias_scores[i],
                # "custom": custom[i]
            }   
    elif category == "Language Understanding": # actually for Ethics
        # weight = [0, 0, 0.1, 0.1, 0, 0.8]
        # custom = custom_score(data, weight)

        for i, model in enumerate(models):
            my_dict[model] = {
                "cluster_acc": "Not Considered",
                "accuracy": "Not Considered",
                "coherence": coherence_scores[i],
                "relevance": relevance_scores[i],
                "creativity": "Not Considered",
                "bias": bias_scores[i],
                # "custom": custom[i]
            }

    return my_dict


@app.route("/api/accuracy", methods=["POST"])
def receive_data():
    data = request.json
    print("DATA HERE", data)
    scores = calculate_all_scores(data)
    if not scores:
        print("NO SCORES")
    print("SCORES HERE", scores)
    return jsonify({
        "message": scores
        })

@app.route("/api/score", methods=["GET"])
def get_score():
    # print("DICTIONARY SCORE", relevance_score)
    return jsonify({
        "message": relevance_score
    })

if __name__ == "__main__":
    app.run(debug=True, port=8080)
    