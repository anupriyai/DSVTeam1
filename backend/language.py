from transformers import pipeline

# Load models
def load_models():
    models = {
        "T5": pipeline("translation", model="t5-small"),
        "Summarization": pipeline("summarization", model="google/pegasus-xsum"),
    }
    print("Models loaded successfully!")
    return models

# Translation
def translate_text(models, text, source_lang="English", target_lang="French"):
    prompt = f"translate {source_lang} to {target_lang}: {text}"
    return models['T5'](prompt)[0]['translation_text']

# Summarization (Abstractive)
def summarize_text(models, text):
    return models['Summarization'](text, max_length=100, min_length=30)[0]['summary_text']

# Grammar Correction
def grammar_correction(models, text):
    prompt = f"fix grammar: {text}"
    return models['T5'](prompt)[0]['translation_text']

# Evaluation
def evaluate_llm_responses(models, llm_responses, reference_texts):
    scores = {
        "translation": [],
        "summarization": [],
        "grammar": []
    }

    # Compare translations
    for llm_response, reference in zip(llm_responses["translation"], reference_texts["translation"]):
        model_translation = translate_text(models, reference)
        # Compute BLEU score or other metric (code omitted for brevity)
        print(f"LLM Translation: {llm_response} | Model Translation: {model_translation}")

    # Compare summaries
    for llm_response, reference in zip(llm_responses["summarization"], reference_texts["summarization"]):
        abstractive_summary = summarize_text(models, reference)
        # Compute ROUGE score or other metric (code omitted for brevity)
        print(f"LLM Summary: {llm_response} | Abstractive Summary: {abstractive_summary}")

    # Compare grammar corrections
    for llm_response, reference in zip(llm_responses["grammar"], reference_texts["grammar"]):
        corrected_text = grammar_correction(models, reference)
        # Compute grammar error rate (code omitted for brevity)
        print(f"LLM Grammar Fix: {llm_response} | Corrected Text: {corrected_text}")

    return scores

if __name__ == "__main__":
    models = load_models()

    # Example Data
    llm_responses = {
        "translation": ["Bonjour, comment ça va ?"],
        "summarization": ["AI refers to the simulation of human intelligence."],
        "grammar": ["He go to school yesterday."]
    }
    reference_texts = {
        "translation": ["Hello, how are you?"],
        "summarization": ["Artificial intelligence refers to the simulation of human intelligence in machines."],
        "grammar": ["He went to school yesterday."]
    }

    # Evaluate LLM responses
    evaluate_llm_responses(models, llm_responses, reference_texts)
