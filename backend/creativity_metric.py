import json
import os
import openai
import numpy as np
from textblob import TextBlob
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables from the .env file
load_dotenv()

# Ensure the API key is set correctly
api_key = os.getenv("OPENAI_API_KEY")
print("API Key loaded:", api_key is not None)

# Create OpenAI client instance
client = OpenAI(api_key=api_key)

# Initialize sentence transformer model for semantic similarity calculation
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')  # A pre-trained model for text similarity

# Analyze feedback and assign a score based on sentiment polarity
def analyze_feedback(feedback):
    sentiment_scores = []
    for feedback_text in feedback:
        # Sentiment analysis using TextBlob
        sentiment = TextBlob(feedback_text).sentiment.polarity
        # Convert sentiment polarity to a score between 1 and 6
        if sentiment > 0.1:
            score = 6  # Positive feedback
        elif sentiment < -0.1:
            score = 1  # Negative feedback
        else:
            score = 3  # Neutral feedback
        sentiment_scores.append(score)
    return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

# Generate GPT-4 Response Based on Writing Prompt
def generate_gpt4_output(writing_prompt, gpt4_model="gpt-4"):
    try:
        # Sending the writing prompt to GPT-4
        completion = client.chat.completions.create(
            model=gpt4_model,  # Specify the GPT-4 model
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": writing_prompt}  # User's writing prompt
            ]
        )
        # Correct way to access the response content
        response_text = completion.choices[0].message.content.strip()
        return response_text
    except Exception as e:
        print(f"Error generating output with GPT-4: {e}")
        return ""

# Calculate Semantic Similarity Between Two Texts
def calculate_semantic_similarity(text1, text2):
    # Encode the texts into embeddings
    embeddings = similarity_model.encode([text1, text2])
    # Compute cosine similarity between the embeddings
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

# Calculate Weighted Score with Normalization
def calculate_weighted_score(human_score, gpt4_score, semantic_similarity, feedback_score):
    weights = {
        'human': 0.4,
        'gpt4': 0.25,
        'semantic_similarity': 0.10,
        'feedback': 0.10
    }

    # Normalize the scores to be between 0 and 1
    normalized_human_score = human_score / 6.0 
    normalized_gpt4_score = gpt4_score / 6.0
    normalized_semantic_similarity = max(0, min(1, float(semantic_similarity)))
    normalized_feedback_score = feedback_score / 6.0 

    # Calculate the final weighted score
    weighted_score = (
        weights['human'] * normalized_human_score +
        weights['gpt4'] * normalized_gpt4_score +
        weights['semantic_similarity'] * normalized_semantic_similarity +
        weights['feedback'] * normalized_feedback_score
    )

    return weighted_score

# Prepare features and labels for training
def prepare_data(data_file):
    features = []
    labels = []
    
    # Open the JSON data file
    with open(data_file, 'r') as f:
        data = json.load(f)

    for entry_id, entry in data.items():
        # Extract the feedback content
        feedback = entry.get('feedback', {})

        # If feedback is missing, skip this entry
        if not feedback:
            print(f"Skipping entry {entry_id} due to missing feedback.")
            continue

        gpt4_feedback = feedback.get('gpt4_feedback', '')
        human_feedback = feedback.get('human_feedback', '')

        # Analyze feedback
        feedback_analysis_score = analyze_feedback([gpt4_feedback, human_feedback])

        # Get human score (assuming it's part of the entry)
        human_score = entry.get("combined_evaluation", {}).get("human_score", 0)

        # Get GPT-4 score (assuming it's part of the entry)
        gpt4_score = entry.get("combined_evaluation", {}).get("gpt4_score", 0)

        # Use the writing prompt to generate GPT-4 output
        writing_prompt = entry.get("writing_prompt", "")
        gpt4_output = generate_gpt4_output(writing_prompt)

        # Calculate semantic similarity between generated response and another model's response or ideal output
        reference_output = "The expected or ideal response that you'd compare with."
        semantic_similarity = calculate_semantic_similarity(gpt4_output, reference_output)

        # Prepare the feature vector (writing prompt, human score, gpt4 score, etc.)
        feature_vector = [
            human_score,
            gpt4_score,
            semantic_similarity,
            feedback_analysis_score
        ]

        # Calculate the final score using the weighted calculation (this is your label)
        final_score = calculate_weighted_score(human_score, gpt4_score, semantic_similarity, feedback_analysis_score)

        # Append features and labels
        features.append(feature_vector)
        labels.append(final_score)

    return np.array(features), np.array(labels)

# Train the model using Random Forest Regressor
def train_model(features, labels):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Create and train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error on Test Set: {mse}")

    return model

# Example usage
data_file_path = 'C:/Users/19254/OneDrive/Desktop/DSV/DSVTeam1/backend/creative_writing_prompts.json'

# Prepare data (features and labels)
features, labels = prepare_data(data_file_path)

# Train the model
model = train_model(features, labels)

# Use the trained model to make predictions on new LLM responses (input list is from here!)
new_llm_responses = [
    {
        'writing_prompt': "Write a poem about the beauty of nature.",
        'feedback': {'gpt4_feedback': "It's a lovely and evocative poem.", 'human_feedback': "Beautiful and soothing imagery."},
        'human_score': 5,
        'gpt4_score': 4.5
    }
]

# Preprocess this new data and feed it to the model like this:
new_features = []
for response in new_llm_responses:
    writing_prompt = response['writing_prompt']
    gpt4_output = generate_gpt4_output(writing_prompt)

    # Calculate feedback analysis score
    feedback_analysis_score = analyze_feedback([response['feedback']['gpt4_feedback'], response['feedback']['human_feedback']])

    # Calculate semantic similarity (optional, depending on your use case)
    reference_output = "The expected or ideal response that you'd compare with."  # You can adjust as needed
    semantic_similarity = calculate_semantic_similarity(gpt4_output, reference_output)

    # Prepare the feature vector
    feature_vector = [
        response['human_score'],
        response['gpt4_score'],
        semantic_similarity,
        feedback_analysis_score
    ]
    new_features.append(feature_vector)

# Predict the creativity scores for the new responses
new_predictions = model.predict(np.array(new_features))
print("Predicted Creativity Scores:", new_predictions)
