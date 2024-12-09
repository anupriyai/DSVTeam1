import json
import os
import numpy as np
from textblob import TextBlob
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from the .env file
load_dotenv()
# Ensure the API key is set correctly
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is missing in the .env file")
# Create OpenAI client instance
client = OpenAI(api_key=api_key)

# Analyze feedback and assign a score based on sentiment polarity
def analyze_feedback(feedback):
    sentiment_scores = []
    for feedback_text in feedback:
        sentiment = TextBlob(feedback_text).sentiment.polarity
        if sentiment > 0.1:
            score = 6
        elif sentiment < -0.1:
            score = 1
        else:
            score = 3
        sentiment_scores.append(score)
    return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
# Generate GPT-4 Response Based on Writing Prompt
def generate_gpt4_output(writing_prompt, gpt4_model="gpt-4"):
    try:
        completion = client.chat.completions.create(
            model=gpt4_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": writing_prompt}
            ]
        )
        response_text = completion.choices[0].message.content.strip()
        return response_text
    except Exception as e:
        print(f"Error generating output with GPT-4: {e}")
        return ""

def generate_gpt4_score(text, score_criteria="creativity"):
    # Modify the prompt to ensure GPT-4 outputs a numerical score
    prompt = f"Rate the following text on a scale of 1 to 6 for {score_criteria} (Please provide only a number between 1 and 6):\n\n{text}"
    
    # Get the GPT-4 output
    response = generate_gpt4_output(prompt)
    
    try:
        # Extract the numeric value from the response
        score = float(response.strip())
        
        # Check if the score is in the expected range (1 to 6)
        if score < 1 or score > 6:
            print(f"Score out of range: {score}")
            return 0  # Default to 0 if the score is out of range
        
        return score
    
    except ValueError:
        # print(f"Invalid response: {response}")
        return 0  # Default to 0 if the response is not a valid number

# Calculate Weighted Score with Normalization
def calculate_weighted_score(gpt4_score, feedback_score):
    weights = {
        'gpt4': 0.65,
        'feedback': 0.35
    }

    # Normalize the scores to be between 0 and 1
    normalized_gpt4_score = gpt4_score / 6.0 
    normalized_feedback_score = feedback_score / 6.0

    # Calculate the final weighted score
    weighted_score = (
        weights['gpt4'] * normalized_gpt4_score +
        weights['feedback'] * normalized_feedback_score
    )

    return weighted_score

# Prepare features and labels for training
def prepare_data(data_file):
    features = []
    labels = []
    with open(data_file, 'r') as f:
        data = json.load(f)
    for entry_id, entry in data.items():
        feedback = entry.get('feedback', {})
        if not feedback:
            print(f"Skipping entry {entry_id} due to missing feedback.")
            continue
        gpt4_feedback = feedback.get('gpt4_feedback', '')
        human_feedback = feedback.get('human_feedback', '')
        feedback_analysis_score = analyze_feedback([gpt4_feedback, human_feedback])

        # Get GPT-4 score (use GPT-4 to generate score based on creativity)
        gpt4_score = generate_gpt4_score(entry.get("writing_prompt", ""), score_criteria="creativity")

        # Prepare the feature vector (gpt4 score, feedback score, etc.)
        feature_vector = [
            gpt4_score,
            feedback_analysis_score
        ]

        # Calculate the final score using the weighted calculation (this is your label)
        final_score = calculate_weighted_score(gpt4_score, feedback_analysis_score)

        # Append features and labels
        features.append(feature_vector)
        labels.append(final_score)
    return np.array(features), np.array(labels)
# Calculate Weighted Score
def calculate_weighted_score(gpt4_score, feedback_score):
    weights = {'gpt4': 0.65, 'feedback': 0.35}
    normalized_gpt4_score = gpt4_score / 6.0
    normalized_feedback_score = feedback_score / 6.0
    return (
        weights['gpt4'] * normalized_gpt4_score +
        weights['feedback'] * normalized_feedback_score
    )
# Train the model
def train_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error on Test Set: {mse}")
    return model
# Generate creativity scores for new responses
def generate_creativity_scores(responses, model):
    new_features = []
    for response in responses:
        gpt4_score = generate_gpt4_score(response)
        feedback_analysis_score = analyze_feedback([response])
        feature_vector = [gpt4_score, feedback_analysis_score]
        new_features.append(feature_vector)
    predicted_scores = model.predict(np.array(new_features)).tolist()
    return predicted_scores
# Example usage with Flask integration
if __name__ == "__main__":
    # Training the model using data from JSON
    data_file_path = 'backend/creative_writing_prompts.json'
    features, labels = prepare_data(data_file_path)
    trained_model = train_model(features, labels)
    # Test responses
    test_responses = [
        "The ocean is a vast blue expanse filled with mystery and life.",
        "A creative idea is one that is novel and inspiring.",
        "The bird soared high above the clouds, singing joyfully."
    ]
