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

# Load environment variables from the .env file
load_dotenv()

# Ensure the API key is set correctly
api_key = os.getenv("OPENAI_API_KEY")
print("API Key loaded:", api_key is not None)

# Create OpenAI client instance
client = OpenAI(api_key=api_key)

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
data_file_path = 'backend/creative_writing_prompts.json'

# Prepare data (features and labels)
features, labels = prepare_data(data_file_path)

# Train the model
model = train_model(features, labels)

# List of strings to test the model
test_strings = [
    "No, not all apples are necessarily red. While every apple is a fruit and some fruits are red, this does not imply that all apples must be red. Apples can come in other colors, such as green or yellow.",
    "The area of a rectangle is calculated using the formula: Area=Length×Width. Substitute the given values: Area=10meters×4meters=40square meters. Thus, the area of the rectangle is 40 square meters.",
    "Sports are awesome."
]

# Preprocess this new data and feed it to the model like this:
new_features = []
for text in test_strings:
    # Generate GPT-4 output (using the same text as the prompt)
    gpt4_output = generate_gpt4_output(text)

    # Calculate feedback analysis score (mocking feedback for this example)
    feedback_analysis_score = analyze_feedback([text, text])  # Example feedback analysis

    # Generate GPT-4 score for creativity
    gpt4_score = generate_gpt4_score(text)

    # Prepare the feature vector
    feature_vector = [gpt4_score, feedback_analysis_score]
    new_features.append(feature_vector)

# Predict the creativity scores for the new responses
# new_predictions = model.predict(np.array(new_features)).tolist()  # Convert to list
# print("Predicted Creativity Scores:", new_predictions)



