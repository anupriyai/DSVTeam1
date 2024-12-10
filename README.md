
# **DSV

https://github.com/user-attachments/assets/67983239-1040-4078-9e4b-23bca13c0abb

Team1: LLM Evaluation Platform**

This project is designed to evaluate LLM outputs across various categories using a Flask backend and a Next.js frontend.

---

## **Steps to Run the Project**

### **1. Running the Flask Backend**
The Flask backend is located in the `backend` folder and runs via `server.py`.

#### Steps:
1. Navigate to the `backend` folder:
   ```bash
   cd backend
   ```
2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask server:
   ```bash
   python server.py
   ```
4. The backend should now be running on:
   ```
   http://127.0.0.1:8080
   ```

---

### **2. Running the Next.js Frontend**
The Next.js frontend is located in the `frontend` folder.

#### Steps:
1. Navigate to the `frontend` folder:
   ```bash
   cd frontend
   ```
2. Install the required Node.js dependencies:
   ```bash
   npm install
   ```
3. Run the development server:
   ```bash
   npm run dev
   ```
4. The frontend should now be running on:
   ```
   http://localhost:3000
   ```

---

## **Additional Steps**

### **3. Download Our Excel File w/ Prompts & Responses**
1. Go to (https://docs.google.com/spreadsheets/d/1P9wv6TKKD_AK9E9MiLZqJ-OR3KfeDzUA/edit?usp=sharing&ouid=104548229055294474288&rtpof=true&sd=true).
2. Download the spreadsheet as an .XLSX to the _backend folder of DSVTeam1_

### **4. First-Time Setup for `accuracy.py`**
1. Open the `accuracy.py` file in the `backend` folder.
2. Uncomment the `nltk.download()` line to download the tokenizer model:
   ```python
   nltk.download('punkt')  # Example line
   ```
3. Run the backend (`server.py`) once to ensure the model downloads successfully.
4. After the model is downloaded, **comment the line back** to avoid downloading it repeatedly:
   ```python
   # nltk.download("popular")
   # nltk.download("stopwords")
   # nltk.download('punkt_tab')
   # nltk.download('wordnet')
   # nltk.download('omw-1.4')
   ```

---

### **5. Download Wikipedia Dump for Accuracy Metric**
1. Download the Wikipedia dump files in XML format from [https://dumps.wikimedia.org/](https://dumps.wikimedia.org/).
2. Save the dump file in your local machine and note its path.
3. Open `accuracy.py` in the `backend` folder.
4. Update the function in `accuracy.py` to reference the path of your XML dump file:
   ```python
   dump_path = "/path/to/your/wikipedia_dump.xml"
   ```
5. Save the file and restart the Flask backend.

---

## **How the Model Works**

### **Technologies Used**
- **Programming Languages & Libraries:** Python, TypeScript, NLTK, Torch, SKLearn, TextBlob, OpenAI Moderation API, and Sentence Transformers.
- **Models & Frameworks:** BM25, RoBERTa Large Model, KMeans Clustering, Random Forest, Perplexity Scoring, and Cosine Similarity.
- **Datasets:** Wikipedia dump files for the accuracy metric.

---

### **Metric Descriptions**

1. **Accuracy Metric**  
   The accuracy metric uses Wikipedia dump files to determine the correctness of responses. BM25 and a lightweight BERT model identify the closest matches between evidence and the user’s claim, providing a confidence score for correctness.

2. **Clustering Accuracy**  
   KMeans clustering is applied to group different LLM responses and detect outliers. Responses that don’t relate well to others are marked as outliers, indicating poor alignment with the group.

3. **Coherence**  
   KMeans clustering is also used for coherence by grouping responses and identifying unrelated ones. Any response isolated from others is flagged as incoherent.

4. **Bias Metric**  
   The OpenAI Moderation API evaluates responses for bias, harmful content, or sensitive language. This ensures that the generated text is ethically appropriate.

5. **Creativity Metric**  
   The creativity metric leverages GPT-4 for scoring writing prompts, TextBlob for feedback sentiment analysis, and a Random Forest model to predict creativity scores dynamically by combining these features.

---

### **Testing the Application**
1. Access the frontend at `http://localhost:3000`.
2. Use the interface to input prompts and compare model outputs.
3. Ensure the backend is running to handle requests.
