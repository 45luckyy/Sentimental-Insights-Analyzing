# YouTube Live Chat Sentiment Analysis

This project provides a real-time sentiment analysis tool for YouTube live stream chats. Using a deep learning model, it fetches chat messages as they happen and classifies them as **Positive**, **Neutral**, or **Negative**. The application is built with Streamlit, providing a simple and interactive web interface.

![Demo Screenshot](https://github.com/45luckyy/Sentimental-Insights-Analyzing/blob/main/Sentimental%20Insights%20Interface.png)

## Features
- **Real-Time Analysis**: Fetches and analyzes YouTube live chat messages in near real-time.
- **Deep Learning Model**: Utilizes a Bidirectional LSTM (Bi-LSTM) network for sentiment classification.
- **Pre-trained Word Embeddings**: Leverages GloVe embeddings to understand the semantic meaning of words.
- **Interactive Web Interface**: A user-friendly interface built with Streamlit to input a video ID and view results.
- **Easy Setup**: Clear instructions for setting up the environment and running the application.

## How It Works
1.  **User Input**: The user provides the Video ID of an active YouTube live stream.
2.  **API Communication**: The application uses the **YouTube Data API v3** to connect to the live stream and fetch chat messages.
3.  **Text Preprocessing**: Each incoming chat message is cleaned and tokenized using a pre-fitted Keras `Tokenizer`.
4.  **Sentiment Prediction**: The processed text is fed into the trained Bi-LSTM model (`sentiment_model.h5`), which outputs a sentiment prediction (Positive, Neutral, or Negative).
5.  **Display Results**: The message, its author, and the predicted sentiment are displayed on the Streamlit web interface.

## Model Details
- **Architecture**: The sentiment analysis model is a Sequential model built with Keras.
    - `Embedding` Layer (initialized with GloVe vectors)
    - `Bidirectional LSTM`
    - `Dropout` for regularization
    - `Bidirectional LSTM`
    - `Dropout`
    - `Dense` layers
    - `Softmax` activation for multi-class classification.
- **Training Data**: The model was trained on the [Reddit User Comments dataset](https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset), which contains comments labeled with positive, neutral, and negative sentiment.
- **Embeddings**: The model uses pre-trained **GloVe 6B 100d** word embeddings. These are not included in the repository and must be downloaded separately.

## File Structure

```
.
├── app.py                      # The main Streamlit web application
├── youtube_chat_analyzer.py    # Class for handling YouTube API and predictions
├── sentiment_model_training.py # Script to train the sentiment model
├── sentiment_model.h5          # The trained Keras model
├── tokenizer.pickle            # Keras Tokenizer object
├── requirements.txt            # Python dependencies
├── Reddit_Data.csv             # Raw dataset for training
└── Preprocessed_Reddit_Data2.csv # Cleaned and preprocessed dataset
```

---

## Setup and Installation

Follow these steps to set up and run the project on your local machine.

### 1. Prerequisites
- Python 3.8+
- `pip` package manager

### 2. Clone the Repository
```bash
git clone [(https://github.com/45luckyy/Sentimental-Insights-Analyzing)]
```

### 3. Install Dependencies
Install all the required Python packages using the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

### 4. Download GloVe Word Embeddings
The model requires pre-trained GloVe word vectors.
1.  Download the "GloVe 6B" vectors from the [official Stanford NLP website](https://nlp.stanford.edu/projects/glove/). The direct download link is: [glove.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip).
2.  Unzip the downloaded file.
3.  From the unzipped folder, copy the `glove.6B.100d.txt` file and place it in the root directory of this project.

### 5. Obtain a YouTube Data API Key
This application requires a YouTube Data API key to fetch chat messages.
1.  Go to the [Google Cloud Console](https://console.cloud.google.com/).
2.  Create a new project (or select an existing one).
3.  Enable the **"YouTube Data API v3"** for your project.
4.  Create credentials for a new **API Key**.
5.  Copy the generated API key.

### 6. Configure the Application
Open the `app.py` file and paste your API key into the `YOUTUBE_API_KEY` variable:

```python
# app.py

# --- PASTE YOUR YOUTUBE API KEY HERE ---
YOUTUBE_API_KEY = "YOUR_API_KEY_GOES_HERE" 
# ------------------------------------
```

## How to Run the Application

1.  Make sure you have completed all the setup steps above.
2.  Open your terminal, navigate to the project directory, and run the Streamlit app:

    ```bash
    streamlit run app.py
    ```
3.  Your web browser should automatically open with the application running. If not, the terminal will provide a local URL (usually `http://localhost:8501`) that you can open manually.

## Usage
1.  Find a YouTube video that is currently live streaming.
2.  Copy the **Video ID** from the YouTube URL. For example, if the URL is `https://www.youtube.com/watch?v=abcdef12345`, the Video ID is `abcdef12345`.
3.  Paste the Video ID into the input box in the web app.
4.  Click the **"Analyze Chat"** button to start seeing the real-time sentiment analysis.

## Retraining the Model
If you wish to retrain the model on different data or with different parameters:
1.  Ensure your dataset is prepared and clean.
2.  Make sure the `glove.6B.100d.txt` file is in the root directory.
3.  Run the training script:
    ```bash
    python sentiment_model_training.py
    ```
This will generate new `sentiment_model.h5` and `tokenizer.pickle` files.

## License
This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.
