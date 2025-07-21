import googleapiclient.discovery
from googleapiclient.errors import HttpError
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import time

class YouTubeChatAnalyzer:
    def __init__(self, model_path, tokenizer_path, api_key):
        """
        Initializes the analyzer by loading the model, tokenizer, and setting up the YouTube API client.
        """
        self.model = load_model(model_path)
        with open(tokenizer_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        
        # Initialize the YouTube API client
        self.youtube = googleapiclient.discovery.build(
            "youtube", "v3", developerKey=api_key
        )

    def get_live_chat_id(self, video_id):
        """
        Gets the active live chat ID for a given video ID.
        Returns None if no active chat is found.
        """
        request = self.youtube.videos().list(
            part="liveStreamingDetails",
            id=video_id
        )
        response = request.execute()
        # Check for valid response and live chat details
        if not response.get('items') or 'liveStreamingDetails' not in response['items'][0] or 'activeLiveChatId' not in response['items'][0]['liveStreamingDetails']:
            return None
        return response['items'][0]['liveStreamingDetails']['activeLiveChatId']

    def get_chat_messages(self, live_chat_id, page_token=None):
        """
        Gets a batch of chat messages from the live chat.
        """
        request = self.youtube.liveChatMessages().list(
            liveChatId=live_chat_id,
            part="snippet,authorDetails",
            pageToken=page_token
        )
        return request.execute()

    def predict_sentiment(self, text):
        """
        Predicts the sentiment of a given text using the loaded model.
        """
        sequence = self.tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=120, padding='post', truncating='post')
        # Set verbose to 0 to prevent printing during prediction
        prediction = self.model.predict(padded_sequence, verbose=0) 
        sentiment_labels = ['Negative', 'Neutral', 'Positive']
        return sentiment_labels[prediction.argmax()]

    def analyze_chat(self, video_id):
        """
        A generator function that yields formatted chat messages with their sentiment.
        Handles API errors and rate limiting.
        """
        live_chat_id = self.get_live_chat_id(video_id)
        if not live_chat_id:
            yield "🔴 **Error:** Could not find an active live chat for this video. It might be a regular video or the stream may have ended."
            return

        page_token = None
        while True:
            try:
                chat_response = self.get_chat_messages(live_chat_id, page_token)
                for item in chat_response['items']:
                    message = item['snippet']['displayMessage']
                    author = item['authorDetails']['displayName']
                    sentiment = self.predict_sentiment(message)
                    yield f"**{author}:** {message} *(Sentiment: {sentiment})*"

                page_token = chat_response.get('nextPageToken')
                if not page_token:
                    yield "--- End of chat stream ---"
                    break
                
                # Wait for a few seconds before the next request to avoid rate limiting
                time.sleep(10)

            except HttpError as e:
                # If we hit a rate limit error, wait longer and continue
                if e.resp.status == 403:
                    yield "⏳ Rate limit reached. Pausing for 10 seconds..."
                    time.sleep(10)
                    continue
                else:
                    yield f"🔴 **API Error:** {e}"
                    break
