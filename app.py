import streamlit as st
from youtube_chat_analyzer import YouTubeChatAnalyzer
import pickle

# --- PASTE YOUR YOUTUBE API KEY HERE ---
YOUTUBE_API_KEY = "AIzaSyCWBviJX89_7F94foFbJtrVDGS7kijaKEM"
# ------------------------------------

st.set_page_config(layout="wide")
st.title("YouTube Live Chat Sentiment Analysis")

if YOUTUBE_API_KEY == "YOUR_API_KEY_GOES_HERE":
    st.error("ERROR: Please replace 'YOUR_API_KEY_GOES_HERE' with your actual YouTube API Key in the app.py file.")
    st.stop()

# Load tokenizer and model
try:
    # --- CHANGE: Load the .h5 model file ---
    analyzer = YouTubeChatAnalyzer(
        model_path='sentiment_model.h5',
        tokenizer_path='tokenizer.pickle',
        api_key=YOUTUBE_API_KEY
    )
except FileNotFoundError as e:
    st.error(f"File not found: {e}. Please make sure 'sentiment_model.h5' and 'tokenizer.pickle' are in the same folder.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model or tokenizer: {e}")
    st.stop()

st.write("Enter the Video ID of a YouTube live stream to begin.")
video_id = st.text_input("YouTube Video ID:")

if st.button("Analyze Chat"):
    if not video_id:
        st.warning("Please enter a YouTube Video ID.")
    else:
        st.info("Fetching live chat... The analysis will appear below.")
        chat_container = st.empty()
        all_messages_display = ""
        try:
            for message in analyzer.analyze_chat(video_id):
                all_messages_display = message + "\n\n" + all_messages_display
                chat_container.markdown(all_messages_display)
        except Exception as e:
            st.error(f"An unexpected error occurred during analysis: {e}")