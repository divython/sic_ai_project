# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 00:00:17 2023

@author: Divyanshu
"""

import re
import textblob


def clean_text(text):
    # Remove special characters, URLs, and non-alphanumeric characters
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text
import nltk

def tokenize_text(text):
    # Tokenize the text into words
    tokens = nltk.word_tokenize(text)
    return tokens
from nltk.corpus import stopwords

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    return tokens
from nltk.stem import WordNetLemmatizer

def lemmatize_words(tokens):
    lemmatizer = WordNetLemmatizer()
    # Lemmatize words
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return lemmatized_tokens
def preprocess_text(comment):
    cleaned_comment = clean_text(comment)
    tokens = tokenize_text(cleaned_comment)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_words(tokens)
    return ' '.join(tokens)

from textblob import TextBlob

def analyze_sentiment(comment):
    analysis = TextBlob(comment)
    # Assign a sentiment score
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'
def perform_sentiment_analysis(preprocessed_comments):
    sentiment_scores = {'positive': 0, 'neutral': 0, 'negative': 0}

    for comment in preprocessed_comments:
        sentiment = analyze_sentiment(comment)
        sentiment_scores[sentiment] += 1

    return sentiment_scores
import matplotlib.pyplot as plt

def plot_sentiment_distribution(sentiment_scores):
    labels = sentiment_scores.keys()
    sizes = [sentiment_scores[label] for label in labels]

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#66b3ff', '#99ff99', '#ff9999'])
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Sentiment Distribution of Comments')
    plt.show()

# Example usage
sentiment_scores = {'positive': 30, 'neutral': 50, 'negative': 20}
plot_sentiment_distribution(sentiment_scores)
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import streamlit as st
# Define the Streamlit app


# Add input for YouTube video URL

import streamlit as st
import re
from textblob import TextBlob
import matplotlib.pyplot as plt
from googleapiclient.discovery import build

# Function to extract video ID from YouTube URL
def extract_video_id(video_url):
    """
    Extracts the video ID from a YouTube video URL.

    Parameters:
        video_url (str): The YouTube video URL.

    Returns:
        str: The video ID.
    """
    video_id_match = re.match(r'^.*(?:youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|&v=)([^#\&\?\n]*).*', video_url)
    if video_id_match and len(video_id_match.groups()) > 0:
        return video_id_match.group(1)
    else:
        return None

# Function to fetch YouTube comments using the YouTube Data API
def fetch_youtube_comments(api_key, video_id):
    """
    Fetches comments for a YouTube video using the YouTube Data API.

    Parameters:
        api_key (str): The YouTube Data API key.
        video_id (str): The YouTube video ID for which to fetch comments.

    Returns:
        list: A list of comments for the specified video.
    """
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        textFormat="plainText"
    )

    while request:
        response = request.execute()

        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)

        request = youtube.commentThreads().list_next(request, response)

    return comments

# Function to analyze sentiment
def analyze_sentiment(comment):
    """
    Analyzes the sentiment of a given comment using TextBlob.

    Parameters:
        comment (str): The comment text to be analyzed.

    Returns:
        str: Sentiment label ('positive', 'negative', or 'neutral').
    """
    analysis = TextBlob(comment)
    
    # Analyze sentiment
    sentiment_polarity = analysis.sentiment.polarity
    
    # Assign a sentiment label based on polarity
    if sentiment_polarity > 0:
        return 'positive'
    elif sentiment_polarity < 0:
        return 'negative'
    else:
        return 'neutral'

# Function to perform sentiment analysis
def perform_sentiment_analysis(comments):
    """
    Performs sentiment analysis on a list of comments.

    Parameters:
        comments (list): A list of comments.

    Returns:
        dict: A dictionary containing sentiment labels and their corresponding counts.
    """
    sentiment_scores = {'positive': 0, 'negative': 0, 'neutral': 0}
    
    for comment in comments:
        sentiment_label = analyze_sentiment(comment)
        sentiment_scores[sentiment_label] += 1
    
    return sentiment_scores

# Function to plot sentiment distribution
def plot_sentiment_distribution(sentiment_scores):
    """
    Plots the sentiment distribution pie chart.

    Parameters:
        sentiment_scores (dict): A dictionary containing sentiment labels and their corresponding counts.
    """
    labels = sentiment_scores.keys()
    sizes = [sentiment_scores[label] for label in labels]

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#66b3ff', '#99ff99', '#ff9999'])
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Sentiment Distribution of Comments')
    st.pyplot()

# Define the Streamlit app
st.title('YouTube Video Analyzer')

# Add input for YouTube video URL
video_url = st.text_input('Enter YouTube Video URL:', key='video_url')

# Fetch comments and display
if video_url:
    # Extract video ID from the URL
    video_id = extract_video_id(video_url)

    if video_id:
        # Replace with your YouTube API key
        api_key = "AIzaSyCSSgeaDnJwCtV4iRHXXTFJWh93wMnaDBE"

        # Fetch comments using the API
        comments = fetch_youtube_comments(api_key, video_id)

        # Perform sentiment analysis
        sentiment_scores = perform_sentiment_analysis(comments)

        # Display sentiment distribution
        st.write('### Sentiment Distribution')
        plot_sentiment_distribution(sentiment_scores)

        # Display comments
        st.write('### Comments')
        for comment in comments:
            st.write('- ' + comment)
    else:
        st.write('Invalid YouTube video URL. Please enter a valid URL.')
