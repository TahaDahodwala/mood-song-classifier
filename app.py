import pickle
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import random

data = pd.read_csv('F:/TAHA/ML/SPOTIFY SONG RECOMMENDER/dataset.xls')

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

mood_features = {
    "happy": {
        "danceability": (0.7, 1.0),
        "energy": (0.7, 1.0),
        "loudness_db": (-5, -2),
        "acousticness": (0.0, 0.3),
        "valence": (0.7, 1.0),
        "tempo_bpm": (0.6, 1.0)
    },
    "sad": {
        "danceability": (0.0, 0.4),
        "energy": (0.0, 0.4),
        "loudness_db": (-12, -6),
        "acousticness": (0.6, 1.0),
        "valence": (0.0, 0.3),
        "tempo_bpm": (0.0, 0.3)
    },
    "motivated":
    {
        "danceability": (0.6, 0.9),
        "energy": (0.8, 1.0),
        "loudness_db": (-5, -1),
        "acousticness": (0.0, 0.3),
        "valence": (0.5, 0.9),
        "tempo_bpm": (0.6, 0.9)
    },

    "angry": {
        "danceability": (0.4, 0.7),
        "energy": (0.7, 1.0),
        "loudness_db": (-6, -2),
        "acousticness": (0.0, 0.2),
        "valence": (0.3, 0.6),
        "tempo_bpm": (0.7, 1.0)
    },
    "calm": {
        "danceability": (0.4, 0.6),
        "energy": (0.2, 0.5),
        "loudness_db": (-12, -6),
        "acousticness": (0.6, 1.0),
        "valence": (0.4, 0.6),
        "tempo_bpm": (0.3, 0.6)
    },
    "neutral": {
        "danceability": (0.4, 0.6),
        "energy": (0.4, 0.6),
        "loudness_db": (-8, -5),
        "acousticness": (0.3, 0.6),
        "valence": (0.4, 0.6),
        "tempo": (0.4, 0.6)
    }
}

mood_vectors = {
    "happy" : [0.8, 0.8, -0.5, 0.2, 0.9, 130],
    "sad" : [0.2, 0.3, -12, 0.8, 0.2, 70],
    "motivated" : [0.7, 0.85, -4, 0.1, 0.7, 140],
    "calm" : [0.3, 0.2, -15, 0.9, 0.5, 60],
    "angry" : [0.5, 0.9, -3, 0.1, 0.2, 150],
    "neutral" : [0.5, 0.5, -9, 0.5, 0.5, 100],
}

def detect_mood(text):
    text = text.lower()
    if "sad" in text:
        return "sad"
    elif "happy" in text or "energetic" in text:
        return "happy"
    elif "calm" in text or "relaxed" in text:
        return "calm"
    elif "angry" in text or "furious" in text:
        return "angry"
    elif "pumped" in text or "motivated" in text:
        return "motivated"
    else:
        return "neutral"

feature_columns = ['danceability', 'energy', 'loudness', 'acousticness', 'valence', 'tempo']
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[feature_columns])
model.fit(data_scaled)
data['cluster'] = model.labels_

def recommend_songs(user_input, model, data_original, scaler):
    # Define 6 relevant features
    feature_columns = ['danceability', 'energy', 'loudness', 'acousticness', 'valence', 'tempo']

    # Validate required columns
    required_columns = feature_columns + ['track_id', 'track_name', 'artists']
    for col in required_columns:
        if col not in data_original.columns:
            raise ValueError(f"Missing column: {col} in dataset")

    # Detect mood
    mood = detect_mood(user_input)
    print(f"Detected mood: {mood}")

    # Create mood vector
    features = mood_vectors[mood]
    mood_vector_df = pd.DataFrame([features], columns=feature_columns)

    # Scale mood vector
    mood_vector_scaled = scaler.transform(mood_vector_df)

    # Predict cluster
    cluster_label = model.predict(mood_vector_scaled)[0]

    # Copy original data with cluster labels
    data_with_clusters = data_original.copy()
    data_with_clusters['cluster'] = model.labels_

    # Filter songs in the same cluster
    cluster_songs = data_with_clusters[data_with_clusters['cluster'] == cluster_label]

    if cluster_songs.empty:
        print("No songs found in the predicted cluster.")
        return pd.DataFrame(columns=['track_name', 'artists'])

    # Scale features of the cluster songs
    cluster_features_scaled = scaler.transform(cluster_songs[feature_columns])

    # Compute distances
    distances = euclidean_distances(cluster_features_scaled, mood_vector_scaled).flatten()

    # Sort by closest distances
    sorted_indices = distances.argsort()

    # Fetch more than needed to handle duplicates
    top_n = 150
    closest_songs = cluster_songs.iloc[sorted_indices[:top_n]]

    unique_recommendations = cluster_songs.drop_duplicates(subset=['track_name', 'artists']).sample(frac = 1).reset_index(drop = True)

    # Prepare final result
    results_df = unique_recommendations[['track_id','track_name', 'artists']].head(5)
    return results_df

# Title
st.title("🎧 Mood-Based Song Recommender")

# Text input or mood selection
user_mood = st.text_input("What's your mood right now?", "")

# When user enters mood and hits "Recommend"
if user_mood:
    all_recs = recommend_songs(user_mood, model, data, scaler)

    # Drop duplicates
    all_recs = all_recs.drop_duplicates(subset='track_id')

    # Randomize and limit to 10 each time
    num_songs_to_show = 10
    final_recommendations = all_recs.sample(n=min(num_songs_to_show, len(all_recs)))

    # Refresh button
    if st.button("🔁 Refresh Recommendations"):
        final_recommendations = all_recs.sample(n=min(num_songs_to_show, len(all_recs)))

    # Display recommendations
    st.subheader("Recommended Tracks 🎶")
    for index, row in final_recommendations.iterrows():
        st.markdown(f"{row['track_name']} - {row['artists']}")

        # Spotify Embed using track_id
        track_id = row['track_id']
        embed_url = f"https://open.spotify.com/embed/track/{track_id}"
        st.components.v1.html(
            f"""
            <iframe style="border-radius:12px" 
                    src="{embed_url}" 
                    width="80%" height="80" frameBorder="0" 
                    allowfullscreen 
                    allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" 
                    loading="lazy">
            </iframe>
            """,
            height=100
        )


