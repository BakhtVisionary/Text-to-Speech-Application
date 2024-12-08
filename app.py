# Set page configuration (MUST be the first Streamlit command)
import streamlit as st
st.set_page_config(
    page_title="Text-to-Speech App",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded",
)

import torch
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import os

# Define synthesizer
@st.cache_resource
def load_synthesizer():
    return pipeline("text-to-speech", "microsoft/speecht5_tts")

synthesiser = load_synthesizer()

@st.cache_resource
def load_embeddings_dataset():
    return load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

embeddings_dataset = load_embeddings_dataset()

# Streamlit UI
st.title("🎤 Killer Text-to-Speech Application")
st.markdown("Convert your text into realistic speech using state-of-the-art AI.")

# Sidebar for input
st.sidebar.header("Input Text")
user_input = st.sidebar.text_area("Enter the text you want to convert to speech:", "Hello, my dog is cooler than you!")

# Speaker selection
st.sidebar.header("Speaker Embedding")
selected_speaker = st.sidebar.slider(
    "Choose Speaker Embedding Index (0-100):",
    min_value=0,
    max_value=100,
    value=30,
)

# Generate Speech Button
if st.sidebar.button("Generate Speech"):
    st.write("🔄 Generating speech... Please wait.")

    try:
        # Load speaker embedding
        speaker_embedding = torch.tensor(embeddings_dataset[selected_speaker]["xvector"]).unsqueeze(0)

        # Generate speech
        speech = synthesiser(
            user_input, forward_params={"speaker_embeddings": speaker_embedding}
        )

        # Save and play audio
        output_path = "output_speech.wav"
        sf.write(output_path, speech["audio"], samplerate=speech["sampling_rate"])
        st.audio(output_path, format="audio/wav")
        st.success("✅ Speech generated successfully!")

    except Exception as e:
        st.error(f"Error: {str(e)}")

# Additional Features Section
st.sidebar.header("Features Coming Soon!")
st.sidebar.write("- **Upload your own speaker embedding**")
st.sidebar.write("- **Choose different languages**")
st.sidebar.write("- **Control pitch and speed**")

st.markdown(
    """
    ---
    ### Why Choose Our Text-to-Speech App?
    - 🧠 **AI-Powered**: Leverages Microsoft SpeechT5 for natural and expressive speech synthesis.
    - 🎨 **Customizable UI**: Choose your speaker and customize your experience.
    - 🚀 **Fast and Efficient**: Generate speech in seconds!
    """
)

# Clean up generated audio
if os.path.exists("output_speech.wav"):
    os.remove("output_speech.wav")
