# speech_emotion.py
import sounddevice as sd
import torch
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification

# --- Model Loading ---
print("Initializing... ⏳")
MODEL_NAME = "superb/wav2vec2-base-superb-er"

print(f"Loading feature extractor: {MODEL_NAME}")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

print(f"Loading audio classification model: {MODEL_NAME}")
model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
print("Model loaded successfully! ✅")

def record_audio(duration=5, samplerate=16000):
    """Record audio and return as numpy array."""
    print(f"\n🎙️  Recording for {duration} seconds... Speak now!")
    # Record audio from the default microphone
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    # Wait for the recording to finish
    sd.wait()
    print("Recording finished. 👍")
    # Remove single-dimensional entries from the shape of an array.
    return np.squeeze(audio)

def get_speech_emotion(duration=3):
    """Record and return predicted speech emotion."""
    # Record audio
    audio = record_audio(duration)
    
    print("🧠  Analyzing emotion...")
    # Process the audio with the feature extractor
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    
    # Make a prediction with the model
    with torch.no_grad():
        logits = model(**inputs).logits
        
    # Get the predicted class index with the highest probability
    predicted_class = torch.argmax(logits, dim=-1).item()
    
    print("Analysis complete.")
    # Return the corresponding emotion label
    return model.config.id2label[predicted_class]

if __name__ == "__main__":
    print("\n--- Live Speech Emotion Recognition ---")
    while True:
        # Get the predicted emotion
        emotion = get_speech_emotion()
        # Print the final result
        print(f"\nEmotion Detected: {emotion.upper()}")
        print("-" * 35)
        
        # Ask the user to continue or quit
        if input("Press Enter to record again or 'q' to quit: ").lower() == "q":
            print("Exiting program. Goodbye! 👋")
            break