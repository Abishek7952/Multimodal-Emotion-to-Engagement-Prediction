import sounddevice as sd
import torch
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification

# Model name
MODEL_NAME = "superb/wav2vec2-base-superb-er"

# Load feature extractor & model (skip tokenizer)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)

# Function to record live audio
def record_audio(duration=10
                 , samplerate=16000):
    print(f"🎙️ Recording {duration} seconds...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    print("✅ Recording done!")
    return np.squeeze(audio)

# Function to predict emotion
def predict_emotion(audio):
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    label = model.config.id2label[predicted_class]
    return label

if __name__ == "__main__":
    while True:
        audio = record_audio()
        emotion = predict_emotion(audio)
        print(f"🎯 Predicted Emotion: {emotion}")
        cont = input("Press Enter to record again or type 'q' to quit: ")
        if cont.lower() == 'q':
            break
