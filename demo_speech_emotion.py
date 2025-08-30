# demo_speech_emotion.py
import librosa
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL = "superb/wav2vec2-base-superb-er"
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL)
model = AutoModelForAudioClassification.from_pretrained(MODEL)

# Load an audio file (replace with your own .wav)
file = "03-01-02-01-01-01-04.wav"
speech, sr = librosa.load(file, sr=16000)  # Resample to 16kHz

# Preprocess
inputs = feature_extractor(
    speech, sampling_rate=16000, return_tensors="pt", padding=True
)

# Run inference
with torch.no_grad():
    logits = model(**inputs).logits
predicted_id = torch.argmax(logits, dim=-1).item()
label = model.config.id2label[predicted_id]

print(f"Predicted Emotion: {label}")
