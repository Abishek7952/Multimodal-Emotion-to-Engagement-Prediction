import threading
import queue
import time
import numpy as np

# Import your existing modules
from demo_speech_emotion import predict_speech_emotion
from demo_facial_emotion import predict_facial_emotion

# Shared queues for predictions
speech_queue = queue.Queue()
face_queue = queue.Queue()

# Thread target for speech emotion
def run_speech():
    for emotion, score in predict_speech_emotion():
        speech_queue.put((emotion, score))

# Thread target for facial emotion
def run_face():
    for emotion, score in predict_facial_emotion():
        face_queue.put((emotion, score))

# Fusion function
def fuse_emotions(speech_data, face_data):
    # Example: Weighted average based on confidence
    if not speech_data or not face_data:
        return speech_data or face_data
    
    speech_emotion, speech_score = speech_data
    face_emotion, face_score = face_data

    # Simple fusion
    if speech_emotion == face_emotion:
        return speech_emotion, (speech_score + face_score) / 2
    else:
        # Choose the one with higher confidence
        return max([speech_data, face_data], key=lambda x: x[1])

# Main fusion loop
def main():
    threading.Thread(target=run_speech, daemon=True).start()
    threading.Thread(target=run_face, daemon=True).start()

    speech_data, face_data = None, None

    while True:
        # Get latest predictions
        try:
            while not speech_queue.empty():
                speech_data = speech_queue.get_nowait()
            while not face_queue.empty():
                face_data = face_queue.get_nowait()
        except queue.Empty:
            pass

        # Fuse predictions
        if speech_data or face_data:
            final_emotion, confidence = fuse_emotions(speech_data, face_data)
            print(f"[FUSED] Emotion: {final_emotion} (Conf: {confidence:.2f})")

        time.sleep(0.5)

if __name__ == "__main__":
    main()
