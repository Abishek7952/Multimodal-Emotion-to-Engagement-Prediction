# facial_emotion.py
import cv2
from deepface import DeepFace

def get_facial_emotion():
    """Capture one frame and return detected facial emotion."""
    print("\nAccessing webcam... 📸")
    # Open a connection to the default webcam (index 0)
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return "Webcam error"

    # Capture a single frame
    ret, frame = cap.read()
    
    # Release the webcam immediately after capturing the frame
    cap.release()
    print("Webcam released.")
    
    # Check if the frame was captured successfully
    if not ret:
        print("Error: Could not read frame from webcam.")
        return "Capture error"
    print("Frame captured successfully. 👍")

    try:
        print("🧠 Analyzing face for emotion...")
        # Note: The model is downloaded and built on the first run, which may take time.
        # We set enforce_detection=True to ensure it raises an error if no face is found.
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=True)
        print("Analysis complete.")
        # The result is a list containing a dictionary for each detected face
        return result[0]['dominant_emotion']
    except ValueError:
        # This error is raised by DeepFace when no face is detected in the image
        print("No face detected in the frame.")
        return "No face detected"
    except Exception as e:
        # Catch any other unexpected errors
        print(f"An unexpected error occurred: {e}")
        return "Analysis error"


if __name__ == "__main__":
    print("\n--- Live Facial Emotion Recognition ---")
    print("The program will capture a single photo from your webcam to analyze.")
    
    while True:
        emotion = get_facial_emotion()
        
        # Print the final result in a formatted way
        print("\n" + "-"*35)
        print(f"Detected Emotion: {emotion.upper()}")
        print("-"*35)
        
        # Ask the user for input to continue or quit
        if input("Press Enter to try again or 'q' to quit: ").lower() == 'q':
            break

    print("Exiting program. Goodbye! 👋")
    