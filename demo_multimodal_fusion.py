# fusion_emotion.py
print("Initializing Multimodal Emotion Fusion System... 🚀")
import time
# Note: Ensure you have your other files named correctly for these imports
from demo_speech_emotion import get_speech_emotion
from demo_facial_emotion import get_facial_emotion
print("Modules imported successfully. ✅")


def fuse_emotions(speech_emotion, facial_emotion):
    """
    Simple fusion logic with print statements explaining the decision.
    - Prioritizes facial emotion in case of disagreement.
    - Handles cases where one modality fails or returns an error.
    """
    # Check if the inputs are valid detections (not None or error messages)
    is_facial_valid = facial_emotion and "error" not in facial_emotion and "detected" not in facial_emotion
    is_speech_valid = speech_emotion and "error" not in speech_emotion

    if is_facial_valid and is_speech_valid:
        print("💡 Both facial and speech emotions were detected.")
        if facial_emotion.lower() == speech_emotion.lower():
            print(f"   - Agreement found: Both indicate '{facial_emotion}'.")
            return facial_emotion
        else:
            print(f"   - Disagreement: Facial is '{facial_emotion}', Speech is '{speech_emotion}'.")
            print("   - Prioritizing the facial emotion as the final result.")
            return facial_emotion  # Prioritize facial emotion in case of conflict
            
    elif is_facial_valid:
        print(f"💡 Only a valid facial emotion ('{facial_emotion}') was detected.")
        return facial_emotion
        
    elif is_speech_valid:
        print(f"💡 Only a valid speech emotion ('{speech_emotion}') was detected.")
        return speech_emotion
        
    else:
        print("🔌 Could not determine a valid emotion from either modality.")
        return "Undetermined"


def main():
    """Main loop to capture and fuse emotions."""
    print("\n" + "="*50)
    print("    Multimodal Emotion Recognition".center(50))
    print("="*50)
    
    while True:
        # --- Step 1: Data Acquisition ---
        print("\n--- [STEP 1: CAPTURING DATA] ---")
        # The print statements from your other scripts will show progress here
        speech = get_speech_emotion(duration=3)
        facial = get_facial_emotion()

        # --- Step 2: Fusion Logic ---
        print("\n--- [STEP 2: FUSING RESULTS] ---")
        fused = fuse_emotions(speech, facial)
        
        # --- Step 3: Final Report ---
        print("\n--- [FINAL REPORT] ---")
        print(f"    🎤 Speech Input: {speech.upper()}")
        print(f"    🙂 Facial Input: {facial.upper()}")
        print("-" * 28)
        print(f"    🔗 Fused Result: {fused.upper()}")
        print("-" * 28)
        
        # Pause before asking for input
        time.sleep(1)
        
        # Ask user to continue
        cont = input("\nPress Enter to run again or 'q' to quit: ")
        if cont.lower() == 'q':
            break
            
    print("\nProgram terminated. Goodbye! 👋")


if __name__ == "__main__":
    main()