#!/usr/bin/env python3
"""
Sign Language Recognition & Text-To-Speech (TTS)
This script loads the pre-trained gesture classifier from models/sign_model.pkl,
captures video from the webcam, processes hand landmarks in real-time,
displays predictions, and uses pyttsx3 to voice confirmed gestures if held for 1.5s.
"""

import os
import sys
import cv2
import time
import joblib
import threading
import numpy as np
# pyrefly: ignore [missing-import]
import mediapipe as mp
import pyttsx3

MODEL_PATH = "models/sign_model.pkl"
LOCK_IN_DURATION = 1.5  # Seconds to hold a gesture to speak it

# Thread-safe Speech synthesis helper
speech_lock = threading.Lock()

def speak(text):
    """Speaks the text in a separate background thread to avoid freezing the camera feed."""
    def _speak_thread():
        with speech_lock:
            try:
                # Re-initialize engine locally within the thread for safety
                engine = pyttsx3.init()
                engine.setProperty("rate", 150) # Set speed of speech
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"Text-to-speech error: {e}")
                
    t = threading.Thread(target=_speak_thread)
    t.daemon = True
    t.start()

def main():
    # 1. Load Pre-trained Sklearn Model
    if not os.path.exists(MODEL_PATH):
        print("\n" + "!"*60)
        print(f"Error: Model not found at '{MODEL_PATH}'")
        print("Please run 'python train_model.py' to train your model first.")
        print("!"*60 + "\n")
        sys.exit(1)

    print(f"Loading sklearn classifier from: {MODEL_PATH}...")
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)
        
    print("Model loaded successfully!")

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Open Webcam
    print("Opening webcam... (this might take a few seconds)")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)

    # State variables for Lock-in & UI
    prev_prediction = None
    lock_start_time = None
    locked_word = ""
    confirmed_word_display = ""
    confirmed_display_timer = 0  # Number of frames to highlight confirmed word

    print("\n" + "="*50)
    print("SignSpeak Real-Time Recognizer Initialized")
    print("Instructions:")
    print("1. Perform a gesture in front of the webcam.")
    print("2. Look at the real-time prediction and confidence bar.")
    print("3. Hold the gesture still for 1.5 seconds to lock it in.")
    print("4. The system will print, speak, and display the locked word.")
    print("5. Press 'q' to quit.")
    print("="*50 + "\n")

    cv2.namedWindow("SignSpeak Real-Time Recognizer", cv2.WINDOW_NORMAL)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Exiting...")
                break

            # Mirror frame horizontally for selfie view
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # RGB conversion for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = hands.process(rgb_frame)
            rgb_frame.flags.writeable = True

            current_prediction = None
            confidence = 0.0
            hand_detected = False

            # Draw hand overlays and run inference
            if results.multi_hand_landmarks:
                hand_detected = True
                hand_landmarks = results.multi_hand_landmarks[0]

                # Draw skeleton
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Extract landmarks
                flat_landmarks = []
                for lm in hand_landmarks.landmark:
                    flat_landmarks.extend([lm.x, lm.y, lm.z])
                
                # Reshape for single sample prediction
                X_sample = np.array(flat_landmarks).reshape(1, -1)

                try:
                    # Predict label
                    current_prediction = model.predict(X_sample)[0]
                    
                    # Estimate confidence probability
                    probabilities = model.predict_proba(X_sample)[0]
                    class_idx = np.where(model.classes_ == current_prediction)[0][0]
                    confidence = probabilities[class_idx]
                except Exception as e:
                    # If model was trained without probability estimates, or index issue occurs
                    # RandomForestClassifier defaults to having predict_proba
                    confidence = 1.0

            # 1.5-Second Lock-in logic
            countdown = LOCK_IN_DURATION
            if hand_detected and current_prediction:
                if current_prediction == prev_prediction:
                    if lock_start_time is None:
                        lock_start_time = time.time()
                    
                    elapsed = time.time() - lock_start_time
                    countdown = max(0.0, LOCK_IN_DURATION - elapsed)

                    if elapsed >= LOCK_IN_DURATION:
                        # Lock in the word!
                        locked_word = current_prediction
                        confirmed_word_display = locked_word.upper()
                        confirmed_display_timer = 45  # Show for 45 frames (approx 1.5s)
                        
                        print(f"[LOCKED] Locked In: '{locked_word}' (speaking...)")
                        speak(locked_word)
                        
                        # Reset lock-in timers so it doesn't repeatedly lock/speak
                        lock_start_time = None
                        prev_prediction = None
                else:
                    lock_start_time = time.time()
                    prev_prediction = current_prediction
            else:
                lock_start_time = None
                prev_prediction = None

            # --- HUD Rendering ---
            # Translucent black top banner
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 140), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

            cv2.putText(frame, "SignSpeak Real-Time Recognizer", (15, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

            if hand_detected and current_prediction:
                pred_text = current_prediction.upper()
                cv2.putText(frame, f"Prediction: {pred_text}", (15, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2, cv2.LINE_AA)

                # Confidence visual progress bar
                bar_width = 180
                bar_height = 15
                bar_x = 15
                bar_y = 75
                filled_width = int(bar_width * confidence)
                
                # Gray background bar
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
                # Colored filled bar
                bar_color = (0, 255, 0) if confidence > 0.75 else (0, 165, 255)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), bar_color, -1)
                # Percent text
                cv2.putText(frame, f"{confidence*100:.0f}% Match", (bar_x + bar_width + 10, bar_y + 12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

                # Speak lock countdown indicator
                if lock_start_time is not None:
                    # Draw visual circular/pie progress or numeric countdown
                    cv2.putText(frame, f"Speaking in: {countdown:.1f}s", (w - 200, 25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

            else:
                cv2.putText(frame, "Waiting for hand...", (15, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (100, 100, 255), 2, cv2.LINE_AA)

            # Prominently display confirmed (spoken) word
            if confirmed_display_timer > 0:
                # Highlighted lock-in block
                box_w = 400
                box_h = 60
                box_x = (w - box_w) // 2
                box_y = h - box_h - 30
                
                # Yellow-green border box
                cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (30, 30, 30), -1)
                cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 0), 2)
                
                msg = f"SPOKEN: {confirmed_word_display}"
                # Center-align text roughly
                text_sz = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
                text_x = box_x + (box_w - text_sz[0]) // 2
                text_y = box_y + (box_h + text_sz[1]) // 2
                
                cv2.putText(frame, msg, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
                confirmed_display_timer -= 1
            else:
                # Standard hint at bottom
                cv2.rectangle(frame, (0, h - 35), (w, h), (30, 30, 30), -1)
                cv2.putText(frame, "Hold gesture still for 1.5s to trigger Speech output | Press 'Q' to quit", (15, h - 12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

            cv2.imshow("SignSpeak Real-Time Recognizer", frame)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("Exiting recognizer...")
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Resources closed. Goodbye!")

if __name__ == "__main__":
    main()
