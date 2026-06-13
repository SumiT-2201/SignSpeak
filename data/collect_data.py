#!/usr/bin/env python3
"""
Sign Language Data Collection and Landmark Extraction Script
This script opens the webcam feed, processes the frames using MediaPipe Hands to detect 
21 hand landmarks, and draws them on the screen.
When the user presses keys '0'-'9', the 63 landmark values (x, y, z for all 21 landmarks)
are extracted and saved along with the class label to data/landmarks.csv.
Press 'q' to quit the application.
"""

import os
import sys
import csv
import cv2
import numpy as np
import mediapipe as mp

# Label mapping for SignSpeak gestures
LABEL_MAPPING = {
    "0": "hello",
    "1": "yes",
    "2": "no",
    "3": "thank_you",
    "4": "i_love_you",
    "5": "please",
    "6": "sorry",
    "7": "good",
    "8": "help",
    "9": "stop"
}

CSV_FILE_PATH = "data/landmarks.csv"

def init_csv_file():
    """
    Initializes the CSV file if it doesn't already exist.
    Creates directory if needed and writes the header row.
    """
    # Ensure directory exists
    dir_name = os.path.dirname(CSV_FILE_PATH)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"Created directory: {dir_name}")

    # Write headers if file doesn't exist or is empty
    if not os.path.exists(CSV_FILE_PATH) or os.path.getsize(CSV_FILE_PATH) == 0:
        headers = []
        for i in range(21):
            headers.extend([f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z"])
        headers.extend(["label_index", "label_name"])

        try:
            with open(CSV_FILE_PATH, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            print(f"Initialized CSV file at: {CSV_FILE_PATH} with headers.")
        except Exception as e:
            print(f"Error initializing CSV file: {e}")
            sys.exit(1)

def main():
    # Initialize the landmarks CSV file
    init_csv_file()

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Open Webcam
    print("Opening webcam... (this might take a few seconds)")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam. Please check if another app is using it or if it is connected.")
        sys.exit(1)

    print("\n" + "="*50)
    print("SignSpeak-BRO Data Collector Initialized Successfully!")
    print("Instructions:")
    print("1. Place your hand in front of the camera.")
    print("2. Press keys 0-9 to record landmarks for that gesture:")
    for key, val in LABEL_MAPPING.items():
        print(f"   [{key}] -> {val}")
    print("3. Press 'q' or 'Q' to quit.")
    print("="*50 + "\n")

    # Set OpenCV window properties
    cv2.namedWindow("SignSpeak-BRO Data Collector", cv2.WINDOW_NORMAL)

    # State variables for screen notifications
    feedback_text = ""
    feedback_color = (0, 255, 0)
    feedback_frames = 0

    # Initialize MediaPipe Hands context
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from webcam. Exiting...")
                break

            # Mirror the frame horizontally for a more intuitive selfie-view
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Convert BGR to RGB for MediaPipe processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # To improve performance, optionally mark the image as not writeable
            rgb_frame.flags.writeable = False
            results = hands.process(rgb_frame)
            rgb_frame.flags.writeable = True

            # Draw UI information on frame
            cv2.putText(frame, "SignSpeak-BRO Data Collector", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Press 0-9 to record, Q to quit", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

            hand_detected = False
            current_landmarks = None

            # Render landmarks on the screen
            if results.multi_hand_landmarks:
                hand_detected = True
                # Focus on the first detected hand
                hand_landmarks = results.multi_hand_landmarks[0]
                current_landmarks = hand_landmarks.landmark

                # Draw the hand landmarks & connections
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                cv2.putText(frame, "Status: Hand Detected", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Status: No Hand Detected", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

            # Display visual feedback for captures or errors
            if feedback_frames > 0:
                cv2.putText(frame, feedback_text, (10, h - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, feedback_color, 2, cv2.LINE_AA)
                feedback_frames -= 1

            # Render the frame
            cv2.imshow("SignSpeak-BRO Data Collector", frame)

            # Handle keyboard inputs
            key = cv2.waitKey(1) & 0xFF
            
            # Quit script
            if key == ord('q') or key == ord('Q'):
                print("Exiting data collector...")
                break

            # Handle gesture logging (keys '0' - '9')
            key_char = chr(key) if 32 <= key < 127 else ""
            if key_char in LABEL_MAPPING:
                label_name = LABEL_MAPPING[key_char]
                
                if hand_detected and current_landmarks:
                    # Extract 21 landmarks (x, y, z) into a list of 63 values
                    landmark_list = []
                    for lm in current_landmarks:
                        landmark_list.extend([lm.x, lm.y, lm.z])
                    
                    # Append the label indices and names
                    row_data = landmark_list + [key_char, label_name]
                    
                    try:
                        with open(CSV_FILE_PATH, mode="a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(row_data)
                        
                        success_msg = f"Recorded: '{label_name}' (Key: {key_char})"
                        print(success_msg)
                        feedback_text = success_msg
                        feedback_color = (0, 255, 0)
                        feedback_frames = 30  # Display feedback for ~30 frames
                    except Exception as e:
                        err_msg = f"Failed to save landmark: {e}"
                        print(err_msg)
                        feedback_text = err_msg
                        feedback_color = (0, 0, 255)
                        feedback_frames = 45
                else:
                    warn_msg = "Warning: No hand detected! Position hand in view to record."
                    print(warn_msg)
                    feedback_text = "No hand detected!"
                    feedback_color = (0, 0, 255)
                    feedback_frames = 30

    # Cleanup resources
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released and window closed. Goodbye!")

if __name__ == "__main__":
    main()
