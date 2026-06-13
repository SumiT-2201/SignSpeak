#!/usr/bin/env python3
"""
SignSpeak Instant Data Collector
Captures hand landmarks INSTANTLY as soon as a hand is detected in frame.
No stability checks, no timers — pure speed.
Automatically cycles through all signs after 100 samples each.
"""

import os
import sys
import pandas as pd
import cv2
import time
import numpy as np
import mediapipe as mp
import pygame

# ============================================================
# Configuration
# ============================================================

LABELS = [
    "hello", "yes", "no", "thank_you", "i_love_you",
    "please", "sorry", "good", "help", "stop"
]

CSV_FILE_PATH = "data/landmarks.csv"
SAMPLES_PER_SIGN = 100
FLASH_DURATION = 0.2  # Seconds for green border flash
CAPTURE_COOLDOWN = 0.08  # Small cooldown between captures to avoid duplicates

# ============================================================
# CSV Initialization
# ============================================================

def init_csv_file():
    """Creates the CSV file with headers if it doesn't exist."""
    dir_name = os.path.dirname(CSV_FILE_PATH)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)

    if not os.path.exists(CSV_FILE_PATH) or os.path.getsize(CSV_FILE_PATH) == 0:
        headers = []
        for i in range(21):
            headers.extend([f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z"])
        headers.extend(["label_index", "label_name"])

        df = pd.DataFrame(columns=headers)
        df.to_csv(CSV_FILE_PATH, index=False)
        print(f"Initialized CSV: {CSV_FILE_PATH}")

# ============================================================
# Sound
# ============================================================

def init_sound():
    """Initialize pygame mixer and generate a short beep sound buffer."""
    try:
        pygame.mixer.init(frequency=22050, size=-16, channels=1)
        duration = 0.08
        freq = 1200
        sr = 22050
        n = int(duration * sr)
        t = np.linspace(0, duration, n, endpoint=False)
        wave = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
        return pygame.sndarray.make_sound(wave)
    except Exception:
        return None

def play_beep(sound):
    """Play the pre-generated beep sound."""
    if sound:
        try:
            sound.play()
        except Exception:
            pass

# ============================================================
# Main Loop
# ============================================================

def main():
    init_csv_file()
    pygame.init()
    beep_sound = init_sound()

    # MediaPipe setup
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)

    # State
    current_label_idx = 0
    sample_count = 0
    total_all = 0
    last_capture_time = 0.0
    flash_end_time = 0.0
    finished = False

    window_name = "SignSpeak Instant Collector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n" + "=" * 55)
    print("  SignSpeak INSTANT Data Collector")
    print("  Hand visible = instant capture. No waiting.")
    print("  SPACE = skip to next sign  |  Q = quit & save")
    print("=" * 55 + "\n")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            now = time.time()

            # MediaPipe processing
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            hand_detected = False
            flat_landmarks = None

            if results.multi_hand_landmarks:
                hand_detected = True
                hand_lm = results.multi_hand_landmarks[0]

                # Draw skeleton
                mp_drawing.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Flatten 21 landmarks to 63 values
                flat_landmarks = []
                for lm in hand_lm.landmark:
                    flat_landmarks.extend([lm.x, lm.y, lm.z])

            # ---- INSTANT CAPTURE ----
            if (hand_detected
                    and flat_landmarks
                    and not finished
                    and (now - last_capture_time) > CAPTURE_COOLDOWN):

                label_name = LABELS[current_label_idx]
                row = flat_landmarks + [str(current_label_idx), label_name]

                try:
                    df = pd.DataFrame([row])
                    df.to_csv(CSV_FILE_PATH, mode="a", header=False, index=False)
                except Exception as e:
                    print(f"CSV write error: {e}")

                sample_count += 1
                total_all += 1
                last_capture_time = now
                flash_end_time = now + FLASH_DURATION
                play_beep(beep_sound)

                # Auto-advance after reaching sample target
                if sample_count >= SAMPLES_PER_SIGN:
                    print(f"  [DONE] '{label_name}' -> {sample_count} samples")
                    current_label_idx += 1
                    sample_count = 0

                    if current_label_idx >= len(LABELS):
                        finished = True
                        print("\nAll signs captured! You can press Q to quit.")

            # ============================================================
            # UI RENDERING
            # ============================================================

            # Dark translucent header bar
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 130), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

            if finished:
                # Completion screen
                cv2.putText(frame, "ALL SIGNS CAPTURED!", (w // 2 - 200, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, f"Total samples: {total_all}", (w // 2 - 100, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(frame, "Press Q to quit and save", (w // 2 - 130, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
            else:
                current_sign = LABELS[current_label_idx].upper().replace("_", " ")
                next_sign = LABELS[current_label_idx + 1].upper().replace("_", " ") if current_label_idx + 1 < len(LABELS) else "DONE"

                # Title
                cv2.putText(frame, "SignSpeak Instant Collector", (15, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

                # Current sign to capture
                cv2.putText(frame, f"Current Sign to Capture: {current_sign}", (15, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

                # Progress counter
                cv2.putText(frame, f"Samples collected: {sample_count}/{SAMPLES_PER_SIGN}", (15, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

                # Progress bar
                bar_x, bar_y, bar_w, bar_h = 15, 108, 250, 12
                progress = sample_count / SAMPLES_PER_SIGN
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h), (0, 255, 0), -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (150, 150, 150), 1)

                # Next sign preview
                cv2.putText(frame, f"Next up: {next_sign}", (w - 250, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

                # Sign number indicator
                cv2.putText(frame, f"Sign {current_label_idx + 1}/{len(LABELS)}", (w - 150, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

            # Hand status indicator
            if not finished:
                if hand_detected:
                    cv2.putText(frame, "HAND DETECTED - CAPTURING", (w // 2 - 160, h - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "Show your hand to start capturing", (w // 2 - 180, h - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

            # Bottom bar with controls
            cv2.rectangle(frame, (0, h - 30), (w, h), (20, 20, 20), -1)
            cv2.putText(frame, f"SPACE: Skip Sign  |  Q: Quit & Save  |  Total: {total_all}",
                        (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

            # Green border flash on capture
            if now < flash_end_time:
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 255, 0), 6)

            # Show frame
            cv2.imshow(window_name, frame)

            # Key handling
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == ord('Q'):
                print(f"\nQuitting. Total samples saved: {total_all}")
                break

            if key == 32 and not finished:  # SPACE to skip
                print(f"  [SKIP] '{LABELS[current_label_idx]}' at {sample_count} samples")
                current_label_idx += 1
                sample_count = 0
                if current_label_idx >= len(LABELS):
                    finished = True
                    print("\nAll signs done! Press Q to quit.")

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    print("Resources released. Goodbye!")

if __name__ == "__main__":
    main()
