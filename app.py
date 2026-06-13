#!/usr/bin/env python3
"""
SignSpeak Communicator (app.py)
This is the final application designed for non-speaking people to communicate using hand signs.
It tracks hand landmarks in real-time, recognizes sign gestures, builds sentences that display
at the bottom of the screen, speaks words out loud when locked-in, and allows saving or clearing sentences.
"""

import os
import sys
import cv2
import time
import joblib
import threading
import numpy as np
import mediapipe as mp
import pyttsx3
import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
from datetime import datetime

MODEL_PATH = "models/sign_model.pkl"
LOCK_IN_DURATION = 1.5  # Seconds to hold a gesture to speak/lock it
SAVE_FILE_PATH = "saved_sentences.txt"

# Dynamic application configurations
app_settings = {
    "confidence_threshold": 0.85,
    "sound_enabled": True,
    "auto_speak_enabled": True,
    "show_landmarks": True,
    "show_confidence_bar": True,
    "restart_camera": False,
    "reload_model": False
}

# Thread-safe Speech Synthesis helper
speech_lock = threading.Lock()

def speak(text):
    """Speaks the text in a separate background thread to avoid freezing the camera feed."""
    if not app_settings.get("sound_enabled", True):
        return
    def _speak_thread():
        with speech_lock:
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", 145)  # Natural speaking rate
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"Text-to-speech error: {e}")
                
    t = threading.Thread(target=_speak_thread)
    t.daemon = True
    t.start()

# ============================================================
# Welcome Guide Screen — Hand Icon Drawing Helpers
# ============================================================

# Color palette
ACCENT_BLUE = (180, 80, 20)     # BGR deep blue
ACCENT_GREEN = (60, 160, 30)    # BGR green
SKIN_COLOR = (180, 210, 235)    # BGR warm skin tone
SKIN_OUTLINE = (100, 130, 160)  # BGR skin outline
DARK_TEXT = (40, 40, 40)        # BGR near-black
LIGHT_GRAY = (180, 180, 180)

def _draw_palm(canvas, cx, cy, sz):
    """Draws a basic palm (filled ellipse) centered at (cx, cy) with scale sz."""
    pw = int(sz * 0.55)
    ph = int(sz * 0.65)
    cv2.ellipse(canvas, (cx, cy + int(sz * 0.1)), (pw, ph), 0, 0, 360, SKIN_COLOR, -1)
    cv2.ellipse(canvas, (cx, cy + int(sz * 0.1)), (pw, ph), 0, 0, 360, SKIN_OUTLINE, 2)

def _draw_finger(canvas, x1, y1, x2, y2, thickness=8):
    """Draws a single finger as a thick rounded line with a circle tip."""
    cv2.line(canvas, (x1, y1), (x2, y2), SKIN_COLOR, thickness + 4)
    cv2.line(canvas, (x1, y1), (x2, y2), SKIN_OUTLINE, 2)
    cv2.circle(canvas, (x2, y2), thickness // 2 + 1, SKIN_COLOR, -1)
    cv2.circle(canvas, (x2, y2), thickness // 2 + 1, SKIN_OUTLINE, 2)

def draw_thumbs_up(canvas, cx, cy, sz):
    """Thumbs up: fist with thumb pointing up."""
    s = sz // 2
    # Fist body
    fx, fy = cx, cy + int(s * 0.3)
    cv2.ellipse(canvas, (fx, fy), (int(s*0.5), int(s*0.4)), 0, 0, 360, SKIN_COLOR, -1)
    cv2.ellipse(canvas, (fx, fy), (int(s*0.5), int(s*0.4)), 0, 0, 360, SKIN_OUTLINE, 2)
    # Thumb pointing up
    _draw_finger(canvas, cx - int(s*0.15), cy + int(s*0.05), cx - int(s*0.15), cy - int(s*0.7), 7)

def draw_open_palm(canvas, cx, cy, sz):
    """Open palm: all 5 fingers spread out."""
    s = sz // 2
    _draw_palm(canvas, cx, cy, s)
    # Five fingers fanning out
    angles = [-55, -25, 0, 25, 50]
    lengths = [s*0.75, s*0.9, s*0.95, s*0.85, s*0.7]
    for i, (ang, length) in enumerate(zip(angles, lengths)):
        rad = np.radians(ang - 90)
        x2 = int(cx + np.cos(rad) * length)
        y2 = int(cy + int(s*0.1) + np.sin(rad) * length)
        base_x = int(cx + np.cos(rad) * s * 0.3)
        base_y = int(cy + int(s*0.1) + np.sin(rad) * s * 0.3)
        _draw_finger(canvas, base_x, base_y, x2, y2, 6)

def draw_wave(canvas, cx, cy, sz):
    """Waving hand: open palm tilted with motion lines."""
    s = sz // 2
    # Tilted palm
    pts = np.array([
        [cx - int(s*0.4), cy + int(s*0.35)],
        [cx + int(s*0.5), cy + int(s*0.2)],
        [cx + int(s*0.45), cy - int(s*0.15)],
        [cx - int(s*0.35), cy]
    ], np.int32)
    cv2.fillPoly(canvas, [pts], SKIN_COLOR)
    cv2.polylines(canvas, [pts], True, SKIN_OUTLINE, 2)
    # Fingers
    offsets = [(-0.2, -0.55), (0.0, -0.7), (0.2, -0.65), (0.38, -0.5)]
    for ox, oy in offsets:
        bx = int(cx + ox * s * 0.5)
        by = int(cy - int(s * 0.05))
        _draw_finger(canvas, bx, by, int(cx + ox * s), int(cy + oy * s), 5)
    # Motion arcs
    cv2.ellipse(canvas, (cx + int(s*0.7), cy - int(s*0.1)), (int(s*0.15), int(s*0.4)), 15, -40, 40, LIGHT_GRAY, 2)
    cv2.ellipse(canvas, (cx + int(s*0.85), cy - int(s*0.05)), (int(s*0.12), int(s*0.3)), 15, -30, 30, LIGHT_GRAY, 2)

def draw_peace(canvas, cx, cy, sz):
    """Peace / victory sign: index + middle finger up, rest curled."""
    s = sz // 2
    _draw_palm(canvas, cx, cy, s)
    # Index finger
    _draw_finger(canvas, cx - int(s*0.15), cy - int(s*0.15), cx - int(s*0.25), cy - int(s*0.8), 6)
    # Middle finger
    _draw_finger(canvas, cx + int(s*0.1), cy - int(s*0.2), cx + int(s*0.15), cy - int(s*0.85), 6)
    # Curled ring + pinky (small arcs)
    cv2.ellipse(canvas, (cx + int(s*0.3), cy + int(s*0.15)), (int(s*0.12), int(s*0.15)), 0, -120, 60, SKIN_OUTLINE, 2)
    cv2.ellipse(canvas, (cx + int(s*0.45), cy + int(s*0.25)), (int(s*0.1), int(s*0.12)), 0, -100, 50, SKIN_OUTLINE, 2)

def draw_fist(canvas, cx, cy, sz):
    """Closed fist."""
    s = sz // 2
    fw, fh = int(s * 0.6), int(s * 0.55)
    cv2.ellipse(canvas, (cx, cy), (fw, fh), 0, 0, 360, SKIN_COLOR, -1)
    cv2.ellipse(canvas, (cx, cy), (fw, fh), 0, 0, 360, SKIN_OUTLINE, 2)
    # Knuckle lines
    for i in range(4):
        kx = cx - int(s*0.3) + int(i * s * 0.2)
        cv2.line(canvas, (kx, cy - int(s*0.15)), (kx + int(s*0.05), cy - int(s*0.35)), SKIN_OUTLINE, 2)
    # Thumb wrap
    cv2.ellipse(canvas, (cx - int(s*0.45), cy + int(s*0.1)), (int(s*0.2), int(s*0.15)), 30, -60, 120, SKIN_OUTLINE, 2)

def draw_call_me(canvas, cx, cy, sz):
    """Call me / shaka: pinky + thumb out, others curled."""
    s = sz // 2
    _draw_palm(canvas, cx, cy, s)
    # Thumb out left
    _draw_finger(canvas, cx - int(s*0.35), cy + int(s*0.1), cx - int(s*0.75), cy + int(s*0.3), 6)
    # Pinky out right
    _draw_finger(canvas, cx + int(s*0.35), cy - int(s*0.1), cx + int(s*0.7), cy - int(s*0.6), 6)
    # Curled fingers
    for i, ox in enumerate([-0.1, 0.05, 0.2]):
        cv2.ellipse(canvas, (int(cx + ox*s), cy - int(s*0.25)), (int(s*0.08), int(s*0.12)), 0, -140, 40, SKIN_OUTLINE, 2)

def draw_point_up(canvas, cx, cy, sz):
    """Pointing up: index finger extended, others curled."""
    s = sz // 2
    _draw_palm(canvas, cx, cy, s)
    # Index finger straight up
    _draw_finger(canvas, cx - int(s*0.05), cy - int(s*0.2), cx - int(s*0.05), cy - int(s*0.9), 7)
    # Curled fingers
    for i, ox in enumerate([0.15, 0.3, 0.42]):
        cv2.ellipse(canvas, (int(cx + ox*s), cy - int(s*0.05)), (int(s*0.1), int(s*0.15)), 10, -130, 30, SKIN_OUTLINE, 2)

def draw_please(canvas, cx, cy, sz):
    """Fingers pressed together (prayer / please gesture)."""
    s = sz // 2
    # Two palms pressed together
    lx, rx = cx - int(s*0.2), cx + int(s*0.2)
    # Left palm
    cv2.ellipse(canvas, (lx, cy + int(s*0.1)), (int(s*0.3), int(s*0.5)), -5, 0, 360, SKIN_COLOR, -1)
    cv2.ellipse(canvas, (lx, cy + int(s*0.1)), (int(s*0.3), int(s*0.5)), -5, 0, 360, SKIN_OUTLINE, 2)
    # Right palm
    cv2.ellipse(canvas, (rx, cy + int(s*0.1)), (int(s*0.3), int(s*0.5)), 5, 0, 360, SKIN_COLOR, -1)
    cv2.ellipse(canvas, (rx, cy + int(s*0.1)), (int(s*0.3), int(s*0.5)), 5, 0, 360, SKIN_OUTLINE, 2)
    # Finger tips meeting at center top
    for i, oy in enumerate([-0.6, -0.45, -0.25]):
        y = int(cy + oy * s)
        cv2.line(canvas, (lx, y), (rx, y), SKIN_OUTLINE, 1)
    # Center seam
    cv2.line(canvas, (cx, cy - int(s*0.35)), (cx, cy + int(s*0.5)), SKIN_OUTLINE, 1)


# ============================================================
# Welcome Guide Screen
# ============================================================

def show_welcome_guide():
    """
    Renders a full-screen welcome/onboarding guide using OpenCV.
    Shows hand sign illustrations (drawn with shapes), gesture names, meanings,
    and waits for SPACE to proceed or Q to quit.
    Returns True to continue to webcam, False to quit.
    """
    # Create a 1280x720 white canvas
    W, H = 1280, 720
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255

    # Helper to center text
    def draw_centered_text(text, y, font, scale, color, thickness):
        size, _ = cv2.getTextSize(text, font, scale, thickness)
        tx = (W - size[0]) // 2
        cv2.putText(canvas, text, (tx, y), font, scale, color, thickness, cv2.LINE_AA)

    # --- Title Section ---
    draw_centered_text("SignSpeak - Learn Hand Signs", 65, cv2.FONT_HERSHEY_DUPLEX, 1.4, ACCENT_BLUE, 3)
    # Decorative underline
    cv2.line(canvas, (W // 2 - 280, 85), (W // 2 + 280, 85), ACCENT_BLUE, 2)

    # --- Column Headers ---
    header_y = 130
    cv2.putText(canvas, "HAND SIGN", (100, header_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, LIGHT_GRAY, 1, cv2.LINE_AA)
    cv2.putText(canvas, "GESTURE", (260, header_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, LIGHT_GRAY, 1, cv2.LINE_AA)
    cv2.putText(canvas, "MEANING", (500, header_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, LIGHT_GRAY, 1, cv2.LINE_AA)
    # Right column
    cv2.putText(canvas, "HAND SIGN", (720, header_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, LIGHT_GRAY, 1, cv2.LINE_AA)
    cv2.putText(canvas, "GESTURE", (880, header_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, LIGHT_GRAY, 1, cv2.LINE_AA)
    cv2.putText(canvas, "MEANING", (1100, header_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, LIGHT_GRAY, 1, cv2.LINE_AA)
    cv2.line(canvas, (60, header_y + 10), (W - 60, header_y + 10), (220, 220, 220), 1)

    # --- Gesture Data (split into 2 columns of 4) ---
    gestures = [
        (draw_thumbs_up, "Thumbs Up",     "Good/Yes"),
        (draw_open_palm,  "Open Palm",     "Stop/Wait"),
        (draw_wave,       "Wave Hand",     "Hello"),
        (draw_peace,      "Two Fingers",   "Peace"),
        (draw_fist,       "Closed Fist",   "No"),
        (draw_point_up,   "Point Up",      "I Agree"),
        (draw_call_me,    "Pinky+Thumb",   "Call Me"),
        (draw_please,     "Flat Hand",     "Please"),
    ]

    row_height = 110
    start_y = 195
    icon_size = 70

    for i, (draw_fn, name, meaning) in enumerate(gestures):
        col = i // 4  # 0 or 1
        row = i % 4
        
        # Column offsets
        col_x_offset = col * 620
        icon_cx = 130 + col_x_offset
        icon_cy = start_y + row * row_height
        text_x_name = 260 + col_x_offset
        text_x_meaning = 500 + col_x_offset

        # Draw the hand icon
        draw_fn(canvas, icon_cx, icon_cy, icon_size)

        # Gesture name in accent color
        cv2.putText(canvas, name, (text_x_name, icon_cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, ACCENT_BLUE, 2, cv2.LINE_AA)
        
        # Arrow
        arrow_x = text_x_meaning - 30
        cv2.arrowedLine(canvas, (arrow_x, icon_cy), (arrow_x + 18, icon_cy), ACCENT_GREEN, 2, tipLength=0.4)

        # Meaning text
        cv2.putText(canvas, meaning, (text_x_meaning, icon_cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, DARK_TEXT, 1, cv2.LINE_AA)

        # Subtle row separator
        sep_y = icon_cy + int(row_height * 0.45)
        if row < 3:
            cv2.line(canvas, (70 + col_x_offset, sep_y), (580 + col_x_offset, sep_y), (235, 235, 235), 1)

    # --- Confidence Info Box ---
    info_y = start_y + 4 * row_height + 10
    cv2.rectangle(canvas, (W // 2 - 200, info_y - 5), (W // 2 + 200, info_y + 30), (240, 248, 255), -1)
    cv2.rectangle(canvas, (W // 2 - 200, info_y - 5), (W // 2 + 200, info_y + 30), ACCENT_BLUE, 1)
    cv2.putText(canvas, "Confidence Required: 85%  |  Hold: 1.5 sec", (W // 2 - 185, info_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ACCENT_BLUE, 1, cv2.LINE_AA)

    # --- Footer Instructions ---
    footer_y = H - 45
    draw_centered_text("Press SPACE to Start | Press Q to Quit", footer_y, cv2.FONT_HERSHEY_SIMPLEX, 0.75, ACCENT_GREEN, 2)

    # --- Show Guide Window ---
    guide_window = "SignSpeak - Learn Hand Signs"
    cv2.namedWindow(guide_window, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(guide_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(guide_window, canvas)

    print("Welcome guide displayed. Press SPACE to start or Q to quit.")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 32:  # SPACE
            cv2.destroyWindow(guide_window)
            return True
        elif key == ord('q') or key == ord('Q'):
            cv2.destroyWindow(guide_window)
            return False


def show_settings_dialog(current_settings, cap):
    """
    Opens a Tkinter settings window to modify confidence threshold, toggles,
    displays model accuracy and type, and provides buttons to retrain the model
    or add new signs. Releases camera temporarily if adding signs.
    """
    # Create the root window
    root = tk.Tk()
    root.title("SignSpeak Settings")
    root.geometry("460x520")
    root.resizable(False, False)
    
    # Set dark blue and light gray theme
    bg_color = "#f4f6f9"
    card_bg = "#ffffff"
    accent_blue = "#1450b4"
    text_color = "#333333"
    
    root.configure(bg=bg_color)
    
    # Style configuration
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TFrame", background=bg_color)
    style.configure("TLabel", background=bg_color, foreground=text_color, font=("Helvetica", 10))
    style.configure("TButton", background=accent_blue, foreground="white", borderwidth=0, font=("Helvetica", 10, "bold"))
    style.map("TButton", background=[("active", "#0f3d8a")])
    style.configure("TCheckbutton", background=card_bg, font=("Helvetica", 10))
    
    # Header
    header_frame = tk.Frame(root, bg=accent_blue, height=60)
    header_frame.pack(fill="x")
    header_label = tk.Label(header_frame, text="SignSpeak Control Settings", bg=accent_blue, fg="white", font=("Helvetica", 14, "bold"))
    header_label.pack(pady=15)
    
    # Main Container
    main_frame = tk.Frame(root, bg=bg_color, padx=20, pady=15)
    main_frame.pack(fill="both", expand=True)
    
    # 1. Info Card (Model name & Accuracy)
    info_card = tk.LabelFrame(main_frame, text=" AI Recognition Model ", bg=card_bg, fg=accent_blue, font=("Helvetica", 10, "bold"), padx=15, pady=10)
    info_card.pack(fill="x", pady=5)
    
    tk.Label(info_card, text="Model: RandomForestClassifier (200 estimators)", bg=card_bg, fg=text_color, font=("Helvetica", 9, "bold")).pack(anchor="w")
    
    # Accuracy Display
    accuracy_text = "N/A"
    accuracy_file = "models/accuracy.txt"
    if os.path.exists(accuracy_file):
        try:
            with open(accuracy_file, "r") as f:
                accuracy_text = f.read().strip()
        except Exception:
            pass
            
    accuracy_var = tk.StringVar(value=f"Model Accuracy: {accuracy_text}")
    accuracy_lbl = tk.Label(info_card, textvariable=accuracy_var, bg=card_bg, fg="#10b981", font=("Helvetica", 9, "bold"))
    accuracy_lbl.pack(anchor="w", pady=(5, 0))
    
    # 2. Confidence Threshold Slider (50% to 95%)
    slider_frame = tk.Frame(main_frame, bg=bg_color)
    slider_frame.pack(fill="x", pady=10)
    
    tk.Label(slider_frame, text="Confidence Threshold:", font=("Helvetica", 10, "bold"), bg=bg_color).pack(anchor="w")
    
    slider_val = tk.DoubleVar(value=current_settings["confidence_threshold"])
    slider_label_var = tk.StringVar(value=f"{int(current_settings['confidence_threshold'] * 100)}%")
    
    def on_slider_change(val):
        pct = int(float(val) * 100)
        slider_label_var.set(f"{pct}%")
        current_settings["confidence_threshold"] = float(val)
        
    slider_row = tk.Frame(slider_frame, bg=bg_color)
    slider_row.pack(fill="x", pady=5)
    
    slider = ttk.Scale(slider_row, from_=0.50, to=0.95, variable=slider_val, command=on_slider_change, orient="horizontal")
    slider.pack(side="left", fill="x", expand=True, padx=(0, 10))
    
    tk.Label(slider_row, textvariable=slider_label_var, font=("Helvetica", 10, "bold"), width=5, bg=bg_color).pack(side="right")
    
    # 3. Toggles Panel
    toggles_card = tk.LabelFrame(main_frame, text=" Toggles / Preferences ", bg=card_bg, fg=accent_blue, font=("Helvetica", 10, "bold"), padx=15, pady=10)
    toggles_card.pack(fill="x", pady=5)
    
    sound_var = tk.BooleanVar(value=current_settings["sound_enabled"])
    speak_var = tk.BooleanVar(value=current_settings["auto_speak_enabled"])
    landmarks_var = tk.BooleanVar(value=current_settings["show_landmarks"])
    confidence_var = tk.BooleanVar(value=current_settings["show_confidence_bar"])
    
    def update_toggles():
        current_settings["sound_enabled"] = sound_var.get()
        current_settings["auto_speak_enabled"] = speak_var.get()
        current_settings["show_landmarks"] = landmarks_var.get()
        current_settings["show_confidence_bar"] = confidence_var.get()
        
    t1 = ttk.Checkbutton(toggles_card, text="Sound ON", variable=sound_var, command=update_toggles)
    t1.grid(row=0, column=0, sticky="w", pady=5, padx=10)
    
    t2 = ttk.Checkbutton(toggles_card, text="Auto-Speak ON", variable=speak_var, command=update_toggles)
    t2.grid(row=0, column=1, sticky="w", pady=5, padx=10)
    
    t3 = ttk.Checkbutton(toggles_card, text="Show Landmarks", variable=landmarks_var, command=update_toggles)
    t3.grid(row=1, column=0, sticky="w", pady=5, padx=10)
    
    t4 = ttk.Checkbutton(toggles_card, text="Show Confidence Bar", variable=confidence_var, command=update_toggles)
    t4.grid(row=1, column=1, sticky="w", pady=5, padx=10)
    
    # 4. Action Buttons (Retrain, Add New Signs)
    btn_frame = tk.Frame(main_frame, bg=bg_color)
    btn_frame.pack(fill="x", pady=15)
    
    def on_retrain():
        confirm = messagebox.askyesno("Retrain Model", "Do you want to retrain the AI model now?\nThis might take a few seconds.")
        if confirm:
            retrain_btn.config(state="disabled", text="Training...")
            root.update()
            
            try:
                # Run train_model.py
                result = subprocess.run([sys.executable, "train_model.py"], capture_output=True, text=True)
                if result.returncode == 0:
                    messagebox.showinfo("Success", "Model retrained successfully!")
                    current_settings["reload_model"] = True
                    # Reload new model accuracy
                    if os.path.exists(accuracy_file):
                        with open(accuracy_file, "r") as f:
                            accuracy_text = f.read().strip()
                        accuracy_var.set(f"Model Accuracy: {accuracy_text}")
                else:
                    messagebox.showerror("Error", f"Model training failed:\n{result.stderr}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to run train_model.py:\n{e}")
                
            retrain_btn.config(state="normal", text="Retrain Model")
            
    def on_add_signs():
        confirm = messagebox.askyesno("Add New Signs", "To capture new signs, we need to launch the Data Collector.\nThe camera recognition will pause temporarily.\n\nDo you want to proceed?")
        if confirm:
            # Release camera so collect_data.py can use it
            cap.release()
            cv2.destroyAllWindows()
            root.withdraw() # Hide settings window during collection
            
            try:
                # Run collect_data.py
                print("Launching collect_data.py...")
                subprocess.run([sys.executable, "data/collect_data.py"])
            except Exception as e:
                messagebox.showerror("Error", f"Failed to launch collect_data.py:\n{e}")
                
            # Reopen settings and let main loop know to restart camera
            root.deiconify()
            messagebox.showinfo("Returned", "Returned to SignSpeak. Click OK to restart camera.")
            current_settings["restart_camera"] = True
            root.destroy()
            
    retrain_btn = tk.Button(btn_frame, text="Retrain Model", bg=accent_blue, fg="white", activebackground="#0f3d8a", font=("Helvetica", 10, "bold"), pady=8, command=on_retrain)
    retrain_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))
    
    add_btn = tk.Button(btn_frame, text="Add New Signs", bg="#10b981", fg="white", activebackground="#059669", font=("Helvetica", 10, "bold"), pady=8, command=on_add_signs)
    add_btn.pack(side="right", fill="x", expand=True, padx=(5, 0))
    
    # Close button at the bottom
    close_btn = tk.Button(main_frame, text="Close & Apply", bg="#6b7280", fg="white", activebackground="#4b5563", font=("Helvetica", 10, "bold"), pady=5, command=root.destroy)
    close_btn.pack(fill="x", pady=(5, 0))

    # Wait for the settings window to close (blocking the OpenCV loop)
    root.focus_force()
    root.mainloop()


def main():
    # 0. Show the Welcome Guide Screen first
    if not show_welcome_guide():
        print("User chose to quit from the welcome guide.")
        sys.exit(0)

    # 1. Load Pre-trained Sklearn Model
    if not os.path.exists(MODEL_PATH):
        print("\n" + "!"*60)
        print(f"Error: Model not found at '{MODEL_PATH}'")
        print("Please train your model first using: python train_model.py")
        print("!"*60 + "\n")
        sys.exit(1)

    print(f"Loading gesture model from: {MODEL_PATH}...")
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
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)

    # State variables
    sentence_words = []
    prev_prediction = None
    lock_start_time = None
    feedback_text = ""
    feedback_frames = 0
    feedback_color = (0, 255, 0)

    # Set up Fullscreen OpenCV window
    window_name = "SignSpeak Communicator"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("\n" + "="*50)
    print("SignSpeak Communicator Launched (Full Screen)")
    print("Control Hotkeys:")
    print("  [SPACE] -> Clear the current sentence")
    print("  [S]     -> Save the sentence to saved_sentences.txt")
    print("  [Q]     -> Quit the application")
    print("="*50 + "\n")

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

            # Mirror for natural camera selfie-view
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

            # MediaPipe Hand Landmarks detection
            if results.multi_hand_landmarks:
                hand_detected = True
                hand_landmarks = results.multi_hand_landmarks[0]

                # Draw Hand Skeleton
                if app_settings["show_landmarks"]:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                # Flatten landmarks
                flat_landmarks = []
                for lm in hand_landmarks.landmark:
                    flat_landmarks.extend([lm.x, lm.y, lm.z])
                
                # Single-sample prediction shape
                X_sample = np.array(flat_landmarks).reshape(1, -1)

                try:
                    # Run classification model
                    current_prediction = model.predict(X_sample)[0]
                    
                    # Estimate confidence probability
                    probabilities = model.predict_proba(X_sample)[0]
                    class_idx = np.where(model.classes_ == current_prediction)[0][0]
                    confidence = probabilities[class_idx]
                except Exception:
                    confidence = 1.0

            # Lock-In Logic
            countdown = LOCK_IN_DURATION
            status_text = "Waiting for Hand"
            status_color = (0, 0, 255) # Red

            if hand_detected and current_prediction:
                status_text = f"Detecting: {current_prediction.upper()}"
                status_color = (0, 165, 255) # Orange
                
                if confidence >= app_settings["confidence_threshold"]:
                    if current_prediction == prev_prediction:
                        if lock_start_time is None:
                            lock_start_time = time.time()
                        
                        elapsed = time.time() - lock_start_time
                        countdown = max(0.0, LOCK_IN_DURATION - elapsed)
                        status_color = (0, 255, 0) # Green (Active Lock-in)

                        if elapsed >= LOCK_IN_DURATION:
                            # Speak out loud and add to sentence list
                            sentence_words.append(current_prediction)
                            print(f"[ACCUMULATED] Word: '{current_prediction}'")
                            speak(current_prediction)

                            # Trigger visual feedback
                            feedback_text = f"Added: {current_prediction.upper()}"
                            feedback_color = (0, 255, 0)
                            feedback_frames = 35

                            # Reset lock states
                            lock_start_time = None
                            prev_prediction = None
                    else:
                        lock_start_time = time.time()
                        prev_prediction = current_prediction
                else:
                    # Confidence below threshold
                    lock_start_time = None
                    prev_prediction = None
            else:
                lock_start_time = None
                prev_prediction = None

            # --- Full-Screen Clean UI Render ---
            # 1. Top HUD bar: status, prediction info
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 140), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            cv2.putText(frame, "SignSpeak Communicator v1.0", (20, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # Draw prediction stats
            cv2.putText(frame, f"Status: {status_text}", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_color, 2, cv2.LINE_AA)

            if hand_detected and current_prediction and app_settings["show_confidence_bar"]:
                # Progress bar for confidence
                bar_w, bar_h = 220, 16
                bar_x, bar_y = 20, 95
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * confidence), bar_y + bar_h), status_color, -1)
                cv2.putText(frame, f"{confidence*100:.0f}% Match", (bar_x + bar_w + 15, bar_y + 13), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

                # Countdown clock helper
                if lock_start_time is not None:
                    cv2.putText(frame, f"Locking in: {countdown:.1f}s", (w - 240, 35), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            # 2. Bottom sentence HUD banner
            cv2.rectangle(frame, (0, h - 120), (w, h), (15, 15, 15), -1)
            cv2.rectangle(frame, (0, h - 120), (w, h - 117), (0, 255, 255), -1) # Colored divider border

            sentence_str = " ".join(sentence_words).upper()
            if not sentence_str:
                sentence_str = "(START GESTURING TO BUILD SENTENCE)"
                text_color = (120, 120, 120)
            else:
                text_color = (255, 255, 255)

            # Draw sentence text
            cv2.putText(frame, "Sentence Output:", (20, h - 85), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, sentence_str, (20, h - 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2, cv2.LINE_AA)

            # Instructions shortcuts
            cv2.putText(frame, "TAB: Settings | SPACE: Clear | S: Save | Q: Quit", (w - 560, h - 85), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

            # 3. Trigger feedback overlays
            if feedback_frames > 0:
                cv2.putText(frame, feedback_text, (w // 2 - 120, h // 2 - 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, feedback_color, 3, cv2.LINE_AA)
                feedback_frames -= 1

            # Render frame
            cv2.imshow(window_name, frame)

            # 4. Handle Key Events
            key = cv2.waitKey(1) & 0xFF
            
            # TAB key: Open settings panel
            if key == 9:
                print("Opening settings panel...")
                show_settings_dialog(app_settings, cap)
                
                # Check reload model
                if app_settings.get("reload_model", False):
                    app_settings["reload_model"] = False
                    print("Reloading trained model...")
                    try:
                        model = joblib.load(MODEL_PATH)
                    except Exception as e:
                        print(f"Failed to load model: {e}")
                
                # Check restart camera
                if app_settings.get("restart_camera", False):
                    app_settings["restart_camera"] = False
                    print("Re-initializing webcam...")
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        print("Error: Could not reopen webcam.")
                        sys.exit(1)
                    # Recreate window to make sure it's active
                    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            # SPACE bar: Clear sentence
            elif key == 32: 
                sentence_words.clear()
                feedback_text = "SENTENCE CLEARED"
                feedback_color = (0, 165, 255) # Orange
                feedback_frames = 25
                print("Sentence cleared.")

            # S key: Save sentence
            elif key == ord('s') or key == ord('S'):
                if sentence_words:
                    full_sentence = " ".join(sentence_words)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    try:
                        with open(SAVE_FILE_PATH, "a") as f:
                            f.write(f"[{timestamp}] {full_sentence}\n")
                        feedback_text = "SENTENCE SAVED!"
                        feedback_color = (0, 255, 0)
                        feedback_frames = 30
                        print(f"Saved sentence to {SAVE_FILE_PATH}: '{full_sentence}'")
                    except Exception as e:
                        feedback_text = f"Save failed: {e}"
                        feedback_color = (0, 0, 255)
                        feedback_frames = 30
                        print(f"Error saving file: {e}")
                else:
                    feedback_text = "EMPTY SENTENCE - NOT SAVED"
                    feedback_color = (0, 0, 255)
                    feedback_frames = 25

            # Q key: Quit
            elif key == ord('q') or key == ord('Q'):
                print("Exiting communicator app...")
                break

    # Resource release
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released. Goodbye!")

if __name__ == "__main__":
    main()
