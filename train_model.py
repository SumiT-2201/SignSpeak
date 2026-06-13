import os
import sys
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CSV_FILE_PATH = "data/landmarks.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "sign_model.pkl")
CLASSES_PATH = os.path.join(MODEL_DIR, "label_classes.txt")

def main():
    # 1. Load landmarks.csv
    if not os.path.exists(CSV_FILE_PATH):
        print(f"Error: Dataset not found at '{CSV_FILE_PATH}'. Please run collect_data.py first.")
        sys.exit(1)
        
    print(f"Loading dataset from: {CSV_FILE_PATH}...")
    try:
        df = pd.read_csv(CSV_FILE_PATH)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    print(f"Dataset loaded. Shape: {df.shape}")
    
    # 2. Split X (landmark columns) and y (label column)
    # Check for landmark columns
    X = df.filter(like="lm_")
    if X.shape[1] != 63:
        # Fallback to first 63 columns
        X = df.iloc[:, :63]
        
    # Check for label column
    if "label_name" in df.columns:
        y = df["label_name"]
    elif "label" in df.columns:
        y = df["label"]
    else:
        # Fallback to last column
        y = df.iloc[:, -1]
        
    print(f"Features (X) shape: {X.shape}")
    print(f"Labels (y) shape: {y.shape}")
    
    # Remove any rows with NaN values if any exist
    nan_rows = df[df.isna().any(axis=1)]
    if not nan_rows.empty:
        print(f"Warning: Found {len(nan_rows)} rows with NaN values. Dropping them.")
        df = df.dropna()
        X = df.iloc[:, :63]
        y = df.iloc[:, -1]

    # Check class distribution
    print("\nClass distribution:")
    print(y.value_counts())
    
    # Check if we have enough data to split
    unique_classes = y.nunique()
    if len(df) < 5:
        print("Error: Too few samples to train a model. Please collect more samples first.")
        sys.exit(1)
        
    # 3. Train/test split and RandomForestClassifier
    # Use stratify only if every class has at least 2 samples
    stratify_y = y if (y.value_counts() >= 2).all() else None
    
    test_size = 0.2
    # Adjust test size if dataset is extremely small
    if len(df) < 10:
        test_size = 0.5
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify_y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    print("Training RandomForestClassifier (200 estimators)...")
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    
    # 4. Evaluate and print metrics
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nEvaluation Accuracy: {accuracy * 100:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # 5. Save model and labels
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"Created directory: {MODEL_DIR}")
        
    try:
        joblib.dump(clf, MODEL_PATH)
        print(f"Model successfully saved to: {MODEL_PATH}")
    except Exception as e:
        print(f"Error saving model: {e}")
        sys.exit(1)
        
    # 6. Save label classes
    classes = sorted(y.unique().tolist())
    try:
        with open(CLASSES_PATH, "w") as f:
            for cls in classes:
                f.write(f"{cls}\n")
        print(f"Label classes saved to: {CLASSES_PATH}")
        print(f"Classes: {classes}")
    except Exception as e:
        print(f"Error saving label classes: {e}")
        sys.exit(1)
        
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
