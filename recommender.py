# curova/ai_services/ml/recommender.py

import os
import joblib  # type: ignore
import numpy as np # type: ignore

from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import accuracy_score, classification_report  # type: ignore
from sklearn.utils.class_weight import compute_class_weight  # type: ignore

from imblearn.over_sampling import SMOTE  # type: ignore

from curova.ai_services.utils.data_loader import load_both, preprocess_data
from curova.ai_services.utils.logger import setup_logger

# ------------------------------------------------
# Logging
# ------------------------------------------------
logger = setup_logger("recommender")

# Base model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "schedule_recommender.pkl")


# ------------------------------------------------
# Training Function
# ------------------------------------------------
def train_model(apply_smote=True):
    """Train, balance, and save the recommender model."""

    logger.info("Loading synthetic + Kaggle datasets...")
    df = load_both()

    # Preprocess (Label encode, extract X, y)
    X, y = preprocess_data(df)

    # VALID COLUMNS – prevents crashes if new columns appear
    VALID_FEATURES = ["doctor_id", "patient_id", "department", "shift", "day_of_week"]
    X = X[VALID_FEATURES]

    logger.info(f"Final training features: {list(X.columns)}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ------------------------------------------------
    # Class balancing
    # ------------------------------------------------
    logger.info("Computing class weights...")
    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_train
    )

    class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weights)}
    logger.info(f"Class Weights Used: {class_weight_dict}")

    # ------------------------------------------------
    # SMOTE Optional Oversampling
    # ------------------------------------------------
    if apply_smote:
        logger.info("Applying SMOTE oversampling to training dataset...")
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        logger.info(f"Resampled dataset size: {X_train.shape}")

    # ------------------------------------------------
    # Model Training
    # ------------------------------------------------
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight=class_weight_dict,  # <-- This mitigates bias
    )

    logger.info("Training RandomForest model...")
    model.fit(X_train, y_train)

    # ------------------------------------------------
    # Evaluation
    # ------------------------------------------------
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    logger.info(f"Model accuracy: {acc:.4f}")
    print("\n--- MODEL REPORT ---")
    print(classification_report(y_test, preds))

    # Save the model
    joblib.dump(model, MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")
    print(f"[INFO] Model trained + saved at: {MODEL_PATH}")

    return acc


# ------------------------------------------------
# Inference Function
# ------------------------------------------------
def predict_slot(features):
    """
    Predict if a slot is recommended.

    features = [
        doctor_id, patient_id, department, shift, day_of_week
    ]
    """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train it using train_model()"
        )

    model = joblib.load(MODEL_PATH)
    prediction = model.predict([features])[0]

    return bool(prediction)


# ------------------------------------------------
# Run as script
# ------------------------------------------------
if __name__ == "__main__":
    train_model(apply_smote=True)
