import os
import sys
from datetime import datetime, timedelta, time
import pandas as pd
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "schedule_recommender.pkl")

from curova.ai_services.utils.data_loader import load_both, preprocess_data

# --------------------------------------------------------
# VALID DEPARTMENTS
# --------------------------------------------------------
VALID_DEPARTMENTS = [
    "Cardiology", "Oncology", "Neurology", "Pediatrics", "Orthopedics",
    "Dermatology", "Gastroenterology", "General Medicine", "ENT",
    "Gynecology", "Urology", "Radiology", "Physiotherapy", "Emergency"
]

# --------------------------------------------------------
# DOCTOR-SPECIFIC BLACKOUT DATES
# --------------------------------------------------------
BLACKOUTS = {
    "Dr. Smith": ["2025-11-22", "2025-11-25"],
    "Dr. Adams": ["2025-11-21"],
    "Dr. Johnson": [],
}

# --------------------------------------------------------
# DOCTOR WORKING HOURS & BREAKS
# --------------------------------------------------------
DOCTOR_SCHEDULE = {
    "default": {
        "start": time(8, 0),
        "end": time(17, 0),
        "break_start": time(12, 0),
        "break_end": time(13, 0)
    }
}

# ---------------------------------------------------------------------
# SIMPLE HELPERS
# ---------------------------------------------------------------------
def time_to_shift_custom(t: str) -> str:
    dt = datetime.strptime(t, "%H:%M").time()
    if dt.hour < 8:
        return "AM"
    if 8 <= dt.hour < 16:
        return "PM"
    return "Night"

def safe_input_choice(prompt, choices):
    while True:
        v = input(prompt).strip()
        if v in choices:
            return v
        print("Invalid option — try again.\n")

def generate_next_n_days(n=30):
    today = datetime.now().date()
    return [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n)]

# ---------------------------------------------------------------------
#  CHECKS
# ---------------------------------------------------------------------
def violates_break(doctor: str, time_str: str) -> bool:
    t = datetime.strptime(time_str, "%H:%M").time()
    schedule = DOCTOR_SCHEDULE.get(doctor, DOCTOR_SCHEDULE["default"])
    return schedule["break_start"] <= t < schedule["break_end"]

def violates_hours(doctor: str, time_str: str) -> bool:
    t = datetime.strptime(time_str, "%H:%M").time()
    schedule = DOCTOR_SCHEDULE.get(doctor, DOCTOR_SCHEDULE["default"])
    return not (schedule["start"] <= t <= schedule["end"])

def is_blackout(doctor: str, date: str) -> bool:
    return date in BLACKOUTS.get(doctor, [])

def find_doctor_conflict(df, doctor, date, time_str):
    """Returns True if this doctor has an overlapping appointment."""
    same_day = df[(df["doctor_id"] == doctor) & (df["date"] == date)]
    if same_day.empty:
        return False

    t = datetime.strptime(time_str, "%H:%M").time()

    for _, row in same_day.iterrows():
        try:
            booked_t = datetime.strptime(row["time"], "%H:%M").time()
            if booked_t == t:
                return True
        except:
            continue

    return False

# ---------------------------------------------------------------------
# CREATE SAMPLE ROW FOR ML MODEL
# ---------------------------------------------------------------------
def build_ml_row(doctor, department, date, time_str, df):
    day_of_week = datetime.strptime(date, "%Y-%m-%d").strftime("%A")
    shift = time_to_shift_custom(time_str)

    new_row = {
        "doctor_id": doctor,
        "patient_id": "Simulated",
        "department": department,
        "shift": shift,
        "date": date,
        "day_of_week": day_of_week,
        "duration": 30,
        "satisfaction_score": 0.8,
        "conflict": 0,
        "overtime": 0,
        "cancelled": 0,
        "missed_break": 0,
        "rota_change": 0,
        "recommended": 0,
        "time": time_str   # <--- ensure time is present
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    X_all, _ = preprocess_data(df)
    sample = X_all.iloc[-1].values.reshape(1, -1)
    return sample

# ---------------------------------------------------------------------
#  SUGGEST ALTERNATIVES
# ---------------------------------------------------------------------
def suggest_later_time(doctor, chosen_time):
    base = datetime.strptime(chosen_time, "%H:%M")
    new_time = (base + timedelta(minutes=30)).strftime("%H:%M")
    schedule = DOCTOR_SCHEDULE.get(doctor, DOCTOR_SCHEDULE["default"])

    if datetime.strptime(new_time, "%H:%M").time() <= schedule["end"]:
        return new_time
    return None

def suggest_later_day(date):
    d = datetime.strptime(date, "%Y-%m-%d").date() + timedelta(days=1)
    return d.strftime("%Y-%m-%d")

def suggest_other_doctors(doctor, doctors):
    return [d for d in doctors if d != doctor]

# ---------------------------------------------------------------------
#  MAIN INTERACTIVE FLOW
# ---------------------------------------------------------------------
def run_single_prediction():
    print("\n=== CUROVA Appointment AI — Smart Scheduling Test ===\n")

    if not os.path.exists(MODEL_PATH):
        print("Model missing — train your model first.")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)
    df = load_both()
    df_copy = df.copy()

    doctors = sorted([d for d in df["doctor_id"].unique() if d not in ["", None, "UNKNOWN"]])
    dates = generate_next_n_days(30)

    # --------------------------------------
    # Doctor
    # --------------------------------------
    print("\nAvailable Doctors:")
    for d in doctors:
        print(" •", d)

    doctor = safe_input_choice("\nChoose doctor: ", doctors)

    # --------------------------------------
    # Department
    # --------------------------------------
    print("\nDepartments:")
    for d in VALID_DEPARTMENTS:
        print(" •", d)

    department = safe_input_choice("\nChoose department: ", VALID_DEPARTMENTS)

    # --------------------------------------
    # Time
    # --------------------------------------
    while True:
        time_str = input("\nEnter preferred time (HH:MM): ").strip()
        try:
            datetime.strptime(time_str, "%H:%M")
            break
        except:
            print("Invalid time.\n")

    # --------------------------------------
    # DATE
    # --------------------------------------
    print("\nNext available dates:")
    print(", ".join(dates[:7]), " ...")

    date_choice = safe_input_choice("\nChoose date (YYYY-MM-DD): ", dates)

    # -------------------- VALIDATION --------------------

    if is_blackout(doctor, date_choice):
        print("\n❌ Doctor is unavailable (BLACKOUT date).")
    elif violates_hours(doctor, time_str):
        print("\n❌ Doctor doesn't work at that hour.")
    elif violates_break(doctor, time_str):
        print("\n❌ Doctor is on break at that time.")
    elif find_doctor_conflict(df_copy, doctor, date_choice, time_str):
        print("\n❌ The doctor already has an appointment at this time.")
    else:
        # --- If schedule is physically possible, ask model ---
        to_model = build_ml_row(doctor, department, date_choice, time_str, df_copy)

        # Fix shape mismatch if needed
        if to_model.shape[1] != model.n_features_in_:
            diff = model.n_features_in_ - to_model.shape[1]
            if diff > 0:
                to_model = np.hstack([to_model, np.zeros((1, diff))])
            else:
                to_model = to_model[:, :model.n_features_in_]

        pred = model.predict(to_model)[0]
        prob = model.predict_proba(to_model)[0].max()

        if pred == 1:
            print(f"\n✅ Booking Confirmed! (Confidence {prob:.2%})")
            return
        else:
            print(f"\n❌ Model did NOT confirm booking (Confidence {prob:.2%}).")

    # =================================================================
    #   FALLBACK OPTIONS
    # =================================================================

    print("\n🔍 Searching for alternatives...\n")

    # 1 — Suggest later same day
    later_time = suggest_later_time(doctor, time_str)
    if later_time:
        print(f"➡ Suggestion: Try later today at {later_time} with {doctor}")
    else:
        print("➡ No more slots today for this doctor.")

    # 2 — Suggest next days
    next_day = suggest_later_day(date_choice)
    if next_day in dates:
        print(f"➡ Alternative day: {next_day}")

    # 3 — Suggest other doctors
    others = suggest_other_doctors(doctor, doctors)
    print("\n➡ Other doctors available:")
    for o in others[:5]:
        print("   -", o)

    print("\n❗ None of the primary options succeeded, but suggestions have been provided.\n")


# ---------------------------------------------------------------------
# SIMULATION
# ---------------------------------------------------------------------
def run_simulation():
    print("\n=== Running 100-case Simulation ===")
    df = load_both()
    df_copy = df.copy()
    model = joblib.load(MODEL_PATH)

    df_copy["department"] = df_copy["department"].apply(
        lambda x: x if x in VALID_DEPARTMENTS else np.random.choice(VALID_DEPARTMENTS)
    )

    sims = []
    for _ in range(100):
        row = df_copy.sample(1).iloc[0].to_dict()
        row["time"] = f"{np.random.randint(8,17):02d}:{np.random.choice(['00','30'])}"
        row["date"] = generate_next_n_days(30)[np.random.randint(0, 30)]
        df_copy = pd.concat([df_copy, pd.DataFrame([row])], ignore_index=True)
        sims.append(row)

    X_all, _ = preprocess_data(df_copy)
    feats = X_all.tail(100).values
    preds = model.predict(feats)

    print(f"Simulation complete — confirmation rate: {preds.mean():.2%}")


# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("\nChoose mode:")
    print("1. Interactive booking test")
    print("2. 100-case simulation")

    c = input("\nEnter choice: ").strip()
    if c == "1":
        run_single_prediction()
    else:
        run_simulation()
