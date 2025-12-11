# ai_services/ml/generate_synthetic_data.py
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from datetime import datetime, timedelta
import random
import os

def generate_synthetic_schedule_data(num_samples=1000):
    doctors = [f"Dr_{i}" for i in range(1, 8)]
    patients = [f"Patient_{i}" for i in range(1, 101)]
    departments = ["Cardiology", "Neurology", "Pediatrics", "Oncology", "Dermatology", "Orthopedics"]

    # Define time shifts
    shifts = {
        "Night": {"start": "00:00", "end": "08:00"},
        "AM": {"start": "08:00", "end": "16:00"},
        "PM": {"start": "16:00", "end": "00:00"},
    }

    start_date = datetime.now().date()
    data = []

    for _ in range(num_samples):
        doctor = random.choice(doctors)
        patient = random.choice(patients)
        dept = random.choice(departments)

        # Randomly select a date within the next 30 days
        date = start_date + timedelta(days=random.randint(0, 29))
        day_of_week = date.strftime("%A")

        # Randomly select a shift
        shift = random.choice(list(shifts.keys()))
        time_start = datetime.strptime(shifts[shift]["start"], "%H:%M").time()
        time_end = datetime.strptime(shifts[shift]["end"], "%H:%M").time()

        # Duration of each appointment (in minutes)
        duration = random.choice([15, 20, 30, 45, 60])
        satisfaction_score = random.uniform(0.4, 1.0)

        # Realistic factors
        conflict = random.choices([0, 1], weights=[0.8, 0.2])[0]  # 20% chance of conflict
        overtime = random.choices([0, 1], weights=[0.9, 0.1])[0]  # 10% overtime
        cancelled = random.choices([0, 1], weights=[0.95, 0.05])[0]  # 5% cancellations
        missed_break = random.choices([0, 1], weights=[0.85, 0.15])[0]  # 15% missed breaks
        rota_change = random.choices([0, 1], weights=[0.9, 0.1])[0]  # 10% rota changes

        # Simulate emergency/free slots — 10% of records are "free"
        is_emergency_slot = random.choices([0, 1], weights=[0.9, 0.1])[0]
        patient_id = "FreeSlot" if is_emergency_slot else patient

        # A recommended slot is one that’s free, non-conflicting, and has high satisfaction
        recommended = 1 if (is_emergency_slot or (satisfaction_score > 0.7 and conflict == 0 and cancelled == 0)) else 0

        data.append([
            doctor, patient_id, dept, shift, str(date), day_of_week,
            duration, satisfaction_score, conflict, overtime, cancelled,
            missed_break, rota_change, recommended
        ])

    df = pd.DataFrame(data, columns=[
        "doctor_id", "patient_id", "department", "shift",
        "date", "day_of_week", "duration", "satisfaction_score",
        "conflict", "overtime", "cancelled", "missed_break",
        "rota_change", "recommended"
    ])

    # Save in the ML folder path
    output_path = os.path.join(os.path.dirname(__file__), "synthetic_schedule_data.csv")
    df.to_csv(output_path, index=False)

    print(f"[INFO] Synthetic dataset created with {len(df)} records at {output_path}")


if __name__ == "__main__":
    generate_synthetic_schedule_data()
