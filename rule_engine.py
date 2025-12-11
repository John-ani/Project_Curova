# ai_services/ml/rule_engine.py

def rule_based_recommendation(satisfaction_score, conflict, duration):
    if satisfaction_score > 0.75 and conflict == 0 and duration <= 45:
        return True
    return False
