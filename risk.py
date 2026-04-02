def classify_risk(score):
    if score > 1.5:
        return "High"
    elif score > 0.7:
        return "Medium"
    else:
        return "Low"
