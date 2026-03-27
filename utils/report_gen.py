import json
from datetime import datetime


def generate_report(image_filename, predictions, val_acc, save_path):
    report = {
        "generated_at"   : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image"          : image_filename,
        "top_prediction" : {
            "class"      : predictions[0]["class"],
            "confidence" : f"{predictions[0]['confidence']:.2f}%",
        },
        "all_predictions": predictions,
        "model_val_acc"  : f"{val_acc * 100:.2f}%",
        "disclaimer"     : (
            "⚠️ AI-generated analysis — NOT a clinical diagnosis. "
            "Consult a licensed dermatologist."
        ),
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    return report
