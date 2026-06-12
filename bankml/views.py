import re
import os
import csv
from datetime import datetime
from googletrans import Translator
import joblib
import numpy as np

from django.conf import settings
from django.shortcuts import render


# =========================
# 1. Load models & encoder
# =========================

tfidf = joblib.load("bankml/models/tfidf_vectorizer.pkl")
label_encoder = joblib.load("bankml/models/label_encoder.pkl")
stacking_clf = joblib.load("bankml/models/stacking_model.pkl")

class_names = label_encoder.classes_.tolist()
translator = Translator()

# =========================
# 2. Text cleaning helper
# =========================

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^\w\s,.!?-]", " ", text)
    text = re.sub(r"([!?.,])\1+", r"\1", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================
# 3. Rule-based adjustment
# =========================

def rule_based_adjustment(text: str, predicted_label: str, max_prob: float) -> str:

    if max_prob >= 0.90:
        return predicted_label

    t = text.lower()

    upi_keywords = [
        "upi", "vpa", "virtual payment address",
        "qr code", "scan and pay", "upi id",
        "bhim", "gpay", "google pay", "phonepe", "paytm"
    ]

    retail_keywords = [
        "savings account", "current account", "bank account",
        "passbook", "cheque", "chequebook",
        "atm", "debit card", "nominee", "aadhaar", "kyc",
        "minimum balance", "branch", "ifsc"
    ]

    has_upi = any(k in t for k in upi_keywords)
    has_retail = any(k in t for k in retail_keywords)

    if has_upi:
        return "upi_transaction_failures"

    if (not has_upi) and has_retail and predicted_label == "upi_transaction_failures":
        return "retail_banking"

    return predicted_label


# =========================
# 4. Next-step recommendation
# =========================

def next_step_for_label(label: str) -> str:
    if label == "upi_transaction_failures":
        return "Keep the UTR/reference ID ready and contact your bank if the amount is not auto-reversed within 24–48 hours."
    elif label == "credit_card":
        return "Review your credit card statement and raise a dispute if something looks wrong."
    elif label == "retail_banking":
        return "Check your account statement or contact branch support if issues persist."
    elif label == "credit_reporting":
        return "Download your latest credit report and raise a dispute with the bureau/bank."
    elif label == "debt_collection":
        return "Ask for a written notice and debt validation before making payments."
    elif label == "mortgages_and_loans":
        return "Keep your loan account details and contact loan servicing support."
    else:
        return "Please contact your bank’s customer support with full complaint details."


# =========================
# 5. Helper: prettify labels
# =========================

def prettify(label: str) -> str:
    """
    Convert machine labels (snake_case) to human-friendly Title Case.
    Also uppercase common acronyms (UPI, ATM, UTR, IFSC, KYC, VPA).
    Examples:
      'credit_card' -> 'Credit Card'
      'upi_transaction_failures' -> 'UPI Transaction Failures'
    """
    if not label:
        return label

    # Replace underscores with spaces
    s = label.replace("_", " ").strip()

    # Words that should be uppercased
    ACRONYMS = {"upi", "atm", "utr", "ifsc", "kyc", "vpa", "gpay", "id"}

    parts = []
    for w in s.split():
        wl = w.lower()
        if wl in ACRONYMS:
            parts.append(wl.upper())
        else:
            parts.append(wl.capitalize())
    return " ".join(parts)


# =========================
# 6. Model proba
# =========================

def model_predict_proba(text_list):
    tfidf_vecs = tfidf.transform(text_list)
    return stacking_clf.predict_proba(tfidf_vecs)


# =========================
# 7. CSV logging
# =========================

LOG_DIR = os.path.join(settings.BASE_DIR, "logs")
LOG_PATH = os.path.join(LOG_DIR, "predictions.csv")


def save_prediction_to_csv(
    timestamp: str,
    user_text: str,
    cleaned_text: str,
    raw_pred: str,
    final_pred: str,
    max_prob: float,
    proba: np.ndarray,
    action: str,
):
    os.makedirs(LOG_DIR, exist_ok=True)

    proba_str = ";".join([f"{label}:{p:.4f}" for label, p in zip(class_names, proba)])
    file_exists = os.path.exists(LOG_PATH)

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp",
                "action",
                "user_text",
                "cleaned_text",
                "raw_prediction",
                "final_prediction",
                "max_prob",
                "proba_per_class",
            ])
        writer.writerow([
            timestamp,
            action,
            user_text,
            cleaned_text,
            raw_pred,
            final_pred,
            f"{max_prob:.4f}",
            proba_str,
        ])


# =========================
# 8. Main view (LIME removed)
# =========================

def classify_complaint(request):
    prediction = None
    raw_prediction = None
    probs_with_labels = None
    max_prob = None
    rule_used = None
    complaint_text = ""
    next_step = None

    if request.method == "POST":
        action = request.POST.get("action", "classify")
        user_raw = request.POST.get("complaint_text", "").strip()
        complaint_clean = clean_text(user_raw)

        if complaint_clean:
            proba = model_predict_proba([complaint_clean])[0]
            pred_idx = int(np.argmax(proba))
            pred_label_raw = label_encoder.classes_[pred_idx]
            max_prob = float(proba[pred_idx])

            # Prettified raw prediction for UI
            raw_prediction = prettify(pred_label_raw)

            # Rule-based final label (machine form)
            final_label = rule_based_adjustment(complaint_clean, pred_label_raw, max_prob)
            rule_used = (final_label != pred_label_raw)

            # Suggested next step based on final machine label
            next_step = next_step_for_label(final_label)

            # Prettified final label for display
            prediction = prettify(final_label)

            # Pair prettified class names with probabilities for UI
            probs_with_labels = [(prettify(lbl), float(p)) for lbl, p in zip(class_names, proba)]

            # ---------- CSV logging ----------
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                save_prediction_to_csv(
                    timestamp=timestamp,
                    user_text=user_raw,
                    cleaned_text=complaint_clean,
                    raw_pred=raw_prediction,
                    final_pred=prediction,
                    max_prob=max_prob,
                    proba=proba,
                    action=action,
                )
            except Exception as e:
                print("Error saving prediction to CSV:", e)

        complaint_text = user_raw

    context = {
        "prediction": prediction,
        "raw_prediction": raw_prediction,
        "probs_with_labels": probs_with_labels,
        "max_prob": max_prob,
        "rule_used": rule_used,
        "complaint_text": complaint_text,
        "next_step": next_step,
    }

    return render(request, "index.html", context)
