#!/usr/bin/env python3
"""
Customer Churn Predictor CLI
============================
Usage:
    python predict.py

Prompts for customer details and outputs the churn probability
and risk classification.

You can also call predict_single() directly from Python:

    from src.prediction import predict_single
    result = predict_single({
        "Tenure Months": 6,
        "Monthly Charges": 80.0,
        "Total Charges": 480.0,
        "CLTV": 3000,
        "Churn Score": 75,
        "contract_length_months": 1,
        "service_count": 2,
        ...
    })
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(message)s")

# Ensure we can import from src/
sys.path.insert(0, str(Path(__file__).parent))


def _prompt_int(label: str, default: int) -> int:
    try:
        val = input(f"  {label} [{default}]: ").strip()
        return int(val) if val else default
    except ValueError:
        return default


def _prompt_float(label: str, default: float) -> float:
    try:
        val = input(f"  {label} [{default}]: ").strip()
        return float(val) if val else default
    except ValueError:
        return default


def _prompt_choice(label: str, choices: list, default: int) -> int:
    print(f"\n  {label}:")
    for i, c in enumerate(choices):
        print(f"    {i}) {c}")
    try:
        val = input(f"  Enter number [{default}]: ").strip()
        idx = int(val) if val else default
        return idx if 0 <= idx < len(choices) else default
    except ValueError:
        return default


def main() -> None:
    print()
    print("=" * 52)
    print("  CUSTOMER CHURN PREDICTION TOOL")
    print("=" * 52)
    print("  Press Enter to accept defaults.\n")

    tenure = _prompt_int("Tenure (months)", 12)
    monthly_charges = _prompt_float("Monthly Charges ($)", 65.0)
    total_charges = _prompt_float("Total Charges ($)", monthly_charges * tenure or 780.0)
    cltv = _prompt_int("CLTV ($)", 3000)
    churn_score = _prompt_int("Churn Score (0-100)", 50)

    contract_choices = ["Month-to-month", "One year", "Two year"]
    contract_idx = _prompt_choice("Contract Type", contract_choices, 0)
    contract_length = [1, 12, 24][contract_idx]

    internet_choices = ["No", "DSL", "Fiber optic"]
    internet_idx = _prompt_choice("Internet Service", internet_choices, 1)
    has_internet = 0 if internet_idx == 0 else 1
    internet_enc = internet_idx

    has_phone = 1 if _prompt_choice("Has Phone Service?", ["No", "Yes"], 1) else 0
    service_count = _prompt_int("Number of subscribed services (0-8)", 3)

    segment_choices = ["Budget", "Standard", "Premium"]
    segment_idx = _prompt_choice("Customer Segment", segment_choices, 1)

    gender_enc = 1 if _prompt_choice("Gender", ["Female", "Male"], 1) == 1 else 0
    senior_enc = 1 if _prompt_choice("Senior Citizen?", ["No", "Yes"], 0) == 1 else 0
    partner_enc = 1 if _prompt_choice("Has Partner?", ["No", "Yes"], 0) == 1 else 0
    dependents_enc = 1 if _prompt_choice("Has Dependents?", ["No", "Yes"], 0) == 1 else 0

    payment_choices = [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ]
    payment_idx = _prompt_choice("Payment Method", payment_choices, 0)

    high_value_flag = 1 if cltv >= 4500 else 0
    long_term_flag = 1 if tenure >= 24 else 0
    digital_customer = 1 if payment_idx == 2 else 0

    customer = {
        "Tenure Months": tenure,
        "Monthly Charges": monthly_charges,
        "Total Charges": total_charges,
        "CLTV": cltv,
        "Churn Score": churn_score,
        "contract_length_months": contract_length,
        "service_count": service_count,
        "has_internet": has_internet,
        "has_phone": has_phone,
        "high_value_flag": high_value_flag,
        "long_term_flag": long_term_flag,
        "digital_customer": digital_customer,
        "gender_enc": gender_enc,
        "senior_enc": senior_enc,
        "partner_enc": partner_enc,
        "dependents_enc": dependents_enc,
        "internet_enc": internet_enc,
        "contract_enc": contract_idx,
        "payment_enc": payment_idx,
        "segment_enc": segment_idx,
    }

    from src.prediction import predict_single
    result = predict_single(customer)

    print()
    print("=" * 52)
    print("  PREDICTION RESULT")
    print("=" * 52)
    print(f"  Model Used           : {result['model_used']}")
    print(f"  Churn Probability    : {result['churn_probability_pct']}%")
    print(f"  Prediction           : {result['prediction_label']}")
    print(f"  Risk Classification  : {result['risk_label']}")
    print("=" * 52)
    print()


if __name__ == "__main__":
    main()
