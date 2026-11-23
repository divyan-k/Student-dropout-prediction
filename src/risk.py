"""Risk classification logic and actionable intervention suggestions.

Threshold rules:
- Low: dropout probability < 0.33
- Medium: 0.33 <= dropout probability < 0.66
- High: dropout probability >= 0.66
"""

from typing import List, Dict, Any
from src.config import RISK_THRESHOLDS


def classify_risk(dropout_prob: float) -> str:
    """Classify risk level from predicted dropout probability."""
    if dropout_prob < RISK_THRESHOLDS["low"]:
        return "Low"
    elif dropout_prob < RISK_THRESHOLDS["high"]:
        return "Medium"
    else:
        return "High"


def suggest_actions(risk_level: str, student_data: Dict[str, Any]) -> List[str]:
    """Generate rule-based intervention recommendations based on student risk and status."""
    actions = []

    tuition_up_to_date = student_data.get("Tuition fees up to date", 1)
    debtor = student_data.get("Debtor", 0)

    # Calculate approximate approval rate if semester units are provided
    approval_rate = 1.0
    enrolled_s1 = student_data.get("Curricular units 1st sem (enrolled)", 0)
    approved_s1 = student_data.get("Curricular units 1st sem (approved)", 0)
    enrolled_s2 = student_data.get("Curricular units 2nd sem (enrolled)", 0)
    approved_s2 = student_data.get("Curricular units 2nd sem (approved)", 0)

    total_enrolled = (enrolled_s1 or 0) + (enrolled_s2 or 0)
    total_approved = (approved_s1 or 0) + (approved_s2 or 0)
    if total_enrolled > 0:
        approval_rate = total_approved / total_enrolled

    if risk_level == "High":
        if tuition_up_to_date == 0:
            actions.append("Refer to financial aid and fee-support office immediately.")
        if debtor == 1:
            actions.append("Review emergency tuition payment plan options with student accounts.")
        if total_enrolled > 0 and approval_rate < 0.6:
            actions.append("Arrange priority academic counseling, tutoring, and remedial coursework support.")
        if not actions:
            actions.append("Schedule urgent one-on-one intervention meeting with department advisor.")

    elif risk_level == "Medium":
        actions.append("Schedule a proactive mentor check-in within the next 2 weeks.")
        if tuition_up_to_date == 0 or debtor == 1:
            actions.append("Advise on upcoming fee deadlines and financial advisory resources.")
        if total_enrolled > 0 and approval_rate < 0.75:
            actions.append("Offer optional academic mentoring and study group recommendations.")

    else:  # Low risk
        actions.append("No immediate intervention needed; continue routine progress monitoring.")

    return actions

