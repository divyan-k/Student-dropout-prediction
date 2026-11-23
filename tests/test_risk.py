"""Unit tests for risk level classification and intervention suggestions."""

import pytest
from src.risk import classify_risk, suggest_actions


def test_risk_levels():
    """Verify threshold boundary mappings."""
    # Low risk (< 0.33)
    assert classify_risk(0.0) == "Low"
    assert classify_risk(0.20) == "Low"
    assert classify_risk(0.329) == "Low"

    # Medium risk (0.33 to < 0.66)
    assert classify_risk(0.33) == "Medium"
    assert classify_risk(0.50) == "Medium"
    assert classify_risk(0.659) == "Medium"

    # High risk (>= 0.66)
    assert classify_risk(0.66) == "High"
    assert classify_risk(0.85) == "High"
    assert classify_risk(1.0) == "High"


def test_suggest_actions_high_risk_unpaid_fees():
    """Test intervention rule when high risk and tuition fees unpaid."""
    student = {
        "Tuition fees up to date": 0,
        "Debtor": 1,
        "Curricular units 1st sem (enrolled)": 6,
        "Curricular units 1st sem (approved)": 1,
    }
    actions = suggest_actions("High", student)
    assert any("financial aid" in a.lower() for a in actions)
    assert any("payment plan" in a.lower() for a in actions)


def test_suggest_actions_high_risk_academic():
    """Test intervention rule for high risk and low academic approval rate."""
    student = {
        "Tuition fees up to date": 1,
        "Debtor": 0,
        "Curricular units 1st sem (enrolled)": 6,
        "Curricular units 1st sem (approved)": 1,
    }
    actions = suggest_actions("High", student)
    assert any("academic counseling" in a.lower() for a in actions)


def test_suggest_actions_low_risk():
    """Test low risk students receive routine monitoring advice."""
    student = {
        "Tuition fees up to date": 1,
        "Debtor": 0,
    }
    actions = suggest_actions("Low", student)
    assert len(actions) == 1
    assert "routine" in actions[0].lower()


def test_suggest_actions_medium_risk():
    """Test medium risk students get proactive mentor check-in."""
    student = {
        "Tuition fees up to date": 1,
        "Debtor": 0,
    }
    actions = suggest_actions("Medium", student)
    assert any("mentor" in a.lower() for a in actions)

