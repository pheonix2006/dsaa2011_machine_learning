import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    add_academic_features,
    add_socioeconomic_features,
    add_financial_risk_features,
    add_demographic_features,
    add_macroeconomic_features,
    add_application_features,
    add_all_engineered_features,
    target_encode_column,
    apply_target_encoding_oof,
    select_features_by_importance,
)


class TestAcademicFeatures:
    def test_adds_expected_columns(self, raw_features):
        result = add_academic_features(raw_features)
        expected_new = [
            "approval_rate_1st", "approval_rate_2nd", "total_approved",
            "total_enrolled", "overall_approval_rate", "grade_improvement",
        ]
        for col in expected_new:
            assert col in result.columns

    def test_does_not_modify_original(self, raw_features):
        original_cols = list(raw_features.columns)
        add_academic_features(raw_features)
        assert list(raw_features.columns) == original_cols

    def test_approval_rate_bounded(self, raw_features):
        result = add_academic_features(raw_features)
        assert result["approval_rate_1st"].min() >= 0
        assert not result["approval_rate_1st"].isna().any()


class TestSocioeconomicFeatures:
    def test_adds_expected_columns(self, raw_features):
        result = add_socioeconomic_features(raw_features)
        assert "parent_max_qualification" in result.columns
        assert "parent_same_occupation" in result.columns

    def test_same_occupation_is_binary(self, raw_features):
        result = add_socioeconomic_features(raw_features)
        assert set(result["parent_same_occupation"].unique()) <= {0, 1}


class TestFinancialRiskFeatures:
    def test_adds_expected_columns(self, raw_features):
        result = add_financial_risk_features(raw_features)
        assert "financial_risk" in result.columns
        assert "no_financial_stress" in result.columns

    def test_binary_output(self, raw_features):
        result = add_financial_risk_features(raw_features)
        assert set(result["financial_risk"].unique()) <= {0, 1}


class TestDemographicFeatures:
    def test_adds_expected_columns(self, raw_features):
        result = add_demographic_features(raw_features)
        assert "is_mature_student" in result.columns

    def test_mature_threshold(self, raw_features):
        result = add_demographic_features(raw_features)
        mature_mask = result["is_mature_student"] == 1
        assert (raw_features.loc[mature_mask, "Age at enrollment"] > 23).all()


class TestMacroeconomicFeatures:
    def test_adds_economic_condition(self, raw_features):
        result = add_macroeconomic_features(raw_features)
        assert "economic_condition" in result.columns


class TestApplicationFeatures:
    def test_adds_expected_columns(self, raw_features):
        result = add_application_features(raw_features)
        assert "is_first_application" in result.columns
        assert "is_late_application" in result.columns


class TestAllEngineeredFeatures:
    def test_adds_all_groups(self, raw_features):
        result = add_all_engineered_features(raw_features)
        assert result.shape[1] > raw_features.shape[1]
        assert "approval_rate_1st" in result.columns
        assert "financial_risk" in result.columns
        assert "economic_condition" in result.columns


class TestTargetEncoding:
    def test_encode_column_returns_series(self, raw_features, raw_targets):
        train_enc, test_enc = target_encode_column(
            raw_features["Course"].iloc[:60],
            raw_features["Course"].iloc[60:],
            raw_targets.iloc[:60],
        )
        assert len(train_enc) == 60
        assert len(test_enc) == 20
        assert not train_enc.isna().any()

    def test_oof_encoding_same_length(self, raw_features, raw_targets):
        X_train = raw_features.iloc[:60].copy()
        X_test = raw_features.iloc[60:].copy()
        y_train = raw_targets.iloc[:60]
        X_train_enc, X_test_enc = apply_target_encoding_oof(
            X_train, X_test, y_train, columns=["Course"], n_folds=3,
        )
        assert X_train_enc.shape == X_train.shape
        assert not X_train_enc["Course"].isna().any()


class TestFeatureSelection:
    def test_select_top_k(self, sample_features, sample_targets):
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(random_state=42, max_depth=3)
        model.fit(sample_features, sample_targets)
        selected, top_features = select_features_by_importance(sample_features, model, top_k=5)
        assert selected.shape[1] == 5
        assert len(top_features) == 5

    def test_raises_without_importances(self, sample_features):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=200)
        model.fit(sample_features, [0]*50 + [1]*50)
        with pytest.raises(ValueError, match="feature_importances_"):
            select_features_by_importance(sample_features, model)
