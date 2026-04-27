import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    get_feature_types,
    handle_missing_values,
    encode_features,
    standardize_features,
    NOMINAL_FEATURES,
    BINARY_FEATURES,
    CONTINUOUS_FEATURES,
)


class TestGetFeatureTypes:
    def test_classifies_known_columns(self, raw_features):
        ft = get_feature_types(raw_features)
        assert set(ft["nominal"]) <= set(raw_features.columns)
        assert set(ft["binary"]) <= set(raw_features.columns)
        assert set(ft["continuous"]) <= set(raw_features.columns)
        for col in ft["count"]:
            assert col in raw_features.columns

    def test_no_overlap_between_groups(self, raw_features):
        ft = get_feature_types(raw_features)
        all_cols = ft["nominal"] + ft["binary"] + ft["count"] + ft["continuous"]
        assert len(all_cols) == len(set(all_cols))

    def test_all_columns_classified(self, raw_features):
        ft = get_feature_types(raw_features)
        all_cols = set(ft["nominal"] + ft["binary"] + ft["count"] + ft["continuous"])
        assert all_cols == set(raw_features.columns)

    def test_returns_dict_with_required_keys(self, raw_features):
        ft = get_feature_types(raw_features)
        assert "nominal" in ft
        assert "binary" in ft
        assert "count" in ft
        assert "continuous" in ft

    def test_nominal_features_match_constants(self, raw_features):
        ft = get_feature_types(raw_features)
        for col in ft["nominal"]:
            assert col in NOMINAL_FEATURES

    def test_binary_features_match_constants(self, raw_features):
        ft = get_feature_types(raw_features)
        for col in ft["binary"]:
            assert col in BINARY_FEATURES

    def test_continuous_features_match_constants(self, raw_features):
        ft = get_feature_types(raw_features)
        for col in ft["continuous"]:
            assert col in CONTINUOUS_FEATURES

    def test_count_features_are_remainder(self, raw_features):
        """count 特征应为不属于其他三类的剩余列。"""
        ft = get_feature_types(raw_features)
        known = set(NOMINAL_FEATURES + BINARY_FEATURES + CONTINUOUS_FEATURES)
        expected_count = set(raw_features.columns) - known
        assert set(ft["count"]) == expected_count


class TestHandleMissingValues:
    def test_no_missing_returns_copy(self, raw_features):
        result, info = handle_missing_values(raw_features)
        assert info["total_missing"] == 0
        assert result.shape == raw_features.shape

    def test_median_fill(self):
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [10, 20, 30, 40]})
        result, info = handle_missing_values(df, strategy="median")
        assert result["a"].isna().sum() == 0
        assert info["total_missing"] == 1
        # median([1.0, 2.0, 4.0]) = 2.0
        assert result["a"].iloc[2] == 2.0

    def test_mean_fill(self):
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 3.0], "b": [10, 20, 30, 40]})
        result, info = handle_missing_values(df, strategy="mean")
        assert result["a"].isna().sum() == 0
        assert info["total_missing"] == 1
        # mean([1.0, 2.0, 3.0]) = 2.0
        assert abs(result["a"].iloc[2] - 2.0) < 1e-9

    def test_drop_strategy(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [10, 20, 30]})
        result, info = handle_missing_values(df, strategy="drop")
        assert len(result) == 2
        assert result["a"].isna().sum() == 0

    def test_info_contains_required_keys(self, raw_features):
        _, info = handle_missing_values(raw_features)
        assert "total_missing" in info
        assert "missing_per_column" in info
        assert "strategy" in info

    def test_returns_copy_not_inplace(self, raw_features):
        """原 DataFrame 不应被修改。"""
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [10, 20, 30]})
        original_null_count = df["a"].isna().sum()
        handle_missing_values(df, strategy="median")
        # 原 df 仍有缺失值
        assert df["a"].isna().sum() == original_null_count

    def test_strategy_recorded_in_info(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        _, info = handle_missing_values(df, strategy="median")
        assert info["strategy"] == "median"


class TestEncodeFeatures:
    def test_onehot_expands_columns(self, raw_features):
        ft = get_feature_types(raw_features)
        encoded = encode_features(raw_features, ft, method="onehot")
        assert encoded.shape[1] > raw_features.shape[1]
        for col in ft["nominal"]:
            assert col not in encoded.columns

    def test_label_encoding_keeps_shape(self, raw_features):
        ft = get_feature_types(raw_features)
        encoded = encode_features(raw_features, ft, method="label")
        assert encoded.shape == raw_features.shape

    def test_invalid_method_raises(self, raw_features):
        ft = get_feature_types(raw_features)
        with pytest.raises(ValueError, match="Unknown encoding"):
            encode_features(raw_features, ft, method="bad")

    def test_feature_types_optional(self, raw_features):
        """feature_types 参数为可选，不传时应自动推断。"""
        encoded = encode_features(raw_features, method="onehot")
        ft = get_feature_types(raw_features)
        for col in ft["nominal"]:
            assert col not in encoded.columns

    def test_onehot_no_null_in_result(self, raw_features):
        ft = get_feature_types(raw_features)
        encoded = encode_features(raw_features, ft, method="onehot")
        assert encoded.isna().sum().sum() == 0

    def test_label_encoding_numeric_output(self, raw_features):
        """label 编码后名义特征应为整数类型。"""
        ft = get_feature_types(raw_features)
        encoded = encode_features(raw_features, ft, method="label")
        for col in ft["nominal"]:
            assert pd.api.types.is_integer_dtype(encoded[col]), (
                f"{col} should be integer after label encoding"
            )

    def test_binary_features_unchanged(self, raw_features):
        """二值特征编码后保持不变。"""
        ft = get_feature_types(raw_features)
        encoded = encode_features(raw_features, ft, method="onehot")
        for col in ft["binary"]:
            pd.testing.assert_series_equal(
                raw_features[col].reset_index(drop=True),
                encoded[col].reset_index(drop=True),
                check_names=False,
            )


class TestStandardizeFeatures:
    def test_fit_returns_scaler(self, raw_features):
        ft = get_feature_types(raw_features)
        result, scaler = standardize_features(raw_features, ft, fit=True)
        assert scaler is not None
        assert result.shape == raw_features.shape

    def test_transform_without_scaler_raises(self, raw_features):
        ft = get_feature_types(raw_features)
        with pytest.raises(ValueError, match="scaler must be provided"):
            standardize_features(raw_features, ft, fit=False, scaler=None)

    def test_continuous_cols_are_standardized(self, raw_features):
        ft = get_feature_types(raw_features)
        result, _ = standardize_features(raw_features, ft, fit=True)
        for col in ft["continuous"]:
            if col in result.columns:
                assert abs(result[col].mean()) < 0.5

    def test_feature_types_optional(self, raw_features):
        """feature_types 参数为可选，不传时应自动推断。"""
        result, scaler = standardize_features(raw_features, fit=True)
        assert scaler is not None
        assert result.shape == raw_features.shape

    def test_transform_mode_uses_existing_scaler(self, raw_features):
        """transform 模式应使用传入的 scaler，不重新拟合。"""
        ft = get_feature_types(raw_features)
        _, fitted_scaler = standardize_features(raw_features, ft, fit=True)
        result2, scaler2 = standardize_features(raw_features, ft, fit=False, scaler=fitted_scaler)
        assert scaler2 is fitted_scaler
        assert result2.shape == raw_features.shape

    def test_binary_features_not_scaled(self, raw_features):
        """二值特征标准化后应保持 0/1，不被缩放。"""
        ft = get_feature_types(raw_features)
        result, _ = standardize_features(raw_features, ft, fit=True)
        for col in ft["binary"]:
            if col in result.columns:
                unique_vals = set(result[col].unique())
                assert unique_vals <= {0, 1}, (
                    f"Binary feature {col} should remain 0/1 but got {unique_vals}"
                )

    def test_count_cols_are_standardized(self, raw_features):
        """count 特征也应被标准化（均值接近 0）。"""
        ft = get_feature_types(raw_features)
        result, _ = standardize_features(raw_features, ft, fit=True)
        for col in ft["count"]:
            if col in result.columns:
                assert abs(result[col].mean()) < 0.5
