import pandas as pd
import streamlit as st
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing

def preprocess_data(df, target_col, protected_col):
    df = df.copy()

    # Handle missing values
    df = df.replace("?", pd.NA)
    df = df.dropna(subset=[target_col, protected_col])

    # Strip spaces from string columns
    for col in df.columns:
        if df[col].dtype == "object" or str(df[col].dtype).startswith("string"):
            df[col] = df[col].astype(str).str.strip()

    # Encode target column
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        unique_vals = sorted(df[target_col].dropna().unique())

        if len(unique_vals) == 2:
            df[target_col] = df[target_col].map({
                unique_vals[0]: 0,
                unique_vals[1]: 1
            })
        else:
            raise ValueError("Target must be binary for fairness analysis")

    # Encode protected column
    if not pd.api.types.is_numeric_dtype(df[protected_col]):
        top = df[protected_col].value_counts().idxmax()
        df[protected_col] = (df[protected_col] == top).astype(int)

    # Save protected + target
    protected_data = df[protected_col]
    target_data = df[target_col]

    # Drop them before encoding
    features = df.drop(columns=[target_col, protected_col])

    # One-hot encode features
    features = pd.get_dummies(features, drop_first=True)

    # Convert booleans to int
    features = features.astype(int)

    # Rebuild dataframe
    df = pd.concat([features, target_data, protected_data], axis=1)

    # Final hard numeric conversion
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    return df


def prepare_dataset(df, label_col, protected_col):

    dataset = BinaryLabelDataset(
        df=df,
        label_names=[label_col],
        protected_attribute_names=[protected_col],
        favorable_label=1,
        unfavorable_label=0
    )

    return dataset


def measure_bias(dataset, protected_col, privileged_value, unprivileged_value):

    metric = BinaryLabelDatasetMetric(
        dataset,
        privileged_groups=[{protected_col: privileged_value}],
        unprivileged_groups=[{protected_col: unprivileged_value}]
    )

    di = metric.disparate_impact()

    if pd.isna(di) or di == float('inf'):
        raise ValueError("Invalid Disparate Impact — check group distribution")
    
    return di

# grouped

def mitigate_bias(dataset, protected_col, privileged_value, unprivileged_value):

    RW = Reweighing(
        unprivileged_groups=[{protected_col: unprivileged_value}],
        privileged_groups=[{protected_col: privileged_value}]
    )
    
    return RW.fit_transform(dataset)

def group_outcome_rates(dataset, protected_col, use_weights=False):
    result = dataset.convert_to_dataframe()
    df = result[0] if isinstance(result, tuple) else result

    df = df.copy().reset_index(drop=True)

    label = dataset.label_names[0]

    if use_weights:
        weights = dataset.instance_weights

        if len(weights) != len(df):
            raise ValueError("Weights and dataframe size mismatch")

        df["weights"] = weights

        grouped = df.groupby(protected_col, group_keys=False).apply(
            lambda x: (x[label] * x["weights"]).sum() / x["weights"].sum()
        )
    else:
        grouped = df.groupby(protected_col)[label].mean()

    return grouped.to_dict()

def map_group_value(col, val):
    col = col.lower()

    # Handle common protected attributes
    if col in ["gender", "sex"]:
        return "Male" if val == 1 else "Female"

    if col in ["race"]:
        return "Majority" if val == 1 else "Others"

    if col in ["age"]:
        return "Older" if val == 1 else "Younger"

    # fallback
    return f"{col}={val}"