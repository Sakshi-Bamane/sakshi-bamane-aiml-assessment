import pandas as pd

# Load data
leads = pd.read_csv("data/leads_with_target.csv")
interactions = pd.read_csv("data/interactions.csv")

# Basic aggregations
features = interactions.groupby("lead_id").agg(
    session_count=("session_id", "nunique"),
    total_duration=("session_duration_seconds", "sum"),
    avg_duration=("session_duration_seconds", "mean"),
    avg_scroll_depth=("scroll_depth_percent", "mean"),
    total_clicks=("click_count", "sum"),
    avg_mouse_activity=("mouse_activity_score", "mean"),
    return_visitor=("is_return_visitor", "max")
).reset_index()

# Event counts
event_counts = interactions.pivot_table(
    index="lead_id",
    columns="event_name",
    aggfunc="size",
    fill_value=0
).reset_index()

# Funnel stage counts
funnel_counts = interactions.pivot_table(
    index="lead_id",
    columns="funnel_stage",
    aggfunc="size",
    fill_value=0
).reset_index()

# Merge everything
features = features.merge(event_counts, on="lead_id", how="left")
features = features.merge(funnel_counts, on="lead_id", how="left")

print("Final Feature Shape:")
print(features.shape)

print("\nColumns:")
print(features.columns.tolist())
# -------------------------
# Merge with leads dataset
# -------------------------

dataset = leads.merge(features, on="lead_id", how="left")

# Fill missing values

# Fill numeric columns with 0
numeric_cols = dataset.select_dtypes(include=["number"]).columns
dataset[numeric_cols] = dataset[numeric_cols].fillna(0)

# Fill text columns with "Unknown"
text_cols = dataset.select_dtypes(include=["object", "string"]).columns
dataset[text_cols] = dataset[text_cols].fillna("Unknown")

print("\nFinal Dataset Shape:")
print(dataset.shape)

print("\nTarget Distribution:")
print(dataset["converted"].value_counts())

dataset.to_csv("data/final_dataset.csv", index=False)

print("\nSaved as data/final_dataset.csv")