import pandas as pd

leads = pd.read_csv("data/leads.csv")
interactions = pd.read_csv("data/interactions.csv")

# Conversion events
conversion_events = [
    "demo_request",
    "contact_form_submit",
    "free_trial_start"
]

converted_leads = interactions[
    interactions["event_name"].isin(conversion_events)
]["lead_id"].unique()

leads["converted"] = leads["lead_id"].isin(converted_leads).astype(int)

print(leads["converted"].value_counts())

leads.to_csv("data/leads_with_target.csv", index=False)