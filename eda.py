import pandas as pd

# Load datasets
leads = pd.read_csv("data/leads.csv")
interactions = pd.read_csv("data/interactions.csv")

print("===== LEADS DATASET =====")
print("Shape:", leads.shape)

print("\nColumns:")
print(leads.columns.tolist())

print("\n===== INTERACTIONS DATASET =====")
print("Shape:", interactions.shape)

print("\nColumns:")
print(interactions.columns.tolist())

print("\n===== EVENT TYPES =====")
print(interactions["event_type"].value_counts())

print("\n===== EVENT NAMES =====")
print(interactions["event_name"].value_counts().head(50))

print("\n===== FUNNEL STAGES =====")
print(interactions["funnel_stage"].value_counts())

print("\n===== LEADS MISSING VALUES =====")
print(leads.isnull().sum())

print("\n===== INTERACTIONS MISSING VALUES =====")
print(interactions.isnull().sum())

# Conversion Plot

import matplotlib.pyplot as plt
import pandas as pd

leads = pd.read_csv("data/leads_with_target.csv")

plt.figure(figsize=(6,4))
leads["converted"].value_counts().plot(kind="bar")

plt.title("Conversion Distribution")
plt.xlabel("Converted")
plt.ylabel("Count")

plt.tight_layout()
plt.savefig("outputs/conversion_distribution.png")
plt.show()

#Lead Source vs Conversion

source_conv = pd.crosstab(
    leads["source"],
    leads["converted"]
)

source_conv.plot(
    kind="bar",
    figsize=(8,5)
)

plt.title("Lead Source vs Conversion")
plt.tight_layout()

plt.savefig("outputs/source_conversion.png")
plt.show()

# Funnel Stage Distribution

interactions = pd.read_csv("data/interactions.csv")

interactions["funnel_stage"].value_counts().plot(
    kind="bar",
    figsize=(8,5)
)

plt.title("Funnel Stage Distribution")
plt.tight_layout()

plt.savefig("outputs/funnel_stage_distribution.png")
plt.show()

# Top Interaction Events

interactions["event_name"].value_counts().head(10).plot(
    kind="bar",
    figsize=(10,5)
)

plt.title("Top Interaction Events")
plt.tight_layout()

plt.savefig("outputs/event_distribution.png")
plt.show()