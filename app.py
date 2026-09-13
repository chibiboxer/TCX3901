import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# -----------------------------
# Page setup
# -----------------------------

st.set_page_config(
    page_title="Olist Order Value Dashboard",
    layout="wide"
)

st.title("Olist Order Value Analytics Dashboard")

st.write(
    "Exploratory and predictive analysis of order value "
    "using the Brazilian E-Commerce Public Dataset."
)

# -----------------------------
# Load data
# -----------------------------

df = pd.read_csv("olist_order_level.csv")

# Make sure order_value exists
if "order_value" not in df.columns:
    df["order_value"] = (
        df["total_price"] + df["total_freight"]
    )

# -----------------------------
# Create high-value classification
# -----------------------------

threshold = df["order_value"].quantile(0.80)

df["high_value_order"] = (
    df["order_value"] >= threshold
).astype(int)

# -----------------------------
# KPI section
# -----------------------------

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Orders",
    f"{len(df):,}"
)

col2.metric(
    "Average Order Value",
    f"R$ {df['order_value'].mean():,.2f}"
)

col3.metric(
    "High-Value Threshold",
    f"R$ {threshold:,.2f}"
)

st.divider()

# -----------------------------
# Chart 1: Order Value Distribution
# -----------------------------

st.subheader("1. Distribution of Order Value")

fig, ax = plt.subplots()

ax.hist(
    df["order_value"],
    bins=50
)

ax.axvline(
    threshold,
    linestyle="--",
    label=f"80th percentile: R$ {threshold:,.2f}"
)

ax.set_xlabel("Order Value (R$)")
ax.set_ylabel("Number of Orders")
ax.set_title("Distribution of Order Value")
ax.legend()

st.pyplot(fig)

# -----------------------------
# Chart 2: Number of Items vs Order Value
# -----------------------------

st.subheader("2. Number of Items vs Order Value")

fig, ax = plt.subplots()

ax.scatter(
    df["number_of_items"],
    df["order_value"],
    alpha=0.3
)

ax.set_xlabel("Number of Items")
ax.set_ylabel("Order Value (R$)")
ax.set_title("Number of Items vs Order Value")

st.pyplot(fig)

# ============================================================
# Chart 3. Order Value Distribution and High-Value Threshold
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42
)

high_value_threshold = (
    train_df["order_value"]
    .quantile(0.80)
)

# Create classification target

train_df["high_value_order"] = (
    train_df["order_value"]
    >= high_value_threshold
).astype(int)

test_df["high_value_order"] = (
    test_df["order_value"]
    >= high_value_threshold
).astype(int)

train_class_distribution = (
    train_df["high_value_order"]
    .value_counts()
    .sort_index()
)

train_class_percentage = (
    train_df["high_value_order"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

st.subheader("3. Order Value Distribution and High-Value Threshold")

# ============================================================
# Distribution of High-Value Orders
# ============================================================

# Count each class
class_counts = (
    train_df["high_value_order"]
    .value_counts()
    .sort_index()
)

# Create figure
fig, ax = plt.subplots(figsize=(8, 5))

class_counts.plot(
    kind="bar",
    ax=ax
)

# X-axis labels
ax.set_xticks([0, 1])
ax.set_xticklabels(
    ["Standard Order", "High-Value Order"],
    rotation=0
)

# Labels and title
ax.set_xlabel("Order Class")
ax.set_ylabel("Number of Orders")
ax.set_title("Distribution of High-Value Orders")

plt.tight_layout()

# Display in Streamlit
st.pyplot(fig)

# Close figure to prevent overlap/memory issues
plt.close(fig)

# -----------------------------
# Chart 4: Standard vs High-Value
# -----------------------------

st.subheader("4. Number of Items by Order Classification")

fig, ax = plt.subplots()

df.boxplot(
    column="number_of_items",
    by="high_value_order",
    ax=ax
)

ax.set_xlabel("Order Classification")
ax.set_ylabel("Number of Items")
ax.set_title("Number of Items by Order-Value Classification")

plt.suptitle("")

ax.set_xticklabels(
    ["Standard-value", "High-value"]
)

st.pyplot(fig)
