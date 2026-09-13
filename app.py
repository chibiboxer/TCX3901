import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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

st.subheader("3. Order Value Distribution and High-Value Threshold")

fig, ax = plt.subplots(figsize=(10, 5))

# Histogram of order values
ax.hist(
    df["order_value"],
    bins=50
)

# High-value threshold
ax.axvline(
    threshold,
    linestyle="--",
    linewidth=2,
    label=f"High-value threshold: R$ {threshold:,.2f}"
)

ax.set_xlabel("Order Value (R$)")
ax.set_ylabel("Number of Orders")

ax.set_title(
    "Order Value Distribution and High-Value Threshold"
)

ax.legend()

st.pyplot(fig)
plt.close(fig)

# Explanation below the chart
st.info(
    f"Orders with a value of R$ {threshold:,.2f} or above "
    "are classified as high-value orders (top 20%)."
)

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
