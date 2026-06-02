# app.py
# DemandIQ - Professional Streamlit E-commerce Demand Forecasting Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="DemandIQ",
    page_icon="📈",
    layout="wide"
)

st.title("📈 DemandIQ - Smart Retail Demand Forecasting")
st.markdown("AI Powered E-commerce Analytics Dashboard")

# ---------------- LOAD DATA ---------------- #
@st.cache_data
def load_data():
    df = pd.read_csv("demand.csv")
    df["week"] = pd.to_datetime(df["week"], format="%d/%m/%y")
    return df

df = load_data()

# ---------------- DATA PREPROCESS ---------------- #
df["revenue"] = df["units_sold"] * df["total_price"]
df["discount"] = df["base_price"] - df["total_price"]

# ---------------- SIDEBAR ---------------- #
st.sidebar.title("Navigation")

page = st.sidebar.selectbox(
    "Select Module",
    [
        "Dashboard",
        "Demand Forecast",
        "Product Analysis",
        "Store Analysis",
        "Promotion Impact",
        "Pricing Insights"
    ]
)

# ==========================================================
# DASHBOARD
# ==========================================================
if page == "Dashboard":

    st.subheader("📊 Business Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Sales Units", int(df["units_sold"].sum()))
    col2.metric("Revenue", f"₹ {round(df['revenue'].sum(),2)}")
    col3.metric("Products", df["sku_id"].nunique())
    col4.metric("Stores", df["store_id"].nunique())

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        weekly_sales = df.groupby("week")["units_sold"].sum().reset_index()

        fig = px.line(
            weekly_sales,
            x="week",
            y="units_sold",
            title="Weekly Sales Trend"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        top_products = df.groupby("sku_id")["units_sold"].sum().reset_index()
        top_products = top_products.sort_values(by="units_sold", ascending=False).head(10)

        fig = px.bar(
            top_products,
            x="sku_id",
            y="units_sold",
            title="Top 10 Products"
        )
        st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# DEMAND FORECAST
# ==========================================================
elif page == "Demand Forecast":

    st.subheader("🔮 Product Demand Forecast")

    sku = st.selectbox("Select Product SKU", df["sku_id"].unique())

    product_df = df[df["sku_id"] == sku]

    ts = product_df.groupby("week")["units_sold"].sum().reset_index()
    ts.columns = ["ds", "y"]

    if len(ts) < 10:
        st.warning("Not enough data for forecasting.")
    else:
        model = Prophet()
        model.fit(ts)

        future = model.make_future_dataframe(periods=12, freq="W")
        forecast = model.predict(future)

        fig = px.line(
            forecast,
            x="ds",
            y="yhat",
            title=f"Next 12 Weeks Forecast for SKU {sku}"
        )

        fig.add_scatter(
            x=ts["ds"],
            y=ts["y"],
            mode="lines+markers",
            name="Actual Sales"
        )

        st.plotly_chart(fig, use_container_width=True)

        future_sales = round(forecast["yhat"].tail(12).sum())
        st.success(f"Expected Next 12 Weeks Sales: {future_sales} Units")

# ==========================================================
# PRODUCT ANALYSIS
# ==========================================================
elif page == "Product Analysis":

    st.subheader("🛍️ Product Performance")

    product_stats = df.groupby("sku_id").agg({
        "units_sold":"sum",
        "revenue":"sum",
        "discount":"mean"
    }).reset_index()

    product_stats = product_stats.sort_values(by="units_sold", ascending=False)

    st.dataframe(product_stats.head(20))

    fig = px.scatter(
        product_stats,
        x="discount",
        y="units_sold",
        size="revenue",
        title="Discount vs Sales"
    )
    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# STORE ANALYSIS
# ==========================================================
elif page == "Store Analysis":

    st.subheader("🏬 Store Performance")

    store_stats = df.groupby("store_id").agg({
        "units_sold":"sum",
        "revenue":"sum"
    }).reset_index()

    st.dataframe(store_stats.sort_values(by="revenue", ascending=False).head(20))

    fig = px.bar(
        store_stats.sort_values(by="revenue", ascending=False).head(15),
        x="store_id",
        y="revenue",
        title="Top Revenue Stores"
    )
    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# PROMOTION IMPACT
# ==========================================================
elif page == "Promotion Impact":

    st.subheader("🎯 Promotion Effectiveness")

    featured = df.groupby("is_featured_sku")["units_sold"].mean().reset_index()
    display = df.groupby("is_display_sku")["units_sold"].mean().reset_index()

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            featured,
            x="is_featured_sku",
            y="units_sold",
            title="Featured Product Impact"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            display,
            x="is_display_sku",
            y="units_sold",
            title="Display Product Impact"
        )
        st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# PRICING INSIGHTS
# ==========================================================
elif page == "Pricing Insights":

    st.subheader("💰 Pricing Optimization")

    avg_discount = round(df["discount"].mean(),2)
    avg_price = round(df["total_price"].mean(),2)

    col1, col2 = st.columns(2)

    col1.metric("Average Discount", avg_discount)
    col2.metric("Average Selling Price", avg_price)

    fig = px.scatter(
        df.sample(3000),
        x="total_price",
        y="units_sold",
        color="discount",
        title="Price vs Units Sold"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info("Use lower pricing where demand is sensitive. Use premium pricing where demand remains stable.")