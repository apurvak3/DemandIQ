# 📈 DemandIQ – AI Powered E-Commerce Demand Forecasting Platform

DemandIQ is an end-to-end retail analytics and demand forecasting platform built using **Streamlit**, **Python**, and **Machine Learning**. It helps e-commerce businesses predict future product demand, analyze pricing strategies, evaluate promotions, and improve business decision-making through interactive dashboards.

---

## 🚀 Project Overview

E-commerce businesses often struggle with:

- Stock shortages during high demand
- Overstocking slow-moving products
- Ineffective pricing strategies
- Poor promotion planning
- Lack of future demand visibility

DemandIQ solves these problems by using historical sales data and time-series forecasting models.

---

## 🎯 Key Features

### 📊 Business Dashboard
- Total Revenue
- Total Units Sold
- Number of Products
- Number of Stores
- Weekly Sales Trend

### 🔮 Demand Forecasting
- Product-wise future demand prediction
- Next 12 weeks sales forecast
- Trend visualization using Prophet model

### 🛍️ Product Analysis
- Top-selling SKUs
- Revenue by product
- Discount vs Sales performance

### 🏬 Store Analysis
- Best performing stores
- Revenue by store
- Store-wise sales insights

### 🎯 Promotion Impact
- Featured product effectiveness
- Display promotion performance
- Promotion vs non-promotion comparison

### 💰 Pricing Insights
- Price vs Demand relationship
- Discount impact analysis
- Revenue optimization insights

---

## 🧠 Machine Learning Model Used

### Prophet (Time Series Forecasting)

Used for forecasting future product demand based on:

- Historical sales trends
- Weekly patterns
- Seasonal behavior
- Growth / decline trends

---

## 🗂️ Dataset Columns

The project uses `demand.csv` with the following fields:

| Column Name         | Description |
|--------------------|-------------|
| record_ID          | Unique record identifier |
| week               | Week/date of sales |
| store_id           | Store identifier |
| sku_id             | Product identifier |
| total_price        | Final selling price |
| base_price         | Original price |
| is_featured_sku    | Featured product flag |
| is_display_sku     | Display promotion flag |
| units_sold         | Quantity sold |

---

## 🛠️ Tech Stack

### Frontend
- Streamlit

### Backend / Logic
- Python

### Data Processing
- Pandas
- NumPy

### Visualization
- Plotly

### Machine Learning
- Prophet

---

## 📁 Project Structure

```bash
DemandIQ/
│── app.py
│── demand.csv
│── requirements.txt
│── README.md