import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import numpy as np
import pandas as pd

st.set_page_config(layout="wide")
st.title("📦 E-Commerce Public Dataset Analysis Dashboard")

# Load and clean data
all_df = pd.read_csv("all_data.csv")

date_columns = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date']
for col in date_columns:
    all_df[col] = pd.to_datetime(all_df[col], errors='coerce')

all_df['order_approved_at'].fillna(all_df['order_purchase_timestamp'], inplace=True)
all_df['order_delivered_customer_date'].fillna(all_df['order_estimated_delivery_date'], inplace=True)
all_df['order_purchase_year_month'] = all_df['order_purchase_timestamp'].dt.to_period('M').astype(str)

# KPI Section
total_sales = all_df['payment_value'].sum()
total_customers = all_df['customer_unique_id'].nunique()
avg_review = all_df['review_score'].mean()

col1, col2, col3 = st.columns(3)
col1.metric("💰 Total Sales", f"${total_sales:,.0f}")
col2.metric("👥 Total Customers", f"{total_customers}")
col3.metric("⭐️ Average Review", f"{avg_review:.2f}")

# Compute customer frequency
customer_unique_ids = all_df['customer_unique_id'].to_numpy().flatten()
unique, counts = np.unique(customer_unique_ids, return_counts=True)
customer_unique_ids_counts = dict(zip(unique, counts))
multiple_orders = dict(zip(*np.unique(counts, return_counts=True)))

# RFM Calculation
last_date = all_df['order_delivered_carrier_date'].max() + pd.to_timedelta(1, 'D')
rfm_df = all_df.dropna(subset=['order_delivered_carrier_date']).groupby('customer_id').agg(
    recency=('order_delivered_carrier_date', lambda x: (last_date - x.max()).days),
    frequency=('order_id', 'size'),
    monetary=('payment_value', 'sum')
)

rfm_df['r_rank'] = rfm_df['recency'].rank(ascending=False)
rfm_df['f_rank'] = rfm_df['frequency'].rank(ascending=True)
rfm_df['m_rank'] = rfm_df['monetary'].rank(ascending=True)

rfm_df['r_rank_norm'] = (rfm_df['r_rank'] / rfm_df['r_rank'].max()) * 100
rfm_df['f_rank_norm'] = (rfm_df['f_rank'] / rfm_df['f_rank'].max()) * 100
rfm_df['m_rank_norm'] = (rfm_df['m_rank'] / rfm_df['m_rank'].max()) * 100

rfm_df['RFM_score'] = 0.15 * rfm_df['r_rank_norm'] + 0.28 * rfm_df['f_rank_norm'] + 0.57 * rfm_df['m_rank_norm']
rfm_df['RFM_score'] *= 0.05
rfm_df = rfm_df.round(2)

rfm_df['customer_segment'] = np.select(
    condlist=[
        rfm_df['RFM_score'] > 4.5,
        rfm_df['RFM_score'] > 4,
        rfm_df['RFM_score'] > 3,
        rfm_df['RFM_score'] > 1.6
    ],
    choicelist=['Top', 'High', 'Medium', 'Low'],
    default='Lost Customers'
)

segmentwise = rfm_df.groupby('customer_segment').agg(
    RecencyMean=('recency', 'mean'),
    FrequencyMean=('frequency', 'mean'),
    MonetaryMean=('monetary', 'mean'),
    GroupSize=('recency', 'size')
)

# Tabs
tab1, tab2, tab3 = st.tabs(["📈 Sales Trend", "📊 Product & Demographics", "🎯 RFM Analysis"])

with tab1:
    st.subheader("Sales Trend")
    unique_months = sorted(all_df['order_purchase_year_month'].dropna().unique())
    selected_months = st.multiselect("Select Months", unique_months, default=unique_months)

    filtered_sales = all_df[all_df['order_purchase_year_month'].isin(selected_months)]
    fig_sales, ax_sales = plt.subplots(figsize=(15, 6))
    filtered_counts = filtered_sales['order_purchase_year_month'].value_counts().sort_index()
    ax_sales.plot(filtered_counts.index, filtered_counts.values)
    ax_sales.set_title('Sales per Month')
    ax_sales.tick_params(axis='x', rotation=90)
    st.pyplot(fig_sales)
    st.caption("📌 Sales increased steadily from Sep 2016 to Mar 2018, peaking at ~8,000 orders. A sharp drop followed through Sep 2018.")

with tab2:
    st.subheader("Best-Selling and Least-Selling Products")

    data = all_df.product_category_name_english.value_counts()

    max_category = data.idxmax()
    sorted_counts = data.sort_values()
    least_sold_category = sorted_counts.idxmin()

    top_10_index = data.index[:10]
    bottom_10_index = sorted_counts.index[:10]

    # Setup figure
    fig_products, axes_products = plt.subplots(1, 2, figsize=(15, 6))

    # Top 10 Best-Selling Products
    sns.barplot(
        x=data.values[:10],
        y=data.index[:10],
        hue=[x == max_category for x in top_10_index],
        palette={True: '#ADD8E6', False: 'gray'},
        ax=axes_products[0],
        legend=False
    )
    axes_products[0].set_title('Top 10 Best-Selling Products')
    axes_products[0].tick_params(axis='x')
    sns.despine(ax=axes_products[0])

    # Bottom 10 Least-Selling Products
    sns.barplot(
        x=sorted_counts.values[:10],
        y=sorted_counts.index[:10],
        hue=[x == least_sold_category for x in bottom_10_index],
        palette={True: '#ADD8E6', False: 'gray'},
        ax=axes_products[1],
        legend=False
    )
    axes_products[1].set_title('Top 10 Least Sold Products')
    axes_products[1].tick_params(axis='y')
    sns.despine(ax=axes_products[1])

    plt.tight_layout()
    st.pyplot(fig_products)

    st.subheader("Customer Demographics by City and State")

    fig_geo, axes_geo = plt.subplots(1, 2, figsize=(15, 6))

    # City: Top 10
    city_counts = all_df['customer_city'].value_counts()
    top_10_cities = city_counts.index[:10]
    max_city = city_counts.idxmax()

    highlight_hue_cities = [x == max_city for x in top_10_cities]

    sns.barplot(
        x=city_counts.values[:10],
        y=top_10_cities,
        hue=highlight_hue_cities,
        palette={True: '#ADD8E6', False: 'gray'},
        ax=axes_geo[0],
        legend=False
    )
    axes_geo[0].set_title('Top 10 Customers by City')
    axes_geo[0].tick_params(axis='x')
    sns.despine(ax=axes_geo[0])

    # State: Top 10
    state_counts = all_df['customer_state'].value_counts()
    top_10_states = state_counts.index[:10]
    max_state = state_counts.idxmax()

    highlight_hue_states = [x == max_state for x in top_10_states]

    sns.barplot(
        x=state_counts.values[:10],
        y=top_10_states,
        hue=highlight_hue_states,
        palette={True: '#ADD8E6', False: 'gray'},
        ax=axes_geo[1],
        legend=False
    )
    axes_geo[1].set_title('Top 10 Customers by State')
    axes_geo[1].tick_params(axis='x')
    sns.despine(ax=axes_geo[1])

    plt.tight_layout()
    st.pyplot(fig_geo)

    st.subheader("Payment Method Distribution")

    payment_counts = all_df['payment_type'].value_counts()

    pastel_colors = sns.color_palette("pastel", len(payment_counts))
    colors = ['magenta' if x == payment_counts.idxmax() else pastel_colors[i]
            for i, x in enumerate(payment_counts.index)]

    fig_payment, ax_payment = plt.subplots(figsize=(8, 8))
    ax_payment.pie(payment_counts.values,
                explode=[0.05] * len(payment_counts),
                labels=payment_counts.index,
                autopct='%1.1f%%',
                shadow=False,
                startangle=90,
                colors=colors)
    ax_payment.set_title('Distribution of Payment Methods')

    st.pyplot(fig_payment)

    st.subheader("Review Scores Distribution")

    review_score_counts = all_df['review_score'].value_counts().sort_values(ascending=False)
    max_review_score = review_score_counts.idxmax()

    review_score_index = [str(i) for i in review_score_counts.index]
    highlight_hue_review = [str(x) == str(max_review_score) for x in review_score_counts.index]

    fig_review, ax_review = plt.subplots(figsize=(15, 8))
    sns.barplot(
        x=review_score_index,
        y=review_score_counts.values,
        hue=highlight_hue_review,
        palette={True: "blue", False: "gray"},
        legend=False,
        ax=ax_review
    )

    ax_review.set_title('Distribution of Review Scores (Descending)')
    ax_review.set_xlabel('Review Scores')
    ax_review.set_ylabel('Total')
    sns.despine(ax=ax_review)

    st.pyplot(fig_review)

    st.subheader("Customer Order Frequency")

    # Hitung ulang secara eksplisit untuk akurasi
    customer_order_counts = all_df.groupby('customer_unique_id')['order_id'].nunique()
    once = (customer_order_counts == 1).sum()
    more_than_once = (customer_order_counts > 1).sum()

    # Pie chart
    labels = ['once', 'more than once']
    sizes = [once, more_than_once]
    explode = (0, 0.1)
    colors = ['#87CEFA', '#FFA07A']

    fig_freq, ax_freq = plt.subplots()
    ax_freq.pie(
        sizes,
        explode=explode,
        labels=labels,
        autopct='%1.1f%%',
        shadow=True,
        startangle=90,
        colors=colors
    )
    ax_freq.set_title('Customer Order Frequency')

    st.pyplot(fig_freq)

with tab3:
    st.subheader("RFM Segments TreeMap")
    fig_rfm, ax_rfm = plt.subplots(figsize=(12, 8))
    squarify.plot(sizes=segmentwise['GroupSize'], label=segmentwise.index,
                  alpha=0.8, color=sns.color_palette("Set3"), ax=ax_rfm)
    ax_rfm.axis('off')
    st.pyplot(fig_rfm)

    st.subheader("RFM Segment Summary Table")
    st.dataframe(segmentwise.style.format("{:.2f}"))

st.caption('Copyright (c) Azaria 2023')