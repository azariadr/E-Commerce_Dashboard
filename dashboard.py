import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency
sns.set(style='dark')

def create_order_purchase_df(all_df):
    #order_purchase_df = all_df["order_purchase_year_month"].value_counts().sort_index()
    order_purchase_df = all_df.order_purchase_year_month.value_counts().sort_index()
    return order_purchase_df

def create_sum_order_items_df(all_df):
    sum_order_items_df = all_df.product_category_name_english.value_counts()
    return sum_order_items_df

def create_bycity_df(all_df):
    bycity_df = all_df.groupby(by="customer_city").customer_id.nunique().reset_index()
    bycity_df.rename(columns={
        "customer_id": "customer_count"
    }, inplace=True)
    
    return bycity_df

def create_bystate_df(all_df):
    bystate_df = all_df.groupby(by="customer_state").customer_id.nunique().reset_index()
    bystate_df.rename(columns={
        "customer_id": "customer_count"
    }, inplace=True)
    
    return bystate_df

def create_bypayment_df(all_df):
    bypayment_df = all_df['payment_type'].value_counts()
    return bypayment_df

all_df = pd.read_csv("all_data.csv")

order_purchase_df = create_order_purchase_df(all_df)
sum_order_items_df = create_sum_order_items_df(all_df)
bycity_df = create_bycity_df(all_df)
bystate_df = create_bystate_df(all_df)
bypayment_df = create_bypayment_df(all_df)

st.header('E-Commerce Dashboard')

st.subheader('Tren Penjualan')

plt.figure(figsize=(15,6))
#plt.plot(order_purchase_df.value_counts().sort_index())
plt.plot(order_purchase_df)
plt.title('Tren Penjualan')
plt.xticks(rotation=90)
fig = plt.gcf()
st.pyplot(fig)

st.subheader('Top 10 Best and Worst Performing Product')

data = sum_order_items_df

# Menentukan warna untuk highlight
highlight_color = ["blue" if x == data.idxmax() else "gray" for x in data.index]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Subplot pertama - Top 10 Produk Paling Banyak Terjual
sns.barplot(x=data.values[:10], y=data.index[:10], palette=highlight_color[:10], ax=axes[0])
axes[0].set_title('Top 10 Produk Paling Banyak Terjual')
axes[0].tick_params(axis='x')
sns.despine(ax=axes[0])

# Subplot kedua - Top 10 Produk Paling Sedikit Terjual
sorted_counts = all_df.product_category_name_english.value_counts().sort_values()
sns.barplot(x=sorted_counts.values[:10], y=sorted_counts.index[:10], palette=highlight_color[:10], ax=axes[1])
axes[1].set_title('Top 10 Produk Paling Sedikit Terjual')
axes[1].tick_params(axis='y')
sns.despine(ax=axes[1])

plt.tight_layout()
st.pyplot(fig)

st.subheader("Customer Demographics")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

highlight_color = ["blue" if x == data.idxmax() else "gray" for x in data.index]

# Subplot pertama - Top 10 Pelanggan berdasarkan Kota
sns.barplot(x=bycity_df["customer_count"].values[:10],
            y=bycity_df["customer_city"][:10],
            palette=highlight_color[:10], ax=axes[0])
axes[0].set_title('Top 10 Pelanggan berdasarkan Kota')
axes[0].tick_params(axis='x')
sns.despine(ax=axes[0])

# Subplot kedua - Top 10 Pelanggan berdasarkan Negara Bagian
sns.barplot(x=bystate_df["customer_count"].values[:10],
            y=bystate_df["customer_state"][:10],
            palette=highlight_color[:10], ax=axes[1])
axes[1].set_title('Top 10 Pelanggan berdasarkan Negara Bagian')
axes[1].tick_params(axis='x')
sns.despine(ax=axes[1])

plt.tight_layout()
fig = plt.gcf()
st.pyplot(fig)
 
st.caption('Copyright (c) Azaria 2023')
