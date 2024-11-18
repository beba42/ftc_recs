
# %%
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
#%%
# Set display options and style
pd.set_option('display.max_columns', None)
plt.style.use('ggplot')

# Define columns to aggregate with specific rules
agg_dict = {
    'Title': 'first',
    'Body (HTML)': 'first',
    'Vendor': 'first',
    'Product Category': 'first',
    'Type': 'first',
    'Tags': 'first',
    'Published': 'first',
    'Option1 Name': 'first',
    'Option1 Value': 'first',
    'Option1 Linked To': 'first',
    'Option2 Name': 'first',
    'Option2 Value': 'first',
    'Option2 Linked To': 'first',
    'Option3 Name': 'first',
    'Option3 Value': 'first',
    'Option3 Linked To': 'first',
    'Variant SKU': 'first',
    'Variant Grams': 'first',
    'Variant Inventory Tracker': 'first',
    'Variant Inventory Qty': 'first',
    'Variant Inventory Policy': 'first',
    'Variant Fulfillment Service': 'first',
    'Variant Price': 'first',
    'Variant Compare At Price': 'first',
    'Variant Requires Shipping': 'first',
    'Variant Taxable': 'first',
    'Variant Barcode': 'first',
    'Image Src': lambda x: '; '.join(x.dropna()),  # Join image links with ';' separator
    'Image Position': 'first',
    'Image Alt Text': 'first',
    'Gift Card': 'first',
    'SEO Title': 'first',
    'SEO Description': 'first',
    'Google Shopping / Google Product Category': 'first',
    'Google Shopping / Gender': 'first',
    'Google Shopping / Age Group': 'first',
    'Google Shopping / MPN': 'first',
    'Google Shopping / Condition': 'first',
    'Google Shopping / Custom Product': 'first',
    'Google Shopping / Custom Label 0': 'first',
    'Google Shopping / Custom Label 1': 'first',
    'Google Shopping / Custom Label 2': 'first',
    'Google Shopping / Custom Label 3': 'first',
    'Google Shopping / Custom Label 4': 'first',
    'Colours (product.metafields.filter.colourtag)': 'first',
    'Eco-conscious products (product.metafields.filter.sustainabletag)': 'first',
    'Complementary products (product.metafields.shopify--discovery--product_recommendation.complementary_products)': 'first',
    'Related products (product.metafields.shopify--discovery--product_recommendation.related_products)': 'first',
    'Related products settings (product.metafields.shopify--discovery--product_recommendation.related_products_display)': 'first',
    'Variant Image': 'first',
    'Variant Weight Unit': 'first',
    'Variant Tax Code': 'first',
    'Cost per item': 'first',
    # Include other region-specific price columns here, e.g.:
    'Included / Ireland': 'first',
    'Price / Ireland': 'first',
    'Compare At Price / Ireland': 'first',
    # Repeat for each country and region column as per your list
    'Status': 'first'
}

# Define constants
START_DATE = pd.Timestamp("2024-09-25 00:00:00")
END_DATE = pd.Timestamp("2024-10-25 23:59:59")
DECAY_FACTOR = 0.3
#%%
# Load sales data with filters
def load_sales_data(filepath, start_date, end_date):
    filters = [
        [('hour', '>=', start_date), ('hour', '<=', end_date), ('fulfillment_status', '==', 'fulfilled'), ('cancelled', '==', 'No')]
    ]
    return pd.read_parquet(filepath, engine='pyarrow', filters=filters)

# Load product data
def load_product_data(filepaths):
    prod_df = pd.concat([pd.read_csv(path) for path in filepaths], ignore_index=True)
    prod_df = prod_df.groupby('Handle', as_index=False).agg(agg_dict).dropna(subset=['Variant SKU'])
    prod_df['Variant SKU'] = pd.to_numeric(prod_df['Variant SKU'].str.replace("'", ""), errors='coerce')
    return prod_df

# One-hot encode tags
def one_hot_encode_tags(df, tag_column, prefix):
    one_hot_encoded = pd.DataFrame(index=df.index)
    for idx, tags in df[tag_column].dropna().items():
        for tag in tags.split(','):
            tag = tag.strip()
            if ':' in tag:
                tag_prefix, tag_value = tag.split(':', 1)
                if tag_prefix.strip() == prefix:
                    tag_value = re.sub(r'^(GG|AG)', '', tag_value).strip().lower()
                    column_name = f"{prefix}_{tag_value}"
                    one_hot_encoded.loc[idx, column_name] = 1
    return one_hot_encoded.fillna(0).astype(int)

# Extract unique tag value for specific tags like Campaign
def extract_unique_tag_value(df, tag_column, prefix):
    campaign_column = pd.Series(index=df.index, dtype='object')
    for idx, tags in df[tag_column].dropna().items():
        tags = str(tags) if not isinstance(tags, str) else tags
        for tag in tags.split(','):
            if ':' in tag:
                tag_prefix, tag_value = tag.split(':', 1)
                if tag_prefix.strip() == prefix:
                    campaign_column[idx] = tag_value.strip()
                    break
    return campaign_column

# Calculate decay scores
def calculate_decay_scores(df, decay_factor, current_time):
    df['time_diff'] = (current_time - df['hour']).dt.total_seconds() / (3600 * 24)
    df['decay_score'] = np.exp(-decay_factor * df['time_diff'])
    return df

# Calculate top recommendations
def get_top_recommendations(df, group_cols, k=9):
    return (
        df.sort_values(by=['shipping_country', 'decay_score'], ascending=[True, False])
          .groupby(group_cols)
          .head(k)
          .groupby(group_cols)['variant_sku']
          .apply(list)
          .to_dict()
    )

def get_general_recommendations(df,k=9):
    recs = (df.groupby('variant_sku')['decay_score'].sum()
            .sort_values( ascending=[False])
            .head(k)
            .index
            .tolist())
    rec_dict = {
        country:recs
        for country in df['shipping_country'].unique()
    }
    return rec_dict

# Precision@K and Recall@K
def precision_at_k(predictions, actuals, k=9):
    if not actuals:
        return 0.0
    hits = len(set(predictions[:k]) & set(actuals))
    return hits / min(k, len(predictions))

def recall_at_k(predictions, actuals, k=9):
    if not actuals:
        return 0.0
    hits = len(set(predictions[:k]) & set(actuals))
    return hits / len(actuals)

# Calculate metrics for each email address
def calculate_metrics(recommendations, actual_data):
    precisions, recalls = [], []
    for _, row in actual_data.iterrows():
        actual_skus = row['variant_sku']
        recommendations_for_country = recommendations.get(row['shipping_country'], [])
        precisions.append(precision_at_k(recommendations_for_country, actual_skus))
        recalls.append(recall_at_k(recommendations_for_country, actual_skus))
    return sum(precisions) / len(precisions), sum(recalls) / len(recalls)
#%%
# Execution Flow
# Step 1: Load data
sales_df = load_sales_data('data/sales/online_sales.parquet', START_DATE, END_DATE)
prod_df = load_product_data(['data/products_export_1.csv', 'data/products_export_2.csv'])

# Step 2: Data Processing
prod_df['Campaign'] = extract_unique_tag_value(prod_df, 'Tags', 'Campaign')

sales_df = calculate_decay_scores(sales_df, DECAY_FACTOR, sales_df['hour'].max())

# Aggregate decay scores by 'shipping_country', 'product_title', and 'variant_sku'
relevance_scores_by_country = (
    sales_df
    .groupby(['shipping_country', 'product_title', 'variant_sku'])['decay_score']
    .sum()
    .reset_index()
)

relevance_with_campaign = pd.merge(
    relevance_scores_by_country,
    prod_df[['Variant SKU', 'Campaign']],
    left_on='variant_sku',
    right_on='Variant SKU',
    how='left'
).drop(columns=['Variant SKU'])
#%%
# Step 3: Generate recommendations
general_recommendations = get_general_recommendations(sales_df, k=9)
#%%

country_specific_recommendations = get_top_recommendations(relevance_scores_by_country, ['shipping_country'], k=9)
#%%
# Step 4: Calculate metrics for general and country-specific predictions
sales_test_df = pd.read_csv('data/sales/sales_2024-10-26_2024-11-01.csv')
actual_purchases = sales_test_df.groupby(['customer_email', 'shipping_country'])['variant_sku'].apply(list).reset_index()

# most frequent campaing in last week of sales df
prev_week = sales_df.query('hour > "2024-11-18"')
prev_week = prev_week.merge(
    prod_df[['Variant SKU','Campaign']], 
    how='inner', 
    left_on='variant_sku', 
    right_on='Variant SKU'
).drop(columns=['Variant SKU'])


prev_week_pop = prev_week[['order_name','Campaign']].groupby('Campaign').count().sort_values(by = 'order_name', ascending=False)


#%%

# General Recommendations Metrics
precision_at_9, recall_at_9 = calculate_metrics(general_recommendations, actual_purchases)
print(f"General Recommendations - Precision@9: {precision_at_9:.4f}, Recall@9: {recall_at_9:.4f}")

# Country-Specific Recommendations Metrics
precision_at_9, recall_at_9 = calculate_metrics(country_specific_recommendations, actual_purchases)
print(f"Country-Specific Recommendations - Precision@9: {precision_at_9:.4f}, Recall@9: {recall_at_9:.4f}")

# Step 5: Campaign-filtered Recommendations Metrics
filtered_campaign_df = relevance_with_campaign[relevance_with_campaign['Campaign'].str.startswith('2411', na=False)]
campaign_recommendations = get_top_recommendations(filtered_campaign_df, ['shipping_country'], k=9)

precision_at_9, recall_at_9 = calculate_metrics(campaign_recommendations, actual_purchases)
print(f"Country-Specific Campaign '2411' Recommendations - Precision@9: {precision_at_9:.4f}, Recall@9: {recall_at_9:.4f}")

# %%
