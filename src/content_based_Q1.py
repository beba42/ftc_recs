#%%
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.sparse import csr_matrix
import faiss
# %%
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
# %%


def load_sales_data(filepath, start_date=None, end_date=None):
    """
    Load sales data from a Parquet file with optional date filtering.
    
    Parameters:
    - filepath: str, path to the Parquet file.
    - start_date: pd.Timestamp or None, optional start date for filtering.
    - end_date: pd.Timestamp or None, optional end date for filtering.
    
    Returns:
    - pd.DataFrame: DataFrame with sales data, optionally filtered by date.
    """
    # Set up the filters list if start_date and end_date are provided
    filters = []
    if start_date and end_date:
        filters = [
            [('hour', '>=', start_date), ('hour', '<=', end_date), ('fulfillment_status', '==', 'fulfilled'), ('cancelled', '==', 'No')]
        ]
    elif not start_date and not end_date:
        # Only apply filters for fulfillment_status and cancelled if no dates are provided
        filters = [
            [('fulfillment_status', '==', 'fulfilled'), ('cancelled', '==', 'No')]
        ]
    
    # Load the data with the filters if specified
    return pd.read_parquet(filepath, engine='pyarrow', filters=filters)


# Load product data
def load_product_data(filepaths):
    prod_df = pd.concat([pd.read_csv(path) for path in filepaths], ignore_index=True)
    prod_df = prod_df.groupby('Handle', as_index=False).agg(agg_dict).dropna(subset=['Variant SKU'])
    prod_df['Variant SKU'] = pd.to_numeric(prod_df['Variant SKU'].str.replace("'", ""), errors='coerce')
    return prod_df

# One-hot encode tags
def one_hot_encode_tags(df, tag_column, prefix):
    """
    Perform one-hot encoding on tags in the specified column that match the given prefix.

    Parameters:
    - df: DataFrame containing the tags column
    - tag_column: Column name containing tags as comma-separated strings
    - prefix: Prefix to filter tags for encoding (e.g., 'Category', 'SKU Type', 'Tag')

    Returns:
    - DataFrame: One-hot encoded columns for each unique tag under the specified prefix.
    """
    # Dictionary to store one-hot encoded columns
    one_hot_dict = {}

    # Iterate over each row in the tag column
    for idx, tags in df[tag_column].dropna().items():
        for tag in tags.split(','):
            tag = tag.strip()
            if isinstance(tag,int):
                print(tag)
                continue
            if ':' in tag:
                tag_prefix, tag_value = tag.split(':', 1)
                if tag_prefix.strip() == prefix:
                    # Clean up the tag value and create a column name
                    tag_value = re.sub(r'^(GG|AG)', '', tag_value).strip().lower()
                    tag_value = re.sub(r'\d+$', '', tag_value)
                    column_name = f"{prefix}_{tag_value}"
                    
                    # Add the value to the dictionary for one-hot encoding
                    if column_name not in one_hot_dict:
                        one_hot_dict[column_name] = pd.Series(0, index=df.index)  # Initialize the column
                    one_hot_dict[column_name].at[idx] = 1  # Set value to 1 for the current row

    # Convert the dictionary to a DataFrame and fill NaNs with 0
    one_hot_encoded = pd.DataFrame(one_hot_dict).fillna(0).astype(int)
    return one_hot_encoded


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

def precision_at_k(predictions, actuals, k=9):
    if len(predictions) == 0:
        return 0.0
    hits = len(set(predictions[:k]) & set(actuals))
    return hits / min(k, len(predictions))


def recall_at_k(predictions, actuals, k=9):
    if not actuals:
        return 0.0
    hits = len(set(predictions[:k]) & set(actuals))
    return hits / len(actuals)

#%%

sales_df = load_sales_data('data/sales/online_sales.parquet')
sales_test_df = pd.read_csv('data/sales/sales_2024-10-26_2024-11-01.csv')
#%%
#removing all emails from sales df that are no in the test set to decrease computational time
# Get the unique customer emails from sales_test_df
test_customers = sales_test_df['customer_email'].unique()

# Filter sales_df to keep only rows with customer_email in test_customers
sales_df = sales_df[sales_df['customer_email'].isin(test_customers)]

overlap_customers = sales_df['customer_email'].unique()

# Filter sales_test_df to keep only rows with customer_email in overlap_customers
sales_test_df = sales_test_df[sales_test_df['customer_email'].isin(overlap_customers)]

#%%
prod_df = load_product_data(['data/products_export_1.csv', 'data/products_export_2.csv'])

# Step 2: Data Processing
prod_df['Campaign'] = extract_unique_tag_value(prod_df, 'Tags', 'Campaign')
# Perform one-hot encoding for the specified tags without 'Campaign'
categories_one_hot = one_hot_encode_tags(prod_df, 'Tags', 'Category')
sku_type_one_hot = one_hot_encode_tags(prod_df, 'Tags', 'SKU Type')
tag_one_hot = one_hot_encode_tags(prod_df, 'Tags', 'Tag')

# Concatenate the one-hot encoded columns with the original DataFrame
prod_df = pd.concat([prod_df, categories_one_hot, sku_type_one_hot, tag_one_hot], axis=1).fillna(0)

# Convert all one-hot encoded columns to integer type
tag_cols = categories_one_hot.columns.union(sku_type_one_hot.columns).union(tag_one_hot.columns)
prod_df = prod_df.astype({col: int for col in tag_cols})


# %%
## Add user purchase history to identify unique items per user
#user_purchase_history = sales_df.groupby('customer_email')['variant_sku'].apply(list).reset_index()
#
#
## Function to calculate a single user profile
#def calculate_user_profile(user, purchased_skus, prod_df, tag_columns):
#    # Extract binary tag vectors for purchased items
#    purchased_items = prod_df[prod_df['Variant SKU'].isin(purchased_skus)]
#    tag_matrix = purchased_items[tag_columns].values  # Matrix of tag values for purchased items
#
#    # Average tag values to create a user profile
#    user_profile = tag_matrix.mean(axis=0) if len(tag_matrix) > 0 else np.zeros(len(tag_columns))
#    return user, user_profile
#
## Build user profiles using parallel processing with progress bar
#def build_user_profiles_parallel(user_purchase_history, prod_df, tag_columns):
#    # Parallel processing of each user profile with tqdm for progress
#    results = Parallel(n_jobs=-1)(delayed(calculate_user_profile)(
#        row['customer_email'], row['variant_sku'], prod_df, tag_columns)
#        for _, row in tqdm(user_purchase_history.iterrows(), total=user_purchase_history.shape[0], desc="Building user profiles")
#    )
#    
#    # Convert results to DataFrame
#    user_profiles = pd.DataFrame.from_dict(dict(results), orient='index', columns=tag_columns)
#    return user_profiles
#
## Define tag columns to include in user profiles
#tag_columns = categories_one_hot.columns.union(sku_type_one_hot.columns).union(tag_one_hot.columns)
#
## Build user profiles DataFrame in parallel
#user_profiles_df = build_user_profiles_parallel(user_purchase_history, prod_df, tag_columns)
#
#
## %%
#user_profiles_df.to_parquet('data/user_profiles.parquet', index=True)
# %%
# Load the DataFrames
user_profiles_df = pd.read_parquet('data/user_profiles.parquet')
prod_vecs = prod_df[['Variant SKU'] + list(tag_cols)]
prod_vecs['Variant SKU'] = prod_vecs['Variant SKU'].astype(int)
prod_vecs = prod_vecs.set_index('Variant SKU')

# Step 1: Store the indices as separate arrays
user_indices = user_profiles_df.index.tolist()
product_indices = prod_vecs.index.tolist()

# Step 2: Convert DataFrames to CSR sparse matrices and then to dense arrays
user_profiles_dense = csr_matrix(user_profiles_df.values).toarray().astype('float32')
product_vectors_dense = csr_matrix(prod_vecs.values).toarray().astype('float32')

# Step 3: Normalize vectors (for cosine similarity)
faiss.normalize_L2(product_vectors_dense)
faiss.normalize_L2(user_profiles_dense)

# Step 4: Initialize the FAISS index for cosine similarity (inner product after normalization)
index = faiss.IndexFlatIP(product_vectors_dense.shape[1])  # IP = inner product, which is cosine similarity after normalization
index.add(product_vectors_dense)

# Step 5: Query with user profiles and get neighbors
D, I = index.search(user_profiles_dense, k=500)

# Step 6: Map results back to original indices, including similarity scores
recommendations_with_scores = {
    user_indices[i]: {
        'recommended_products': [
            {'variant_sku': product_indices[neighbor], 'similarity_score': score}
            for neighbor, score in zip(I[i], D[i])
        ]
    }
    for i in range(len(I))
}

# 'recommendations_with_scores' is now a dictionary where each key is a user identifier (from user_indices)
# and each value is a list of dictionaries, each containing:
# - 'product_id': the recommended product identifier (from product_indices)
# - 'similarity_score': the cosine similarity score between the user profile and the recommended product

# %%

sales_test_df = sales_test_df.merge(
    prod_df[['Variant SKU','Campaign']], 
    how='inner', 
    left_on='variant_sku', 
    right_on='Variant SKU'
).drop(columns=['Variant SKU'])



# Step 1: Retrieve the first user's recommendations
first_user_recommendations = next(iter(recommendations_with_scores.items()))

# Unpack to get user_id and recommendations data
user_id, recommendations_data = first_user_recommendations
recommendations_df = pd.DataFrame(recommendations_data['recommended_products'][:500])

# Step 2: Add the user_id column
recommendations_df['user_id'] = user_id
recommendations_df = recommendations_df[['user_id', 'variant_sku', 'similarity_score']]

# Step 3: Merge to add product titles
recommendations_df = recommendations_df.merge(
    prod_df[['Variant SKU', 'Title','Campaign']], 
    how='inner', 
    left_on='variant_sku', 
    right_on='Variant SKU'
).drop(columns=['Variant SKU'])

# Step 4: Retrieve products bought by the user
prods_bought = sales_df.query('customer_email == @user_id')['variant_sku']

# Step 5: Filter out products that the user has already bought
recommendations_df = recommendations_df[~recommendations_df['variant_sku'].isin(prods_bought)]

camp_recs = recommendations_df[recommendations_df['Campaign'].str.startswith('2411', na=False)]
# %%
# Initialize lists to store precision and recall scores for each user
p_at9 = []
r_at9 = []
p_at9_camp = []
r_at9_camp = []

# Counter for warnings
no_campaign_pred_count = 0

# Iterate over each user and their orders in the test data
for email, orders in sales_test_df.groupby('customer_email'):
    ordered_skus = orders['variant_sku'].tolist()  # Convert to list

    recommendations_data = recommendations_with_scores[email]
    
    # Build a DataFrame for the current user's recommendations
    recommendations_df = pd.DataFrame(recommendations_data['recommended_products'][:500])
    recommendations_df['email'] = email
    recommendations_df = recommendations_df[['email', 'variant_sku', 'similarity_score']]

    # Step 3: Merge to add product titles and campaign information
    recommendations_df = recommendations_df.merge(
        prod_df[['Variant SKU', 'Title', 'Campaign']], 
        how='inner', 
        left_on='variant_sku', 
        right_on='Variant SKU'
    ).drop(columns=['Variant SKU'])

    # Step 4: Retrieve products bought by the user
    prods_bought = sales_df.query('customer_email == @email')['variant_sku']

    # Step 5: Filter out products that the user has already bought
    recommendations_df = recommendations_df[~recommendations_df['variant_sku'].isin(prods_bought)]

    # General recommendations precision and recall
    gen_pred = recommendations_df['variant_sku'][:9].tolist()  # Top 9 general recommendations
    if len(gen_pred) == 0:
        print(f"Warning: No general predictions for user {email}")
    p_at9.append(precision_at_k(gen_pred, ordered_skus))
    r_at9.append(recall_at_k(gen_pred, ordered_skus))

    # Campaign-specific recommendations precision and recall
    recommendations_df['Campaign'] = recommendations_df['Campaign'].astype(str)
    camp_recs = recommendations_df[recommendations_df['Campaign'].str.startswith('2411', na=False)]['variant_sku'][:9].tolist()
    
    if len(camp_recs) == 0:
        no_campaign_pred_count += 1  # Increment the counter
        #print(f"Warning: No campaign-specific predictions for user {email}")
    p_at9_camp.append(precision_at_k(camp_recs, ordered_skus))
    r_at9_camp.append(recall_at_k(camp_recs, ordered_skus))

# Print the averages for each measure
print(f"Average Precision@9 (General): {np.mean(p_at9):.4f}")
print(f"Average Recall@9 (General): {np.mean(r_at9):.4f}")
print(f"Average Precision@9 (Campaign-specific): {np.mean(p_at9_camp):.4f}")
print(f"Average Recall@9 (Campaign-specific): {np.mean(r_at9_camp):.4f}")

# Print the total count of warnings
print(f"Total number of users with no campaign-specific predictions: {no_campaign_pred_count}")



# %%
