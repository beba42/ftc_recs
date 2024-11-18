#%%
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pandas as pd
from scipy.sparse import csr_matrix
import implicit
import gc
from tqdm import tqdm
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
test_customers = sales_test_df['customer_email'].unique()
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
# Concatenate the one-hot encoded columns into a binary string representation

# %%
frequent_tag_cols = prod_df[tag_cols].sum()[prod_df[tag_cols].sum() > 50].index
prod_df['ftag_id'] = prod_df[frequent_tag_cols].astype(str).agg(''.join, axis=1)
#gettit frequen catogory columns
freq_cat_tag_cols = set(frequent_tag_cols).intersection(categories_one_hot)
prod_df['fcat_id'] = prod_df[list(freq_cat_tag_cols)].astype(str).agg(''.join, axis=1)

# %%
sales_df = sales_df.merge(prod_df[['Variant SKU', 'ftag_id','fcat_id','Campaign']],
               how = 'left',left_on='variant_sku',
               right_on='Variant SKU').drop(columns=['Variant SKU'])
#%%
test_customers = sales_test_df['customer_email'].unique()

# Filter sales_df to keep only rows with customer_email in test_customers
test_customers = sales_df[sales_df['customer_email'].isin(test_customers)]['customer_email'].unique()

# %%
# Step 1: Group data to get interaction counts (or just mark interaction as 1)
user_item_counts = sales_df.groupby(['customer_email', 'ftag_id']).size().reset_index(name='interaction')

# Step 2: Create mappings for user and item IDs
user_mapping = {email: idx for idx, email in enumerate(user_item_counts['customer_email'].unique())}
item_mapping = {cat_id: idx for idx, cat_id in enumerate(user_item_counts['ftag_id'].unique())}

# Step 3: Map user and item columns to integer indices
user_item_counts['user_idx'] = user_item_counts['customer_email'].map(user_mapping)
user_item_counts['item_idx'] = user_item_counts['ftag_id'].map(item_mapping)

# Step 4: Create a sparse matrix
user_item_sparse_matrix = csr_matrix(
    (user_item_counts['interaction'], (user_item_counts['user_idx'], user_item_counts['item_idx'])),
    shape=(len(user_mapping), len(item_mapping))
)
# %%
# I will use this to pick the most popular item with the ftag_id
item_popularity_df = sales_df.groupby(['ftag_id','fcat_id','variant_sku','product_title','Campaign'])[['ordered_item_quantity']].sum() \
    .sort_values(by = 'ordered_item_quantity',ascending = False)\
    .reset_index()

item_id_to_category = {v: k for k, v in item_mapping.items()}
# %%
# Step 1: Initialize the ALS model
model = implicit.als.AlternatingLeastSquares(factors=10, iterations=15, regularization=0.1)
#%%
# Step 2: Train the model with the user-item matrix
model.fit(user_item_sparse_matrix)
#%%
# Step 3: Recommend items for a users
# Choose a test user
gen_precisions= []
gen_recalls= []
for test_user_email in tqdm(test_customers):

    test_user_idx = user_mapping.get(test_user_email)

    # Slice the user-item matrix to contain only the test user's interactions
    user_interactions = user_item_sparse_matrix[test_user_idx]
    
    # Get recommendations
    recommendations = model.recommend(
        test_user_idx,
        user_interactions,
        N=20,  # Number of recommendations
        filter_already_liked_items=False  # I will filter on those later
    )

    # Map back the item indices to category IDs
    user_skus = sales_df.query('customer_email == @test_user_email')['variant_sku'].tolist()
    actual_skus = sales_test_df.query('customer_email == @test_user_email')['variant_sku'].tolist()
    rec_skus = []
    for item_id in recommendations[0]:
        # Filter item popularity for the same `ftag_id` and exclude already interacted items
        filtered_items = item_popularity_df.query(
            f'ftag_id == "{item_id_to_category[item_id]}" and variant_sku not in @user_skus'
        )

        # Ensure there are items remaining after the filter
        if not filtered_items.empty:
            # Select the most popular item for the given `ftag_id`
            rec = filtered_items.iloc[0]
            # Print the recommended item details
            rec_skus.append(rec['variant_sku'])
            #print(rec[['variant_sku', 'product_title', 'ordered_item_quantity','Campaign']])
        
            #print(f"No suitable recommendation found for ftag_id: {item_id_to_category[item_id]}")

    gen_precisions.append(precision_at_k(rec_skus[:9],actual_skus))
    gen_recalls.append(recall_at_k(rec_skus[:9],actual_skus))

print("General collaborative filtering precisoin is: ",str(np.mean(gen_precisions)))
print("General collaborative filtering recall is: ",str(np.mean(gen_recalls)))

# %%
#calulating recommendtaions for campaign specific where I only consider category tags
#deleting model so the new one can fit into memory
del model
gc.collect()
# %%
# Step 1: Group data to get interaction counts (or just mark interaction as 1)
user_item_counts = sales_df.groupby(['customer_email', 'fcat_id']).size().reset_index(name='interaction')

# Step 2: Create mappings for user and item IDs
user_mapping = {email: idx for idx, email in enumerate(user_item_counts['customer_email'].unique())}
item_mapping = {cat_id: idx for idx, cat_id in enumerate(user_item_counts['fcat_id'].unique())}

# Step 3: Map user and item columns to integer indices
user_item_counts['user_idx'] = user_item_counts['customer_email'].map(user_mapping)
user_item_counts['item_idx'] = user_item_counts['fcat_id'].map(item_mapping)

# Step 4: Create a sparse matrix
user_item_sparse_matrix = csr_matrix(
    (user_item_counts['interaction'], (user_item_counts['user_idx'], user_item_counts['item_idx'])),
    shape=(len(user_mapping), len(item_mapping))
)
camp_item_popularity_df = item_popularity_df[item_popularity_df['Campaign'].str.startswith('2411', na=False)]
# %%
# Step 1: Initialize the ALS model
camp_model = implicit.als.AlternatingLeastSquares(factors=10, iterations=15, regularization=0.1)
#%%
# Step 2: Train the model with the user-item matrix
camp_model.fit(user_item_sparse_matrix)
#%%
camp_precisions= []
camp_recalls= []
for test_user_email in tqdm(test_customers):

    test_user_idx = user_mapping.get(test_user_email)

    # Slice the user-item matrix to contain only the test user's interactions
    user_interactions = user_item_sparse_matrix[test_user_idx]
    
    # Get recommendations
    recommendations = camp_model.recommend(
        test_user_idx,
        user_interactions,
        N=20,  # Number of recommendations
        filter_already_liked_items=False  # I will filter on those later
    )

    # Map back the item indices to category IDs
    user_skus = sales_df.query('customer_email == @test_user_email')['variant_sku'].tolist()
    actual_skus = sales_test_df.query('customer_email == @test_user_email')['variant_sku'].tolist()
    rec_skus = []
    for item_id in recommendations[0]:
        # Filter item popularity for the same `ftag_id` and exclude already interacted items
        filtered_items = camp_item_popularity_df.query(
        f'fcat_id == "{item_id_to_category[item_id]}" and variant_sku not in @user_skus'
        )

        # Ensure there are items remaining after the filter
        if not filtered_items.empty:
            # Select the most popular item for the given `ftag_id`
            rec = filtered_items.iloc[0]
            # Print the recommended item details
            rec_skus.append(rec['variant_sku'])
        #    #print(rec[['variant_sku', 'product_title', 'ordered_item_quantity','Campaign']])
        #else:
        #    print(f"No suitable recommendation found for ftag_id: {item_id_to_category[item_id]}")
            
    camp_precisions.append(precision_at_k(rec_skus[:9],actual_skus))
    camp_recalls.append(recall_at_k(rec_skus[:9],actual_skus))
print("Camapaign collaborative filtering precisoin is: ",str(np.mean(camp_precisions)))
print("Camapaign collaborative filtering recall is: ",str(np.mean(camp_recalls)))


# %%
del camp_model
gc.collect()
# %%
#%%
plt.hist([i for i in prod_df[tag_cols].sum() if i <2000],bins= 100)

# %%
plt.hist([i for i in prod_df[tag_cols].sum() if i <200],bins= 100)
