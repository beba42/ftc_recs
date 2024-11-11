#%%
import pandas as pd
import re
import dask.dataframe as dd

pd.set_option('display.max_columns', None)
#%%
# Load and concatenate the two CSV files
prod_df_1 = pd.read_csv('data/products_export_1.csv')
prod_df_2 = pd.read_csv('data/products_export_2.csv')
prod_df = pd.concat([prod_df_1, prod_df_2], ignore_index=True)

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

# Group by 'Handle' and aggregate
prod_df = prod_df.groupby('Handle', as_index=False).agg(agg_dict)
prod_df = prod_df.dropna(subset=['Variant SKU'])

# %%
prefix_values = {}

# Iterate over each product's tags
for tags in prod_df['Tags'].dropna():
    # Split each tag by commas and iterate through each tag element
    for tag in tags.split(','):
        tag = tag.strip()  # Remove leading/trailing whitespace
        if ':' in tag:
            prefix, value = tag.split(':', 1)  # Split by ':' to separate prefix and value
            prefix = prefix.strip()
            value = value.strip()
            
            # If prefix is 'Tag', remove leading 'GG' or 'AG' and trailing numbers
            if prefix == "Tag":
                value = re.sub(r'^(GG|AG)', '', value).strip()  # Remove 'GG' or 'AG' at the start
                value = re.sub(r'\d+$', '', value).strip()  # Remove any trailing numbers
            
            value = value.lower()
            # Collect unique values
            if prefix not in prefix_values:
                prefix_values[prefix] = set()
            prefix_values[prefix].add(value)

# Convert sets to lists for easier viewing/usage
prefix_values = {k: list(v) for k, v in prefix_values.items()}

# Display the collected values
for prefix, values in prefix_values.items():
    print(f"{prefix}: {values}")
# %%
# Assuming 'prod_df' is your initial DataFrame

# Helper function to create one-hot encoded columns for each prefix tag
def one_hot_encode_tags(df, tag_column, prefix):
    # Initialize a DataFrame to store one-hot encoded columns
    one_hot_encoded = pd.DataFrame(index=df.index)

    # Iterate over each row in the tag column
    for idx, tags in df[tag_column].dropna().items():
        # Split each tag and check if it matches the prefix
        for tag in tags.split(','):
            tag = tag.strip()
            if ':' in tag:
                tag_prefix, tag_value = tag.split(':', 1)
                tag_prefix = tag_prefix.strip()
                tag_value = tag_value.strip()

                # Process only if the tag matches the specified prefix
                if tag_prefix == prefix:
                    # If prefix is 'Tag', remove 'GG' or 'AG' from the start and any trailing numbers
                    if prefix == 'Tag':
                        tag_value = re.sub(r'^(GG|AG)', '', tag_value).strip()
                        tag_value = re.sub(r'\d+$', '', tag_value).strip()
                        tag_value = tag_value.lower()
                    
                    # Define the column name and set value to 1
                    column_name = f"{prefix}_{tag_value}"
                    one_hot_encoded.loc[idx, column_name] = 1

    # Fill NaN with 0 (for tags that are not present in the row)
    one_hot_encoded = one_hot_encoded.fillna(0).astype(int)
    return one_hot_encoded

# Generate one-hot encoded columns for each tag category and merge them with the original DataFrame
categories_one_hot = one_hot_encode_tags(prod_df, 'Tags', 'Category')
sku_type_one_hot = one_hot_encode_tags(prod_df, 'Tags', 'SKU Type')
campaign_one_hot = one_hot_encode_tags(prod_df, 'Tags', 'Campaign')
tag_one_hot = one_hot_encode_tags(prod_df, 'Tags', 'Tag')

# Concatenate the one-hot encoded columns with the original DataFrame
prod_df = pd.concat([prod_df, categories_one_hot, sku_type_one_hot, campaign_one_hot, tag_one_hot], axis=1).fillna(0)

# Convert all one-hot encoded columns to integer type
prod_df = prod_df.astype({col: int for col in categories_one_hot.columns.union(sku_type_one_hot.columns).union(campaign_one_hot.columns).union(tag_one_hot.columns)})
# %%

# Remove the ' character and convert valid numeric values to int, leaving others as NaN
prod_df['Variant SKU'] = pd.to_numeric(prod_df['Variant SKU'].str.replace("'", ""), errors='coerce')

# Drop rows where 'Variant SKU' could not be converted to an integer (NaN values)
prod_df = prod_df.dropna(subset=['Variant SKU'])

# Convert 'Variant SKU' column to integer type
prod_df['Variant SKU'] = prod_df['Variant SKU'].astype(int)
# %%

## Online transactions
#sales_files = glob.glob("data/sales/sales_*.csv")
#
## Read and concatenate all files into a single DataFrame
#sales_df = pd.concat((pd.read_csv(file) for file in sales_files), ignore_index=True)
#
## Save the combined DataFrame to a new CSV file
#sales_df.to_csv("data/sales/online_sales.csv", index=False, sep=';')
#

# %%
sales_df = pd.read_csv('data/sales/online_sales.csv',sep=";")
# %%
country_order =["Germany", "United Kingdom", "France", 
    "Italy", "Spain", "Poland", "Netherlands",
    "Belgium", "Czech Republic", "Sweden", "Greece", 
    "Portugal", "Hungary", "Austria", "Denmark", "Finland",
     "Slovakia", "Norway", "Ireland", "Lithuania", "Latvia", "Estonia"]


sales_df = sales_df.query('fulfillment_status == "fulfilled" and cancelled == "No"')
#%%
sales_df = pd.read_csv('data/sales/online_sales.csv',sep=";")

country_order =["Germany", "United Kingdom", "France", 
    "Italy", "Spain", "Poland", "Netherlands",
    "Belgium", "Czech Republic", "Sweden", "Greece", 
    "Portugal", "Hungary", "Austria", "Denmark", "Finland",
     "Slovakia", "Norway", "Ireland", "Lithuania", "Latvia", "Estonia"]


sales_df = sales_df.query('fulfillment_status == "fulfilled" and cancelled == "No"')
#%%
sales_df['hour'] = pd.to_datetime(sales_df['hour'])
sales_df['variant_sku'] =sales_df['variant_sku'].astype(int)
sales_df.to_parquet('data/sales/online_sales.parquet')
# %%
#email subscriber list

email_df = pd. read_csv('data/masters_email_list.csv')


# %%
# Count the number of missing values for each column
missing_counts = email_df.isna().sum()

# Display the result
print(missing_counts)
# %%
email_df = email_df.query('Email.notna() and Country.notna()')
# %%
#Define a function to fill missing 'Locale: Language' with the most common value within State or Country
# Most common 'Locale: Language' by 'State / Region'
state_language_mode = (
    email_df.dropna(subset=['State / Region', 'Locale: Language'])
    .groupby('State / Region')['Locale: Language']
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    .to_dict()
)

# Most common 'Locale: Language' by 'Country'
country_language_mode = (
    email_df.dropna(subset=['Country', 'Locale: Language'])
    .groupby('Country')['Locale: Language']
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    .to_dict()
)

# Step 2: Define a function to fill missing values using the dictionaries
def fill_locale_language(row):
    if pd.notna(row['Locale: Language']):
        return row['Locale: Language']
    elif pd.notna(row['State / Region']) and row['State / Region'] in state_language_mode:
        return state_language_mode[row['State / Region']]
    elif pd.notna(row['Country']) and row['Country'] in country_language_mode:
        return country_language_mode[row['Country']]
    else:
        return row['Locale: Language']  # Leave as NaN if no match is found
#%%
# Apply the function to fill missing 'Locale: Language' values
email_df['Locale: Language'] = email_df.apply(fill_locale_language, axis=1)
# %%
email_df = email_df.query('`Locale: Language`.notna()')
# %%
import dask.dataframe as dd
#file_path_2023 = r"data/offline_sales/Trans_2023.csv"
#ddf_2023 = dd.read_csv(file_path_2023)
## %%
#file_path_2024 = r"data/offline_sales/Trans_2024_v2.csv"
#ddf_2024 = dd.read_csv(file_path_2024)
## %%
#ddf = dd.concat([ddf_2023, ddf_2024], axis=0)
## %%
#first_partition = ddf.get_partition(0).compute()
# %%
ddf = dd.read_parquet('data/offline_sales/transactions_data.parquet', engine='pyarrow')

ddf.dtypes
# %%
missing_values = ddf.isnull().sum().compute()
# %%
# Use map_partitions to apply the duplicated function on each partition
duplicate_count = ddf.map_partitions(lambda df: df.duplicated().sum()).compute().sum()


# %%
nrows = ddf.shape[0].compute()
# %%
#To find the unique values of Return Code when it is not NaN and the unique values of Flag Canceled, you can use the following code in Dask:

# Get unique values of 'Return Code' where it's not NaN
unique_return_codes = ddf['Return Code'].dropna().unique().compute()
print("Unique values in 'Return Code' (non-NaN):", unique_return_codes)

# Get unique values of 'Flag Canceled'
unique_flag_canceled = ddf['Flag Canceled'].unique().compute()
print("Unique values in 'Flag Canceled':", unique_flag_canceled)
# %%
ddf['Date'] = dd.to_datetime(ddf['Date'], errors='coerce')
# Convert 'Sales Header Pos Id' to string to ensure consistency
ddf['Sales Header Pos Id'] = ddf['Sales Header Pos Id'].astype(str)

#%%
#filtered_ddf = ddf[(ddf['Return Code'].isna()) & (ddf['Flag Canceled'] == 0)]
#
## Save the filtered DataFrame to a Parquet file
#filtered_ddf.to_parquet('data//offline_sales/transactions_data.parquet', engine='pyarrow', write_index=False)
# %%
