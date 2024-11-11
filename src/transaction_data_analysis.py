#%%
import pandas as pd
import dask.dataframe as dd
import dask
# %%
file_path_2023 = r"data/offline_sales/Trans_2023.csv"
ddf_2023 = dd.read_csv(file_path_2023)
# %%
file_path_2024 = r"data/offline_sales/Trans_2024_v2.csv"
ddf_2024 = dd.read_csv(file_path_2024)
# %%
ddf = dd.concat([ddf_2023, ddf_2024], axis=0)
# %%
unique_sales_header_pos_id = ddf['Sales Header Pos Id'].nunique().compute()
# %%
first_date = ddf['Date'].min().compute()
# %%
last_date = ddf['Date'].max().compute()
# %%
entities = ddf['Entity'].unique().compute()
# %%

total_rows = ddf.shape[0].compute()
# %%
