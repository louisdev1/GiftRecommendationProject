import pandas as pd

ratings = pd.read_csv('ratings.csv')
products = pd.read_csv('products.csv')

print(f'Ratings: {len(ratings):,}')
print(f'Products: {len(products):,}')
print(f'Unique products in ratings: {ratings["product_idx"].nunique():,}')

# Filter products die in ratings zitten
valid_ids = set(ratings['product_idx'].unique())
filtered = products[products['product_idx'].isin(valid_ids)]
print(f'Products with ratings: {len(filtered):,}')

# Save
filtered.to_csv('products.csv', index=False)
print('Saved filtered products.csv')