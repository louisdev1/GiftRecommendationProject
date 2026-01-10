"""
Create smaller dataset for Azure deployment
Takes a random sample of ratings while keeping all products that have ratings
"""
import pandas as pd
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
RATINGS_FILE = SCRIPT_DIR / "GiftApi" / "ratings.csv"
PRODUCTS_FILE = SCRIPT_DIR / "GiftApi" / "products.csv"
OUTPUT_RATINGS = SCRIPT_DIR / "GiftApi" / "ratings_small.csv"
OUTPUT_PRODUCTS = SCRIPT_DIR / "GiftApi" / "products_small.csv"

SAMPLE_SIZE = 2_000_000  # 2 million ratings

print("=" * 50)
print("CREATE AZURE DEPLOYMENT DATASET")
print("=" * 50)

# Load ratings
print(f"Loading ratings from {RATINGS_FILE}...")
ratings_df = pd.read_csv(RATINGS_FILE)
print(f"  Total ratings: {len(ratings_df):,}")

# Sample ratings
print(f"\nSampling {SAMPLE_SIZE:,} ratings...")
np.random.seed(42)
if len(ratings_df) > SAMPLE_SIZE:
    ratings_sample = ratings_df.sample(n=SAMPLE_SIZE, random_state=42)
else:
    ratings_sample = ratings_df
print(f"  Sampled: {len(ratings_sample):,}")

# Get unique products in sample
product_ids = set(ratings_sample['product_idx'].unique())
print(f"  Unique products in sample: {len(product_ids):,}")

# Load and filter products
print(f"\nLoading products from {PRODUCTS_FILE}...")
products_df = pd.read_csv(PRODUCTS_FILE)
print(f"  Total products: {len(products_df):,}")

products_filtered = products_df[products_df['product_idx'].isin(product_ids)]
print(f"  Products with ratings: {len(products_filtered):,}")

# Reindex to consecutive IDs
print("\nReindexing...")
old_to_new_product = {old: new for new, old in enumerate(sorted(product_ids))}
old_to_new_user = {old: new for new, old in enumerate(sorted(ratings_sample['user_idx'].unique()))}

ratings_sample = ratings_sample.copy()
ratings_sample['product_idx'] = ratings_sample['product_idx'].map(old_to_new_product)
ratings_sample['user_idx'] = ratings_sample['user_idx'].map(old_to_new_user)

products_filtered = products_filtered.copy()
products_filtered['product_idx'] = products_filtered['product_idx'].map(old_to_new_product)

# Save
print(f"\nSaving to {OUTPUT_RATINGS}...")
ratings_sample.to_csv(OUTPUT_RATINGS, index=False)

print(f"Saving to {OUTPUT_PRODUCTS}...")
products_filtered.to_csv(OUTPUT_PRODUCTS, index=False)

print("\n" + "=" * 50)
print("DONE!")
print("=" * 50)
print(f"Ratings: {len(ratings_sample):,}")
print(f"Products: {len(products_filtered):,}")
print(f"Users: {len(old_to_new_user):,}")
print("\nNext steps:")
print("  1. cd GiftApi")
print("  2. copy ratings_small.csv ratings.csv")
print("  3. copy products_small.csv products.csv")
print("  4. Deploy to Azure")