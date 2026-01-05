"""
Generate ratings.csv that matches products.csv
Only includes ratings for products that exist in products.csv
"""
import gzip
import json
import csv
from pathlib import Path
from collections import defaultdict
import random

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "amazon_raw" / "reviews"
PRODUCTS_FILE = SCRIPT_DIR / "products.csv"
OUTPUT_FILE = SCRIPT_DIR / "ratings.csv"

# Config
MAX_RATINGS_PER_PRODUCT = 100  # Limit ratings per product to keep file size manageable
MAX_TOTAL_RATINGS = 3_000_000  # Target ~3M ratings total

def load_product_asins():
    """Load ASIN to product_idx mapping from products.csv"""
    asin_to_idx = {}
    with open(PRODUCTS_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            asin_to_idx[row['asin']] = int(row['product_idx'])
    print(f"Loaded {len(asin_to_idx)} products from products.csv")
    return asin_to_idx

def get_category_from_asin(asin, asin_to_idx, products_by_idx):
    """Get category for an ASIN"""
    if asin in asin_to_idx:
        idx = asin_to_idx[asin]
        if idx in products_by_idx:
            return products_by_idx[idx]['category']
    return None

def load_products_by_idx():
    """Load products indexed by product_idx"""
    products = {}
    with open(PRODUCTS_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            products[int(row['product_idx'])] = row
    return products

CATEGORIES = [
    "Baby_Products",
    "Beauty_and_Personal_Care",
    "Clothing_Shoes_and_Jewelry",
    "Electronics",
    "Home_and_Kitchen",
    "Office_Products",
    "Pet_Supplies",
    "Sports_and_Outdoors",
    "Toys_and_Games"
]

def main():
    print("=" * 50)
    print("GENERATING RATINGS.CSV")
    print("=" * 50)
    
    # Load product ASINs
    asin_to_idx = load_product_asins()
    
    # Collect ratings
    all_ratings = []
    user_id_map = {}  # Map original user IDs to sequential IDs
    next_user_id = 0
    ratings_per_product = defaultdict(int)
    
    for category in CATEGORIES:
        reviews_file = DATA_DIR / f"{category}.jsonl.gz"
        
        if not reviews_file.exists():
            print(f"Skipping {category} - file not found")
            continue
        
        print(f"\nProcessing {category}...")
        category_ratings = 0
        
        with gzip.open(reviews_file, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    review = json.loads(line)
                    asin = review.get('parent_asin') or review.get('asin')
                    user_id = review.get('user_id')
                    rating = review.get('rating')
                    
                    # Skip if product not in our products.csv
                    if asin not in asin_to_idx:
                        continue
                    
                    product_idx = asin_to_idx[asin]
                    
                    # Limit ratings per product
                    if ratings_per_product[product_idx] >= MAX_RATINGS_PER_PRODUCT:
                        continue
                    
                    # Map user ID
                    if user_id not in user_id_map:
                        user_id_map[user_id] = next_user_id
                        next_user_id += 1
                    
                    all_ratings.append({
                        'user_idx': user_id_map[user_id],
                        'product_idx': product_idx,
                        'rating': float(rating)
                    })
                    
                    ratings_per_product[product_idx] += 1
                    category_ratings += 1
                    
                except Exception as e:
                    continue
        
        print(f"  Found {category_ratings} ratings")
    
    print(f"\nTotal ratings collected: {len(all_ratings)}")
    
    # Shuffle and limit if needed
    if len(all_ratings) > MAX_TOTAL_RATINGS:
        print(f"Limiting to {MAX_TOTAL_RATINGS} ratings...")
        random.shuffle(all_ratings)
        all_ratings = all_ratings[:MAX_TOTAL_RATINGS]
    
    # Write CSV
    print(f"\nWriting {len(all_ratings)} ratings to {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['user_idx', 'product_idx', 'rating'])
        for r in all_ratings:
            writer.writerow([r['user_idx'], r['product_idx'], r['rating']])
    
    # Summary
    products_with_ratings = len(ratings_per_product)
    unique_users = len(user_id_map)
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total ratings: {len(all_ratings)}")
    print(f"Products with ratings: {products_with_ratings} / 135000")
    print(f"Unique users: {unique_users}")
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 50)

if __name__ == "__main__":
    main()
