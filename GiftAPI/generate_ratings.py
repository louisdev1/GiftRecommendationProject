import gzip
import json
import csv
from pathlib import Path
from collections import defaultdict
import random

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "amazon_raw" / "reviews"
PRODUCTS_FILE = SCRIPT_DIR / "products.csv"
OUTPUT_FILE = SCRIPT_DIR / "ratings.csv"

MAX_RATINGS_PER_PRODUCT = 100  # Limit per product to avoid super popular items
MAX_TOTAL_RATINGS = 3_000_000  # Target approximately 3 million ratings total

def load_product_asins():
    asin_to_idx = {}
    with open(PRODUCTS_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Map each ASIN to its product index
            asin_to_idx[row['asin']] = int(row['product_idx'])
    
    print(f"Loaded {len(asin_to_idx)} products from products.csv")
    return asin_to_idx

def load_products_by_idx():
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
    # Load products thatexist in our dataset
    asin_to_idx = load_product_asins()
    
    # All ratings we'll collect
    all_ratings = []
    
    # Map original Amazon user IDs to sequential numbers
    user_id_map = {}
    next_user_id = 0
    
    ratings_per_product = defaultdict(int)
    
    # Process each categories review file
    for category in CATEGORIES:
        reviews_file = DATA_DIR / f"{category}.jsonl.gz"
        
        if not reviews_file.exists():
            print(f"Skipping {category} - file not found")
            continue
        
        print(f"\nProcessing {category}...")
        category_ratings = 0
        
        # Open the compressed review file
        # Each line is a JSON object = one review
        with gzip.open(reviews_file, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    review = json.loads(line)
                    
                    # Extract key fields
                    asin = review.get('parent_asin') or review.get('asin')
                    user_id = review.get('user_id')
                    rating = review.get('rating')
                    
                    # Skip if product isn't in products.csv
                    if asin not in asin_to_idx:
                        continue
                    
                    product_idx = asin_to_idx[asin]
                    
                    if ratings_per_product[product_idx] >= MAX_RATINGS_PER_PRODUCT:
                        continue
                    
                    # Map this user to a sequential ID
                    # Never seen this user = give them the next ID
                    if user_id not in user_id_map:
                        user_id_map[user_id] = next_user_id
                        next_user_id += 1
                    
                    # Add this rating to our collection
                    all_ratings.append({
                        'user_idx': user_id_map[user_id],
                        'product_idx': product_idx,
                        'rating': float(rating)
                    })
                    
                    ratings_per_product[product_idx] += 1
                    category_ratings += 1
                    
                except Exception as e:
                    # Skip any reviews that cause errors
                    continue

        print(f"  Found {category_ratings} ratings")
    print(f"\nTotal ratings collected: {len(all_ratings)}")
    
    # If we collected too many ratings, randomly sample
    if len(all_ratings) > MAX_TOTAL_RATINGS:
        print(f"Limiting to {MAX_TOTAL_RATINGS} ratings...")
        random.shuffle(all_ratings)  # Randomize order
        all_ratings = all_ratings[:MAX_TOTAL_RATINGS]  # Keep first N
    
    # Write all ratings to CSV file
    print(f"\nWriting {len(all_ratings)} ratings to {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)    
        writer.writerow(['user_idx', 'product_idx', 'rating'])
        
        for r in all_ratings:
            writer.writerow([r['user_idx'], r['product_idx'], r['rating']])
    
    products_with_ratings = len(ratings_per_product)
    unique_users = len(user_id_map)

    print(f"Total ratings: {len(all_ratings)}")
    print(f"Products with ratings: {products_with_ratings}")
    print(f"Unique users: {unique_users}")
    print(f"Output: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()