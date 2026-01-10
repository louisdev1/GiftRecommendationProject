"""
Rating Data Generator
This script reads Amazon review data and creates a ratings.csv file
that only includes ratings for products that exist in our products.csv
Limits to 3 million ratings total for manageable file size
"""
import gzip
import json
import csv
from pathlib import Path
from collections import defaultdict
import random

# Set up file paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "amazon_raw" / "reviews"
PRODUCTS_FILE = SCRIPT_DIR / "products.csv"
OUTPUT_FILE = SCRIPT_DIR / "ratings.csv"

# Configuration settings
MAX_RATINGS_PER_PRODUCT = 100  # Limit per product to avoid super-popular items dominating
MAX_TOTAL_RATINGS = 3_000_000  # Target approximately 3 million ratings total

def load_product_asins():
    """
    Load the mapping from ASIN to product_idx from products.csv
    Returns a dictionary: {asin: product_idx}
    """
    asin_to_idx = {}
    
    with open(PRODUCTS_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Map each ASIN to its product index
            asin_to_idx[row['asin']] = int(row['product_idx'])
    
    print(f"Loaded {len(asin_to_idx)} products from products.csv")
    return asin_to_idx

def load_products_by_idx():
    """
    Load products indexed by their product_idx
    Used for looking up product categories
    """
    products = {}
    
    with open(PRODUCTS_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            products[int(row['product_idx'])] = row
    
    return products

# Same 9 categories as in generate_products.py
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
    """Main function that coordinates rating data generation"""
    print("=" * 50)
    print("GENERATING RATINGS.CSV")
    print("=" * 50)
    
    # Load which products exist in our dataset
    asin_to_idx = load_product_asins()
    
    # Storage for all ratings we'll collect
    all_ratings = []
    
    # Map original Amazon user IDs to sequential numbers (0, 1, 2, ...)
    # This saves space and makes the ML model work better
    user_id_map = {}
    next_user_id = 0
    
    # Track how many ratings each product has (for limiting)
    ratings_per_product = defaultdict(int)
    
    # Process each category's review file
    for category in CATEGORIES:
        reviews_file = DATA_DIR / f"{category}.jsonl.gz"
        
        # Skip if file doesn't exist
        if not reviews_file.exists():
            print(f"Skipping {category} - file not found")
            continue
        
        print(f"\nProcessing {category}...")
        category_ratings = 0
        
        # Open the compressed review file
        # Each line is a JSON object representing one review
        with gzip.open(reviews_file, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    # Parse the review JSON
                    review = json.loads(line)
                    
                    # Extract key fields
                    asin = review.get('parent_asin') or review.get('asin')
                    user_id = review.get('user_id')
                    rating = review.get('rating')
                    
                    # Skip if this product isn't in our products.csv
                    if asin not in asin_to_idx:
                        continue
                    
                    product_idx = asin_to_idx[asin]
                    
                    # Limit ratings per product
                    # This prevents super popular products from dominating the dataset
                    if ratings_per_product[product_idx] >= MAX_RATINGS_PER_PRODUCT:
                        continue
                    
                    # Map this user to a sequential ID
                    # If we've never seen this user, give them the next ID
                    if user_id not in user_id_map:
                        user_id_map[user_id] = next_user_id
                        next_user_id += 1
                    
                    # Add this rating to our collection
                    all_ratings.append({
                        'user_idx': user_id_map[user_id],
                        'product_idx': product_idx,
                        'rating': float(rating)
                    })
                    
                    # Update counters
                    ratings_per_product[product_idx] += 1
                    category_ratings += 1
                    
                except Exception as e:
                    # Skip any reviews that can't be parsed
                    continue
        
        print(f"  Found {category_ratings} ratings")
    
    print(f"\nTotal ratings collected: {len(all_ratings)}")
    
    # If we collected too many ratings, randomly sample to limit size
    if len(all_ratings) > MAX_TOTAL_RATINGS:
        print(f"Limiting to {MAX_TOTAL_RATINGS} ratings...")
        random.shuffle(all_ratings)  # Randomize order
        all_ratings = all_ratings[:MAX_TOTAL_RATINGS]  # Keep first N
    
    # Write all ratings to CSV file
    print(f"\nWriting {len(all_ratings)} ratings to {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header row
        writer.writerow(['user_idx', 'product_idx', 'rating'])
        
        # Write each rating as a row
        for r in all_ratings:
            writer.writerow([r['user_idx'], r['product_idx'], r['rating']])
    
    # Print summary statistics
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

# Run the script
if __name__ == "__main__":
    main()