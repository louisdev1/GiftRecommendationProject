"""
Product Data Generator
This script reads Amazon product metadata and creates a clean products.csv file
with 15,000 gift-appropriate products from each of 9 categories (135,000 total)
"""
import gzip
import json
import csv
from pathlib import Path

# Set up file paths relative to this script
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "amazon_raw"
METADATA_DIR = DATA_DIR / "metadata"
OUTPUT_FILE = SCRIPT_DIR / "products.csv"

# How many products to take from each category
PRODUCTS_PER_CATEGORY = 15000  # 15k per category = roughly 135k total

# The 9 categories we're using for our gift recommendation system
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

# Words that indicate the product is NOT a good gift (accessories, parts, etc.)
EXCLUDE_KEYWORDS = [
    'cable', 'adapter', 'charger', 'connector', 'cord', 'replacement part'
]

def process_category(category):
    """
    Process one category and extract valid gift products
    Returns a list of product dictionaries
    """
    # Build the path to the metadata file for this category
    meta_file = METADATA_DIR / f"meta_{category}.jsonl.gz"
    
    # Check if file exists
    if not meta_file.exists():
        print(f"  Metadata file not found: {meta_file}")
        return []
    
    products = []
    
    # Open the compressed JSON file
    # Each line is a separate JSON object representing one product
    with gzip.open(meta_file, 'rt', encoding='utf-8') as f:
        for line in f:
            # Stop when we have enough products from this category
            if len(products) >= PRODUCTS_PER_CATEGORY:
                break
                
            try:
                # Parse the JSON line
                item = json.loads(line)
                
                # Extract the ASIN (Amazon Standard Identification Number)
                # parent_asin groups product variations, fallback to regular asin
                asin = item.get('parent_asin') or item.get('asin')
                
                # Extract product name and price
                name = item.get('title', '')
                price = item.get('price')
                
                # Skip products with missing essential data
                if not name or not price or not asin:
                    continue
                
                # Convert price to float (handle both string and number formats)
                try:
                    if isinstance(price, str):
                        # Remove $ and commas, then convert to float
                        price = float(price.replace('$', '').replace(',', ''))
                    else:
                        price = float(price)
                except:
                    # If price conversion fails, skip this product
                    continue
                
                # Only keep products in reasonable gift price range
                # Too cheap (under $5) or too expensive (over $500) filtered out
                if price < 5 or price > 500:
                    continue
                
                # Filter out non-gift items like cables and chargers
                name_lower = name.lower()
                if any(kw in name_lower for kw in EXCLUDE_KEYWORDS):
                    continue
                
                # Extract product image URL
                images = item.get('images', [])
                image_url = ''
                if images:
                    # Handle different image formats in the data
                    if isinstance(images[0], dict):
                        # Try to get large image first, fallback to thumbnail
                        image_url = images[0].get('large', images[0].get('thumb', ''))
                    elif isinstance(images[0], str):
                        # Sometimes images are just strings
                        image_url = images[0]
                
                # Add this product to our list
                products.append({
                    'asin': asin,
                    'name': name,
                    'category': category,
                    'price': round(price, 2),  # Round to 2 decimal places
                    'image_url': image_url
                })
                
            except:
                # If anything goes wrong parsing this line, skip it
                continue
    
    return products

def main():
    """Main function that coordinates the entire process"""
    print("=" * 50)
    print("GENERATING PRODUCTS.CSV (ALL PRODUCTS)")
    print(f"Max per category: {PRODUCTS_PER_CATEGORY}")
    print("=" * 50)
    
    # List to hold all products from all categories
    all_products = []
    
    # Process each category one at a time
    for category in CATEGORIES:
        print(f"\n{category}:")
        products = process_category(category)
        print(f"  Found {len(products)} products")
        all_products.extend(products)
    
    # Assign sequential product IDs (0, 1, 2, ...)
    # These IDs are what the ML model will use
    for i, p in enumerate(all_products):
        p['product_id'] = i
    
    # Write all products to CSV file
    print(f"\nWriting {len(all_products)} products...")
    
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header row
        writer.writerow(['product_idx', 'asin', 'name', 'category', 'price', 'image_url'])
        
        # Write each product as a row
        for p in all_products:
            writer.writerow([
                p['product_id'], 
                p['asin'], 
                p['name'], 
                p['category'], 
                p['price'], 
                p['image_url']
            ])
    
    # Print summary statistics
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    # Count products per category
    from collections import Counter
    for cat, count in sorted(Counter(p['category'] for p in all_products).items()):
        print(f"  {cat}: {count}")
    
    print(f"\nTotal: {len(all_products)} products")
    print(f"Output: {OUTPUT_FILE}")

# Run the script
if __name__ == "__main__":
    main()