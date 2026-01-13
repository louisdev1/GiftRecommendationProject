import gzip
import json
import csv
from pathlib import Path

# File paths needed for this script
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "amazon_raw"
METADATA_DIR = DATA_DIR / "metadata"
OUTPUT_FILE = SCRIPT_DIR / "products.csv"

PRODUCTS_PER_CATEGORY = 15000
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
EXCLUDE_KEYWORDS = [
    'cable', 'adapter', 'charger', 'connector', 'cord', 'replacement part'
]

def process_category(category):
    meta_file = METADATA_DIR / f"meta_{category}.jsonl.gz"
    
    if not meta_file.exists():
        print(f"  Metadata file not found: {meta_file}")
        return []
    
    products = []
    
    with gzip.open(meta_file, 'rt', encoding='utf-8') as f:
        for line in f:
            if len(products) >= PRODUCTS_PER_CATEGORY:
                break
                
            try:
                item = json.loads(line)
                asin = item.get('asin')
                name = item.get('title', '')
                price = item.get('price')
                
                # Skip products met missende data
                if not name or not price or not asin:
                    continue
                
                # Price naar float
                try:
                    if isinstance(price, str):
                        # Remove $ and commas, then convert to float
                        price = float(price.replace('$', '').replace(',', ''))
                    else:
                        price = float(price)
                except:
                    # If price conversion fails,just skip product
                    continue
                if price < 5 or price > 500:
                    continue
                
                name_lower = name.lower()
                if any(kw in name_lower for kw in EXCLUDE_KEYWORDS):
                    continue
                
                # Product image URL
                images = item.get('images', [])
                image_url = ''
                if images:
                    # Handle different image formats
                    if isinstance(images[0], dict):
                        image_url = images[0].get('large', images[0].get('thumb', ''))
                    elif isinstance(images[0], str):
                        image_url = images[0]
                
                # Add this product to our list
                products.append({
                    'asin': asin,
                    'name': name,
                    'category': category,
                    'price': round(price, 2),
                    'image_url': image_url
                })
                
            except:
                # Enkele fout = skip product
                continue
    
    return products

def main():
    print("PRODUCTS.CSV")
    print(f"Max per category: {PRODUCTS_PER_CATEGORY}")
    
    # List to hold all products from all categories
    all_products = []
    
    # Process each category one at a time
    for category in CATEGORIES:
        print(f"\n{category}:")
        products = process_category(category)
        print(f"  Found {len(products)} products")
        all_products.extend(products)
    
    # Assign sequential product IDs
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
    
    
    # Count products per category
    from collections import Counter
    for cat, count in sorted(Counter(p['category'] for p in all_products).items()):
        print(f"  {cat}: {count}")
    
    print(f"\nTotal: {len(all_products)} products")
    print(f"Output: {OUTPUT_FILE}")

# Run the script
if __name__ == "__main__":
    main()