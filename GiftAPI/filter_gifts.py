"""
Filter Amazon metadata to gift-appropriate products
Creates products.csv from metadata files
"""
import gzip
import json
import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
META_DIR = SCRIPT_DIR / "data" / "amazon_raw" / "metadata"
OUTPUT_FILE = SCRIPT_DIR / "GiftApi" / "products.csv"

# Gift categories
CATEGORIES = [
    "Baby_Products", "Beauty_and_Personal_Care", "Clothing_Shoes_and_Jewelry",
    "Electronics", "Home_and_Kitchen", "Office_Products", "Pet_Supplies",
    "Sports_and_Outdoors", "Toys_and_Games"
]

# Blacklist - not suitable as gifts
BLACKLIST = [
    "replacement", "refurbished", "part only", "for repair", "compatible with",
    "cable", "cables", "adapter", "charger cable", "screen protector",
    "case for", "cover for", "battery for", "cartridge", "ink for", "toner",
    "manual", "guide book", "repair kit", "tool kit", "mounting", "bracket"
]

def is_gift_appropriate(item):
    """Check if product is suitable as a gift"""
    name = item.get('title', '').lower()
    price = item.get('price')
    
    # Must have name and reasonable price
    if not name or len(name) < 5:
        return False
    if price is None or price < 5 or price > 500:
        return False
    
    # Check blacklist
    for word in BLACKLIST:
        if word in name:
            return False
    
    return True

def extract_price(item):
    """Extract price from item"""
    price = item.get('price')
    if isinstance(price, (int, float)):
        return float(price)
    return None

def main():
    print("=" * 50)
    print("FILTER AMAZON PRODUCTS FOR GIFTS")
    print("=" * 50)
    
    if not META_DIR.exists():
        print(f"ERROR: {META_DIR} not found!")
        return
    
    products = []
    product_idx = 0
    
    for category in CATEGORIES:
        meta_file = META_DIR / f"meta_{category}.jsonl.gz"
        if not meta_file.exists():
            print(f"  Skipping {category} - file not found")
            continue
        
        print(f"Processing {category}...")
        count = 0
        
        with gzip.open(meta_file, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    price = extract_price(item)
                    if price is None:
                        continue
                    
                    item['price'] = price
                    if not is_gift_appropriate(item):
                        continue
                    
                    products.append({
                        'product_idx': product_idx,
                        'asin': item.get('parent_asin') or item.get('asin', ''),
                        'name': item.get('title', '')[:200],  # Limit name length
                        'category': category,
                        'price': price,
                        'image_url': (item.get('images', [{}])[0].get('large', '') if item.get('images') else '')
                    })
                    product_idx += 1
                    count += 1
                    
                except (json.JSONDecodeError, KeyError):
                    continue
        
        print(f"  -> {count:,} products")
    
    # Save to CSV
    print(f"\nSaving {len(products):,} products to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['product_idx', 'asin', 'name', 'category', 'price', 'image_url'])
        writer.writeheader()
        writer.writerows(products)
    
    print("Done!")
    print(f"\nProducts per category:")
    from collections import Counter
    cats = Counter(p['category'] for p in products)
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count:,}")

if __name__ == "__main__":
    main()