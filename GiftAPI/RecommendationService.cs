using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace GiftApi;

public class RecommendationService
{
    // Matrix Factorization model - this is our machine learning algorithm
    private readonly MatrixFactorization _mf;
    
    // List of all products loaded from products.csv
    private readonly List<Product> _products;
    
    // Total number of users in our dataset
    private readonly int _numUsers;

    // Constructor - runs when the service is created
    // Loads all data and trains the recommendation model
    public RecommendationService(string dataPath)
    {
        Console.WriteLine("Loading recommendation data...");
        
        // Load ratings from CSV file
        // Returns arrays of user IDs, product IDs, and rating values
        var (users, items, ratings) = LoadRatings(Path.Combine(dataPath, "ratings.csv"));
        
        // Load product information from CSV file
        _products = LoadProducts(Path.Combine(dataPath, "products.csv"));

        Console.WriteLine($"  Loaded {_products.Count:N0} products");
        
        // Find the highest user ID to know how many users we have
        _numUsers = users.Max() + 1;
        
        // Find the highest product ID to know how many products we have
        int numItems = items.Max() + 1;
        
        Console.WriteLine($"  {ratings.Length:N0} ratings, {_numUsers:N0} users");

        // Create and train the Matrix Factorization model
        // 20 factors = how many hidden features to learn (genre, price range, style, etc.)
        _mf = new MatrixFactorization(_numUsers, numItems, numFactors: 20);
        
        // Train the model for 15 epochs (passes through the data)
        _mf.Train(users, items, ratings, epochs: 15);
        
        Console.WriteLine("Ready!");
    }

    // Main recommendation method - returns a list of products based on various filters
    public List<Product> GetRecommendations(
        int? userId = null,           // Optional: specific user ID for personalized recommendations
        double? minPrice = null,      // Optional: minimum price filter
        double? maxPrice = null,      // Optional: maximum price filter
        double? targetPrice = null,   // Optional: ideal price (will boost products near this price)
        string? category = null,      // Optional: single category filter
        List<string>? categories = null,  // Optional: multiple categories filter
        string? gender = null,        // Optional: male/female filter
        int? age = null,              // Optional: age filter (filters inappropriate products)
        int maxPerCategory = 2,       // Maximum products to show per category (for diversity)
        int topN = 10)                // Total number of products to return
    {
        // Start with all products
        var candidates = _products.AsEnumerable();

        // Apply price filters if specified
        if (minPrice.HasValue) 
            candidates = candidates.Where(p => p.Price >= minPrice.Value);
        if (maxPrice.HasValue) 
            candidates = candidates.Where(p => p.Price <= maxPrice.Value);

        // Apply category filters
        // If multiple categories specified, product must be in one of them
        if (categories?.Count > 0)
            candidates = candidates.Where(p => 
                categories.Contains(p.Category, StringComparer.OrdinalIgnoreCase));
        // If single category specified, product must match exactly
        else if (!string.IsNullOrEmpty(category))
            candidates = candidates.Where(p => 
                p.Category.Equals(category, StringComparison.OrdinalIgnoreCase));

        // Apply gender filter
        // This checks both the product name and category to filter inappropriate items
        if (!string.IsNullOrEmpty(gender))
            candidates = candidates.Where(p => 
                IsGenderOk(p.Name, gender, p.Category));

        // Apply age filter
        // This removes products meant for different age groups
        if (age.HasValue)
            candidates = candidates.Where(p => 
                IsAgeOk(p.Name, p.Category, age.Value));

        // Create a random number generator seeded by age
        // This ensures each age group gets different randomized recommendations
        // Using age as seed means the same age always gets similar (but randomized) results
        int ageSeed = age ?? 0;
        var random = new Random(ageSeed * 104729); // Multiply by large prime for better distribution

        // Score each product
        var scored = candidates.Select(p => {
            // Base score from Matrix Factorization
            // If we have a user ID, predict their rating for this product
            // Otherwise, just use the product's overall popularity bias
            double score = userId.HasValue && userId.Value < _numUsers 
                ? _mf.Predict(userId.Value, p.ProductId) 
                : _mf.GetItemBias(p.ProductId);
            
            // Bonus/penalty based on how close the price is to target
            if (targetPrice.HasValue && targetPrice.Value > 0)
            {
                // Calculate how far the price is from target
                double diff = Math.Abs(p.Price - targetPrice.Value);
                
                // Allow up to 30% difference from target price
                double maxDiff = targetPrice.Value * 0.3;
                
                // If within acceptable range, boost score based on closeness
                if (diff < maxDiff)
                    score += 1.0 * (1 - diff / maxDiff);
                else
                    score -= 0.5; // Penalize if too far from target
            }

            // Add strong randomness to ensure variety
            // This is multiplied by 3.0 so it has a big impact on final ranking
            score += random.NextDouble() * 3.0;

            return (product: p, score);
        }).OrderByDescending(x => x.score).ToList();

        // Apply diversity rule - limit products per category
        // This prevents showing 10 books or 10 toys in a row
        var result = new List<Product>();
        var catCount = new Dictionary<string, int>();
        
        foreach (var item in scored)
        {
            string cat = item.product.Category;
            
            // Check how many products we've already added from this category
            catCount.TryGetValue(cat, out int count);
            
            // Only add if we haven't hit the limit for this category
            if (count < maxPerCategory)
            {
                result.Add(item.product);
                catCount[cat] = count + 1;
                
                // Stop once we have enough products
                if (result.Count >= topN) break;
            }
        }
        
        return result;
    }

    // Returns list of all unique categories in our dataset
    public List<string> GetCategories() => 
        _products.Select(p => p.Category)
                .Distinct()
                .OrderBy(c => c)
                .ToList();

    // Check if a product is appropriate for the specified gender
    private static bool IsGenderOk(string name, string gender, string category)
    {
        var lower = name.ToLower();
        
        // Words that indicate products specifically for males
        var maleWords = new[] { 
            "men's", "mens", "for men", "for him", 
            "beard", "shaving", "shave", "razor", "aftershave", "cologne" 
        };
        
        // Words that indicate products not appropriate for males
        var notForMaleWords = new[] { 
            "women's", "womens", "for women", "for her", 
            "lipstick", "mascara", "purse", "handbag", 
            "girl", "girls", "girl's",
            "hair extension", "hair clip", "hair tie",
            "cute", "pink", "crochet", "braids", "braid",
            "dress", "skirt", "leotard", "ballet",
            "panty", "panties", "bra ", "lingerie",
            "lace ", "ruffle"
        };

        // For male recipients
        if (gender == "male")
        {
            // Exclude entire categories that are typically female-oriented
            if (category == "Home_and_Kitchen" || category == "Beauty_and_Personal_Care")
                return false;
            
            // Check if product name contains female-specific words
            if (notForMaleWords.Any(w => lower.Contains(w))) 
                return false;
        }

        // For female recipients, exclude male-specific products
        if (gender == "female" && maleWords.Any(w => lower.Contains(w))) 
            return false;
            
        return true;
    }

    // Check if a product is appropriate for the specified age
    private static bool IsAgeOk(string name, string category, int age)
    {
        var lower = name.ToLower();

        // For adults (18+), filter out ALL products meant for children
        if (age >= 18)
        {
            var adultFilter = new[] { 
                "baby", "babies", "infant", "toddler", "toddlers",
                "kids", "kid's", "for kids",
                "children", "child's", "for children",
                "boys", "boy's", "for boys",
                "girls", "girl's", "for girls",
                "year old", "years old",
                "newborn", "nursery"
            };
            
            if (adultFilter.Any(w => lower.Contains(w)))
                return false;
        }
        // For teenagers (13-17), filter baby items but allow some kid products
        else if (age >= 13)
        {
            var teenFilter = new[] { 
                "baby", "babies", "infant", "toddler", "toddlers",
                "newborn", "nursery", 
                "ages 1", "ages 2", "ages 3", "ages 4", "ages 5"
            };
            
            if (teenFilter.Any(w => lower.Contains(w)))
                return false;
        }

        // Extra filtering for Toys category since age is very important there
        if (category == "Toys_and_Games")
        {
            var babyToys = new[] { "rattle", "teether", "0-3", "1-3" };
            var preschoolToys = new[] { "preschool", "3-5", "paw patrol", "peppa pig", "duplo" };
            var youngKidToys = new[] { "6-8", "ages 6", "ages 7", "ages 8" };

            // Teens shouldn't see baby or young kid toys
            if (age >= 13 && 
                (babyToys.Any(w => lower.Contains(w)) || 
                 preschoolToys.Any(w => lower.Contains(w)) || 
                 youngKidToys.Any(w => lower.Contains(w))))
                return false;
            
            // 9-12 year olds shouldn't see baby or preschool toys
            if (age >= 9 && 
                (babyToys.Any(w => lower.Contains(w)) || 
                 preschoolToys.Any(w => lower.Contains(w))))
                return false;
            
            // 6-8 year olds shouldn't see baby toys
            if (age >= 6 && babyToys.Any(w => lower.Contains(w)))
                return false;
        }
        
        return true;
    }

    // Load ratings data from CSV file
    // Returns three parallel arrays: user IDs, product IDs, and ratings
    private static (int[] users, int[] items, double[] ratings) LoadRatings(string path)
    {
        // Read all lines except the header row
        var lines = File.ReadAllLines(path).Skip(1).ToArray();
        
        // Create arrays to store the data
        var users = new int[lines.Length];
        var items = new int[lines.Length];
        var ratings = new double[lines.Length];

        // Parse each line (format: user_id,product_id,rating)
        for (int i = 0; i < lines.Length; i++)
        {
            var p = lines[i].Split(',');
            users[i] = int.Parse(p[0]);
            items[i] = int.Parse(p[1]);
            
            // Use InvariantCulture to handle decimal points correctly
            ratings[i] = double.Parse(p[2], CultureInfo.InvariantCulture);
        }
        
        return (users, items, ratings);
    }

    // Load product information from CSV file
    private static List<Product> LoadProducts(string path)
    {
        var products = new List<Product>();
        
        // Read each line except the header
        foreach (var line in File.ReadAllLines(path).Skip(1))
        {
            try
            {
                // Parse CSV line (handles commas inside quotes)
                var p = ParseCsvLine(line);
                
                // Skip if line doesn't have all required fields
                if (p.Length < 6) continue;
                
                // Create product object
                products.Add(new Product {
                    ProductId = int.Parse(p[0]),
                    Asin = p[1],
                    Name = p[2].Trim('"'),  // Remove quotes around product name
                    Category = p[3],
                    
                    // Try to parse price, default to 0 if it fails
                    Price = double.TryParse(p[4], NumberStyles.Any, 
                                          CultureInfo.InvariantCulture, out var pr) ? pr : 0,
                    
                    // Only set image URL if it's not empty
                    ImageUrl = string.IsNullOrWhiteSpace(p[5]) ? null : p[5]
                });
            }
            catch 
            {
                // Skip lines that can't be parsed (corrupted data)
            }
        }
        
        return products;
    }

    // Custom CSV parser that handles commas inside quoted strings
    // Standard Split(',') doesn't work for: ProductId,Asin,"Name, with comma",Category,Price
    private static string[] ParseCsvLine(string line)
    {
        var result = new List<string>();
        bool inQuotes = false;
        var current = new System.Text.StringBuilder();
        
        // Go through each character
        foreach (char c in line)
        {
            if (c == '"')
            {
                // Toggle quote state
                inQuotes = !inQuotes;
            }
            else if (c == ',' && !inQuotes)
            {
                // Comma outside quotes = field separator
                result.Add(current.ToString());
                current.Clear();
            }
            else
            {
                // Regular character, add to current field
                current.Append(c);
            }
        }
        
        // Don't forget the last field
        result.Add(current.ToString());
        
        return result.ToArray();
    }
}