using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace GiftApi;

public class RecommendationService
{
    private readonly MatrixFactorization _mf;
    private readonly List<Product> _products;
    private readonly int _numUsers;

    public RecommendationService(string dataPath)
    {
        Console.WriteLine("Loading recommendation data...");
        
        var (users, items, ratings) = LoadRatings(Path.Combine(dataPath, "ratings.csv"));
        _products = LoadProducts(Path.Combine(dataPath, "products.csv"));

        Console.WriteLine($"  Loaded {_products.Count:N0} products");
        
        _numUsers = users.Max() + 1;
        int numItems = items.Max() + 1;
        Console.WriteLine($"  {ratings.Length:N0} ratings, {_numUsers:N0} users");

        _mf = new MatrixFactorization(_numUsers, numItems, numFactors: 20);
        _mf.Train(users, items, ratings, epochs: 15);
        Console.WriteLine("Ready!");
    }

    public List<Product> GetRecommendations(
        int? userId = null, double? minPrice = null, double? maxPrice = null,
        double? targetPrice = null, string? category = null, List<string>? categories = null,
        string? gender = null, int? age = null, int maxPerCategory = 2, int topN = 10)
    {
        var candidates = _products.AsEnumerable();

        // Price filter
        if (minPrice.HasValue) candidates = candidates.Where(p => p.Price >= minPrice.Value);
        if (maxPrice.HasValue) candidates = candidates.Where(p => p.Price <= maxPrice.Value);

        // Category filter
        if (categories?.Count > 0)
            candidates = candidates.Where(p => categories.Contains(p.Category, StringComparer.OrdinalIgnoreCase));
        else if (!string.IsNullOrEmpty(category))
            candidates = candidates.Where(p => p.Category.Equals(category, StringComparison.OrdinalIgnoreCase));

        // Gender filter (includes category exclusions)
        if (!string.IsNullOrEmpty(gender))
            candidates = candidates.Where(p => IsGenderOk(p.Name, gender, p.Category));

        // Age filter - applies to ALL categories
        if (age.HasValue)
            candidates = candidates.Where(p => IsAgeOk(p.Name, p.Category, age.Value));

        // Age-based randomizer - each age gets completely different random seed
        int ageSeed = age ?? 0;
        var random = new Random(ageSeed * 104729); // Large prime for better distribution

        // Score and rank
        var scored = candidates.Select(p => {
            double score = userId.HasValue && userId.Value < _numUsers 
                ? _mf.Predict(userId.Value, p.ProductId) 
                : _mf.GetItemBias(p.ProductId);
            
            // Price scoring
            if (targetPrice.HasValue && targetPrice.Value > 0)
            {
                double diff = Math.Abs(p.Price - targetPrice.Value);
                double maxDiff = targetPrice.Value * 0.3;
                
                if (diff < maxDiff)
                    score += 1.0 * (1 - diff / maxDiff);
                else
                    score -= 0.5;
            }

            // Very strong randomness - dominates the scoring to ensure different products per age
            score += random.NextDouble() * 3.0;

            return (product: p, score);
        }).OrderByDescending(x => x.score).ToList();

        // Diversity - limit per category
        var result = new List<Product>();
        var catCount = new Dictionary<string, int>();
        foreach (var item in scored)
        {
            string cat = item.product.Category;
            catCount.TryGetValue(cat, out int count);
            if (count < maxPerCategory)
            {
                result.Add(item.product);
                catCount[cat] = count + 1;
                if (result.Count >= topN) break;
            }
        }
        return result;
    }

    public List<string> GetCategories() => 
        _products.Select(p => p.Category).Distinct().OrderBy(c => c).ToList();

    private static bool IsGenderOk(string name, string gender, string category)
    {
        var lower = name.ToLower();
        
        // Words that indicate male products - filter these for females
        var maleWords = new[] { "men's", "mens", "for men", "for him", "beard", "shaving", "shave", "razor", "aftershave", "cologne" };
        
        // Words that indicate female/child products - filter these for males
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

        if (gender == "male")
        {
            // Exclude entire categories for males
            if (category == "Home_and_Kitchen" || category == "Beauty_and_Personal_Care")
                return false;
            // Check product name
            if (notForMaleWords.Any(w => lower.Contains(w))) 
                return false;
        }

        if (gender == "female" && maleWords.Any(w => lower.Contains(w))) 
            return false;
            
        return true;
    }

    private static bool IsAgeOk(string name, string category, int age)
    {
        var lower = name.ToLower();

        // For adults (18+), filter out ALL kid/baby/child products across ALL categories
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
        // For teens (13-17), filter baby/toddler but allow some kid stuff
        else if (age >= 13)
        {
            var teenFilter = new[] { 
                "baby", "babies", "infant", "toddler", "toddlers",
                "newborn", "nursery", "ages 1", "ages 2", "ages 3", "ages 4", "ages 5"
            };
            if (teenFilter.Any(w => lower.Contains(w)))
                return false;
        }

        // Additional Toys-specific age filtering
        if (category == "Toys_and_Games")
        {
            var babyToys = new[] { "rattle", "teether", "0-3", "1-3" };
            var preschoolToys = new[] { "preschool", "3-5", "paw patrol", "peppa pig", "duplo" };
            var youngKidToys = new[] { "6-8", "ages 6", "ages 7", "ages 8" };

            if (age >= 13 && (babyToys.Any(w => lower.Contains(w)) || preschoolToys.Any(w => lower.Contains(w)) || youngKidToys.Any(w => lower.Contains(w))))
                return false;
            if (age >= 9 && (babyToys.Any(w => lower.Contains(w)) || preschoolToys.Any(w => lower.Contains(w))))
                return false;
            if (age >= 6 && babyToys.Any(w => lower.Contains(w)))
                return false;
        }
        
        return true;
    }

    private static (int[] users, int[] items, double[] ratings) LoadRatings(string path)
    {
        var lines = File.ReadAllLines(path).Skip(1).ToArray();
        var users = new int[lines.Length];
        var items = new int[lines.Length];
        var ratings = new double[lines.Length];

        for (int i = 0; i < lines.Length; i++)
        {
            var p = lines[i].Split(',');
            users[i] = int.Parse(p[0]);
            items[i] = int.Parse(p[1]);
            ratings[i] = double.Parse(p[2], CultureInfo.InvariantCulture);
        }
        return (users, items, ratings);
    }

    private static List<Product> LoadProducts(string path)
    {
        var products = new List<Product>();
        foreach (var line in File.ReadAllLines(path).Skip(1))
        {
            try
            {
                var p = ParseCsvLine(line);
                if (p.Length < 6) continue;
                products.Add(new Product {
                    ProductId = int.Parse(p[0]),
                    Asin = p[1],
                    Name = p[2].Trim('"'),
                    Category = p[3],
                    Price = double.TryParse(p[4], NumberStyles.Any, CultureInfo.InvariantCulture, out var pr) ? pr : 0,
                    ImageUrl = string.IsNullOrWhiteSpace(p[5]) ? null : p[5]
                });
            }
            catch { }
        }
        return products;
    }

    private static string[] ParseCsvLine(string line)
    {
        var result = new List<string>();
        bool inQuotes = false;
        var current = new System.Text.StringBuilder();
        foreach (char c in line)
        {
            if (c == '"') inQuotes = !inQuotes;
            else if (c == ',' && !inQuotes) { result.Add(current.ToString()); current.Clear(); }
            else current.Append(c);
        }
        result.Add(current.ToString());
        return result.ToArray();
    }
}