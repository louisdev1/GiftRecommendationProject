using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Microsoft.VisualBasic.FileIO;

class Program
{
    static void Main()
    {
        // LOAD RATINGS 
        var ratingLines = File.ReadAllLines("ratings.csv").Skip(1);

        List<int> users = new();
        List<int> items = new();
        List<double> ratings = new();

        foreach (var line in ratingLines)
        {
            var parts = line.Split(',');
            users.Add(int.Parse(parts[0]));
            items.Add(int.Parse(parts[1]));
            ratings.Add(double.Parse(parts[2]));
        }

        int numUsers = users.Max() + 1;
        int numItems = items.Max() + 1;

        // TRAIN MF 
        var mf = new MatrixFactorization(numUsers, numItems);
        mf.Train(users.ToArray(), items.ToArray(), ratings.ToArray());

        // LOAD PRODUCTS 
        var products = LoadProducts("products.csv");

        // TEST SCENARIOS 
        RunScenario(0, "partner", "anniversary", 250, products, mf);
        RunScenario(0, "child", "birthday", 50, products, mf);
        RunScenario(0, "colleague", "thank_you", 30, products, mf);
        RunScenario(0, "friend", "birthday", 80, products, mf);
        RunScenario(0, "parent", "christmas", 120, products, mf);
    }

    // DATA LOADING

    static List<Product> LoadProducts(string path)
    {
        var products = new List<Product>();
        int idx = 0;

        using (var parser = new TextFieldParser(path))
        {
            parser.TextFieldType = FieldType.Delimited;
            parser.SetDelimiters(",");
            parser.HasFieldsEnclosedInQuotes = true;

            parser.ReadLine(); // skip header

            while (!parser.EndOfData)
            {
                var fields = parser.ReadFields();
                if (fields == null || fields.Length < 4)
                    continue;

                products.Add(new Product
                {
                    ProductIdx = idx++,
                    ProductId = fields[0],
                    Title = fields[1],
                    Category = fields[2],
                    Price = double.Parse(fields[3])
                });
            }
        }

        return products;
    }

    // SCENARIO RUNNER

    static void RunScenario(
        int userId,
        string relationship,
        string occasion,
        double budget,
        List<Product> products,
        MatrixFactorization mf
    )
    {
        Console.WriteLine($"\n Scenario: {relationship} | {occasion} | budget €{budget}");

        var recommendations = GetTopNGifts(
            userId,
            relationship,
            occasion,
            budget,
            products,
            mf,
            topN: 5,
            maxPerCategory: 2
        );

        foreach (var (product, score) in recommendations)
        {
            Console.WriteLine(
                $"{product.Title} ({product.Category}) | €{product.Price} | score {score:F2}"
            );
        }
    }

    // RECOMMENDATION LOGIC (WITH DIVERSITY)

    static List<(Product, double)> GetTopNGifts(
        int userId,
        string relationship,
        string occasion,
        double maxBudget,
        List<Product> products,
        MatrixFactorization mf,
        int topN,
        int maxPerCategory
    )
    {
        var allowedCategories = RelationshipCategories(relationship);
        var budgetRange = OccasionBudget(occasion);

        var scored = new List<(Product, double)>();
        var seenTitles = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        //  FILTER + SCORE 
        foreach (var product in products)
        {
            if (!allowedCategories.Contains(product.Category))
                continue;

            if (product.Price > maxBudget)
                continue;

            if (product.Price < budgetRange.min || product.Price > budgetRange.max)
                continue;

            if (!seenTitles.Add(product.Title))
                continue;

            double score = mf.Predict(userId, product.ProductIdx);
            scored.Add((product, score));
        }

        // DIVERSITY RULE 
        var result = new List<(Product, double)>();
        var categoryCount = new Dictionary<string, int>();

        foreach (var item in scored.OrderByDescending(x => x.Item2))
        {
            var category = item.Item1.Category;

            if (!categoryCount.ContainsKey(category))
                categoryCount[category] = 0;

            if (categoryCount[category] >= maxPerCategory)
                continue;

            result.Add(item);
            categoryCount[category]++;

            if (result.Count == topN)
                break;
        }

        return result;
    }

    // RULES

    static HashSet<string> RelationshipCategories(string relationship)
    {
        return relationship switch
        {
            "partner" => new HashSet<string> { "Electronics", "Clothing & Jewelry" },
            "parent" => new HashSet<string> { "Home & Kitchen", "Electronics" },
            "child" => new HashSet<string> { "Toys & Games" },
            "friend" => new HashSet<string> { "Electronics", "Home & Kitchen", "Toys & Games" },
            "colleague" => new HashSet<string> { "Home & Kitchen" },
            _ => new HashSet<string>()
        };
    }

    static (double min, double max) OccasionBudget(string occasion)
    {
        return occasion switch
        {
            "birthday" => (20, 100),
            "christmas" => (30, 150),
            "anniversary" => (60, 300),
            "housewarming" => (20, 80),
            "thank_you" => (10, 40),
            _ => (0, double.MaxValue)
        };
    }
}
