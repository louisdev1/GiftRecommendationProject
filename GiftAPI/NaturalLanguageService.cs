using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace GiftApi;

public class NaturalLanguageService
{
    // HTTP client for making API calls to DeepL translation service
    private readonly HttpClient _http = new();
    
    // DeepL API key for translation (loaded from config file)
    private readonly string? _apiKey;

    // Constructor - loads API key from appsettings.json
    public NaturalLanguageService()
    {
        try 
        {
            // Read the configuration file
            var json = File.ReadAllText("appsettings.json");
            using var doc = JsonDocument.Parse(json);
            
            // Extract the DeepL API key
            _apiKey = doc.RootElement
                .GetProperty("ApiKeys")
                .GetProperty("DeepL")
                .GetString();
        } 
        catch 
        {
            _http.DefaultRequestHeaders.Add("Authorization", $"DeepL-Auth-Key {_apiKey}");
            
            // Send request to DeepL API
            var resp = await _http.PostAsync("https://api-free.deepl.com/v2/translate", content);
            var json = await resp.Content.ReadAsStringAsync();
            
            // Parse response and extract translated text
            using var doc = JsonDocument.Parse(json);
            return doc.RootElement
                .GetProperty("translations")[0]
                .GetProperty("text")
                .GetString() ?? text;
        } 
        catch 
        {
            // If translation fails, return original text
            return text;
        }
    }

    // Extract all filters from the text
    private GiftQuery ParseFilters(string text)
    {
        var query = new GiftQuery { OriginalText = text };
        query.Relationship = ExtractMatch(text, _relationships);
        query.Occasion = ExtractMatch(text, _occasions);
        query.Age = ExtractAge(text);
        query.Gender = GetGender(text, query.Relationship);
        query.SuggestedCategories = GetCategories(query.Relationship, query.Occasion, query.Age);
        return query;
    }

    // Generic pattern matching - checks if any pattern matches the text
    private string? ExtractMatch(string text, Dictionary<string, string[]> patterns)
    {
        foreach (var (key, words) in patterns)
        {
            // If any of the patterns for this key match, return the key
            if (words.Any(w => Regex.IsMatch(text, w))) 
                return key;
        }
        return null;
    }

    // Determine gender from text or relationship
    private string? GetGender(string text, string? rel)
    {
        // Check for explicit gender words in text
        if (Regex.IsMatch(text, @"\b(woman|women|female|girl|for her)\b")) 
            return "female";
        if (Regex.IsMatch(text, @"\b(man|men|male|boy|for him)\b")) 
            return "male";
        
        // Infer gender from relationship
        if (new[] { "girlfriend", "wife", "mom", "mother", "grandma", "sister", "daughter" }.Contains(rel)) 
            return "female";
        if (new[] { "boyfriend", "husband", "dad", "father", "grandpa", "brother", "son" }.Contains(rel)) 
            return "male";
        
        return null;  // Gender unknown
    }

    // Extract age from text
    private int? ExtractAge(string text)
    {
        // Handle age keywords
        if (Regex.IsMatch(text, @"\bteen")) return 15;
        if (Regex.IsMatch(text, @"\btoddler")) return 2;
        if (Regex.IsMatch(text, @"\bbaby|\binfant")) return 0;
        
        // Extract  age numbers
        var m = Regex.Match(text, @"(\d+)\s*(?:year|yr)s?\s*old|age\s*(\d+)");
        if (m.Success)
        {
            // Extract the number from whichever group matched
            string ageStr = m.Groups[1].Success ? m.Groups[1].Value : m.Groups[2].Value;
            return int.Parse(ageStr);
        }
        
        return null;  // Age not found
    }

    // Extract budget from text
    private (double? min, double? max, double? target) ExtractBudget(string text)
    {
        // Match patterns like "$50"
        var m = Regex.Match(text, 
            @"[\$€]\s*(\d+)|(\d+)\s*[\$€]|(\d+)\s*(?:dollar|euro)s?", 
            RegexOptions.IgnoreCase);
        
        if (m.Success) 
        {
            // Extract the number, if not 1 use 2, and otherwise use 3
            string valueStr = m.Groups[1].Success ? m.Groups[1].Value : 
                            m.Groups[2].Success ? m.Groups[2].Value : 
                            m.Groups[3].Value;
            var val = double.Parse(valueStr);
            return (null, val, val * 0.85);
        }
        
        return (null, null, null);  // No budget found
    }

    // Suggest categories based on relationship, occasion, and age
    private List<string> GetCategories(string? rel, string? occasion, int? age)
    {
        // Age-based categories
        if (age.HasValue && age < 16) 
        {
            // Babies (0-2 years)
            if (age < 3) 
                return new() { "Baby_Products", "Toys_and_Games" };
            
            // (3-8 years)
            if (age < 9) 
                return new() { "Toys_and_Games", "Sports_and_Outdoors" };
            
            // (9-15 years)
            return new() { "Toys_and_Games", "Electronics", "Sports_and_Outdoors" };
        }

        // Occasion-based categories
        if (occasion == "valentines" || occasion == "anniversary")
            return new() { "Clothing_Shoes_and_Jewelry", "Beauty_and_Personal_Care", "Home_and_Kitchen" };
        
        if (occasion == "christmas") 
        {
            if (_female.Contains(rel)) 
                return new() { "Home_and_Kitchen", "Beauty_and_Personal_Care", "Clothing_Shoes_and_Jewelry" };
            if (_male.Contains(rel)) 
                return new() { "Electronics", "Home_and_Kitchen", "Sports_and_Outdoors" };
            return new() { "Home_and_Kitchen", "Electronics", "Clothing_Shoes_and_Jewelry" };
        }
        
        if (occasion == "birthday") 
        {
            if (_female.Contains(rel)) 
                return new() { "Clothing_Shoes_and_Jewelry", "Beauty_and_Personal_Care", "Electronics" };            
            if (_male.Contains(rel)) 
                return new() { "Electronics", "Sports_and_Outdoors", "Clothing_Shoes_and_Jewelry" };
        }

        // Relationship-based categories (fallback)
        if (_female.Contains(rel)) 
            return new() { "Clothing_Shoes_and_Jewelry", "Beauty_and_Personal_Care", "Home_and_Kitchen" };
        
        if (_male.Contains(rel)) 
            return new() { "Electronics", "Sports_and_Outdoors", "Clothing_Shoes_and_Jewelry", "Home_and_Kitchen" };
        
        if (new[] { "child", "kid", "son", "daughter" }.Contains(rel)) 
            return new() { "Toys_and_Games", "Electronics" };
        
        if (new[] { "dog", "cat", "pet" }.Contains(rel)) 
            return new() { "Pet_Supplies" };

        // No specific categories
        return new();
    }

    // Lists of relationships by gender (used for gender inference)
    private readonly string[] _female = { 
        "girlfriend", "wife", "mom", "mother", "grandma", "sister", "aunt", "daughter" 
    };
    
    private readonly string[] _male = { 
        "boyfriend", "husband", "dad", "father", "grandpa", "brother", "uncle", "son" 
    };

    // Dictionary of relationship patterns for text matching
    // Key = relationship name, Value = regex patterns to match
    private readonly Dictionary<string, string[]> _relationships = new() 
    {
        ["girlfriend"] = new[] { @"\bgirlfriend\b", @"\bgf\b" },
        ["boyfriend"] = new[] { @"\bboyfriend\b", @"\bbf\b" },
        ["wife"] = new[] { @"\bwife\b" },
        ["husband"] = new[] { @"\bhusband\b" },
        ["mom"] = new[] { @"\bmom\b", @"\bmother\b" },
        ["dad"] = new[] { @"\bdad\b", @"\bfather\b" },
        ["sister"] = new[] { @"\bsister\b" },
        ["brother"] = new[] { @"\bbrother\b" },
        ["grandma"] = new[] { @"\bgrandma\b", @"\bgrandmother\b" },
        ["grandpa"] = new[] { @"\bgrandpa\b", @"\bgrandfather\b" },
        ["son"] = new[] { @"\bson\b" },
        ["daughter"] = new[] { @"\bdaughter\b" },
        ["child"] = new[] { @"\bchild\b", @"\bkid\b" },
        ["friend"] = new[] { @"\bfriend\b" },
        ["dog"] = new[] { @"\bdog\b" },
        ["cat"] = new[] { @"\bcat\b" },
        ["pet"] = new[] { @"\bpet\b" }
    };

    private readonly Dictionary<string, string[]> _occasions = new() 
    {
        ["birthday"] = new[] { @"\bbirthday\b" },
        ["christmas"] = new[] { @"\bchristmas\b", @"\bxmas\b" },
        ["valentines"] = new[] { @"\bvalentine" },
        ["anniversary"] = new[] { @"\banniversary\b" }
    };
}

public class GiftQuery
{
    public string OriginalText { get; set; } = "";
    public double? MinPrice { get; set; }
    public double? MaxPrice { get; set; }
    public double? TargetPrice { get; set; }
    public List<string> SuggestedCategories { get; set; } = new();
    public string? Occasion { get; set; }
    public string? Relationship { get; set; }
    public string? Gender { get; set; }
    public int? Age { get; set; }
}