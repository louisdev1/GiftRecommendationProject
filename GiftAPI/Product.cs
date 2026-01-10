namespace GiftApi;

// Simple data model representing a product in our recommendation system
// This is a POCO (Plain Old CLR Object) - just a container for data with no logic
public class Product
{
    // Unique numeric ID for this product (0 to 134,999)
    // This is what the Matrix Factorization model uses internally
    public int ProductId { get; set; }
    
    // Amazon Standard Identification Number - Amazon's unique product code
    // Used for generating Amazon purchase links
    public string Asin { get; set; } = "";
    
    // Human-readable product name
    // Example: "Wireless Bluetooth Headphones with Noise Cancelling"
    public string Name { get; set; } = "";
    
    // Product category (one of our 9 categories)
    // Example: "Electronics", "Toys_and_Games", "Beauty_and_Personal_Care"
    public string Category { get; set; } = "";
    
    // Product price in dollars/euros
    // Example: 29.99
    public double Price { get; set; }
    
    // URL to product image (nullable because some products might not have images)
    // Example: "https://m.media-amazon.com/images/I/61abc123.jpg"
    // The ? means this can be null
    public string? ImageUrl { get; set; }
}