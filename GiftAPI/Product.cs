namespace GiftApi;

public class Product
{
    public int ProductId { get; set; }
    public string Asin { get; set; } = "";
    public string Name { get; set; } = "";
    public string Category { get; set; } = "";
    public double Price { get; set; }
    public string? ImageUrl { get; set; }
}