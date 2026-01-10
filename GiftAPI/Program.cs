using System;
using System.IO;
using System.Linq;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.FileProviders;
using GiftApi;

// Create the web application builder - this sets up our web server
var builder = WebApplication.CreateBuilder(args);

// Add CORS policy so our frontend can talk to the API from any domain
// This is needed because the HTML frontend might be served from a different port or domain
builder.Services.AddCors(options => {
    options.AddDefaultPolicy(policy => 
        policy.AllowAnyOrigin()
              .AllowAnyMethod()
              .AllowAnyHeader());
});

// Set up the data path where our CSV files are located
var dataPath = "./";

// Create instances of our main services
// RecommendationService handles all the machine learning and product recommendations
var recommender = new RecommendationService(dataPath);

// NaturalLanguageService translates user queries like "gift for my mom" into filters
var nlp = new NaturalLanguageService();

// Register these services so they can be used in our API endpoints
builder.Services.AddSingleton(recommender);
builder.Services.AddSingleton(nlp);

// Build the actual web application
var app = builder.Build();

// Enable CORS so browser requests work
app.UseCors();

// Set up static file serving so we can serve the index.html file
// This makes the API also serve the frontend webpage
var fileProvider = new PhysicalFileProvider(Directory.GetCurrentDirectory());
app.UseDefaultFiles(new DefaultFilesOptions { FileProvider = fileProvider });
app.UseStaticFiles(new StaticFileOptions { FileProvider = fileProvider });

// API endpoint for natural language search
// User sends text like "birthday gift for my 25 year old sister under 50 euros"
app.MapPost("/api/search", async (NaturalLanguageService nlpService, 
                                   RecommendationService svc, 
                                   SearchRequest request) =>
{
    // Parse the user's natural language query into structured filters
    var query = await nlpService.ParseQueryAsync(request.Query);
    
    // If the user specified categories, show more items per category
    // Otherwise keep it at 3 per category for more diversity
    int maxPerCat = (query.SuggestedCategories?.Count ?? 0) > 0 ? 4 : 3;
    
    // Get product recommendations based on the parsed query
    var results = svc.GetRecommendations(
        categories: query.SuggestedCategories,
        gender: query.Gender,
        age: query.Age,
        minPrice: query.MinPrice,
        maxPrice: query.MaxPrice,
        targetPrice: query.TargetPrice,
        maxPerCategory: maxPerCat,
        topN: 9
    );
    
    // Return both the parsed query (so user can see what we understood) and the products
    return Results.Ok(new { parsedQuery = query, products = results });
});

// API endpoint for filter-based search (not natural language)
// User can directly specify price ranges, categories, etc.
app.MapGet("/api/recommend", (RecommendationService svc, 
                               double? minPrice, 
                               double? maxPrice, 
                               string? category, 
                               int? count) =>
{
    var results = svc.GetRecommendations(
        minPrice: minPrice, 
        maxPrice: maxPrice, 
        category: category, 
        topN: count ?? 9
    );
    return Results.Ok(results);
});

// Simple endpoint to get all available product categories
app.MapGet("/api/categories", (RecommendationService svc) => 
    Results.Ok(svc.GetCategories()));

// Health check endpoint to verify the API is running
app.MapGet("/api/health", () => 
    Results.Ok(new { status = "ok" }));

Console.WriteLine("\nAPI running...");

// Start the web server on port 8080, listening on all network interfaces
// 0.0.0.0 means it accepts connections from anywhere (needed for Azure deployment)
app.Run("http://0.0.0.0:8080");

// Simple record to define the structure of search requests
record SearchRequest(string Query);