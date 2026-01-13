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

var dataPath = "./";

var recommender = new RecommendationService(dataPath);

var nlp = new NaturalLanguageService();

// Register these services so they can be used in our API endpoints
builder.Services.AddSingleton(recommender);
builder.Services.AddSingleton(nlp);

// Build the actual web application
var app = builder.Build();

// Enable CORS so browser requests work
app.UseCors();

var fileProvider = new PhysicalFileProvider(Directory.GetCurrentDirectory());
app.UseDefaultFiles(new DefaultFilesOptions { FileProvider = fileProvider });
app.UseStaticFiles(new StaticFileOptions { FileProvider = fileProvider });

// API endpoint for natural language search
app.MapPost("/api/search", async (NaturalLanguageService nlpService, 
                                   RecommendationService svc, 
                                   SearchRequest request) =>
{
    var query = await nlpService.ParseQueryAsync(request.Query);
    
    int maxPerCat = (query.SuggestedCategories?.Count ?? 0) > 0 ? 4 : 3;
    
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
    
    return Results.Ok(new { parsedQuery = query, products = results });
});

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

// 0.0.0.0 means it accepts connections from anywhere (needed for Azure deployment)
app.Run("http://0.0.0.0:8080");

// Simple record to define the structure of search requests
record SearchRequest(string Query);