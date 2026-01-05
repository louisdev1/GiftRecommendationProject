using System;
using System.IO;
using System.Linq;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.FileProviders;
using GiftApi;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddCors(options => {
    options.AddDefaultPolicy(policy => policy.AllowAnyOrigin().AllowAnyMethod().AllowAnyHeader());
});

var dataPath = "./";
var recommender = new RecommendationService(dataPath);
var nlp = new NaturalLanguageService();

builder.Services.AddSingleton(recommender);
builder.Services.AddSingleton(nlp);

var app = builder.Build();
app.UseCors();

// Serve static files from current directory (where index.html is)
var fileProvider = new PhysicalFileProvider(Directory.GetCurrentDirectory());
app.UseDefaultFiles(new DefaultFilesOptions { FileProvider = fileProvider });
app.UseStaticFiles(new StaticFileOptions { FileProvider = fileProvider });

// Natural language search
app.MapPost("/api/search", async (NaturalLanguageService nlpService, RecommendationService svc, SearchRequest request) =>
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

// Filter-based search
app.MapGet("/api/recommend", (RecommendationService svc, double? minPrice, double? maxPrice, string? category, int? count) =>
{
    var results = svc.GetRecommendations(minPrice: minPrice, maxPrice: maxPrice, category: category, topN: count ?? 9);
    return Results.Ok(results);
});

app.MapGet("/api/categories", (RecommendationService svc) => Results.Ok(svc.GetCategories()));
app.MapGet("/api/health", () => Results.Ok(new { status = "ok" }));

Console.WriteLine("\nAPI running...");

app.Run("http://0.0.0.0:8080");

record SearchRequest(string Query);