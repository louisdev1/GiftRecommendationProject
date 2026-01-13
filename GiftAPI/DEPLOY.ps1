# Azure Deployment

Write-Host "GiftFinder Azure Deployment"
Write-Host ""

# Check of we in de juiste folder zijn

$requiredFiles = @("GiftAPI.csproj", "Program.cs", "products.csv", "ratings.csv")
$missing = $requiredFiles | Where-Object { -not (Test-Path $_) }
if ($missing) {
    Write-Host "ERROR: Missing files: $($missing -join ', ')" -ForegroundColor Red
    exit 1
}
Write-Host "1/5 Checking required files" -ForegroundColor Green

Write-Host "2/5 Verifying updated files" -ForegroundColor Green

if (Test-Path "index.html") {
    Write-Host "  - index.html: Found" -ForegroundColor Green
} else {
    Write-Host "  - index.html is missing" -ForegroundColor Red
    exit 1
}

if (Test-Path "RecommendationService.cs") {
    Write-Host "  - RecommendationService.cs: Found" -ForegroundColor Green
} else {
    Write-Host "  - RecommendationService.cs is missing" -ForegroundColor Red
    exit 1
}

# Clean old build and create new one
Write-Host "3/5 Building project..." -ForegroundColor Green
Remove-Item -Recurse -Force ./publish -ErrorAction SilentlyContinue
dotnet publish -c Release -o ./publish
# Check if build worked
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed" -ForegroundColor Red
    exit 1
}
Write-Host "  Build successful" -ForegroundColor Green

# Copy data files to publish folder
Write-Host "4/5 Copying data files"
Copy-Item products.csv ./publish/
Copy-Item ratings.csv ./publish/
Copy-Item appsettings.json ./publish/
Copy-Item index.html ./publish/
Write-Host "  Files copied:" -ForegroundColor Green
Get-ChildItem ./publish | ForEach-Object { Write-Host "    - $($_.Name)" }
# Create zip file for Azure deployment
Write-Host "5/5 Creating deployment package" -ForegroundColor Green
Remove-Item ./deploy.zip -ErrorAction SilentlyContinue
# Compress everything into deploy.zip
Compress-Archive -Path ./publish/* -DestinationPath ./deploy.zip -Force

$zipSize = (Get-Item ./deploy.zip).Length / 1MB
Write-Host "  deploy.zip created ($([math]::Round($zipSize, 2)) MB)" -ForegroundColor Green
Write-Host ""
Write-Host "Deployment ready" -ForegroundColor Green