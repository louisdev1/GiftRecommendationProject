# GiftFinder Azure Deployment Script
# Run this from your GiftAPI folder

Write-Host "GiftFinder Azure Deployment" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the right folder by looking for required files
$requiredFiles = @("GiftAPI.csproj", "Program.cs", "products.csv", "ratings.csv")
$missing = $requiredFiles | Where-Object { -not (Test-Path $_) }

if ($missing) {
    Write-Host "ERROR: Missing files: $($missing -join ', ')" -ForegroundColor Red
    Write-Host "Make sure you're in the GiftAPI folder!" -ForegroundColor Red
    exit 1
}

Write-Host "[1/5] Checking required files... OK" -ForegroundColor Green

# Verify updated files are present
Write-Host "[2/5] Verifying updated files..." -ForegroundColor Yellow

if (Test-Path "index.html") {
    Write-Host "  - index.html: Found" -ForegroundColor Green
} else {
    Write-Host "  - index.html: MISSING - Copy from deploy_package!" -ForegroundColor Red
    exit 1
}

if (Test-Path "RecommendationService.cs") {
    Write-Host "  - RecommendationService.cs: Found" -ForegroundColor Green
} else {
    Write-Host "  - RecommendationService.cs: MISSING!" -ForegroundColor Red
    exit 1
}

# Clean old build and create new one
Write-Host "[3/5] Building project..." -ForegroundColor Yellow

# Delete old publish folder if it exists
Remove-Item -Recurse -Force ./publish -ErrorAction SilentlyContinue

# Build in Release mode (optimized for production)
dotnet publish -c Release -o ./publish

# Check if build succeeded
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "  Build successful!" -ForegroundColor Green

# Copy data files to the publish folder
# These files need to be included in the deployment package
Write-Host "[4/5] Copying data files..." -ForegroundColor Yellow

Copy-Item products.csv ./publish/
Copy-Item ratings.csv ./publish/
Copy-Item appsettings.json ./publish/
Copy-Item index.html ./publish/

Write-Host "  Files copied:" -ForegroundColor Green
Get-ChildItem ./publish | ForEach-Object { Write-Host "    - $($_.Name)" }

# Create zip file for Azure deployment
Write-Host "[5/5] Creating deployment package..." -ForegroundColor Yellow

# Remove old zip if it exists
Remove-Item ./deploy.zip -ErrorAction SilentlyContinue

# Compress everything in publish folder into deploy.zip
Compress-Archive -Path ./publish/* -DestinationPath ./deploy.zip -Force

# Show zip file size
$zipSize = (Get-Item ./deploy.zip).Length / 1MB
Write-Host "  deploy.zip created ($([math]::Round($zipSize, 2)) MB)" -ForegroundColor Green

Write-Host ""
Write-Host "DEPLOYMENT READY!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Go to: https://giftrecommenderlb.scm.azurewebsites.net" -ForegroundColor White
Write-Host "2. Navigate to: Tools > Zip Push Deploy" -ForegroundColor White
Write-Host "3. Drag and drop: deploy.zip" -ForegroundColor White
Write-Host "4. Wait for deployment to complete" -ForegroundColor White
Write-Host "5. Test at: https://giftrecommenderlb.azurewebsites.net" -ForegroundColor White
Write-Host ""