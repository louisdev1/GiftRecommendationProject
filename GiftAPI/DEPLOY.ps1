# GiftFinder Azure Deployment Script
# Run this from your GiftAPI folder

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  GiftFinder Azure Deployment Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check we're in the right folder
$requiredFiles = @("GiftAPI.csproj", "Program.cs", "products.csv", "ratings.csv")
$missing = $requiredFiles | Where-Object { -not (Test-Path $_) }

if ($missing) {
    Write-Host "ERROR: Missing files: $($missing -join ', ')" -ForegroundColor Red
    Write-Host "Make sure you're in the GiftAPI folder!" -ForegroundColor Red
    exit 1
}

Write-Host "[1/5] Checking required files... OK" -ForegroundColor Green

# Step 2: Copy new files (index.html and RecommendationService.cs should already be updated)
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

# Step 3: Clean and build
Write-Host "[3/5] Building project..." -ForegroundColor Yellow
Remove-Item -Recurse -Force ./publish -ErrorAction SilentlyContinue
dotnet publish -c Release -o ./publish

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Build failed!" -ForegroundColor Red
    exit 1
}
Write-Host "  Build successful!" -ForegroundColor Green

# Step 4: Copy data files to publish folder
Write-Host "[4/5] Copying data files..." -ForegroundColor Yellow
Copy-Item products.csv ./publish/
Copy-Item ratings.csv ./publish/
Copy-Item appsettings.json ./publish/
Copy-Item index.html ./publish/

Write-Host "  Files copied:" -ForegroundColor Green
Get-ChildItem ./publish | ForEach-Object { Write-Host "    - $($_.Name)" }

# Step 5: Create deployment zip
Write-Host "[5/5] Creating deployment package..." -ForegroundColor Yellow
Remove-Item ./deploy.zip -ErrorAction SilentlyContinue
Compress-Archive -Path ./publish/* -DestinationPath ./deploy.zip -Force

$zipSize = (Get-Item ./deploy.zip).Length / 1MB
Write-Host "  deploy.zip created ($([math]::Round($zipSize, 2)) MB)" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  DEPLOYMENT READY!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Go to: https://giftrecommenderlb.scm.azurewebsites.net" -ForegroundColor White
Write-Host "2. Navigate to: Tools > Zip Push Deploy" -ForegroundColor White
Write-Host "3. Drag and drop: deploy.zip" -ForegroundColor White
Write-Host "4. Wait for deployment to complete" -ForegroundColor White
Write-Host "5. Test at: https://giftrecommenderlb.azurewebsites.net" -ForegroundColor White
Write-Host ""
