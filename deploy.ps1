Write-Host "🚀 Deploying OMR Evaluation System to Vercel" -ForegroundColor Green

# Build the frontend
Write-Host "📦 Building frontend..." -ForegroundColor Yellow
Set-Location frontend
npm run build
Set-Location ..

Write-Host "✅ Build complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Next Steps:" -ForegroundColor Cyan
Write-Host "1. Push your code to GitHub:"
Write-Host "   git add ."
Write-Host "   git commit -m 'Deploy to Vercel'"
Write-Host "   git push origin main"
Write-Host ""
Write-Host "2. Go to vercel.com and import your repository"
Write-Host ""
Write-Host "3. Set environment variables:"
Write-Host "   VITE_API_URL=https://your-app-name.vercel.app"
Write-Host "   DATABASE_URL=your_postgresql_url"
Write-Host "   JWT_SECRET_KEY=your_secret_key"
Write-Host ""
Write-Host "🎉 Your app will be live in minutes!" -ForegroundColor Green