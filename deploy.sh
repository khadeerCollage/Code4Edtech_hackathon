#!/bin/bash

echo "🚀 Deploying OMR Evaluation System to Vercel"

# Build the frontend
echo "📦 Building frontend..."
cd frontend
npm run build
cd ..

echo "✅ Build complete!"
echo ""
echo "📋 Next Steps:"
echo "1. Push your code to GitHub:"
echo "   git add ."
echo "   git commit -m 'Deploy to Vercel'"
echo "   git push origin main"
echo ""
echo "2. Go to vercel.com and import your repository"
echo ""
echo "3. Set environment variables:"
echo "   VITE_API_URL=https://your-app-name.vercel.app"
echo "   DATABASE_URL=your_postgresql_url"
echo "   JWT_SECRET_KEY=your_secret_key"
echo ""
echo "🎉 Your app will be live in minutes!"