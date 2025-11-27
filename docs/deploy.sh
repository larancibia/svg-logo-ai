#!/bin/bash

# LLM-QD Logo System - Web Visualization Deployment Script
# Supports Cloudflare Pages, GitHub Pages, Vercel, and Netlify

set -e

echo "🚀 LLM-QD Logo System - Deployment Script"
echo "=========================================="
echo ""

# Check if we're in the web directory
if [ ! -f "results_visualization.html" ]; then
    echo "❌ Error: Must run from the web/ directory"
    echo "   cd web && ./deploy.sh"
    exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Display deployment options
echo "Select deployment target:"
echo "1) Cloudflare Pages (via CLI)"
echo "2) GitHub Pages (git push)"
echo "3) Vercel"
echo "4) Netlify"
echo "5) Local preview (Python HTTP server)"
echo "6) Local preview (Node.js HTTP server)"
echo ""
read -p "Enter choice [1-6]: " choice

case $choice in
    1)
        echo ""
        echo "📦 Deploying to Cloudflare Pages..."
        echo ""

        if ! command_exists wrangler; then
            echo "❌ Wrangler CLI not found. Installing..."
            npm install -g wrangler
        fi

        echo "🔐 Logging in to Cloudflare..."
        wrangler login

        read -p "Enter project name (default: llm-qd-logo-system): " project_name
        project_name=${project_name:-llm-qd-logo-system}

        echo "🚢 Publishing to Cloudflare Pages..."
        wrangler pages publish . --project-name="$project_name"

        echo ""
        echo "✅ Deployed successfully!"
        echo "🌐 Your site is live at: https://$project_name.pages.dev"
        ;;

    2)
        echo ""
        echo "📦 Preparing for GitHub Pages deployment..."
        echo ""

        if [ ! -d "../.git" ]; then
            echo "❌ Error: Not a git repository"
            echo "   Initialize git first: git init"
            exit 1
        fi

        echo "1. Ensure your changes are committed:"
        echo "   git add ."
        echo "   git commit -m 'Add web visualization'"
        echo ""
        echo "2. Push to GitHub:"
        echo "   git push origin main"
        echo ""
        echo "3. Enable GitHub Pages:"
        echo "   - Go to repository Settings > Pages"
        echo "   - Source: Deploy from a branch"
        echo "   - Branch: main, Folder: /web"
        echo "   - Click Save"
        echo ""
        echo "4. Your site will be live at:"
        echo "   https://USERNAME.github.io/REPOSITORY/"
        echo ""

        read -p "Push to GitHub now? (y/n): " push_choice
        if [ "$push_choice" = "y" ]; then
            cd ..
            git add web/
            git commit -m "Add LLM-QD web visualization"
            git push
            echo "✅ Pushed to GitHub!"
        fi
        ;;

    3)
        echo ""
        echo "📦 Deploying to Vercel..."
        echo ""

        if ! command_exists vercel; then
            echo "❌ Vercel CLI not found. Installing..."
            npm install -g vercel
        fi

        echo "🚢 Deploying..."
        vercel --prod

        echo ""
        echo "✅ Deployed successfully!"
        ;;

    4)
        echo ""
        echo "📦 Deploying to Netlify..."
        echo ""

        if ! command_exists netlify; then
            echo "❌ Netlify CLI not found. Installing..."
            npm install -g netlify-cli
        fi

        echo "🚢 Deploying..."
        netlify deploy --prod --dir=.

        echo ""
        echo "✅ Deployed successfully!"
        ;;

    5)
        echo ""
        echo "🌐 Starting local preview (Python)..."
        echo ""

        if ! command_exists python3; then
            echo "❌ Python 3 not found"
            exit 1
        fi

        echo "✅ Server running at: http://localhost:8000"
        echo "   Press Ctrl+C to stop"
        echo ""
        python3 -m http.server 8000
        ;;

    6)
        echo ""
        echo "🌐 Starting local preview (Node.js)..."
        echo ""

        if ! command_exists npx; then
            echo "❌ Node.js/npm not found"
            exit 1
        fi

        echo "✅ Server running at: http://localhost:8000"
        echo "   Press Ctrl+C to stop"
        echo ""
        npx http-server -p 8000
        ;;

    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "🎉 Done!"
