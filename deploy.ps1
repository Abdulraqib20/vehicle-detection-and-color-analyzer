# DEPLOY.ps1 - One-Click GitHub Deployment
param(
    [string]$RepoName,
    [string]$Description = "Vehicle Detection & Color Analyzer"
)

# 1. GitHub Connection
$GitHubUser = "Abdulraqib20"
git remote set-url origin "https://github.com/$GitHubUser/$RepoName.git"

# 2. Project Branding
@"
# $RepoName

$Description

## Features
- [ ] TODO: Add features

## Setup
\`\`\`bash
git clone https://github.com/$GitHubUser/$RepoName.git
pip install -r requirements.txt
\`\`\`
"@ | Out-File README.md -Encoding UTF8

# 3. Atomic Commit
git add --all
git commit -m "🚀 Initial deploy: $RepoName

- Core implementation
- Deployment pipeline
- Documentation baseline"

# 4. Force Push
git push -u origin main --force

Write-Host "✅ Success! Live at: https://github.com/$GitHubUser/$RepoName" -ForegroundColor Green


#------------------------------------
# Steps to Deploy
#------------------------------------

# 1. Remove git repository
# rm -Recurse -Force .git OR Remove-Item -Recurse -Force .git

# 2. initialize with Main Branch
# git init -b main

# 3. Show current branch
# git branch --show-current  # Should output "main"

# 4. Set remote URL 
# git remote add origin https://github.com/Abdulraqib20/vehicle-detection-and-color-analyzer.git

# 5. Run the below in PowerShell:
# ./deploy.ps1 -RepoName "your-repo-name" -Description "Project purpose"

# ./deploy.ps1 -RepoName "vehicle-detection-and-color-analyzer" -Description "This application uses AI to detect vehicles in images, identify vehicle types & predict vehicle colors"

# Credential Caching (One-Time Setup):
# git config --global credential.helper wincred


