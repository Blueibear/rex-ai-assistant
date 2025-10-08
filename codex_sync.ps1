Write-Host "Starting Codex Git sync..."

# Check if inside a Git repo
try {
    git rev-parse --is-inside-work-tree > $null
} catch {
    Write-Host "Not a Git repo. Exiting."
    exit 1
}

# Get current branch
$branch = git rev-parse --abbrev-ref HEAD
Write-Host "Current branch: $branch"

# Check if remote 'origin' exists
$originExists = $true
try {
    git remote get-url origin > $null 2>&1
} catch {
    $originExists = $false
}

if (-not $originExists) {
    Write-Host "Adding remote 'origin'..."
    git remote add origin https://github.com/Blueibear/rex-ai-assistant.git
} else {
    Write-Host "Remote 'origin' already exists."
}

# Backup BEFORE modifying anything
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$backupFile = "backup-$timestamp.zip"
Compress-Archive -Path .\* -DestinationPath $backupFile
Write-Host "Project backed up as $backupFile"

# Fetch + set upstream
git fetch origin

try {
    git branch --set-upstream-to="origin/master" $branch
    Write-Host "Upstream set to origin/master."
} catch {
    Write-Host "Upstream already set or branch missing."
}

# Commit any local unstaged changes BEFORE pulling
if (git status --porcelain) {
    git add .
    git commit -m "Auto-sync: Local changes before pulling"
    Write-Host "Local changes committed."
} else {
    Write-Host "No local changes to commit."
}

# Pull + push
Write-Host "Pulling changes from origin..."
git pull origin master

Write-Host "Pushing changes to origin/$branch..."
git push origin $branch

Write-Host "Codex Git sync complete."

