# GitHub Pages Deployment Guide

## Setup Instructions

### Step 1: Ensure You Have a GitHub Repository

If you haven't created a GitHub repo yet:

```bash
cd /Users/prakashsaini/Desktop/AI-for-Interview

# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Add AI interview preparation guides"

# Add GitHub remote (replace with your actual repo URL)
git remote add origin https://github.com/YOUR_USERNAME/AI-for-Interview.git

# Push to GitHub
git push -u origin main
```

### Step 2: Enable GitHub Pages

1. Go to your repository on GitHub
2. Navigate to **Settings** → **Pages**
3. Under "Build and deployment":
   - **Source:** Select "GitHub Actions"
   - This will automatically use the workflow file we created
4. Save

### Step 3: Configure Repository Settings

1. In **Settings** → **Pages**:
   - The deployment will happen automatically on every push to `main`
   - GitHub Actions will build and deploy the Jekyll site
2. Once deployed, your site will be available at:
   ```
   https://YOUR_USERNAME.github.io/AI-for-Interview
   ```

### Step 4: Verify Deployment

After pushing:
1. Go to your repository
2. Click on the **Actions** tab
3. You should see the Jekyll build workflow running
4. Once it completes (green checkmark), your site is live!
5. Visit `https://YOUR_USERNAME.github.io/AI-for-Interview` to see your site

---

## What Was Set Up

✅ **_config.yml** - Jekyll configuration file
✅ **index.md** - Home page with links to all guides
✅ **Gemfile** - Ruby dependencies for Jekyll
✅ **.gitignore** - Ignore unnecessary files
✅ **.github/workflows/jekyll.yml** - GitHub Actions workflow for automatic deployment

---

## File Structure

```
AI-for-Interview/
├── .github/
│   └── workflows/
│       └── jekyll.yml          # GitHub Actions workflow
├── README.md                    # AI/ML Fundamentals
├── GenAI.md                    # Generative AI Guide
├── AgenticAI.md                # Agentic AI Guide
├── Prompt Engineering.md       # Prompt Engineering Guide
├── index.md                    # Home page (NEW)
├── _config.yml                 # Jekyll config (NEW)
├── Gemfile                     # Ruby dependencies (NEW)
└── .gitignore                  # Ignore files (NEW)
```

---

## Local Development (Optional)

To test the site locally before pushing:

### Install Dependencies
```bash
cd /Users/prakashsaini/Desktop/AI-for-Interview
bundle install
```

### Run Jekyll Locally
```bash
bundle exec jekyll serve
```

Then visit `http://localhost:4000/AI-for-Interview` in your browser.

---

## Customization

### Change Theme
In `_config.yml`, modify the `theme` line:
```yaml
# Available themes:
theme: jekyll-theme-minimal          # Current (minimal)
theme: jekyll-theme-dinky
theme: jekyll-theme-slate
theme: jekyll-theme-cayman
theme: jekyll-theme-architect
theme: jekyll-theme-merlot
theme: jekyll-theme-leap-day
theme: jekyll-theme-midnight
theme: jekyll-theme-primer
theme: jekyll-theme-time-machine
```

### Update Site Title/Description
In `_config.yml`:
```yaml
title: "Your Title"
description: "Your Description"
author: "Your Name"
```

### Update Base URL
If your repository name changes:
```yaml
baseurl: "/your-repo-name"
```

---

## Next Steps

1. **Push all changes to GitHub:**
   ```bash
   git add .
   git commit -m "Add GitHub Pages deployment configuration"
   git push
   ```

2. **Monitor the deployment:**
   - Go to Actions tab in GitHub
   - Wait for the workflow to complete

3. **Access your site:**
   - Once deployment is complete, visit:
   - `https://YOUR_USERNAME.github.io/AI-for-Interview`

4. **Share the link:**
   - Your AI interview prep guide is now live for everyone!

---

## Troubleshooting

### Build Fails
- Check the Actions tab for error messages
- Ensure all markdown files have proper front matter
- Verify `_config.yml` syntax is valid (YAML)

### Site Not Showing Up
- Enable GitHub Pages in repository Settings
- Ensure workflow has run successfully
- Check that source is set to "GitHub Actions"
- Give it a few minutes to deploy

### Links Not Working
- Update baseurl in `_config.yml` to match your repo name
- Ensure markdown filenames don't have spaces (use %20 in links or hyphens)

---

## Additional Resources

- [Jekyll Documentation](https://jekyllrb.com/docs/)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

---

**Your AI Interview Preparation Guide is now ready for the world! 🚀**
