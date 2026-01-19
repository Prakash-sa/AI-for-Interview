# 🚀 Quick Start: Deploy to GitHub Pages

## Fastest Way (5 minutes)

### 1. Create a GitHub Repository
- Go to [github.com/new](https://github.com/new)
- Name: `AI-for-Interview`
- Keep default settings
- Click "Create repository"

### 2. Configure Git Remote
```bash
cd /Users/prakashsaini/Desktop/AI-for-Interview

# Set your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/AI-for-Interview.git

# Or if already set, verify:
git remote -v
```

### 3. Push to GitHub
```bash
# Make sure all files are committed
git add .
git commit -m "Initial commit: AI Interview Prep Guide"

# Push to GitHub
git push -u origin main
```

### 4. Enable GitHub Pages
1. Go to your repo: `github.com/YOUR_USERNAME/AI-for-Interview`
2. Click **Settings** (top right)
3. Click **Pages** (left sidebar)
4. Under "Build and deployment":
   - Source: **GitHub Actions** ✅
5. Done! GitHub will automatically build and deploy

### 5. View Your Site
After ~1-2 minutes, your site will be live at:
```
https://YOUR_USERNAME.github.io/AI-for-Interview
```

---

## What You Just Deployed ✨

Your AI interview prep guide is now publicly accessible with:
- ✅ Beautiful markdown rendering
- ✅ Automatic table of contents
- ✅ Full-text search (if theme supports)
- ✅ Mobile responsive design
- ✅ Free hosting on GitHub Pages
- ✅ Automatic updates (push to GitHub = auto-deploy)

---

## Verify It Worked

### Check GitHub Actions
1. Go to your repo
2. Click **Actions** tab
3. You should see a workflow running
4. Once it has a green ✅, your site is live!

### Check Your Site
Visit: `https://YOUR_USERNAME.github.io/AI-for-Interview`

You should see your index page with links to all 4 guides!

---

## Update Your Site

To update your content:
```bash
# Edit any .md file locally
# e.g., Edit README.md, GenAI.md, etc.

# Commit and push
git add .
git commit -m "Update content"
git push

# GitHub automatically rebuilds your site! 🎉
```

---

## Troubleshooting

**My site isn't showing up?**
- Check Actions tab for build errors
- Verify Pages is set to "GitHub Actions"
- Wait 2-3 minutes for deployment
- Clear browser cache

**Links not working?**
- The site might be in a subdirectory
- Try: `https://YOUR_USERNAME.github.io/AI-for-Interview/`

**Still having issues?**
- See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed guide
- Check [Jekyll Docs](https://jekyllrb.com/)
- Check [GitHub Pages Docs](https://docs.github.com/en/pages)

---

## Next Steps

1. ✅ Push to GitHub (do this now!)
2. ✅ Enable GitHub Pages (takes 1 minute)
3. ✅ Share your link! (You've got an amazing guide)
4. 📝 Keep updating with new interview tips
5. 🎓 Use it to ace your AI interviews!

---

**You're all set! Your AI interview guide is now live on the internet! 🌟**
