# Data Sharing Guide

The S&P 500 historical data files are **too large for GitHub** (353 MB+). Instead of committing them to the repository, follow these instructions to download or share the data.

---

## 🚫 Why Data Files Are Not in Git

GitHub has file size limits:
- ⚠️ Files > 50 MB trigger warnings
- ❌ Files > 100 MB are rejected
- 📦 Total repo should be < 1 GB

Our data files:
- `spx500_usd_m1_5years.ndjson` - **353 MB** ❌
- `spx500_usd_m1_2years.ndjson` - **139 MB** ❌
- `usd_sgd_hourly_2025.ndjson` - **1.9 MB** ✅

---

## ✅ Solution: Download Data Using Scripts

### Option 1: Download Directly from OANDA (Recommended)

**Anyone can reproduce the exact same data** using the download scripts included in this repo:

```bash
# Clone the repository
git clone https://github.com/your-username/fx-ml-pipeline.git
cd fx-ml-pipeline

# Set up environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Add your OANDA credentials to .env
cat > .env << EOF
OANDA_TOKEN=your_token_here
OANDA_ENV=practice
EOF

# Download 5 years of S&P 500 data (takes ~5 minutes)
python src/download_sp500_historical.py --years 5 --granularity M1

# Or use the convenience script
./scripts/download_sp500_data.sh --years 5
```

**Output:**
- File: `data/bronze/prices/spx500_usd_m1_5years.ndjson`
- Size: 353 MB
- Candles: 1,705,276
- Time: ~5 minutes

---

## 📤 Option 2: Share via Cloud Storage

If you need to share the pre-downloaded data files with team members, use these options:

### A. Google Drive / Dropbox

1. **Upload the data file:**
   ```bash
   # Compress first to reduce size
   gzip -c data/bronze/prices/spx500_usd_m1_5years.ndjson > spx500_5years.ndjson.gz
   # This reduces size to ~70-80 MB
   ```

2. **Upload to Google Drive/Dropbox**

3. **Share the link** in your team documentation

4. **Team members download:**
   ```bash
   # After downloading spx500_5years.ndjson.gz
   gunzip spx500_5years.ndjson.gz -c > data/bronze/prices/spx500_usd_m1_5years.ndjson
   ```

### B. AWS S3 / Google Cloud Storage

```bash
# Upload to S3
aws s3 cp data/bronze/prices/spx500_usd_m1_5years.ndjson \
  s3://your-bucket/data/spx500_usd_m1_5years.ndjson

# Team members download
aws s3 cp s3://your-bucket/data/spx500_usd_m1_5years.ndjson \
  data/bronze/prices/spx500_usd_m1_5years.ndjson
```

### C. Git LFS (Git Large File Storage)

If you want to use Git but with large files:

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "data/bronze/prices/*.ndjson"
git add .gitattributes

# Now you can commit large files
git add data/bronze/prices/spx500_usd_m1_5years.ndjson
git commit -m "Add data via Git LFS"
git push
```

**Note:** Git LFS has storage limits on free plans (1 GB storage, 1 GB bandwidth/month).

---

## 📋 Data File Inventory

| File | Size | Candles | Date Range | How to Get |
|------|------|---------|------------|------------|
| `spx500_usd_m1_5years.ndjson` | 353 MB | 1.7M | Oct 2020 - Oct 2025 | Download script |
| `spx500_usd_m1_2years.ndjson` | 139 MB | 672K | Oct 2023 - Oct 2025 | Download script |
| `usd_sgd_hourly_2025.ndjson` | 1.9 MB | ~8K | Jan - Oct 2025 | Live collector |

---

## 🔧 Quick Setup for New Team Members

```bash
# 1. Clone repo
git clone https://github.com/your-username/fx-ml-pipeline.git
cd fx-ml-pipeline

# 2. Setup Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure OANDA (get token from https://www.oanda.com/)
echo "OANDA_TOKEN=your_token_here" > .env
echo "OANDA_ENV=practice" >> .env

# 4. Download data (choose one)
# Option A: 5 years (recommended)
python src/download_sp500_historical.py --years 5 --granularity M1

# Option B: 2 years (faster)
python src/download_sp500_historical.py --years 2 --granularity M1

# 5. Validate
python src/download_sp500_historical.py --validate-only

# 6. You're ready to train!
python inspect_sp500_data.py
```

---

## 📦 Alternative: Compressed Data Archive

You can create a compressed archive for sharing:

```bash
# Create compressed archive
tar -czf sp500_data.tar.gz data/bronze/prices/*.ndjson

# Share sp500_data.tar.gz (much smaller)

# Team members extract
tar -xzf sp500_data.tar.gz
```

**Compression ratio:** ~5-6× smaller (353 MB → ~60-70 MB)

---

## 🎓 Best Practices

### For Repository Maintainers:

✅ **DO:**
- Keep download scripts in Git
- Keep documentation in Git
- Keep small sample data (< 10 MB)
- Document how to get full data
- Use `.gitignore` for large files

❌ **DON'T:**
- Commit files > 50 MB to Git
- Use Git for frequently changing data
- Commit binary files without Git LFS

### For Data Users:

✅ **DO:**
- Download data fresh when possible
- Validate checksums if provided
- Keep data files in `.gitignore`
- Document data sources

❌ **DON'T:**
- Expect data in Git repo
- Commit data to personal branches
- Share credentials in commits

---

## 🔐 Security Note

**Never commit your OANDA API token!**

The `.env` file is in `.gitignore` to prevent this. Always double-check before pushing:

```bash
# Check what will be committed
git status
git diff --staged

# If you accidentally added .env
git reset HEAD .env
git checkout .env
```

---

## 📊 Data Verification

After downloading data, verify it:

```bash
# Check file size
ls -lh data/bronze/prices/spx500_usd_m1_5years.ndjson

# Count candles
wc -l data/bronze/prices/spx500_usd_m1_5years.ndjson
# Expected: 1705276

# Run validation
python src/download_sp500_historical.py --validate-only

# Analyze data
python inspect_sp500_data.py
```

**Expected checksums** (for verification):
```bash
# SHA256 (may vary slightly by download date)
shasum -a 256 data/bronze/prices/spx500_usd_m1_5years.ndjson
```

---

## 💡 FAQ

**Q: Why not use Git LFS?**
A: Git LFS is great but has limits on free tiers (1 GB storage, 1 GB bandwidth/month). With our data size, you'd hit limits quickly with multiple team members.

**Q: Can I commit sample data?**
A: Yes! Small samples (< 10 MB) are fine. For example, first 10,000 candles for testing.

**Q: What if OANDA data changes?**
A: Historical data should be stable, but if needed, we can version data files with timestamps: `spx500_usd_m1_5years_20251012.ndjson`

**Q: How to automate downloads in CI/CD?**
A: Store OANDA token in CI secrets and run download script as part of pipeline setup.

---

## 📞 Support

If you have issues downloading data:

1. **Check OANDA credentials** in `.env`
2. **Verify API access** - test with small download
3. **See documentation:** [README_SP500_DOWNLOAD.md](src/README_SP500_DOWNLOAD.md)
4. **Open an issue** with error details

---

**Remember:** The download scripts are designed to be reproducible - anyone with OANDA access can generate the exact same dataset! 🚀
