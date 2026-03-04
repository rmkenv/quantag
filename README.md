# QuantAgri — Live Satellite Monitor

Automated daily pipeline pulling Sentinel-2 and Sentinel-1 data from
**Microsoft Planetary Computer** (free, no account needed) to keep
agricultural commodity spectral metrics live.

---

## Repo Structure

```
quantagri/
├── quantagri_spectral_velocity_pc.py   ← Sentinel-2 data layer
├── quantagri_commodity_config.py       ← 12 growing season configs
├── quantagri_metrics_engine_pc.py      ← NDVI/LSWI/velocity/SAR metrics
├── quantagri_batch_runner_pc.py        ← Historical batch runner
├── quantagri_live_monitor.py           ← ← LIVE MONITOR (this is the new one)
├── quantagri_backtest_aggregator.py    ← Excel workbook builder
└── official_yields.csv                 ← USDA/CONAB yield forecasts

.github/
└── workflows/
    └── quantagri_daily.yml             ← Runs every day at 06:00 UTC

quantagri_live/                         ← Created automatically by the monitor
├── quantagri_live_results.csv          ← Rolling live results (appended daily)
├── quantagri_live_alerts.csv           ← Alert rows only
└── quantagri_monitor.log               ← Run log
```

---

## One-Time Setup (do this once)

### Step 1 — Create the GitHub repo

1. Go to **github.com → New repository**
2. Name it `quantagri` (or anything you like)
3. Set to **Private** (your yield data and signals are proprietary)
4. Do NOT initialise with README — you'll push your files directly

### Step 2 — Upload your files

On your machine, open Terminal and run:

```bash
# Clone the empty repo
git clone https://github.com/YOUR_USERNAME/quantagri.git
cd quantagri

# Copy all the QuantAgri .py files into the quantagri/ subfolder
mkdir quantagri
cp /path/to/your/files/*.py quantagri/
cp /path/to/your/files/official_yields.csv quantagri/

# Add the workflow and config files
mkdir -p .github/workflows
cp /path/to/quantagri_daily.yml .github/workflows/

cp requirements.txt .
cp .gitignore .

# Create an empty live results folder so git tracks it
mkdir quantagri_live
touch quantagri_live/.gitkeep

# Push everything
git add .
git commit -m "Initial commit — QuantAgri live monitor setup"
git push origin main
```

### Step 3 — Enable GitHub Actions

1. Go to your repo on GitHub
2. Click the **Actions** tab
3. If prompted "Workflows aren't running", click **"I understand my workflows, go ahead and enable them"**
4. You'll see **QuantAgri Daily Monitor** listed

### Step 4 — Give Actions permission to push commits

The workflow commits updated CSV results back to your repo. You need to allow this:

1. Go to **Settings → Actions → General**
2. Scroll to **Workflow permissions**
3. Select **"Read and write permissions"**
4. Click **Save**

### Step 5 — Test it manually

Don't wait for 06:00 UTC — trigger it now:

1. Click **Actions → QuantAgri Daily Monitor**
2. Click **"Run workflow"** (top right)
3. Select mode: `daily`, leave commodities blank
4. Click **"Run workflow"** (green button)
5. Watch the run — should take 10–20 minutes

---

## What Happens Each Day

```
06:00 UTC
    ↓
GitHub spins up Ubuntu runner (free)
    ↓
Installs Python dependencies (~2 min)
    ↓
Restores quantagri_live/ from cache
    ↓
Checks which of the 12 seasons are active today
    ↓
For each active season:
    Fetches new Sentinel-2 scenes from Planetary Computer
    Recomputes season-to-date NDVI/LSWI/velocity composites
    Computes yield surprise vs USDA/CONAB
    Appends row to quantagri_live_results.csv
    ↓
Commits updated CSV back to repo
    ↓
Uploads CSV as downloadable artifact
    ↓
If any alert thresholds breached → opens a GitHub Issue
```

---

## Viewing Results

**Option A — GitHub UI (easiest)**
Go to your repo → `quantagri_live/quantagri_live_results.csv` → click the file
→ GitHub renders it as a table.

**Option B — Raw CSV URL**
```
https://raw.githubusercontent.com/YOUR_USERNAME/quantagri/main/quantagri_live/quantagri_live_results.csv
```
Load this URL directly into Excel, Google Sheets, or pandas.

**Option C — Download artifact**
Actions → your run → scroll to Artifacts → download zip.

**Option D — Clone and read locally**
```bash
git pull
python -c "
import pandas as pd
df = pd.read_csv('quantagri_live/quantagri_live_results.csv')
print(df.tail(10).to_string())
"
```

---

## Alerts

When a signal exceeds the thresholds set in `quantagri_live_monitor.py`:

```python
SURPRISE_ALERT_BPS = 1.5   # bpa absolute value
VELOCITY_ALERT     = 0.015  # dNDVI/day
```

GitHub will automatically **open an Issue** in your repo with the alert details.
You'll receive an email notification (GitHub's default behaviour for Issues).

To change thresholds, edit those two lines in `quantagri/quantagri_live_monitor.py`,
commit, and push.

---

## Updating official_yields.csv

When USDA/CONAB releases new forecasts (WASDE, etc.):

1. Edit `quantagri/official_yields.csv` locally
2. `git add quantagri/official_yields.csv`
3. `git commit -m "Update yields — WASDE March 2026"`
4. `git push`

The next daily run will automatically use the updated values for yield surprise calibration.

---

## GitHub Actions Free Tier Limits

| Item | Free limit | QuantAgri usage |
|------|-----------|-----------------|
| Minutes/month | 2,000 | ~20 min/day × 30 = ~600 min ✅ |
| Storage | 500MB | CSVs are tiny (~50KB/month) ✅ |
| Concurrent jobs | 20 | 1 ✅ |

You are very unlikely to hit any free tier limits.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Workflow not running at scheduled time | GitHub can delay scheduled runs by up to 30 min. Also check Actions is enabled. |
| `Permission denied` on git push step | Settings → Actions → General → Workflow permissions → Read and write |
| `No active seasons` in log | Check today's date vs season windows in `quantagri_commodity_config.py` |
| `No S2 scenes found` | Cloud cover >80% for that region. Try increasing `max_cloud_pct` in the monitor call |
| Run takes >60 min and gets killed | Reduce to `--commodities corn soy` to limit scope, or split into two workflows |
| Alert issues not being created | Check Actions has Issues write permission: Settings → Actions → General |
