# QuantAgri — Live Satellite Monitor

Automated daily pipeline pulling Sentinel-2 data from
**Microsoft Planetary Computer** (free, no account needed) to keep
agricultural commodity spectral metrics live.

---

## Repo Structure

```
quantagri/
├── quantagri_spectral_velocity_pc.py   ← Sentinel-2 STAC query, cloud mask, NDVI/LSWI composites
├── quantagri_commodity_config.py       ← 12 growing season configs
├── quantagri_metrics_engine_pc.py      ← NDVI/LSWI/velocity/SAR metrics engine
├── quantagri_batch_runner_pc.py        ← Historical batch runner
├── quantagri_live_monitor.py           ← Daily live monitor
├── quantagri_historical_monthly.py     ← 10-year historical monthly aggregator
└── quantagri_backtest_aggregator.py    ← Excel workbook builder

.github/
└── workflows/
    ├── quantagri_daily.yml             ← Runs every day at 06:00 UTC
    └── quantagri_historical.yml        ← One-time 10-year historical run (manual trigger)

official_yields.csv                     ← USDA/CONAB yield forecasts (repo root)
requirements.txt                        ← Python dependencies (repo root)

quantagri_live/                         ← Created automatically by the daily monitor
├── quantagri_live_results.csv          ← Rolling live results (one row per region per day)
├── quantagri_live_alerts.csv           ← Alert rows only
└── quantagri_monitor.log               ← Run log

quantagri_historical/                   ← Created by the historical workflow
├── quantagri_monthly_soy_mato_grosso_br.csv
├── quantagri_monthly_wheat_kansas_us.csv
├── quantagri_monthly_wheat_rostov_ru.csv
├── quantagri_monthly_corn_iowa_us.csv
├── quantagri_monthly_corn_illinois_us.csv
├── quantagri_monthly_cotton_*.csv
└── quantagri_monthly_ALL.csv           ← All commodities combined
```

---

## Active Seasons by Month

| Commodity | Region | Season Window |
|-----------|--------|---------------|
| Soy | Mato Grosso, BR | Oct → Mar |
| Wheat | Kansas, US | Mar → Jun |
| Wheat | Rostov, RU | Mar → Jul |
| Corn | Iowa, US | May → Oct |
| Corn | Illinois, US | May → Oct |
| Cotton | South Texas, US | Apr → Oct |
| Cotton | North Texas, US | Apr → Oct |
| Cotton | Xinjiang, CN | Apr → Oct |

The daily monitor automatically detects which seasons are active and only runs those.

---

## One-Time Setup

### Step 1 — Upload files to the repo

Your repo layout must be:
```
quantagri/         ← all .py files go here
official_yields.csv   ← repo root
requirements.txt      ← repo root
.github/workflows/quantagri_daily.yml
.github/workflows/quantagri_historical.yml
```

Upload the `.py` files via GitHub UI:
1. Go to `https://github.com/YOUR_USERNAME/quantag/tree/main/quantagri`
2. Click **Add file → Upload files**
3. Upload all `.py` files individually (not inside a folder)

### Step 2 — Give Actions permission to push commits

1. Go to **Settings → Actions → General**
2. Scroll to **Workflow permissions**
3. Select **"Read and write permissions"**
4. Click **Save**

### Step 3 — Test the daily monitor

1. Click **Actions → QuantAgri Daily Monitor**
2. Click **"Run workflow"** → mode: `daily` → **Run workflow**
3. You should see 5 parallel jobs: soy, wheat, corn, cotton, commit-results
4. Runtime: ~10–30 min depending on which seasons are active

---

## Daily Workflow — What Happens at 06:00 UTC

```
06:00 UTC
    ↓
5 commodity jobs spin up in parallel (soy, wheat, corn, cotton)
    ↓
Each job:
    Installs Python dependencies (~2 min)
    Restores quantagri_live/ from cache
    Checks which seasons are active today
    Fetches new Sentinel-2 scenes (re-signs SAS tokens to avoid expiry)
    Builds season-to-date NDVI/LSWI/velocity composites
    Computes yield surprise vs USDA/CONAB forecast
    Writes row to quantagri_live_results.csv
    ↓
commit-results job:
    Downloads all 4 commodity artifacts
    Merges CSVs — deduplicates by commodity + region + date (keeps latest)
    Commits quantagri_live/ back to repo
    Uploads merged artifact (30-day retention)
    ↓
If alert thresholds breached → opens a GitHub Issue (email notification)
```

---

## Resolution Settings

Large ROIs need coarser resolution to fit in the GitHub Actions runner (7GB RAM).
These are set automatically — no manual configuration needed.

| Commodity | Resolution | Reason |
|-----------|-----------|--------|
| Soy (Mato Grosso) | 500m | ~1.1M km² ROI |
| Sugar | 300m | Two large tropical regions |
| Corn, Wheat, Cotton | 200m | Medium ROIs |

---

## Live Results — What the CSV Contains

One row per active region per day. Key columns:

| Column | Example | Notes |
|--------|---------|-------|
| `as_of_date` | 2026-03-06 | Date of latest satellite data |
| `current_ndvi` | 0.407 | Season-to-date latest composite |
| `current_ndvi_velocity` | -0.0086 | dNDVI/day — rate of change |
| `peak_ndvi` | 0.497 | Highest NDVI seen this season |
| `peak_ndvi_date` | 2026-01-05 | Date peak was reached |
| `tercile_mean_early/mid/late` | 0.372 / 0.480 / 0.454 | Season thirds — fills in over time |
| `yield_surprise` | +1.2 | bpa vs official forecast |
| `surprise_pct` | +2.8% | Relative surprise |
| `calibration_r2` | 0.78 | How well RS predicts yield historically |

**Note:** `tercile_mean_*`, `velocity_std`, `yield_surprise` and `calibration_r2` are blank
early in the season — they require multiple composites to compute and fill in over weeks.

---

## Viewing Results

**Option A — GitHub UI**
Go to `quantagri_live/quantagri_live_results.csv` in the repo → GitHub renders as a table.

**Option B — Raw CSV (paste into Excel or Google Sheets)**
```
https://raw.githubusercontent.com/YOUR_USERNAME/quantag/main/quantagri_live/quantagri_live_results.csv
```

**Option C — Download artifact**
Actions → latest run → Artifacts → `quantagri-live-merged-N`

---

## Alerts

Thresholds in `quantagri/quantagri_live_monitor.py`:
```python
SURPRISE_ALERT_BPS = 1.5   # bpa absolute value
VELOCITY_ALERT     = 0.015  # dNDVI/day
```

When breached, GitHub automatically opens an Issue → you get an email.
To change thresholds, edit those two lines, commit, push.

---

## Historical 10-Year Run

To build a full monthly NDVI/LSWI history (2016–2025):

1. **Actions → QuantAgri Historical Monthly → Run workflow**
2. Pick a commodity (start with `wheat` — fastest)
3. Leave start/end year as 2016/2025
4. Runtime: ~30–90 min per commodity
5. Results committed to `quantagri_historical/` in the repo

Run one commodity at a time. The job is resumable — if it times out,
re-run it and it skips already-completed season-years.

**Recommended order:** wheat → corn → cotton → soy → sugar

Output per commodity: one CSV with monthly `ndvi_mean`, `ndvi_max`, `lswi_mean`,
`velocity_mean` per region per year — 10 years × growing season months per row.

---

## Updating official_yields.csv

When USDA/CONAB releases new forecasts (WASDE etc.):

1. Edit `official_yields.csv` in the repo root
2. Commit with message: `"Update yields — WASDE March 2026"`

The next daily run automatically uses the updated values.
Add rows for the current season year (e.g. 2026) so yield_surprise populates.

---

## GitHub Actions Free Tier

| Item | Free limit | QuantAgri usage |
|------|-----------|-----------------|
| Minutes/month | 2,000 | ~30 min/day × 30 = ~900 min ✅ |
| Storage | 500MB | CSVs ~50KB/month ✅ |
| Concurrent jobs | 20 | 5 parallel ✅ |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Permission denied` on git push | Settings → Actions → General → Workflow permissions → Read and write |
| `No active seasons` in log | Normal outside growing season windows. Corn/cotton start April/May. |
| `No S2 scenes found` | Cloud cover >80% that day. Monitor will pick up next clear day automatically. |
| Job killed after ~3 min (exit 143) | Out of memory. Resolution override in `COMMODITY_RESOLUTION` dict in `quantagri_historical_monthly.py`. |
| `RasterioIOError: not a supported format` | SAS token expired — fixed by `pc.sign()` re-sign in `quantagri_spectral_velocity_pc.py`. |
| Duplicate rows in CSV | Fixed — merge step deduplicates by commodity + region + date. |
| Workflow not running at 06:00 UTC | GitHub delays scheduled runs up to 30 min under load. Check Actions tab. |
| Historical job times out | Re-run — it resumes from where it left off (skips completed season-years). |
| `Failed to queue workflow run` | YAML syntax error — validate with `python3 -c "import yaml; yaml.safe_load(open('file.yml'))"` |
