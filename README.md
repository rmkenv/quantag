# QuantAgri — Live Satellite Monitor

Automated daily pipeline pulling Sentinel-2 data from
**Microsoft Planetary Computer** (free, no account needed) to keep
agricultural commodity spectral metrics live, with monthly statistical
analysis and automatic yield forecast updates.

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
├── quantagri_backtest_aggregator.py    ← Excel workbook builder
├── quantagri_yields_updater.py         ← Auto-updates official_yields.csv from USDA/CONAB APIs
└── quantagri_monthly_analysis.py       ← Monthly aggregation + statistical analysis

.github/
└── workflows/
    ├── quantagri_daily.yml             ← Runs every day at 06:00 UTC
    ├── quantagri_historical.yml        ← One-time 10-year historical run (manual trigger)
    └── quantagri_analysis.yml          ← Runs 1st of each month at 08:00 UTC

official_yields.csv                     ← USDA/CONAB yield forecasts (repo root)
requirements.txt                        ← Python dependencies (repo root)

quantagri_live/                         ← Created by daily monitor
├── quantagri_live_results.csv          ← Rolling live results (one row per region per day)
├── quantagri_live_alerts.csv           ← Alert rows only
└── quantagri_monitor.log               ← Run log

quantagri_historical/                   ← Created by historical workflow
├── quantagri_monthly_soy_mato_grosso_br.csv
├── quantagri_monthly_wheat_kansas_us.csv
├── quantagri_monthly_wheat_rostov_ru.csv
├── quantagri_monthly_corn_iowa_us.csv
├── quantagri_monthly_corn_illinois_us.csv
├── quantagri_monthly_cotton_*.csv
└── quantagri_monthly_ALL.csv           ← All commodities combined

quantagri_analysis/                     ← Created by monthly analysis workflow
├── quantagri_monthly_summary.csv       ← Daily results aggregated to monthly
├── quantagri_anomaly_report.csv        ← Rows flagged as anomalous (|z| >= 1.5σ)
├── quantagri_correlations.csv          ← R² by month for each commodity/region
└── quantagri_stats_report.txt          ← Human-readable summary report
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
.github/workflows/quantagri_analysis.yml
```

Upload `.py` files via GitHub UI:
1. Go to `https://github.com/rmkenv/quantag/tree/main/quantagri`
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
3. You will see 5 parallel jobs: soy, wheat, corn, cotton, commit-results
4. Runtime: ~10–30 min depending on which seasons are active

---

## Workflow 1 — Daily Monitor (06:00 UTC)

```
06:00 UTC
    ↓
4 commodity jobs spin up in parallel (soy, wheat, corn, cotton)
    ↓
Each job:
    Installs Python dependencies (~2 min)
    Restores quantagri_live/ from cache
    Checks which seasons are active today
    Fetches new Sentinel-2 scenes (re-signs SAS tokens to prevent expiry)
    Builds season-to-date NDVI/LSWI/velocity composites
    Computes yield surprise vs USDA/CONAB forecast
    Writes row to quantagri_live_results.csv
    ↓
commit-results job:
    Downloads all 4 commodity artifacts
    Merges and deduplicates CSVs (by commodity + region + date, keeps latest)
    Commits quantagri_live/ back to repo
    Uploads merged artifact (30-day retention)
    ↓
If alert thresholds breached → opens GitHub Issue (triggers email)
```

---

## Workflow 2 — Historical 10-Year Run (manual)

Builds a full monthly NDVI/LSWI history for 2016–2025.
Run once per commodity — resumable if it times out.

1. **Actions → QuantAgri Historical Monthly → Run workflow**
2. Pick a commodity (start with `wheat` — fastest)
3. Leave start/end year as 2016/2025
4. Runtime: ~30–90 min per commodity

**Recommended order:** wheat → corn → cotton → soy → sugar

Output: monthly `ndvi_mean`, `ndvi_max`, `lswi_mean`, `velocity_mean`
per region per year — 10 years × growing season months.

This data powers the z-score and R² calculations in the analysis workflow.
Run the historical workflow before expecting those columns to populate.

---

## Workflow 3 — Monthly Analysis (1st of each month, 08:00 UTC)

Runs automatically on the 1st of every month. Can also be triggered manually.

**Step 1 — Yield updater**
Hits USDA PSD API and CONAB API, updates `official_yields.csv` automatically.
Regions without a free API (Xinjiang cotton, Indian sugar) are flagged for manual update.

**Step 2 — Monthly aggregation**
Groups daily live results by commodity + region + year + month:
- End-of-month NDVI/LSWI/velocity
- Intra-month mean, std, min, max
- Season context (tercile means, peak NDVI)

**Step 3 — Statistical analysis**

| Output | What it shows |
|--------|--------------|
| Z-score | Is this month's NDVI high/low vs same month historically |
| Percentile rank | Where does this month sit in the 10-year distribution |
| R² by month | Which month of the season best predicts final yield |
| Mann-Kendall test | Is NDVI velocity trending up or down this season |
| Anomaly flags | Anything beyond ±1.5σ flagged with severity (MODERATE/SIGNIFICANT/EXTREME) |

**Sample stats report output:**
```
[3] NDVI → YIELD R² BY MONTH

  wheat / kansas_us
  Month    R²       r       p    n   OLS slope
  ----------------------------------------------------
  Mar    0.412   0.642   0.045   9     28.4141*
  Apr    0.631   0.794   0.011   9     41.2203* ← BEST
  May    0.589   0.767   0.016   9     38.8901*
  Jun    0.521   0.722   0.028   9     34.1122*
```
The `← BEST` month is your highest-signal timing window.
The OLS slope = bushels/acre per 0.1 NDVI unit change.

---

## Resolution Settings

Set automatically per commodity — no manual configuration needed.

| Commodity | Resolution | Reason |
|-----------|-----------|--------|
| Soy (Mato Grosso) | 500m | ~1.1M km² ROI, OOMs at finer resolution |
| Sugar | 300m | Two large tropical regions |
| Corn, Wheat, Cotton | 200m | Medium ROIs |

---

## Live Results — Column Reference

One row per active region per day in `quantagri_live_results.csv`:

| Column | Example | Notes |
|--------|---------|-------|
| `as_of_date` | 2026-03-06 | Date of latest satellite data |
| `current_ndvi` | 0.407 | Latest composite value |
| `current_ndvi_velocity` | -0.0086 | dNDVI/day — rate of change |
| `peak_ndvi` | 0.497 | Season high |
| `peak_ndvi_date` | 2026-01-05 | Date peak was reached |
| `tercile_mean_early/mid/late` | 0.372 / 0.480 / 0.454 | Season thirds |
| `yield_surprise` | +1.2 | bpa vs official forecast |
| `surprise_pct` | +2.8% | Relative surprise |
| `calibration_r2` | 0.78 | Historical NDVI→yield fit |

**Note:** `tercile_mean_*`, `velocity_std`, `yield_surprise`, `calibration_r2`
are blank early in the season — they need multiple composites to compute
and fill in naturally over 3–4 weeks.

---

## Viewing Results

**Option A — GitHub UI**
`quantagri_live/quantagri_live_results.csv` → GitHub renders as a table.

**Option B — Raw CSV (paste into Excel or Google Sheets)**
```
https://raw.githubusercontent.com/rmkenv/quantag/main/quantagri_live/quantagri_live_results.csv
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
When breached, GitHub opens an Issue and you receive an email.

---

## Updating official_yields.csv

The monthly analysis workflow updates yields automatically from USDA/CONAB APIs.
For manual updates (e.g. after a major WASDE revision):

1. Edit `official_yields.csv` in the repo root
2. Commit: `"Update yields — WASDE March 2026"`

**Key WASDE dates to watch:**
| Month | What changes |
|-------|-------------|
| March | Winter wheat baseline (Kansas) |
| February/March | Brazilian soy CONAB monthly |
| May | First corn/cotton new-crop forecast |
| August | Most important — first field-survey corn/soy estimate |
| November | Near-final corn/soy/cotton |

Regions without API coverage (Xinjiang cotton, Indian sugar, Rostov wheat)
must be updated manually from IGC or local ministry reports.

---

## GitHub Actions Free Tier

| Item | Free limit | QuantAgri usage |
|------|-----------|-----------------|
| Minutes/month | 2,000 | ~30 min/day × 30 + ~30 min/month analysis = ~930 min ✅ |
| Storage | 500MB | CSVs ~50KB/month ✅ |
| Concurrent jobs | 20 | 5 parallel (daily) ✅ |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Permission denied` on git push | Settings → Actions → General → Workflow permissions → Read and write |
| `No active seasons` in log | Normal outside growing season. Corn/cotton start April/May. |
| `No S2 scenes found` | Cloud cover >80% that day — monitor picks up next clear day automatically |
| Job killed after ~3 min (exit 143) | Out of memory — resolution overrides in `COMMODITY_RESOLUTION` dict handle this automatically |
| `RasterioIOError: not a supported format` | SAS token expiry — fixed by `pc.sign()` re-sign in `quantagri_spectral_velocity_pc.py` |
| Duplicate rows in CSV | Fixed — merge step deduplicates by commodity + region + date |
| `Failed to queue workflow run` | YAML syntax error — validate with `python3 -c "import yaml; yaml.safe_load(open('file.yml'))"` |
| Historical job times out | Re-run — resumes from where it left off, skips completed season-years |
| Z-scores all blank in analysis | Run historical workflow first to build the baseline |
| `yield_surprise` blank | Add current season year rows to `official_yields.csv` with forecast values |
| Workflow not running at scheduled time | GitHub delays scheduled runs up to 30 min under load |
