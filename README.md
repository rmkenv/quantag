# QuantAgri — Live Satellite Monitor + ML Signal Engine

Automated daily pipeline pulling Sentinel-2 data from
**Microsoft Planetary Computer** (free, no account needed) to keep
agricultural commodity spectral metrics live, with monthly statistical
analysis, automatic yield forecast updates, and a machine learning layer
that generates LONG/SHORT/NEUTRAL trading signals from satellite data.

---

## Repo Structure

```
quantag/
├── .github/
│   └── workflows/
│       ├── quantagri_daily.yml          ← Runs every day at 06:00 UTC
│       ├── quantagri_historical.yml     ← One-time 10-year historical run (manual)
│       └── quantagri_analysis.yml       ← Runs 1st of each month at 08:00 UTC
│
├── quantagri/
│   ├── quantagri_spectral_velocity_pc.py   ← Sentinel-2 STAC query, cloud mask, NDVI/LSWI composites
│   ├── quantagri_commodity_config.py       ← 12 growing season configs
│   ├── quantagri_metrics_engine_pc.py      ← NDVI/LSWI/velocity/SAR metrics engine
│   ├── quantagri_batch_runner_pc.py        ← Historical batch runner
│   ├── quantagri_live_monitor.py           ← Daily live monitor
│   ├── quantagri_historical_monthly.py     ← 10-year historical monthly aggregator
│   ├── quantagri_backtest_aggregator.py    ← Excel workbook builder
│   ├── quantagri_yields_updater.py         ← Auto-updates official_yields.csv from USDA/CONAB APIs
│   ├── quantagri_monthly_analysis.py       ← Monthly aggregation + statistical analysis
│   │
│   └── ml/                                 ← ML signal engine
│       ├── __init__.py
│       ├── features.py                     ← Season-level feature engineering
│       ├── models.py                       ← LightGBM + Ridge yield prediction ensemble
│       ├── anomaly.py                      ← Isolation Forest anomaly detector
│       ├── phenology.py                    ← Crop stage / changepoint extraction
│       ├── signals.py                      ← Yield surprise classifier (LONG/SHORT/NEUTRAL)
│       ├── score.py                        ← Signal scorer — combines ML + z-scores into conviction score
│       └── train.py                        ← CLI trainer — run via workflow or manually
│
├── quantagri_live/                         ← Created by daily monitor
│   ├── quantagri_live_results.csv          ← Rolling live results (one row per region per day)
│   ├── quantagri_live_alerts.csv           ← Alert rows only
│   ├── quantagri_signal_scorecard.csv      ← Conviction scores per commodity/region
│   └── quantagri_monitor.log              ← Run log
│
├── quantagri_historical/                   ← Created by historical workflow
│   ├── quantagri_monthly_soy_mato_grosso_br.csv
│   ├── quantagri_monthly_wheat_kansas_us.csv
│   ├── quantagri_monthly_wheat_rostov_ru.csv
│   ├── quantagri_monthly_corn_iowa_us.csv
│   ├── quantagri_monthly_corn_illinois_us.csv
│   ├── quantagri_monthly_cotton_*.csv
│   └── quantagri_monthly_ALL.csv           ← All commodities combined
│
├── quantagri_analysis/                     ← Created by monthly analysis workflow
│   ├── quantagri_monthly_summary.csv       ← Daily results aggregated to monthly
│   ├── quantagri_anomaly_report.csv        ← Rows flagged as anomalous (|z| >= 1.5σ)
│   ├── quantagri_correlations.csv          ← R² by month for each commodity/region
│   └── quantagri_stats_report.txt          ← Human-readable summary report
│
├── ml_models/                              ← Persisted trained models (auto-updated daily)
│   └── .gitkeep
│
├── official_yields.csv                     ← USDA/CONAB yield forecasts
└── requirements.txt                        ← Python dependencies
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
quantagri/            ← all .py files go here (including ml/ subfolder)
official_yields.csv   ← repo root
requirements.txt      ← repo root
.github/workflows/quantagri_daily.yml
.github/workflows/quantagri_historical.yml
.github/workflows/quantagri_analysis.yml
```

Upload `.py` files via GitHub UI:
1. Go to `https://github.com/rmkenv/quantag/tree/main/quantagri`
2. Click **Add file → Upload files**
3. Upload all `.py` files. For the `ml/` subfolder, create it first via **Add file → Create new file**, type `ml/__init__.py`, then upload the rest.

### Step 2 — Give Actions permission to push commits

1. Go to **Settings → Actions → General**
2. Scroll to **Workflow permissions**
3. Select **"Read and write permissions"**
4. Click **Save**

### Step 3 — Run the historical baseline (do this first)

The ML models and z-score analysis need historical data to train on. Run the historical workflow before expecting ML signals to populate.

1. Click **Actions → QuantAgri Historical Monthly → Run workflow**
2. Pick `wheat` first (fastest), then repeat for `corn`, `cotton`, `soy`
3. Runtime: ~30–90 min per commodity
4. Recommended order: wheat → corn → cotton → soy

### Step 4 — Test the daily monitor

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
    Installs Python dependencies
    Restores quantagri_live/ from cache
    Checks which seasons are active today
    Fetches new Sentinel-2 scenes from Planetary Computer
    Builds season-to-date NDVI/LSWI/velocity composites
    Computes yield surprise vs USDA/CONAB forecast
    Writes row to quantagri_live_results.csv
    ↓
commit-results job:
    Downloads all 4 commodity artifacts
    Merges and deduplicates CSVs (by commodity + region + date, keeps latest)
    Installs ML dependencies (lightgbm, scikit-learn, shap, ruptures)
    Retrains ML models on updated data → saves to ml_models/
    Generates signal scorecard → saves to quantagri_live/quantagri_signal_scorecard.csv
    Commits quantagri_live/ and ml_models/ back to repo
    Uploads merged artifact (30-day retention)
    ↓
If strong signal (conviction ±3 or ±4) → opens GitHub Issue (triggers email)
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

Output: monthly `ndvi_mean`, `ndvi_max`, `lswi_mean`, `velocity_mean`
per region per year — 10 years × growing season months.

This data is required before the ML models and z-score columns will populate.

---

## Workflow 3 — Monthly Analysis (1st of each month, 08:00 UTC)

Runs automatically on the 1st of every month. Can also be triggered manually.

**Step 1 — Yield updater**
Hits USDA PSD API and CONAB API, updates `official_yields.csv` automatically.
Regions without a free API (Xinjiang cotton, Indian sugar, Rostov wheat) are flagged for manual update.

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

## ML Signal Engine

The `quantagri/ml/` layer runs automatically after each daily data merge and produces three types of output, combined into a single conviction score in `quantagri_signal_scorecard.csv`.

### Conviction Score

Each commodity/region gets a score from -4 to +4 built from four components:

| Component | Max contribution | Source |
|-----------|-----------------|--------|
| ML signal direction (LONG/SHORT) | ±2 | YieldSurpriseClassifier |
| ML confidence (high/medium) | ±1 | Classifier probability |
| Anomaly flag | +1 | IsolationForest score < -0.05 |
| Z-score alignment | ±1 | Historical monthly analysis |

| Score | Conviction | Action |
|-------|-----------|--------|
| ±4 | Maximum | Act with full size |
| ±3 | Strong | Act with normal size |
| ±2 | Moderate | Small position or wait |
| ±1 | Weak lean | Watch only |
| 0 | Neutral | Stay flat |

A GitHub Issue is opened (email sent) whenever any commodity hits ±3 or ±4.

### What the models tell you

**1. YieldModel → predicted yield**

A number in the crop's native unit (bu/ac for US corn/soy/wheat, bag/ha for Brazil soy, t/ha for Rostov wheat). Compare against the current USDA WASDE or CONAB forecast — the gap is your surprise estimate.

```
corn/iowa_us     → predicted: 174.1 bu/ac  |  USDA: 178.0  →  -3.9  bearish
soy/illinois_us  → predicted:  53.4 bu/ac  |  USDA:  52.0  →  +1.4  bullish
```

Uses a LightGBM + Ridge ensemble with time-series cross-validation. LightGBM gets 60% weight because it captures non-linear stress interactions (e.g. high NDVI but low LSWI = green but water-stressed) that Ridge misses.

**2. AnomalyDetector → anomaly score**

A score between roughly -0.3 and +0.2. Below -0.05 means the current NDVI/LSWI/velocity pattern is outside the historical envelope — something unusual is happening. Fires 4–6 weeks before yield impacts appear in official crop condition reports. Does not tell you direction — pair with the yield model for that.

```
wheat/kansas_us  → anomaly_score: -0.18  ⚠️ FLAG
soy/illinois_us  → anomaly_score: +0.04  ✓ normal
```

**3. YieldSurpriseClassifier → LONG / SHORT / NEUTRAL**

The most directly actionable output. Predicts whether the crop will beat or miss the official consensus forecast, with a probability and confidence tier.

```
corn/illinois_us    → LONG    | prob: 0.71 | confidence: high
soy/mato_grosso_br  → SHORT   | prob: 0.28 | confidence: medium
wheat/kansas_us     → NEUTRAL | prob: 0.51 | confidence: low
```

Confidence tiers:
- **high** — probability ≥ 0.75 or ≤ 0.25 → consider full position
- **medium** — 0.65–0.75 or 0.25–0.35 → consider half position
- **low / NEUTRAL** — 0.35–0.65 → no edge, stay flat

### Reading signals together

The highest-conviction setups are when multiple models agree:

| Scenario | Interpretation | Action |
|----------|---------------|--------|
| Anomaly flagged + model predicts miss + SHORT | Strong bearish consensus | High conviction short |
| No anomaly + model predicts beat + LONG | Strong bullish consensus | High conviction long |
| Anomaly flagged + NEUTRAL signal | Unusual pattern, direction unclear | Watch, wait |
| Yield miss predicted, no anomaly | Gradual underperformance | Mild bearish lean |
| Models disagree | Conflicting signals | Stay flat |

### Signal timing

The edge is sharpest mid-season when you have enough satellite composites to be confident but USDA hasn't yet revised its numbers:

| Crop | Best signal window |
|------|-------------------|
| US corn / soy | June – August |
| US wheat (Kansas) | April – May |
| Rostov wheat | April – May |
| Brazil soy | November – January |

### Running the trainer manually

```bash
python3 quantagri/ml/train.py \
  --sat_csv quantagri_live/quantagri_live_results.csv \
  --historical_csv quantagri_historical/quantagri_monthly_ALL.csv \
  --yield_csv official_yields.csv \
  --model_dir ml_models
```

### Getting SHAP explanations

```python
from quantagri.ml.signals import YieldSurpriseClassifier
import pickle, pandas as pd

clf = pickle.load(open("ml_models/corn_iowa_us_classifier.pkl", "rb"))
feat_row = pd.Series({...})  # one row of season features
print(clf.explain(feat_row))
# → {'ndvi_max_peak': 0.42, 'lswi_mean_avg': 0.18, 'vel_mean_avg': -0.09, ...}
```

---

## Live Results — Column Reference

One row per active region per day in `quantagri_live_results.csv`:

| Column | Example | Notes |
|--------|---------|-------|
| `as_of_date` | 2026-03-06 | Date of latest satellite data |
| `commodity` | corn | Crop |
| `region_id` | iowa_us | Region |
| `current_ndvi` | 0.407 | Latest composite value |
| `current_ndvi_velocity` | -0.0086 | dNDVI/day — rate of change |
| `peak_ndvi` | 0.497 | Season high |
| `peak_ndvi_date` | 2026-01-05 | Date peak was reached |
| `tercile_mean_early/mid/late` | 0.372 / 0.480 / 0.454 | Season thirds |
| `yield_surprise` | +1.2 | bpa vs official forecast |
| `surprise_pct` | +2.8% | Relative surprise |
| `calibration_r2` | 0.78 | Historical NDVI→yield fit |

**Note:** `tercile_mean_*`, `velocity_std`, `yield_surprise`, and `calibration_r2` are blank early in the season — they need multiple composites to compute and fill in naturally over 3–4 weeks.

---

## Signal Scorecard — Column Reference

One row per commodity/region in `quantagri_live/quantagri_signal_scorecard.csv`:

| Column | Example | Notes |
|--------|---------|-------|
| `commodity` | corn | Crop |
| `region_id` | iowa_us | Region |
| `conviction_score` | +3 | -4 to +4 — the bottom line |
| `conviction` | STRONG LONG 🟢 | Human-readable label |
| `ml_signal` | LONG | LONG / SHORT / NEUTRAL |
| `ml_beat_prob` | 0.76 | Probability of beating consensus |
| `ml_confidence` | high | high / medium / low |
| `anomaly_score` | -0.14 | Below -0.05 = flagged |
| `yield_surprise` | +2.1 | bpa vs USDA |

---

## Viewing Results

**Option A — GitHub UI**
`quantagri_live/quantagri_live_results.csv` → GitHub renders as a table.

**Option B — Raw CSV (paste into Excel or Google Sheets)**
```
https://raw.githubusercontent.com/rmkenv/quantag/main/quantagri_live/quantagri_live_results.csv
https://raw.githubusercontent.com/rmkenv/quantag/main/quantagri_live/quantagri_signal_scorecard.csv
```

**Option C — Download artifact**
Actions → latest run → Artifacts → `quantagri-live-merged-N`

**Option D — pandas**
```python
import pandas as pd
df = pd.read_csv(
    "https://raw.githubusercontent.com/rmkenv/quantag/main/quantagri_live/quantagri_live_results.csv"
)
print(df.tail(10).to_string())
```

---

## Alerts

Thresholds in `quantagri/quantagri_live_monitor.py`:
```python
SURPRISE_ALERT_BPS = 1.5   # bpa absolute value
VELOCITY_ALERT     = 0.015  # dNDVI/day
```
When breached, GitHub opens an Issue and you receive an email notification.

Strong ML signals (conviction ±3 or ±4) open a separate Issue with the full scorecard table.

---

## Updating official_yields.csv

The monthly analysis workflow updates yields automatically from USDA/CONAB APIs.
For manual updates after a major WASDE revision:

1. Edit `official_yields.csv` in the repo root
2. Commit: `"Update yields — WASDE March 2026"`

**Key WASDE dates to watch:**

| Month | What changes |
|-------|-------------|
| March | Winter wheat baseline (Kansas) |
| February / March | Brazilian soy CONAB monthly |
| May | First corn/cotton new-crop forecast |
| August | Most important — first field-survey corn/soy estimate |
| November | Near-final corn/soy/cotton |

Regions without API coverage (Xinjiang cotton, Indian sugar, Rostov wheat) must be updated manually from IGC or local ministry reports.

---

## Resolution Settings

Set automatically per commodity — no manual configuration needed.

| Commodity | Resolution | Reason |
|-----------|-----------|--------|
| Soy (Mato Grosso) | 500m | ~1.1M km² ROI, OOMs at finer resolution |
| Sugar | 300m | Two large tropical regions |
| Corn, Wheat, Cotton | 200m | Medium ROIs |

---

## GitHub Actions Free Tier

| Item | Free limit | QuantAgri usage |
|------|-----------|-----------------|
| Minutes/month | 2,000 | ~30 min/day × 30 + ~30 min/month analysis = ~930 min ✅ |
| Storage | 500MB | CSVs + model files ~5MB/month ✅ |
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
| ML models all show NEUTRAL | Needs 3+ seasons per commodity — run historical workflow first |
| `lightgbm` not found in ML step | Added to workflow install step — re-run the daily workflow to pick up |
| `ImportError: attempted relative import with no known parent package` | PYTHONPATH must be set to `${{ github.workspace }}/quantagri/ml` (not just `quantagri`) in the Train and Scorecard workflow steps |
| Workflow not running at scheduled time | GitHub delays scheduled runs up to 30 min under load |
| `fatal: pathspec 'ml_models/' did not match any files` | Create `ml_models/.gitkeep` in repo root — the commit step now handles this automatically |
