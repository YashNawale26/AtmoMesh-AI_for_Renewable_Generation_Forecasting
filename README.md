# AI Forecasting Layer for Renewable Solar and Wind Generation

## 1. Problem Overview

Karnataka has rapidly scaled up grid-connected solar and wind capacity, with assets spread across multiple regions and clusters. As renewable penetration grows, generation becomes highly variable because output is driven by local weather (irradiance, cloud cover, wind speed, temperature) rather than dispatchable fuel.

Grid operators and planning departments need accurate, granular forecasts at both **plant level** and **cluster/region level** to:

- Schedule generation day-ahead and intra-day.
- Reduce curtailment and over-reliance on backup thermal sources.
- Minimize system imbalances and deviation penalties.
- Improve visibility into when and where renewable peaks will occur.

The challenge is that, although historical generation and weather data are available, turning them into **reliable, location-specific forecasts with uncertainty and explanations** is non-trivial. Asset behaviour varies by technology (solar vs wind), plant design, geography, and historical performance.

This project proposes an **AI-based forecasting layer** that sits alongside existing SLDC/utility systems and provides:

- Day-ahead and intra-day **hourly forecasts** for solar and wind plants and clusters.
- **Uncertainty ranges** (e.g., P10/P50/P90) for operational risk management.
- **Explainable outputs** that highlight key drivers such as cloud cover, wind speed, and seasonality.
- A design that can be deployed on-prem / in a secure environment without using hosted LLMs on sensitive data.

---

## 2. Dataset Description

For the hackathon prototype, we use the public dataset:

> **Solar and wind power data from the Chinese State Grid Renewable Energy Generation Forecasting Competition**

This dataset contains:

- **8 solar stations** with nominal capacities (e.g. 50 MW, 130 MW, 30 MW, etc.).
- **6 wind farms** with nominal capacities (e.g. 99 MW, 200 MW, 66 MW, etc.).
- Time series spanning multiple years at 15‑minute resolution.

Each **solar site file** includes, for each timestamp:

- Time (year-month-day hour:minute:second)
- Total solar irradiance (W/m²)
- Direct normal irradiance (W/m²)
- Global horizontal irradiance (W/m²)
- Air temperature (°C)
- Atmospheric pressure (hPa)
- Relative humidity (%)
- Plant power output (MW)

Each **wind site file** includes, for each timestamp:

- Time (year-month-day hour:minute:second)
- Wind speed and direction at multiple heights (10 m, 30 m, 50 m)
- Wind speed and direction at hub height
- Air temperature (°C)
- Atmospheric pressure (hPa)
- Relative humidity (%)
- Plant power output (MW)

Why this dataset is suitable for the hackathon:

- Multi-plant: multiple solar and wind sites, matching the “plant + cluster” requirement.
- Rich features: weather + power, ideal for ML-based forecasting.
- Forecasting-oriented: originally constructed for a renewable generation forecasting competition.
- Non-sensitive: can be used openly while the **architecture** remains applicable to Karnataka plants.

In the project, we treat each site as a “plant” and define **virtual clusters** by grouping plants, so that we can demonstrate both plant-level and cluster-level forecasting and aggregation.

---

## 3. Planned Architecture and Modeling Approach

### 3.1 High-Level Architecture

```text
Raw Site Files (8 Solar + 6 Wind)
        |
        v
[Data Ingestion & Cleaning]
        |
        v
[Standardization & Metadata Tagging]
  - plant_id, tech_type (solar/wind)
  - capacity_mw, cluster_id
        |
        v
[Unified Master Dataset (hourly)]
        |
   +----+----+
   |         |
   v         v
Solar Subset   Wind Subset
(tech_type=solar) (tech_type=wind)
   |         |
   v         v
Solar Global Model   Wind Global Model
   |         |
   +----+----+
        |
        v
[Plant-Level Forecasts (hourly)]
        |
        v
[Cluster-Level Aggregation]
        |
        v
APIs / CSV Exports / Demo Dashboard
```

Key ideas:

- One **unified master dataset** for all plants with a consistent schema.
- Two **technology-specific pipelines and models**:
  - One **global solar model** trained on all 8 solar stations.
  - One **global wind model** trained on all 6 wind farms.
- Plant-level forecasts are **aggregated by cluster/region** to produce cluster-level forecasts.

This matches the requirement to **generalise across assets and geographies** without maintaining a separate bespoke model for every single plant.

---

### 3.2 Data Cleaning & Preparation

For each site file (solar or wind), the pipeline will:

- Parse and standardize timestamps.
- Sort records chronologically and remove duplicates.
- Check and handle missing timestamps / missing values.
- Convert all numeric fields to proper types.
- Flag or clip physically impossible values (e.g. negative irradiance, humidity > 100%, power far above nominal capacity).
- Resample 15‑minute data to **hourly** resolution for main forecasts (e.g. sum/average as appropriate).
- Add metadata columns:
  - `plant_id`, `tech_type` (`"solar"` or `"wind"`),
  - `capacity_mw`,
  - `cluster_id` (logical grouping for aggregation).

From the cleaned per-site tables, we build:

- A **master dataset** containing all solar and wind plants.
- A **solar subset** (`tech_type = "solar"`) for solar modeling.
- A **wind subset** (`tech_type = "wind"`) for wind modeling.

---

### 3.3 Feature Engineering

Common feature design:

- **Time features**
  - Hour of day, day of week, month, day of year, season flags (e.g. monsoon vs non‑monsoon in deployment).
- **Lag features**
  - Past power values: previous 1h, 2h, 3h, previous day same hour.
  - Optional rolling mean/standard deviation over recent hours.
- **Plant metadata**
  - `capacity_mw`.
  - Encoded `plant_id`, `cluster_id`, `tech_type`.

Solar-specific features:

- Total solar irradiance, DNI, GHI.
- Temperature, humidity, atmospheric pressure.

Wind-specific features:

- Wind speeds & directions at multiple heights (10m, 30m, 50m, hub height).
- Temperature, humidity, atmospheric pressure.

Optionally, we predict **capacity factor**:

$$
capacity factor = \frac{power(MW)}{capacity(MW)}
$$

and then convert forecasts back to MW by multiplying with `capacity_mw`. This makes plants of different sizes comparable in a single global model.

---

### 3.4 Modeling Strategy

We use **two global models**:

- **Solar model**
  - Single model trained on all solar plants.
  - Inputs: solar-specific weather, time features, plant metadata, lagged power features.
  - Outputs: hourly forecasts for each plant (and quantiles for uncertainty).

- **Wind model**
  - Single model trained on all wind plants.
  - Inputs: wind-specific weather, time features, plant metadata, lagged power features.
  - Outputs: hourly forecasts for each plant (and quantiles for uncertainty).

Model families (subject to iteration):

- Baselines:
  - Persistence (e.g., “tomorrow = today” or previous-day same hour).
  - Simple historical averages (seasonal/hourly profiles).
- Main ML models:
  - Gradient boosting models (LightGBM / XGBoost) with:
    - Either point regression (MAE/MSE loss), or
    - Quantile regression for P10/P50/P90.
  - Optionally, a deep time-series model (e.g., LSTM / TFT) in a later phase.

Cluster-level forecasts are computed by **summing plant-level forecasts** for all plants in a cluster. This keeps the modeling simple while supporting flexible aggregation.

---

### 3.5 Uncertainty & Explainability

To satisfy the requirement for uncertainty-aware and explainable forecasts, we plan:

- **Uncertainty**
  - Quantile forecasts (e.g. P10, P50, P90) from the models, or
  - Ensembles of models with prediction interval estimation.

- **Explainability**
  - Global feature importance to understand main drivers across plants.
  - Local explanations (e.g. SHAP values) for specific forecasts:
    - “High GHI and clear-sky hours drive this solar peak.”
    - “Hub-height wind speed and direction changes drive this wind ramp.”

These explanations will be surfaced in notebooks and a simple UI/demo so that operators can understand *why* the forecast has a particular shape or why it changed between runs.

---

## 4. Phase II Prototype Plan

Phase II will focus on turning this design into a working end‑to‑end prototype.

### 4.1 Goals

- Implement a robust **data ingestion + cleaning pipeline** for all 14 site files.
- Train and evaluate **baseline models** and **main ML models** for solar and wind.
- Generate **plant-level and cluster-level hourly forecasts** (day-ahead and intra-day).
- Produce **uncertainty bands** and **explainability plots**.
- Provide a **demo interface** (notebook/Streamlit/Lightweight API) showcasing forecasts.

### 4.2 Milestones

1. **Data ingestion and unified schema**
   - Load all solar and wind Excel/CSV files.
   - Clean and resample to hourly.
   - Build a unified master dataset with metadata.

2. **Exploratory data analysis (EDA)**
   - Understand distribution, seasonality, and weather–generation relationships for solar and wind.
   - Inspect missing data patterns and potential outliers.

3. **Baselines**
   - Implement persistence and simple historical-average baselines.
   - Compute MAE/MAPE/RMSE for reference at plant and cluster levels.

4. **Main models**
   - Train global solar and global wind models.
   - Evaluate on held-out time periods; compare against baselines.

5. **Uncertainty & explainability**
   - Add quantile/regression models or ensembles.
   - Generate feature importance and SHAP plots.

6. **Demo**
   - Build a notebook or small app that:
     - Lets users pick a plant/cluster and date range.
     - Shows actual vs forecast with uncertainty band.
     - Shows top drivers for selected hours.

---

## 5. Project Structure (Skeleton)

This is the planned structure of the repository. In Phase I, many folders/files will be empty placeholders; they will be filled in during Phase II.

```text
renewable-forecasting-layer/
├── README.md
├── requirements.txt              # Will list Python dependencies
├── data/
│   ├── raw/
│   │   ├── solar/                # Original solar site files (not committed if large)
│   │   └── wind/                 # Original wind site files
│   ├── processed/                # Cleaned & resampled datasets
│   └── metadata/                 # Plant/cluster metadata (CSV/JSON)
├── notebooks/
│   ├── 01_data_ingestion_and_cleaning.ipynb
│   ├── 02_eda_solar.ipynb
│   ├── 03_eda_wind.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_baseline_models.ipynb
│   ├── 06_solar_model_training.ipynb
│   ├── 07_wind_model_training.ipynb
│   ├── 08_uncertainty_and_explainability.ipynb
│   └── 09_demo_forecasting_pipeline.ipynb
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   ├── clean_solar.py
│   │   ├── clean_wind.py
│   │   └── build_master_dataset.py
│   ├── features/
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── baseline.py
│   │   ├── train_solar.py
│   │   ├── train_wind.py
│   │   ├── predict.py
│   │   └── uncertainty.py
│   ├── aggregation/
│   │   └── cluster_forecast.py
│   ├── explain/
│   │   └── shap_analysis.py
│   ├── api/
│   │   └── app.py                # Optional FastAPI for forecasts
│   └── utils/
│       └── config.py
├── outputs/
│   ├── plots/                    # Figures for reports / demo
│   ├── metrics/                  # Evaluation metrics and logs
│   └── forecasts/                # Sample forecast CSVs/JSONs
└── docs/
    ├── architecture_diagram.png  # Exported from pptx or draw.io later
    └── demo_notes.md
```
