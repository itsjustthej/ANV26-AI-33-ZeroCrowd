# ZeroCrowd (Track AI-16): AI Public Transport Overcrowding Predictor
**ANVATION 2026 Hackathon Submission**

> **Note:** All passenger history, demand figures, and impact metrics in this repository are generated from a deterministic synthetic dataset (`seed=42`) and a simulated live feed.

---

## 1. Overview
**ZeroCrowd** is a predictive decision-support and capacity-optimization command center. Instead of acting as a passive monitoring dashboard, it queries historical hourly ridership from SQLite, computes probabilistic demand forecasts with 90% prediction intervals, detects unexpected demand spikes and vehicle breakdowns, and recommends fairness-aware allocations across a limited spare bus fleet.

---

## 2. System Architecture
- **Frontend (`frontend/`):** React, Vite, TypeScript, Tailwind CSS, Recharts, Lucide React
- **Backend (`backend/`):** Python, FastAPI, Pydantic, SQLite (`transport.db`)
- **Database Tables:**
  - `route_baselines`: Active fleet and seat capacity per route (`R001`–`R006`)
  - `route_history`: 1,440 synthetic hourly records (30 days × 6 routes × 8 time windows from `06:00` to `21:00`, seeded with `42`)
  - `allocation_log`: Persistent SQLite audit trail of human-approved vehicle dispatches

---

## 3. Mathematical Engine Specification

1. **Clear-Day Baseline & Event Factor:**
   - Filters `route_history` for `weather == 'Clear'` and `event == 'None'` to compute `mean_clear` and sample standard deviation `sd_clear`.
   - Data-driven event multiplier measured directly from historical rows:
     ```text
     event_factor = mean(event_rows) / mean_clear
     ```

2. **Uncertainty & 90% Prediction Interval:**
   - Standard deviation with a 4% minimum floor, widened by `1.5x` during spike conditions:
     ```text
     sigma = max(sd_clear * factor * widen, 0.04 * predicted, 1.0)
     90% Interval = predicted ± 1.645 * sigma
     ```

3. **Overcrowding Risk & Exceedance Probability:**
   - Computes capacity utilization and the probability that demand exceeds seat capacity using the standard normal CDF (`math.erf`):
     ```text
     Utilization (%) = (predicted / capacity) * 100
     P(Overcrowded)  = 1 - 0.5 * (1 + erf((capacity - predicted) / (sigma * sqrt(2))))
     ```
   - **Risk Tiers:** `LOW` (<80%), `MEDIUM` (80%–100%), `HIGH` (>100%–120%), `CRITICAL` (>120%)
   - **Eligibility for Dispatch:** `Utilization > 100%` OR `P(Overcrowded) >= 50%`

4. **Fairness-Aware Greedy Allocator (Per-Vehicle Re-scoring):**
   - Allocates up to `5` spare vehicles (`100` seats each) one vehicle at a time, recomputing route priority after every single assignment:
     ```text
     severity = min(max(utilization - 80, 0) / 100, 1.5)
     pressure = predicted / max_predicted_across_routes
     service  = min(capacity / predicted, 1.5) / 1.5
     fairness = 0.5 / (1 + extra_vehicles) + 0.5 * (1 - service)

     Priority Score = 0.60 * severity + 0.25 * pressure + 0.15 * fairness
     ```

5. **Hold-Out Backtest Validation (`/api/backtest`):**
   - Trains on days 1–23 and evaluates on held-out days 24–30, reporting per-route **MAE**, **MAPE (%)**, and **90% Interval Coverage**.

---

## 4. Local Setup & Run Instructions

### Start the FastAPI Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Interactive Swagger documentation will be available at: **http://localhost:8000/docs**

### Start the React Frontend
```bash
cd frontend
npm install
npm run dev
```
Dashboard UI will be available at: **http://localhost:5173**

---

## 5. API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/analyze` | `POST` | Evaluates network demand, interval bands, risk tiers, and fairness dispatch |
| `/api/intraday/{route_id}` | `GET` | Returns 8-window profile (`06:00`–`21:00`) for intraday curves vs capacity |
| `/api/history/{route_id}` | `GET` | Returns 30-day historical time-series with festival and weather indicators |
| `/api/backtest` | `GET` | Computes hold-out evaluation metrics (MAE, MAPE, 90% coverage) |
| `/api/log` | `GET` | Retrieves committed dispatch history from SQLite `allocation_log` |
| `/api/reset` | `POST` | Resets the persistent dispatch log table |

---

## 6. Live Demo Scenarios & Features
1. **Intraday Window Switching:** Select any of the 8 time windows (`06:00` to `21:00`) to observe peak commuting vs off-peak patterns.
2. **Stackable Scenarios:** Toggle **Demand Spike** (R002 festival surge) and **Capacity Loss** (R004 loses 2 buses) independently or simultaneously.
3. **Multi-Route Resource Competition:** When stacked, R002 and R004 compete dynamically for the 5 spare buses; the optimizer balances severity with under-service fairness.
4. **Before vs. After Impact Analysis:** Measures overcrowding reduction percentage, critical route count reduction, and utilization spread reduction.
5. **Live Tick Mode (2s):** Simulates real-time streaming vehicle arrivals with continuous re-scoring.
