# ZeroCrowd (Track AI-16): AI Public Transport Overcrowding Predictor
**ANVATION 2026 Hackathon Submission**

🚀 **Live Command Center UI (Render):** https://zerocrowd-cij1.onrender.com/  
📘 **Interactive Swagger API Docs:** https://zerocrowd-cij1.onrender.com/docs  

> **Note:** Calibrated on Bengaluru BMTC high-density corridors (`KIA-9`, `252-F`, `226-M`, `401-M`, `KBS-3A`, `500-D`) with distinct urban land-use demand curves, live Open-Meteo Bengaluru weather telemetry, and Conformal Quantile GBDT forecasting.

---

## 1. Overview
**ZeroCrowd** is a predictive decision-support and capacity-optimization command center. Instead of acting as a passive monitoring dashboard, it queries historical hourly ridership from SQLite, computes probabilistic demand forecasts with 90% prediction intervals, detects unexpected demand spikes and vehicle breakdowns, and recommends fairness-aware allocations across a limited spare bus fleet.

---

## 2. System Architecture
- **Frontend (`frontend/`):** React, Vite, TypeScript, Tailwind CSS, Recharts, Lucide React
- **Backend (`backend/`):** Python, FastAPI, Pydantic, SQLite (`transport.db`), Scikit-Learn (Quantile HistGradientBoostingRegressor)
- **Database Tables:**
  - `route_baselines`: Active fleet and seat capacity per route (`R001`–`R006`)
  - `route_history`: 1,440 calibrated corridor-hour baseline records (30 days × 6 BMTC corridors × 8 intraday windows) paired with live weather telemetry and dynamic stress-test injection.
  - `allocation_log`: Persistent SQLite audit trail of human-approved vehicle dispatches

---

## 3. Mathematical Engine Specification

1. **Quantile Gradient Boosted Regression (HistGBDT):**
   - Fits three pinball loss models ($q=0.05, 0.50, 0.95$) taking route identity embeddings, diurnal continuous hours, cyclic sine/cosine harmonics, weekend flags, rainfall telemetry, and autoregressive lag/rolling momentum features.
   - Non-conformity calibration produces guaranteed 90% conformal prediction intervals:
     ```text
     90% Interval = [q05 - q_hat, q95 + q_hat]
     ```

2. **Overcrowding Risk & Exceedance Probability:**
   - Computes capacity utilization and the closed-form probability that demand exceeds effective seat capacity using the standard Gaussian CDF (`math.erf`):
     ```text
     Utilization (%) = (predicted / capacity) * 100
     P(Overcrowded)  = 1 - 0.5 * (1 + erf((capacity - predicted) / (sigma * sqrt(2))))
     ```
   - **Risk Tiers:** `LOW` (<80%), `MEDIUM` (80%–100%), `HIGH` (>100%–120%), `CRITICAL` (>120%)
   - **Eligibility for Dispatch:** `Utilization > 100%` OR `P(Overcrowded) >= 50%`

3. **Bi-Directional Fleet Optimization Algorithm:**
   - Allocates spare standby vehicles (100 seats each) and safely harvests surplus vehicles from under-utilized donor routes ($\le 65\%$ utilization) to critical receiver routes, recomputing route priority dynamically:
     ```text
     severity = min(max(utilization - 80, 0) / 100, 1.5)
     pressure = predicted / max_predicted_across_routes
     service  = min(capacity / predicted, 1.5) / 1.5
     fairness = 0.5 / (1 + extra_vehicles) + 0.5 * (1 - service)

     Priority Score = 0.60 * severity + 0.25 * pressure + 0.15 * fairness
     ```

4. **Hold-Out Backtest Validation (`/api/backtest`):**
   - Evaluates performance on held-out days 24–30, reporting per-route **MAE**, **MAPE (%)**, and **90% Interval Coverage**.

---

## 4. Deployment & Access Instructions

### Cloud Production Deployment (Render)
ZeroCrowd is deployed natively on Render Cloud Infrastructure:
- 🚀 **Live Command Center UI & API:** https://zerocrowd-cij1.onrender.com/
- 📘 **Live OpenAPI / Swagger Docs:** https://zerocrowd-cij1.onrender.com/docs

The cloud service is deployed via `render.yaml` as a native Python web service:
- Provisions Python 3.11 with `requirements.txt` and `nodeenv`.
- Automatically builds the React/Vite single-page application into `backend/static/`.
- Starts the production Uvicorn ASGI server serving the Single Page Application on root `/` alongside all `/api/*` endpoints and live Open-Meteo telemetry.
- Runs an instant health check probe on `/api/health`.

### Local Development (Python + Vite)

#### 1. Start the FastAPI Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```
API Documentation: Access `/docs` on the active backend port.

#### 2. Start the React Frontend
```bash
cd frontend
npm install
npm run dev
```

---

## 5. API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | `GET` | Lightweight production service health check probe |
| `/api/analyze` | `POST` | Evaluates network demand, interval bands, risk tiers, and fleet optimization |
| `/api/commit` | `POST` | Executes and logs fleet rebalancing with supervisory verification |
| `/api/undo-last-commit` | `POST` | 10-second safety rollback mechanism for recent dispatches |
| `/api/intraday/{route_id}` | `GET` | Returns 8-window profile (`06:00`–`21:00`) for intraday curves vs capacity |
| `/api/history/{route_id}` | `GET` | Returns 30-day historical time-series with festival and weather indicators |
| `/api/backtest` | `GET` | Computes hold-out evaluation metrics (MAE, MAPE, 90% coverage) |
| `/api/log` | `GET` | Retrieves committed dispatch audit history from SQLite |
| `/api/reset` | `POST` | Resets the persistent dispatch log table |

---

## 6. Live Demo Scenarios & Features
1. **Intraday Window Switching:** Select any of the 8 time windows (`06:00` to `21:00`) to observe route-specific land-use curves (Airport, Peenya Industrial, ORR IT Corridor, Majestic Hub).
2. **Stackable Scenarios:** Toggle **Demand Spike** (R002 festival surge) and **Capacity Loss** (R004 breakdown) independently or simultaneously.
3. **Bi-Directional Fleet Optimization:** Generates Pareto-optimal allocations combining standby dispatches and surplus harvesting without creating secondary bottlenecks.
4. **Before vs. After Impact Analysis:** Measures overcrowding reduction percentage, critical route count reduction, and real-time seat capacity elevation.
5. **Two-Person Co-Sign & Safety Rollback:** Enforces two-person operational approvals for major reallocations ($\ge 4$ buses) with a 10-second instant undo safeguard.
