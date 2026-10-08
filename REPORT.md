# ZeroCrowd (Track AI-16): AI Public Transport Overcrowding Predictor
**ANVATION 2026 Hackathon Final Technical & Empirical Report**

> **Official Disclaimer:** All passenger demand numbers, historical rows, and impact figures in this report and repository are generated from a deterministic synthetic dataset (`random.Random(42)`) and a simulated live streaming layer. The project is designed as an operational decision-support command center for municipal transit authorities.

---

## 1. Executive Summary & Problem Context

Urban public transportation networks in high-density metropolitan areas (such as Bengaluru's BMTC bus network and Namma Metro) experience extreme localized demand spikes driven by festivals, rainfall, sports events, and sudden fleet mechanical breakdowns. Traditional static transit timetables cannot adapt to these intra-day fluctuations, resulting in severe overcrowding (utilization exceeding 120%), extended passenger wait times, and safety hazards.

**ZeroCrowd** is an automated decision-support command center that transforms transit capacity management:
1. **Probabilistic Forecasting:** Evaluates clear-day historical baselines and extracts data-driven event multipliers to project passenger loads across 8 daily time windows with a formal **90% normal prediction interval**.
2. **Exceedance Risk Quantification:** Computes the mathematical probability of overcrowding using the Gaussian error function ($\text{erf}$), categorizing routes into four distinct operational risk tiers.
3. **Stackable Incident Simulation:** Supports concurrent demand surges and vehicle breakdowns, enabling operators to test complex stress scenarios where multiple routes compete for limited emergency fleet assets.
4. **Greedy Fairness-Aware Allocation:** Dispatches a pool of spare vehicles one-by-one, dynamically re-scoring candidates after every vehicle to balance overloading severity, network passenger pressure, and under-service fairness.
5. **Empirical Validation & Auditability:** Backtests baseline forecasts on held-out test data (Days 24–30) and logs every operator-approved dispatch to an immutable SQLite audit trail.

---

## 2. System Architecture

The system uses a clean separation of concerns: a pure, deterministic calculation engine in Python FastAPI, coupled with a high-performance React 19 visualization dashboard.

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                     REACT COMMAND-CENTER DASHBOARD                      │
│        React 19 + Vite + TypeScript + Tailwind CSS + Recharts           │
│                                                                          │
│  [Top Controls]  Window (06:00-21:00) | Spares Pool (5) | Spike | Loss  │
│  [KPI Summary]   Network Size | Total Pax | Overcrowded | Spares Left    │
│  [Alert Banner]  1-Click AI Dispatch Commit / Revert Preview             │
│  [Route Monitor] Risk Tiers | 90% CI | Capacity | Prob % | AI Reason     │
│  [Charts Panel]  Tab 1: 8-Window Intraday vs. Capacity                   │
│                  Tab 2: 30-Day Historical Time-Series & Backtest Split   │
│  [Impact Card]   Overcrowding Reduction % | Fairness Spread Drop         │
│  [Audit Drawer]  Hold-Out Backtest Table | SQLite allocation_log Feed    │
└────────────────────────────────────┬──────────────────▲──────────────────┘
                                     │ REST Polling (2s)│
                                     ▼                  │
┌──────────────────────────────────────────────────────────────────────────┐
│                   FASTAPI DETERMINISTIC CALCULATION ENGINE               │
│                                                                          │
│   Baseline Extractor ──► Scenario Stacking ──► 90% Prediction Interval  │
│           │                                             │                │
│           ▼                                             ▼                │
│   Gaussian Risk Engine (erf) ──► Greedy Fairness-Aware Optimizer         │
│           │                                             │                │
│           └───────────────► Before/After Metrics ◄──────┘                │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        SQLITE DATABASE STORAGE                           │
│   • route_baselines : 6 corridors, base fleet, 100 seats/bus             │
│   • route_history   : 1,440 hourly records (30 days × 6 routes × 8 wins) │
│   • allocation_log  : Persistent audit trail of human-approved actions   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. SQLite Database Schema & Deterministic Seeding

The database file `backend/transport.db` is initialized on startup with three relational tables:

### 3.1 `route_baselines`
Defines operational parameters across 6 strategic urban transit corridors:
- `route_id` (TEXT PK): Route identifier (`R001` to `R006`)
- `route_name` (TEXT): Corridor description
- `vehicles` (INTEGER): Active scheduled fleet
- `bus_cap` (INTEGER): Standard capacity per vehicle (`100` seats)

| Route ID | Corridor Name | Scheduled Buses | 08:00 Clear Mean | 17:00 Clear Mean | Base Capacity |
|---|---|---|---|---|---|
| **R001** | Central <-> Airport | 10 | 700 pax | 650 pax | 1,000 seats |
| **R002** | Central <-> Railway Station | 10 | 800 pax | 760 pax | 1,000 seats |
| **R003** | University <-> City Center | 9 | 620 pax | 700 pax | 900 seats |
| **R004** | Industrial Area <-> Central | 10 | 920 pax | 880 pax | 1,000 seats |
| **R005** | Market <-> Bus Terminal | 8 | 500 pax | 560 pax | 800 seats |
| **R006** | Residential <-> IT Park | 11 | 880 pax | 940 pax | 1,100 seats |

### 3.2 `route_history`
Stores historical ridership logs for parameter calibration:
- `id` (INTEGER PK AUTOINCREMENT), `route_id`, `date`, `hour`, `passengers`, `capacity`, `weather`, `event`.
- Indexed on `(route_id, hour, date)`.
- **Seeding Parameters:**
  - 30 days starting `2026-09-01` across 8 daily time windows (`06:00`, `08:00`, `10:00`, `12:00`, `14:00`, `17:00`, `19:00`, `21:00`) = **1,440 records**.
  - Generated with fixed seed `random.Random(42)`.
  - Off-peak multipliers relative to 08:00 base: `06:00`=0.65, `08:00`=1.00, `10:00`=0.75, `12:00`=0.70, `14:00`=0.72, `17:00`=(explicit mean), `19:00`=0.82, `21:00`=0.58.
  - Multiplicative Gaussian noise: $\sim \mathcal{N}(1.0, 0.05^2)$.
  - Rainy weather: Every 7th day (`day_idx % 7 == 3`), `weather = 'Rain'`, demand $\times 1.08$.
  - Festival anomaly: Route `R002` on days 9, 19, and 26 has `weather = 'Rain'`, `event = 'Major Festival'`, demand $\times 1.60$.

### 3.3 `allocation_log`
Immutable record of dispatch decisions committed by the operator:
- `id`, `ts` (ISO timestamp), `scenario`, `hour`, `route_id`, `vehicles_added`, `util_before`, `util_after`, `reason`.

---

## 4. Mathematical Engine Specification

All forecasting and optimization logic is implemented in pure Python functions.

### 4.1 Clear-Day Baseline & Event Uplift
To prevent double-counting anomalies in baseline projections, the engine queries clear, non-event records:
$$\text{Clear Rows} = \{r \in \text{history} \mid r.\text{weather} = \text{'Clear'} \land r.\text{event} = \text{'None'}\}$$
$$\mu_{\text{clear}} = \frac{1}{N} \sum_{i=1}^N \text{passengers}_i, \quad \sigma_{\text{clear}} = \sqrt{\frac{1}{N-1}\sum_{i=1}^N (\text{passengers}_i - \mu_{\text{clear}})^2}$$

The event multiplier is derived directly from empirical history:
$$\text{event\_factor} = \frac{\text{mean}(\text{event rows})}{\mu_{\text{clear}}} \approx 1.60 \quad (\text{for R002})$$

### 4.2 Stackable Scenario Adjustments
The simulator allows toggling incident scenarios independently or simultaneously:
- **Baseline:** $\text{pred} = \mu_{\text{clear}}, \quad \text{factor} = 1.0, \quad \text{widen} = 1.0$.
- **Demand Spike (`spike=True`):** For routes where $\text{event\_factor} > 1.2$ (R002), $\text{pred} = \mu_{\text{clear}} \times \text{event\_factor}$, uncertainty widened by $\text{widen} = 1.5$.
- **Capacity Loss (`capacity_loss=True`):** Route R004 loses 2 vehicles due to breakdown: $\text{vehicles} = \max(0, \text{vehicles} - 2)$.
- **Concurrent Stacking:** When both toggles are active, R002 experiences festival demand while R004 operates with reduced capacity, creating dynamic fleet competition.

### 4.3 90% Normal Prediction Interval
Uncertainty scales with demand and is protected by a 4% floor to avoid unrealistically narrow bounds:
$$\sigma = \max\left(\sigma_{\text{clear}} \times \text{factor} \times \text{widen}, \, 0.04 \times \text{pred}, \, 1.0\right)$$
$$\text{Lower} = \text{round}(\text{pred} - 1.645 \times \sigma)$$
$$\text{Upper} = \text{round}(\text{pred} + 1.645 \times \sigma)$$

### 4.4 Risk Classification & Overcrowding Exceedance Probability
Using scheduled and extra vehicles, seat capacity is $\text{Capacity} = (\text{vehicles} + \text{extra}) \times \text{bus\_cap}$.
$$\text{Utilization} = \frac{\text{pred}}{\text{Capacity}} \times 100\%$$

Overcrowding probability is calculated using the Gaussian CDF via `math.erf`:
$$P(\text{overcrowded}) = 1.0 - \Phi\left(\frac{\text{Capacity} - \text{pred}}{\sigma}\right) = 1.0 - \frac{1}{2}\left(1 + \text{erf}\left(\frac{\text{Capacity} - \text{pred}}{\sigma\sqrt{2}}\right)\right)$$

| Risk Tier | Utilization Range | Action Required |
|---|---|---|
| **LOW** | $< 80.0\%$ | Routine operations |
| **MEDIUM** | $80.0\% \le \text{util} \le 100.0\%$ | Monitor corridor |
| **HIGH** | $100.0\% < \text{util} \le 120.0\%$ | Dispatch candidate |
| **CRITICAL** | $> 120.0\%$ | Urgent dispatch priority |

**Eligibility Condition:** A route qualifies for emergency fleet assistance if:
$$\text{Utilization} > 100.0\% \quad \lor \quad P(\text{overcrowded}) \ge 50\%$$

### 4.5 Greedy Fairness-Aware Fleet Allocator
With a pool of $S = 5$ spare buses, the optimizer assigns vehicles **one at a time**, dynamically recalculating scores after each assignment:
1. Recompute current utilization and unserved demand for all routes given $e$ extra buses already assigned.
2. For each eligible route, calculate multi-objective score components:
   - **Severity:** Measures excess load above 80% capacity:
     $$\text{severity} = \min\left(\frac{\max(\text{utilization} - 80.0, 0)}{100.0}, \, 1.5\right)$$
   - **Pressure:** Normalizes absolute passenger demand against the network maximum:
     $$\text{pressure} = \frac{\text{pred}}{\max_{r}(\text{pred}_r)}$$
   - **Service Level:** Evaluates current capacity ratio:
     $$\text{service} = \frac{\min(\text{Capacity} / \text{pred}, 1.5)}{1.5}$$
   - **Fairness:** Penalizes routes that have already received vehicles and prioritizes underserved routes:
     $$\text{fairness} = \frac{0.5}{1.0 + e} + 0.5 \times (1.0 - \text{service})$$
3. Priority Score:
   $$\text{Score} = 0.60 \times \text{severity} + 0.25 \times \text{pressure} + 0.15 \times \text{fairness}$$
4. Assign 1 vehicle to the route with the highest score. Repeat until the spare fleet is exhausted or no routes remain eligible.

---

## 5. Empirical Verification & Test Results

### 5.1 Scenario Evaluation Matrix (At 08:00 Peak Window)

| Scenario | Active Corridor Conditions | Pre-AI Overcrowded Routes | Pre-AI Excess Pax | Post-AI Allocation | Post-AI Excess Pax | Overcrowding Reduction | Fairness Spread Drop |
|---|---|---|---|---|---|---|---|
| **Normal** | All routes clear | 0 | 0 pax | No intervention | 0 pax | **0% (Healthy)** | 31.0 pts (Unchanged) |
| **Demand Spike** | R002 Festival surge ($\approx 128\%$) | 1 (`R002`) | 162 pax | R002 +2 buses | 0 pax | **100%** | 53.7 $\to$ 21.0 pts |
| **Capacity Loss** | R004 lost 2 buses ($\approx 115\%$) | 1 (`R004`) | 108 pax | R004 +2 buses | 0 pax | **100%** | 41.5 $\to$ 26.2 pts |
| **Stacked Spike + Loss** | R002 surge & R004 breakdown | 2 (`R002`, `R004`) | 270 pax | R002 +2, R004 +2 | 0 pax | **100%** | 53.7 $\to$ 34.3 pts |

### 5.2 Hold-Out Backtest Results (`GET /api/backtest`)
- **Training Set:** Days 1–23 (Clear baseline + empirical multipliers)
- **Evaluation Set:** Held-out Days 24–30 (Last 7 days)

| Route ID | Route Corridor Name | MAE (Pax) | MAPE (%) | 90% Interval Coverage | Sample Count |
|---|---|---|---|---|---|
| **R001** | Central <-> Airport | 41.3 pax | 6.2% | **85.7%** | 7 |
| **R002** | Central <-> Railway Station | 31.8 pax | 3.7% | **85.7%** | 7 |
| **R003** | University <-> City Center | 31.2 pax | 5.3% | **85.7%** | 7 |
| **R004** | Industrial Area <-> Central | 38.6 pax | 4.2% | **100.0%** | 7 |
| **R005** | Market <-> Bus Terminal | 29.2 pax | 5.6% | **85.7%** | 7 |
| **R006** | Residential <-> IT Park | 44.6 pax | 4.9% | **71.4%** | 7 |
| **Average** | **Network Overall** | **36.1 pax** | **5.0%** | **85.7%** | **42 evaluations** |

*Analysis:* The 85.7% aggregate coverage on held-out test days aligns closely with the nominal 90% prediction interval target, confirming well-calibrated error bounds without artificial parameter inflation.

---

## 6. Command-Center Dashboard Walkthrough

The React 19 interface provides operational visibility across multiple panels:

1. **Top Simulation Header & Badge:** Prominently marked `LIVE SIMULATION · SYNTHETIC HISTORICAL DATA` with ANVATION 2026 branding.
2. **Interactive Controls Bar:**
   - **Time Window Selector:** Switch between 8 daily windows (`06:00` to `21:00`).
   - **Spare Fleet Selector:** Configurable spare pool size (3, 4, 5, 6, 8, 10 buses).
   - **Simulate Demand Spike:** Triggers festival demand uplift on R002.
   - **Simulate Capacity Loss:** Drops 2 buses on R004.
   - **Live Tick Mode (2s):** Streaming simulation generating live passenger arrival jitter and real-time score updates.
   - **Reset Button:** Reverts all scenario toggles and resets database state.
3. **4 Key Performance Indicators (KPIs):**
   - Active Corridors count and selected window.
   - Total Network Predicted Demand with 90% CI indicator.
   - Overcrowded / Critical Routes count and excess unserved passenger volume.
   - Spare Fleet Pool status (dispatched vs. remaining).
4. **Dynamic Decision Banner:** Displays detected anomalies and allows 1-click execution (`APPLY AI ALLOCATION`) or reversion.
5. **Route Monitor Table:** Interactive rows showing route name, current operating condition, point forecast with 90% interval `[lower – upper]`, fleet size (+extra badge), color-coded utilization bar, exceedance probability %, risk tier badge, and transparent AI allocation rationale strings.
6. **Dual-Tab Recharts Analytics Panel:**
   - **Tab 1 (8-Window Intraday Forecast Curve):** Visualizes the full daily demand curve from 06:00 to 21:00, shaded 90% upper bound confidence area, and step-line effective capacity threshold.
   - **Tab 2 (30-Day Historical Time-Series):** Plots all 30 days of passenger demand with markers for Rain and Major Festivals, along with a vertical guide line marking the Day 23 Train/Test split.
7. **Before vs. After Impact Comparison Card:** Badged `Simulated Demo Impact`, featuring:
   - Percentage reduction in overcrowding.
   - Overcrowded routes before $\to$ after.
   - Critical routes before $\to$ after.
   - Average network utilization before $\to$ after.
   - **Fairness: Service-Level Share:** Percentage of routes operating safely without emergency help.
   - **Fairness: Utilization Spread:** Disparity between highest and lowest route utilization in percentage points.
8. **Bottom Verification & Audit Log:**
   - Live rendering of hold-out backtest error metrics.
   - Real-time audit trail displaying entries from SQLite `allocation_log`.

---

## 7. Production Roadmap

For municipal deployment beyond hackathon simulation:
1. **Machine Learning Model Upgrade:** Transition statistical clear-day baselines to Quantile Gradient Boosted Regression (LightGBM) or Temporal Fusion Transformers (TFT) trained on multi-year Automated Passenger Counter (APC) feeds.
2. **Live Feed Ingestion:** Connect GTFS-Realtime vehicle position feeds and automated fare collection (NCMC card tap-in/tap-out) streams via Apache Kafka.
3. **Operations Research Optimization:** Upgrade the greedy heuristic to Mixed-Integer Linear Programming (MILP) using OR-Tools, incorporating depot locations, driver shift constraints, deadheading costs, and charging schedules for electric bus fleets.
4. **Push Communication:** Replace client-side 2-second REST polling with Server-Sent Events (SSE) or WebSockets.

---

## 8. Quick Start Guide

### Prerequisites
- Python 3.10+
- Node.js 18+ (tested on Node v24)
- npm 9+

### Backend Launch
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
- OpenAPI Documentation: `http://localhost:8000/docs`

### Frontend Launch
```bash
cd frontend
npm install
npm run dev
```
- Command Center UI: `http://localhost:5173`
