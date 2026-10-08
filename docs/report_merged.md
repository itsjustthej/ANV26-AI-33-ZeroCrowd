# ZeroCrowd (Track AI-16): AI Public Transport Overcrowding Predictor

**Full Project Report: Requirements Cross-Check, Architecture, Mathematical Engine, Test Results, and Demo Specification**  
ANVATION 2026 Hackathon · Prepared October 2026

> **Notice:** All data, demand metrics, and impact indicators in this project are simulated. The historical records are synthetically seeded (`random.Random(42)`) and the "live" stream is a deterministic simulation. The system is designed as an operational decision-support tool for public transport command centers.

---

## 1. Executive Summary

**ZeroCrowd** is an automated decision-support command-center dashboard for urban transit operators. For each route and intraday time window, it forecasts passenger demand with a 90% normal prediction interval, quantifies overcrowding exceedance risk, dynamically flags demand spikes or fleet capacity losses, and dispatches spare vehicles using a multi-objective greedy fairness-aware allocation algorithm.

### Key Capabilities
- **Deterministic Baseline & Measured Event Uplift:** Calculates clear-day averages and standard deviations, extracting data-driven multipliers for festivals and rainy weather without hardcoded inflation.
- **90% Normal Prediction Interval:** Measures parameter uncertainty ($\pm 1.645\sigma$) with a statistical variance floor.
- **Risk Classification & Exceedance Probability:** Computes continuous probability of overcrowding $P(\text{demand} > \text{capacity})$ via the Gaussian error function ($\text{erf}$).
- **Multi-Objective Fairness Allocator:** Prioritizes vehicles across competing overloaded routes, balancing severity, network demand pressure, and under-service fairness with per-vehicle re-scoring.
- **Human-in-the-Loop Override:** Operator steppers (`[-]`/`[+]`) and interactive standby fleet badges (`UNIT-101`–`UNIT-105`) allowing manual intervention with live impact recalculation.
- **Nearby Vehicle Dispatch Broadcast Telemetry:** Real-time dispatch broadcast banners and audit trail categorization (`[AI_DISPATCH]` vs `[HUMAN_OVERRIDE]`).
- **Hold-Out Backtest Engine:** Evaluates predictive accuracy (MAE, MAPE %, and 90% interval coverage) across a held-out test split (Days 24–30).
- **Universal Single-Port Docker & Render Cloud Deployment:** Single container serving both the built React SPA and the FastAPI backend on port `8000`, with automated Render Blueprint (`render.yaml`).

---

## 2. System Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│                    REACT COMMAND-CENTER UI                   │
│   React 19 + Vite + TypeScript + Tailwind CSS + Recharts     │
│   • 4 KPI Cards: Network, Predicted Pax, Overcrowding, Spare │
│   • Admin Console: Mode Badge, Interactive Units (101-105)   │
│   • Stackable Scenarios: Demand Spike & Capacity Loss        │
│   • Route Monitor Table with Inline [-]/[+] Steppers         │
│   • Nearby Standby Bus Dispatch Broadcast Telemetry          │
│   • Tab 1: 8-Window Intraday Forecast Curve (Live Tick Sync) │
│   • Tab 2: 30-Day Historical Time-Series & Backtest View     │
│   • Before vs. After Impact & Fairness Comparison Card       │
│   • Hold-Out Backtest Validation & SQLite Audit Log Drawer   │
└───────────────┬──────────────────────────────▲───────────────┘
                │ REST: /api/analyze, /api/intraday,           │
                │ /api/history, /api/backtest, /api/log        │
                ▼                                              │
┌──────────────────────────────────────────────────────────────┐
│                  FASTAPI DETERMINISTIC ENGINE                │
│                                                              │
│  Baseline Extractor ──► Scenario Stacking ──► Uncertainty    │
│        │                                           │         │
│        ▼                                           ▼         │
│  Risk & Overcrowd Probability ──► Fairness Allocator         │
│        │                                           │         │
│        ├───────────────► Impact Metrics ◄──────────┘         │
│        ▼                                                     │
│  Static Asset Mount ("/") ──► Single-Port Web Service (:8000)│
└───────────────────────────────┬──────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────┐
│                     SQLITE DATABASE LAYER                    │
│   • route_baselines (6 routes, 100 seats/bus)                │
│   • route_history (1,440 rows, 30 days, 8 windows, seed 42)  │
│   • allocation_log (immutable audit trail of AI & overrides) │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. SQLite Database Schema & Seeding

### 3.1 `route_baselines`
| Column | Type | Details |
|---|---|---|
| `route_id` | TEXT PRIMARY KEY | Route identifier (`R001`–`R006`) |
| `route_name` | TEXT | Human-readable route corridor |
| `vehicles` | INTEGER | Active base vehicle count |
| `bus_cap` | INTEGER | Standard capacity per vehicle (100 seats) |

**Seeded Routes:**
- `R001`: Central <-> Airport (10 vehicles, 08:00 mean=700, 17:00 mean=650)
- `R002`: Central <-> Railway Station (10 vehicles, 08:00 mean=800, 17:00 mean=760)
- `R003`: University <-> City Center (9 vehicles, 08:00 mean=620, 17:00 mean=700)
- `R004`: Industrial Area <-> Central (10 vehicles, 08:00 mean=920, 17:00 mean=880)
- `R005`: Market <-> Bus Terminal (8 vehicles, 08:00 mean=500, 17:00 mean=560)
- `R006`: Residential <-> IT Park (11 vehicles, 08:00 mean=880, 17:00 mean=940)

### 3.2 `route_history`
- 1,440 records across 30 days starting `2026-09-01` over 8 time windows: `06:00`, `08:00`, `10:00`, `12:00`, `14:00`, `17:00`, `19:00`, `21:00`.
- Indexed on `(route_id, hour, date)`.
- 5% Gaussian noise (`random.Random(42)`).
- Every 7th day (`day_idx % 7 == 3`): `weather = 'Rain'`, demand $\times 1.08$.
- Route `R002` on days 9, 19, 26: `weather = 'Rain'`, `event = 'Major Festival'`, demand $\times 1.60$.

### 3.3 `allocation_log`
Records operator-committed AI and human-approved dispatches:
- `id` (INTEGER PK), `ts` (ISO string), `scenario`, `hour`, `route_id`, `vehicles_added`, `util_before`, `util_after`, `reason` (`[AI_DISPATCH]` or `[HUMAN_OVERRIDE]`).

---

## 4. Mathematical Engine

### 4.1 Clear-Day Baseline & Event Multipliers
- Filter records for `weather == 'Clear'` and `event == 'None'`.
- Calculate $\text{mean\_clear}$ and sample standard deviation $\text{sd\_clear}$.
- Extract empirical event multiplier:
  $$\text{event\_factor} = \frac{\text{mean}(\text{event rows})}{\text{mean\_clear}}$$

### 4.2 Stackable Scenario Adjustments
- **Normal:** $\text{pred} = \text{mean\_clear}, \text{factor} = 1.0, \text{widen} = 1.0$.
- **Demand Spike:** If $\text{event\_factor} > 1.2$ (triggers on R002), $\text{factor} = \text{event\_factor}, \text{widen} = 1.5$.
- **Capacity Loss:** R004 loses 2 vehicles ($\max(0, \text{vehicles} - 2)$).
- **Stacking:** Both can be active simultaneously, creating multi-route resource competition.

### 4.3 90% Prediction Interval
$$\sigma = \max(\text{sd\_clear} \times \text{factor} \times \text{widen}, 0.04 \times \text{pred}, 1.0)$$
$$\text{Lower} = \text{round}(\text{pred} - 1.645 \times \sigma)$$
$$\text{Upper} = \text{round}(\text{pred} + 1.645 \times \sigma)$$

### 4.4 Risk & Overcrowding Exceedance Probability
$$\text{Capacity} = (\text{vehicles} + \text{extra}) \times \text{bus\_cap}$$
$$\text{Utilization} = \frac{\text{pred}}{\text{capacity}} \times 100\%$$
$$P(\text{overcrowded}) = 1.0 - 0.5 \times \left(1.0 + \text{erf}\left(\frac{\text{capacity} - \text{pred}}{\sigma \sqrt{2}}\right)\right)$$

- Risk tiers: `LOW` (<80%), `MEDIUM` (80–100%), `HIGH` (100–120%), `CRITICAL` (>120%).
- Eligibility for spare dispatch: $\text{Utilization} > 100\%$ OR $P(\text{overcrowded}) \ge 50\%$.

### 4.5 Greedy Fairness-Aware Allocator
Dispatches 1 vehicle per iteration from a pool of 5 spare vehicles. At each iteration, recomputes for all eligible routes:
$$\text{severity} = \min\left(\frac{\max(\text{utilization} - 80.0, 0.0)}{100.0}, 1.5\right)$$
$$\text{pressure} = \frac{\text{pred}}{\max(\text{pred})}$$
$$\text{service} = \frac{\min(\text{capacity} / \text{pred}, 1.5)}{1.5}$$
$$\text{fairness} = \frac{0.5}{1.0 + e} + 0.5 \times (1.0 - service)$$
$$\text{Score} = 0.60 \times \text{severity} + 0.25 \times \text{pressure} + 0.15 \times \text{fairness}$$

### 4.6 Human Intervention & Override Mechanics
`POST /api/analyze` accepts an optional `manual_alloc: Optional[Dict[str, int]]`:
- If omitted, computes optimal AI distribution.
- If provided, enforces `spare_vehicles` quota, recomputes capacities and network impact under operator overrides, and preserves the AI's `recommendation` for comparison.

---

## 5. Hold-Out Backtest Validation

The backtest splits the 30-day synthetic dataset into:
- **Training Set:** Days 1–23 (clear baselines and measured multipliers).
- **Test Set:** Held-out Days 24–30 (last 7 days).

| Route ID | Route Corridor Name | MAE (Pax) | MAPE (%) | 90% Interval Coverage | Sample Count |
|---|---|---|---|---|---|
| **R001** | Central <-> Airport | 41.3 pax | 6.2% | **85.7%** | 7 |
| **R002** | Central <-> Railway Station | 31.8 pax | 3.7% | **85.7%** | 7 |
| **R003** | University <-> City Center | 31.2 pax | 5.3% | **85.7%** | 7 |
| **R004** | Industrial Area <-> Central | 38.6 pax | 4.2% | **100.0%** | 7 |
| **R005** | Market <-> Bus Terminal | 29.2 pax | 5.6% | **85.7%** | 7 |
| **R006** | Residential <-> IT Park | 44.6 pax | 4.9% | **71.4%** | 7 |
| **Average** | **Network Overall** | **36.1 pax** | **5.0%** | **85.7%** | **42 evaluations** |

---

## 6. Endpoints Reference

- `POST /api/analyze`: Evaluates network state, applies scenario stacking, runs fairness optimizer or manual override, and logs actions.
- `GET /api/intraday/{route_id}?tick_step=0`: Returns 8-window daily curve (06:00–21:00) with confidence intervals and real-time tick sync.
- `GET /api/history/{route_id}?hour=08:00`: Returns 30-day history and intraday profile.
- `GET /api/backtest?hour=08:00`: Computes hold-out evaluation metrics.
- `GET /api/log`: Retrieves recent entries from `allocation_log`.
- `POST /api/reset`: Resets logs to initial state.

---

## 7. Deployment Instructions

### 7.1 Single-Port Docker (Recommended)
```bash
docker compose up -d --build
```
Access full-stack app and API at `http://localhost:8000`.

### 7.2 1-Click Render Cloud Deployment
Connect GitHub repository `https://github.com/itsjustthej/CodeVanta` in [Render](https://dashboard.render.com) using Blueprint `render.yaml`.
