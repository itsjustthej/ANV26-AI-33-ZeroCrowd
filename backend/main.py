import math
import os
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="ZeroCrowd: AI Public Transport Overcrowding Predictor (Track AI-16)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "transport.db"

WINDOWS = ["06:00", "08:00", "10:00", "12:00", "14:00", "17:00", "19:00", "21:00"]

# Base route definitions from Section 2 & Section 5.2
ROUTE_SPECS = [
    {"id": "R001", "name": "Central <-> Airport",         "vehicles": 10, "m08": 700, "m17": 650},
    {"id": "R002", "name": "Central <-> Railway Station", "vehicles": 10, "m08": 800, "m17": 760},
    {"id": "R003", "name": "University <-> City Center",  "vehicles": 9,  "m08": 620, "m17": 700},
    {"id": "R004", "name": "Industrial Area <-> Central", "vehicles": 10, "m08": 920, "m17": 880},
    {"id": "R005", "name": "Market <-> Bus Terminal",     "vehicles": 8,  "m08": 500, "m17": 560},
    {"id": "R006", "name": "Residential <-> IT Park",     "vehicles": 11, "m08": 880, "m17": 940},
]

# Off-peak multipliers relative to 08:00 base per Section 2 specification
WINDOW_FACTORS = {
    "06:00": 0.65,
    "08:00": 1.00,
    "10:00": 0.75,
    "12:00": 0.70,
    "14:00": 0.72,
    "17:00": 1.00,  # Uses explicit m17 mean
    "19:00": 0.82,
    "21:00": 0.58,
}


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS route_baselines (
            route_id TEXT PRIMARY KEY,
            route_name TEXT,
            vehicles INTEGER,
            bus_cap INTEGER
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS route_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            route_id TEXT,
            date TEXT,
            hour TEXT,
            passengers INTEGER,
            capacity INTEGER,
            weather TEXT,
            event TEXT
        )
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_route_hour_date
        ON route_history (route_id, hour, date)
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS allocation_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            scenario TEXT,
            hour TEXT,
            route_id TEXT,
            vehicles_added INTEGER,
            util_before REAL,
            util_after REAL,
            reason TEXT
        )
    """)

    cur.execute("SELECT COUNT(*) FROM route_baselines")
    if cur.fetchone()[0] == 0:
        for r in ROUTE_SPECS:
            cur.execute(
                "INSERT INTO route_baselines VALUES (?, ?, ?, ?)",
                (r["id"], r["name"], r["vehicles"], 100),
            )

    cur.execute("SELECT COUNT(*) FROM route_history")
    if cur.fetchone()[0] == 0:
        rng = random.Random(42)
        start_date = datetime(2026, 9, 1)
        rows_to_insert = []
        for day_idx in range(30):
            date_str = (start_date + timedelta(days=day_idx)).strftime("%Y-%m-%d")
            is_rain_day = (day_idx % 7 == 3)
            for r in ROUTE_SPECS:
                cap = r["vehicles"] * 100
                for w in WINDOWS:
                    base = r["m17"] if w == "17:00" else r["m08"] * WINDOW_FACTORS[w]
                    weather = "Clear"
                    event = "None"
                    mult = 1.0

                    if r["id"] == "R002" and day_idx in (9, 19, 26):
                        weather = "Rain"
                        event = "Major Festival"
                        mult = 1.60
                    elif is_rain_day:
                        weather = "Rain"
                        mult = 1.08

                    noise = rng.gauss(1.0, 0.05)
                    passengers = max(50, round(base * mult * noise))
                    rows_to_insert.append(
                        (r["id"], date_str, w, passengers, cap, weather, event)
                    )

        cur.executemany(
            """INSERT INTO route_history
               (route_id, date, hour, passengers, capacity, weather, event)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            rows_to_insert,
        )
        conn.commit()
    conn.close()


init_db()


def phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def classify_risk(util: float) -> str:
    if util < 80.0:
        return "LOW"
    if util <= 100.0:
        return "MEDIUM"
    if util <= 120.0:
        return "HIGH"
    return "CRITICAL"


def compute_route_stats(
    conn: sqlite3.Connection,
    route_id: str,
    hour: str,
    max_date: Optional[str] = None,
):
    cur = conn.cursor()
    date_filter = "AND date <= ?" if max_date else ""
    params = (route_id, hour, max_date) if max_date else (route_id, hour)

    cur.execute(
        f"""SELECT passengers, weather, event FROM route_history
            WHERE route_id = ? AND hour = ? {date_filter}""",
        params,
    )
    rows = cur.fetchall()

    clear_vals = [r["passengers"] for r in rows if r["weather"] == "Clear" and r["event"] == "None"]
    event_vals = [r["passengers"] for r in rows if r["event"] != "None"]
    rain_vals = [r["passengers"] for r in rows if r["weather"] == "Rain" and r["event"] == "None"]

    if not clear_vals:
        clear_vals = [r["passengers"] for r in rows] or [700]

    mean_clear = sum(clear_vals) / len(clear_vals)
    variance = sum((x - mean_clear) ** 2 for x in clear_vals) / max(1, len(clear_vals) - 1)
    sd_clear = math.sqrt(variance)

    event_factor = (sum(event_vals) / len(event_vals)) / mean_clear if event_vals else 1.0
    rain_factor = (sum(rain_vals) / len(rain_vals)) / mean_clear if rain_vals else 1.08

    return {
        "mean_clear": mean_clear,
        "sd_clear": sd_clear,
        "event_factor": event_factor,
        "rain_factor": rain_factor,
    }


def build_forecasts(hour: str, spike: bool, capacity_loss: bool, tick_step: int = 0):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM route_baselines ORDER BY route_id")
    baselines = [dict(r) for r in cur.fetchall()]

    routes = []
    rng = random.Random(42 + tick_step)

    for b in baselines:
        stats = compute_route_stats(conn, b["route_id"], hour)
        factor = 1.0
        widen = 1.0
        condition = "Clear"
        vehicles = b["vehicles"]

        if spike and stats["event_factor"] > 1.2:
            ramp = min(1.0, tick_step / 3.0) if tick_step > 0 else 1.0
            factor = 1.0 + (stats["event_factor"] - 1.0) * ramp
            widen = 1.5
            condition = "Rain + Major Festival"

        if capacity_loss and b["route_id"] == "R004":
            vehicles = max(0, vehicles - 2)
            condition = (
                "2 Vehicles Out of Service"
                if condition == "Clear"
                else condition + " + 2 Vehicles Lost"
            )

        pred = stats["mean_clear"] * factor
        if tick_step > 0:
            pred *= rng.gauss(1.0, 0.015)

        sigma = max(stats["sd_clear"] * factor * widen, 0.04 * pred, 1.0)

        routes.append({
            "route_id": b["route_id"],
            "route_name": b["route_name"],
            "base_vehicles": b["vehicles"],
            "vehicles": vehicles,
            "bus_cap": b["bus_cap"],
            "mean_clear": round(stats["mean_clear"]),
            "sd_clear": round(stats["sd_clear"], 1),
            "event_factor": round(stats["event_factor"], 2),
            "pred": pred,
            "sigma": sigma,
            "condition": condition,
        })

    conn.close()
    return routes


def assess_route(r: dict, extra: int = 0, priority_score: float = 0.0, before_util: Optional[float] = None):
    eff_vehicles = r["vehicles"] + extra
    cap = eff_vehicles * r["bus_cap"]
    pred = max(r["pred"], 1.0)
    sigma = max(r["sigma"], 1.0)

    util = (pred / cap) * 100.0 if cap > 0 else 999.0
    prob = (1.0 - phi((cap - pred) / sigma)) if cap > 0 else 1.0
    excess = max(0, round(pred - cap))
    risk = classify_risk(util)

    if extra > 0 and before_util is not None:
        why = f"+{extra} vehicle(s): {before_util:.1f}% → {util:.1f}% utilization (priority {priority_score:.2f})"
    elif util > 100.0 or prob >= 0.50:
        why = f"Eligible ({util:.1f}% util, {round(prob * 100)}% overflow risk) — spare pool exhausted"
    else:
        why = f"No vehicles needed: {util:.1f}% utilization, {round(prob * 100)}% overcrowding probability"

    return {
        "id": r["route_id"],
        "name": r["route_name"],
        "condition": r["condition"],
        "baseline_mean": r["mean_clear"],
        "predicted": round(pred),
        "lower": round(pred - 1.645 * sigma),
        "upper": round(pred + 1.645 * sigma),
        "sigma": round(sigma, 1),
        "base_vehicles": r["base_vehicles"],
        "vehicles": eff_vehicles,
        "extra": extra,
        "capacity": cap,
        "utilization": round(min(util, 999.0), 1),
        "risk": risk,
        "probability": round(prob * 100),
        "excess": excess,
        "why": why,
    }


def allocate_vehicles(routes: List[dict], spare: int):
    extra = {r["route_id"]: 0 for r in routes}
    last_score = {r["route_id"]: 0.0 for r in routes}
    max_pred = max(r["pred"] for r in routes) if routes else 1.0

    for _ in range(spare):
        best_id = None
        best_score = -1.0

        for r in routes:
            rid = r["route_id"]
            e = extra[rid]
            m = assess_route(r, e)
            if not (m["utilization"] > 100.0 or m["probability"] >= 50):
                continue

            severity = min(max(m["utilization"] - 80.0, 0.0) / 100.0, 1.5)
            pressure = r["pred"] / max_pred
            service = min(m["capacity"] / max(r["pred"], 1.0), 1.5) / 1.5
            fairness = 0.5 / (1.0 + e) + 0.5 * (1.0 - service)

            score = 0.60 * severity + 0.25 * pressure + 0.15 * fairness
            if score > best_score:
                best_score = score
                best_id = rid

        if best_id is None:
            break
        extra[best_id] += 1
        last_score[best_id] = best_score

    return extra, last_score


def summarize_metrics(rows: List[dict]):
    if not rows:
        return {}
    utils = [r["utilization"] for r in rows]
    healthy_count = sum(
        1 for r in rows if r["utilization"] <= 100.0 and r["probability"] < 50
    )
    return {
        "overcrowded_routes": sum(1 for r in rows if r["utilization"] > 100.0),
        "critical_routes": sum(1 for r in rows if r["risk"] == "CRITICAL"),
        "passengers_affected": sum(r["excess"] for r in rows),
        "avg_utilization": round(sum(utils) / len(utils), 1),
        "service_level_share": round((healthy_count / len(rows)) * 100.0, 1),
        "utilization_spread": round(max(utils) - min(utils), 1),
    }


class AnalyzeReq(BaseModel):
    spike: bool = False
    capacity_loss: bool = False
    hour: str = "08:00"
    spare: int = 5
    apply: bool = False
    tick_step: int = 0


@app.post("/api/analyze")
def analyze(req: AnalyzeReq):
    hour = req.hour if req.hour in WINDOWS else "08:00"
    routes = build_forecasts(hour, req.spike, req.capacity_loss, req.tick_step)
    rec, scores = allocate_vehicles(routes, req.spare)

    before_rows = [assess_route(r, 0) for r in routes]
    before_map = {r["id"]: r["utilization"] for r in before_rows}

    after_rows = [
        assess_route(r, rec[r["route_id"]], scores[r["route_id"]], before_map[r["route_id"]])
        for r in routes
    ]

    display_rows = after_rows if req.apply else [
        {**b, "why": a["why"], "recommended_extra": rec[b["id"]]}
        for b, a in zip(before_rows, after_rows)
    ]

    before_summary = summarize_metrics(before_rows)
    after_summary = summarize_metrics(after_rows)

    eb = before_summary["passengers_affected"]
    ea = after_summary["passengers_affected"]
    reduction_pct = round(((eb - ea) / eb) * 100.0) if eb > 0 else 0

    scenario_label = "+".join(
        [s for s, active in [("spike", req.spike), ("capacity_loss", req.capacity_loss)] if active]
    ) or "normal"

    if req.apply and sum(rec.values()) > 0:
        conn = get_db()
        cur = conn.cursor()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for a in after_rows:
            if a["extra"] > 0:
                cur.execute(
                    """SELECT id, vehicles_added FROM allocation_log
                       WHERE scenario = ? AND hour = ? AND route_id = ?
                       ORDER BY id DESC LIMIT 1""",
                    (scenario_label, hour, a["id"]),
                )
                last_row = cur.fetchone()
                if not last_row or last_row[1] != a["extra"]:
                    cur.execute(
                        """INSERT INTO allocation_log
                           (ts, scenario, hour, route_id, vehicles_added, util_before, util_after, reason)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (ts, scenario_label, hour, a["id"], a["extra"], before_map[a["id"]], a["utilization"], a["why"]),
                    )
        conn.commit()
        conn.close()

    return {
        "scenario": scenario_label,
        "spike": req.spike,
        "capacity_loss": req.capacity_loss,
        "hour": hour,
        "applied": req.apply,
        "spare_vehicles": req.spare,
        "spare_used": sum(rec.values()),
        "spare_left": req.spare - (sum(rec.values()) if req.apply else 0),
        "routes": display_rows,
        "recommendation": {k: v for k, v in rec.items() if v > 0},
        "before": before_summary,
        "after": after_summary,
        "reduction_pct": reduction_pct,
        "method": "Historical hourly clear-day baseline + data-driven event multiplier + 90% normal interval (erf)",
        "label": "Simulated Demo Impact",
    }


@app.get("/api/intraday/{route_id}")
def get_intraday_profile(
    route_id: str,
    spike: bool = Query(False),
    capacity_loss: bool = Query(False),
    apply: bool = Query(False),
    spare: int = Query(5),
):
    profile = []
    for w in WINDOWS:
        routes = build_forecasts(w, spike, capacity_loss)
        rec, scores = allocate_vehicles(routes, spare)
        target = next((r for r in routes if r["route_id"] == route_id), routes[0])
        before_m = assess_route(target, 0)
        after_m = assess_route(
            target,
            rec[target["route_id"]] if apply else 0,
            scores[target["route_id"]],
            before_m["utilization"],
        )
        profile.append({
            "hour": w,
            "baseline": target["mean_clear"],
            "predicted": after_m["predicted"],
            "lower": after_m["lower"],
            "upper": after_m["upper"],
            "capacity": after_m["capacity"],
            "utilization": after_m["utilization"],
            "risk": after_m["risk"],
            "probability": after_m["probability"],
        })
    return profile


@app.get("/api/history/{route_id}")
def get_route_history(
    route_id: str,
    hour: str = Query("08:00"),
    spike: bool = Query(False),
    capacity_loss: bool = Query(False),
    apply: bool = Query(False),
    spare: int = Query(5),
):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """SELECT date, hour, passengers, capacity, weather, event 
           FROM route_history
           WHERE route_id = ? AND hour = ?
           ORDER BY date ASC""",
        (route_id, hour),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    # Also build the 8-window profile for convenient dual consumption
    profile = get_intraday_profile(route_id, spike, capacity_loss, apply, spare)

    return {
        "route_id": route_id,
        "hour": hour,
        "history": rows,
        "profile": profile,
    }


@app.get("/api/backtest")
def run_backtest(hour: str = "08:00"):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT date FROM route_history ORDER BY date")
    all_dates = [r["date"] for r in cur.fetchall()]
    cutoff_date = all_dates[22]  # First 23 days for training

    cur.execute("SELECT route_id, route_name FROM route_baselines ORDER BY route_id")
    routes = [dict(r) for r in cur.fetchall()]

    results = []
    for r in routes:
        rid = r["route_id"]
        stats = compute_route_stats(conn, rid, hour, max_date=cutoff_date)

        cur.execute(
            """SELECT date, passengers, weather, event FROM route_history
               WHERE route_id = ? AND hour = ? AND date > ? ORDER BY date""",
            (rid, hour, cutoff_date),
        )
        test_rows = cur.fetchall()

        abs_errors = []
        pct_errors = []
        covered = 0

        for tr in test_rows:
            factor = 1.0
            widen = 1.0
            if tr["event"] != "None":
                factor = stats["event_factor"]
                widen = 1.5
            elif tr["weather"] == "Rain":
                factor = stats["rain_factor"]

            pred = stats["mean_clear"] * factor
            sigma = max(stats["sd_clear"] * factor * widen, 0.04 * pred, 1.0)
            lower = pred - 1.645 * sigma
            upper = pred + 1.645 * sigma

            actual = tr["passengers"]
            err = abs(actual - pred)
            abs_errors.append(err)
            pct_errors.append(err / max(actual, 1))
            if lower <= actual <= upper:
                covered += 1

        n = max(1, len(test_rows))
        results.append({
            "route_id": rid,
            "route_name": r["route_name"],
            "mae": round(sum(abs_errors) / n, 1),
            "mape": round((sum(pct_errors) / n) * 100.0, 1),
            "coverage_90": round((covered / n) * 100.0, 1),
            "samples": n,
        })

    conn.close()
    return {"hour": hour, "train_days": 23, "test_days": 7, "routes": results}


@app.get("/api/log")
def get_allocation_log():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM allocation_log ORDER BY id DESC LIMIT 20")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


@app.post("/api/reset")
def reset_log():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM allocation_log")
    conn.commit()
    conn.close()
    return {"status": "reset"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
