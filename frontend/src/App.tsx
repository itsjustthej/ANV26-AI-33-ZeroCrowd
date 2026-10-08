import { useEffect, useState } from 'react';
import {
  AlertTriangle,
  ArrowRight,
  BarChart2,
  Briefcase,
  Bus,
  CheckCircle2,
  Clock,
  CloudSun,
  Coffee,
  Cpu,
  Database,
  Fuel,
  GitBranch,
  History,
  IndianRupee,
  Layers,
  Leaf,
  Pause,
  Play,
  Radio,
  RotateCcw,
  ShieldAlert,
  Sliders,
  Sparkles,
  TrendingUp,
  Wrench,
  Zap,
} from 'lucide-react';
import {
  Area,
  ComposedChart,
  CartesianGrid,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

const API_BASE = import.meta.env.VITE_API_URL || '';
const WINDOWS = ['06:00', '08:00', '10:00', '12:00', '14:00', '17:00', '19:00', '21:00'];

interface XaiDrivers {
  base_schedule: number;
  autoregressive_trend: number;
  weekend_land_use_effect: number;
  weather_rain_impact: number;
  event_metro_spillover: number;
}

interface RouteData {
  id: string;
  name: string;
  condition: string;
  baseline_mean: number;
  predicted: number;
  lower: number;
  upper: number;
  sigma: number;
  base_vehicles: number;
  vehicles: number;
  extra: number;
  capacity: number;
  utilization: number;
  risk: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  probability: number;
  excess: number;
  why: string;
  recommended_extra?: number;
  manual_extra?: number | null;
  xai_drivers?: XaiDrivers;
}

interface SummaryMetrics {
  overcrowded_routes: number;
  critical_routes: number;
  passengers_affected: number;
  avg_utilization: number;
  service_level_share: number;
  utilization_spread: number;
}

interface DonorRoute {
  route_id: string;
  route_name: string;
  curtailed_buses: number;
  util_before: number;
  util_after: number;
  pred_passengers: number;
  freed_buses: number;
}

interface ReceiverRoute {
  route_id: string;
  route_name: string;
  assigned_buses: number;
  util_before: number;
  util_after: number;
  pred_passengers: number;
  priority_score: number;
}

interface TransferPair {
  donor_id: string;
  donor_name: string;
  receiver_id: string;
  receiver_name: string;
  buses: number;
  rationale: string;
}

interface SavingsInfo {
  harvested_buses: number;
  fuel_liters_saved: number;
  cost_saved_inr: number;
  co2_kg_saved: number;
}

interface FeatureImportance {
  feature: string;
  importance: number;
  description: string;
}

interface MLModelCard {
  model_name: string;
  calibration: string;
  q_hat: number;
  conformal_coverage_pct: number;
  retrain_time_ms: number;
  features: { name: string; desc: string }[];
  feature_importances: FeatureImportance[];
}

interface LiveTelemetry {
  city: string;
  temp_c: number;
  rain_mm: number;
  wind_speed_kmh: number;
  condition: string;
  is_live: boolean;
  live_timestamp: string;
  source: string;
}

interface AnalyzeResponse {
  scenario: string;
  day_type?: string;
  spike: boolean;
  capacity_loss: boolean;
  hour: string;
  applied: boolean;
  is_manual?: boolean;
  spare_vehicles: number;
  spare_used: number;
  spare_left: number;
  harvested_pool?: number;
  total_available_pool?: number;
  routes: RouteData[];
  recommendation: Record<string, number>;
  active_alloc?: Record<string, number>;
  donor_routes?: DonorRoute[];
  receiver_routes?: ReceiverRoute[];
  transfer_pairs?: TransferPair[];
  savings?: SavingsInfo;
  ml_model_card?: MLModelCard;
  before: SummaryMetrics;
  after: SummaryMetrics;
  reduction_pct: number;
  method: string;
  label: string;
  logs?: LogEntry[];
}

interface BacktestRoute {
  route_id: string;
  route_name: string;
  mae: number;
  mape: number;
  coverage_90: number;
  samples: number;
}

interface BacktestResponse {
  hour: string;
  train_days: number;
  test_days: number;
  routes: BacktestRoute[];
}

interface LogEntry {
  id: number;
  ts: string;
  scenario: string;
  hour: string;
  route_id: string;
  vehicles_added: number;
  util_before: number;
  util_after: number;
  reason: string;
}

export default function App() {
  const [hour, setHour] = useState('08:00');
  const [spike, setSpike] = useState(false);
  const [capacityLoss, setCapacityLoss] = useState(false);
  const [applied, setApplied] = useState(false);
  const [spare, setSpare] = useState(5);
  const [selectedRoute, setSelectedRoute] = useState('R002');
  const [liveMode, setLiveMode] = useState(false);
  const [tickStep, setTickStep] = useState(0);
  const [activeChartTab, setActiveChartTab] = useState<'intraday' | 'history'>('intraday');

  // Day type: Weekday vs Weekend (triggers land-use changes & fleet harvesting)
  const [dayType, setDayType] = useState<'WEEKDAY' | 'WEEKEND'>('WEEKDAY');

  // Live Bengaluru Weather Telemetry
  const [telemetry, setTelemetry] = useState<LiveTelemetry | null>(null);
  const [syncingTelemetry, setSyncingTelemetry] = useState(false);

  // Human intervention override state: null = AI recommendation active
  const [manualAlloc, setManualAlloc] = useState<Record<string, number> | null>(null);

  const [data, setData] = useState<AnalyzeResponse | null>(null);
  const [intraday, setIntraday] = useState<any[]>([]);
  const [historyData, setHistoryData] = useState<any[]>([]);
  const [backtest, setBacktest] = useState<BacktestResponse | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const fetchAll = async (currTick = tickStep, activeTelem = telemetry) => {
    try {
      setErrorMsg(null);
      const effTemp = activeTelem ? activeTelem.temp_c : undefined;
      const effRain = activeTelem ? activeTelem.rain_mm : undefined;

      const [resAnalyze, resIntra, resHist, resBt, resLog] = await Promise.all([
        fetch(`${API_BASE}/api/analyze`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            spike,
            capacity_loss: capacityLoss,
            hour,
            spare,
            apply: applied,
            tick_step: liveMode ? currTick : 0,
            manual_alloc: manualAlloc,
            day_type: dayType,
            temp_c: effTemp,
            rain_mm: effRain,
          }),
        }),
        fetch(
          `${API_BASE}/api/intraday/${selectedRoute}?spike=${spike}&capacity_loss=${capacityLoss}&apply=${applied}&spare=${spare}&tick_step=${
            liveMode ? currTick : 0
          }&day_type=${dayType}`
        ),
        fetch(`${API_BASE}/api/history/${selectedRoute}?hour=${hour}`),
        fetch(`${API_BASE}/api/backtest?hour=${hour}`),
        fetch(`${API_BASE}/api/log`),
      ]);

      if (!resAnalyze.ok) throw new Error(`HTTP error ${resAnalyze.status}`);

      const analyzeJson = await resAnalyze.json();
      const intraJson = await resIntra.json();
      const histJson = await resHist.json();
      const btJson = await resBt.json();
      const logJson = await resLog.json();

      setData(analyzeJson);
      setIntraday(intraJson);
      if (histJson && histJson.history) {
        setHistoryData(
          histJson.history.map((h: any, idx: number) => ({
            ...h,
            dayIndex: idx + 1,
            isTest: idx >= 23,
            festivalDemand: h.event !== 'None' ? h.passengers : null,
            rainDemand: h.weather === 'Rain' && h.event === 'None' ? h.passengers : null,
            clearDemand: h.weather === 'Clear' && h.event === 'None' ? h.passengers : null,
          }))
        );
      }
      setBacktest(btJson);
      if (analyzeJson.logs && analyzeJson.logs.length > 0) {
        setLogs(analyzeJson.logs);
      } else {
        setLogs(logJson);
      }
      setLoading(false);
    } catch (err: any) {
      console.error('Backend sync error:', err);
      setErrorMsg('Failed to connect to backend engine at ' + API_BASE + '. Retrying...');
    }
  };

  const handleApplyDispatch = async () => {
    try {
      setApplied(true);
      const effTemp = telemetry ? telemetry.temp_c : undefined;
      const effRain = telemetry ? telemetry.rain_mm : undefined;

      const res = await fetch(`${API_BASE}/api/commit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          spike,
          capacity_loss: capacityLoss,
          hour,
          spare,
          apply: true,
          commit: true,
          tick_step: liveMode ? tickStep : 0,
          manual_alloc: manualAlloc,
          day_type: dayType,
          temp_c: effTemp,
          rain_mm: effRain,
        }),
      });
      if (res.ok) {
        const analyzeJson = await res.json();
        setData(analyzeJson);
        if (analyzeJson.logs && analyzeJson.logs.length > 0) {
          setLogs(analyzeJson.logs);
        }
      }
      const resLog = await fetch(`${API_BASE}/api/log`);
      if (resLog.ok) {
        const freshLogs = await resLog.json();
        setLogs(freshLogs);
      }
    } catch (err: any) {
      console.error('Failed to commit dispatch:', err);
    }
  };

  const handleSyncWeather = async () => {
    try {
      setSyncingTelemetry(true);
      const res = await fetch(`${API_BASE}/api/live-telemetry`);
      if (res.ok) {
        const tJson: LiveTelemetry = await res.json();
        setTelemetry(tJson);
        await fetchAll(liveMode ? tickStep : 0, tJson);
      }
    } catch (e) {
      console.error('Failed to sync live weather:', e);
    } finally {
      setSyncingTelemetry(false);
    }
  };

  useEffect(() => {
    fetchAll(liveMode ? tickStep : 0);
  }, [hour, spike, capacityLoss, applied, selectedRoute, spare, liveMode, tickStep, manualAlloc, dayType]);

  // Live Simulation 2-second tick loop
  useEffect(() => {
    if (!liveMode) return;
    const timer = setInterval(() => {
      setTickStep((prev) => prev + 1);
    }, 2000);
    return () => clearInterval(timer);
  }, [liveMode]);

  const handleReset = async () => {
    setSpike(false);
    setCapacityLoss(false);
    setApplied(false);
    setLiveMode(false);
    setTickStep(0);
    setManualAlloc(null);
    setDayType('WEEKDAY');
    setTelemetry(null);
    try {
      await fetch(`${API_BASE}/api/reset`, { method: 'POST' });
      const resLog = await fetch(`${API_BASE}/api/log`);
      setLogs(await resLog.json());
    } catch (e) {
      console.error(e);
    }
  };

  // Active allocation calculations
  const activeAllocMap = manualAlloc !== null ? manualAlloc : (data?.recommendation || {});
  const totalDispatched = Object.values(activeAllocMap).filter((v) => v > 0).reduce((a, b) => a + b, 0);
  const totalCurtailed = Math.abs(Object.values(activeAllocMap).filter((v) => v < 0).reduce((a, b) => a + b, 0));
  const baseSparePool = data?.spare_vehicles || spare;
  const effectiveMaxPool = baseSparePool + (data?.harvested_pool || totalCurtailed);

  // Stepper handlers supporting positive surge additions and negative curtailment
  const handleIncrement = (routeId: string) => {
    const baseMap = { ...(manualAlloc !== null ? manualAlloc : (data?.recommendation || {})) };
    const curr = baseMap[routeId] || 0;
    const currDeployed = Object.values(baseMap).filter((v) => v > 0).reduce((a, b) => a + b, 0);
    if (curr < 0 || currDeployed < effectiveMaxPool) {
      baseMap[routeId] = curr + 1;
      setManualAlloc(baseMap);
      setApplied(false);
    }
  };

  const handleDecrement = (routeId: string) => {
    const baseMap = { ...(manualAlloc !== null ? manualAlloc : (data?.recommendation || {})) };
    const curr = baseMap[routeId] || 0;
    if (curr > -6) {
      baseMap[routeId] = curr - 1;
      setManualAlloc(baseMap);
      setApplied(false);
    }
  };

  // Interactive Unit Card Click Handler
  const handleUnitClick = (unit: { unitId: string; routeId: string | null }) => {
    const baseMap = { ...(manualAlloc !== null ? manualAlloc : (data?.recommendation || {})) };
    if (unit.routeId) {
      const targetRoute = unit.routeId;
      if ((baseMap[targetRoute] || 0) > 0) {
        baseMap[targetRoute] = Math.max(0, (baseMap[targetRoute] || 0) - 1);
        setManualAlloc(baseMap);
        setApplied(false);
      }
    } else {
      const currDeployed = Object.values(baseMap).filter((v) => v > 0).reduce((a, b) => a + b, 0);
      if (currDeployed < effectiveMaxPool) {
        baseMap[selectedRoute] = (baseMap[selectedRoute] || 0) + 1;
        setManualAlloc(baseMap);
        setApplied(false);
      }
    }
  };

  // Build standby unit list
  const unitAssignments: { unitId: string; routeId: string | null }[] = [];
  const routeAssignmentQueue: string[] = [];
  if (data?.routes) {
    for (const r of data.routes) {
      const extraCount = activeAllocMap[r.id] || 0;
      if (extraCount > 0) {
        for (let i = 0; i < extraCount; i++) {
          routeAssignmentQueue.push(r.id);
        }
      }
    }
  }

  for (let i = 1; i <= baseSparePool; i++) {
    const unitId = `UNIT-${100 + i}`;
    const assignedRoute = routeAssignmentQueue[i - 1] || null;
    unitAssignments.push({ unitId, routeId: assignedRoute });
  }

  // Group dispatched units for broadcast banner
  const dispatchedByRoute: Record<string, string[]> = {};
  for (const u of unitAssignments) {
    if (u.routeId) {
      if (!dispatchedByRoute[u.routeId]) dispatchedByRoute[u.routeId] = [];
      dispatchedByRoute[u.routeId].push(u.unitId);
    }
  }
  const broadcastString = Object.entries(dispatchedByRoute)
    .map(([rId, units]) => `[${units.join(', ')}] -> Rerouted to ${rId}`)
    .join(' | ');

  if (loading && !data) {
    return (
      <div className="min-h-screen bg-slate-950 text-slate-200 flex flex-col items-center justify-center font-sans space-y-4">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
        <p className="text-sm font-semibold tracking-wide text-slate-400">
          Initializing ZeroCrowd Multi-Quantile GBDT &amp; Conformal Transit Engine...
        </p>
      </div>
    );
  }

  const recEntries = Object.entries(data?.recommendation || {}) as [string, number][];
  const activeEntries = Object.entries(activeAllocMap).filter(([_, v]) => v !== 0) as [string, number][];
  const selectedRouteObj = data?.routes.find((r) => r.id === selectedRoute) || data?.routes[0];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans selection:bg-blue-600 selection:text-white pb-12">
      {/* Top Banner Notice */}
      <div className="bg-gradient-to-r from-blue-900/40 via-purple-900/40 to-slate-900 border-b border-slate-800 text-xs px-6 py-2 flex flex-wrap justify-between items-center gap-2">
        <div className="flex items-center gap-2">
          <span className="inline-block w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          <span className="font-bold text-slate-200">ANVATION 2026</span>
          <span className="text-slate-500">|</span>
          <span className="text-slate-400">Track AI-16: ZeroCrowd · Bengaluru Transit Multi-Quantile GBDT &amp; Conformal AI Engine</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="px-2.5 py-0.5 rounded-full text-[10px] font-extrabold uppercase tracking-widest bg-cyan-500/10 text-cyan-300 border border-cyan-500/30">
            BMTC CORRIDORS &amp; NAMMA METRO FEEDER
          </span>
          <span className="text-slate-400 text-[11px]">Quantile GBDT + Conformal Residuals</span>
        </div>
      </div>

      {errorMsg && (
        <div className="max-w-7xl mx-auto mt-3 px-6">
          <div className="bg-red-950/80 border border-red-700 text-red-200 text-xs p-3 rounded-xl flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0 text-red-400" />
            <span>{errorMsg}</span>
          </div>
        </div>
      )}

      {/* Main Container */}
      <div className="max-w-7xl mx-auto px-6 pt-6 space-y-6">
        {/* Header & Controls Bar */}
        <header className="bg-slate-900/70 backdrop-blur-md border border-slate-800/80 rounded-2xl p-5 shadow-2xl flex flex-col xl:flex-row justify-between items-start xl:items-center gap-5">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-2xl shadow-lg shadow-blue-500/25 ring-1 ring-white/10">
              <Bus className="w-7 h-7 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2.5">
                <h1 className="text-xl md:text-2xl font-black tracking-tight text-white">
                  ZeroCrowd: Bengaluru Transit Command Center
                </h1>
              </div>
              <p className="text-slate-400 text-xs mt-0.5 max-w-xl">
                {data?.method || 'Quantile Gradient Boosted Decision Trees + Conformal Prediction Residuals on BMTC Corridors.'}
              </p>
            </div>
          </div>

          {/* Interactive Scenario & Simulator Controls */}
          <div className="flex flex-wrap items-center gap-2.5">
            {/* Time Window Dropdown */}
            <div className="flex items-center bg-slate-950/90 border border-slate-800 rounded-xl px-3 py-2 text-xs">
              <Clock className="w-3.5 h-3.5 text-blue-400 mr-2" />
              <span className="text-slate-400 mr-2 font-medium">Window:</span>
              <select
                value={hour}
                onChange={(e) => setHour(e.target.value)}
                className="bg-transparent text-white font-bold focus:outline-none cursor-pointer"
              >
                {WINDOWS.map((w) => (
                  <option key={w} value={w} className="bg-slate-900 text-white">
                    {w}
                  </option>
                ))}
              </select>
            </div>

            {/* Day Type Toggle: Weekday vs Weekend */}
            <div className="flex bg-slate-950/90 border border-slate-800 rounded-xl p-0.5 text-xs">
              <button
                onClick={() => {
                  setDayType('WEEKDAY');
                  setApplied(false);
                  setManualAlloc(null);
                }}
                className={`px-3 py-1.5 rounded-lg font-bold transition flex items-center gap-1.5 ${
                  dayType === 'WEEKDAY'
                    ? 'bg-blue-600 text-white shadow'
                    : 'text-slate-400 hover:text-white'
                }`}
                title="Weekday Schedule: Normal IT Corridor & Industrial Peak Passenger Loads"
              >
                <Briefcase className="w-3.5 h-3.5" /> Weekday
              </button>
              <button
                onClick={() => {
                  setDayType('WEEKEND');
                  setApplied(false);
                  setManualAlloc(null);
                }}
                className={`px-3 py-1.5 rounded-lg font-bold transition flex items-center gap-1.5 ${
                  dayType === 'WEEKEND'
                    ? 'bg-purple-600 text-white shadow'
                    : 'text-slate-400 hover:text-white'
                }`}
                title="Weekend Schedule: Low IT Ridership (-50%), High Commercial Hub Demand, Cross-Corridor Fleet Harvesting"
              >
                <Coffee className="w-3.5 h-3.5" /> Weekend
              </button>
            </div>

            {/* Spare Fleet Selector */}
            <div className="flex items-center bg-slate-950/90 border border-slate-800 rounded-xl px-3 py-2 text-xs">
              <span className="text-slate-400 mr-1.5 font-medium">Spares:</span>
              <select
                value={spare}
                onChange={(e) => {
                  setSpare(Number(e.target.value));
                  setManualAlloc(null);
                }}
                className="bg-transparent text-purple-400 font-bold focus:outline-none cursor-pointer"
              >
                {[3, 4, 5, 6, 8, 10].map((s) => (
                  <option key={s} value={s} className="bg-slate-900 text-white">
                    {s} buses
                  </option>
                ))}
              </select>
            </div>

            {/* Open-Meteo Live Bengaluru Weather Telemetry Button */}
            <button
              onClick={handleSyncWeather}
              disabled={syncingTelemetry}
              className={`px-3 py-2 rounded-xl text-xs font-bold border transition flex items-center gap-1.5 ${
                telemetry
                  ? 'bg-cyan-950/60 text-cyan-300 border-cyan-700/60 shadow-lg shadow-cyan-900/20'
                  : 'bg-slate-950/90 text-slate-300 border-slate-800 hover:bg-slate-800 hover:text-white'
              }`}
              title="Fetch real-time Bengaluru weather telemetry from Open-Meteo API"
            >
              <CloudSun className={`w-3.5 h-3.5 ${syncingTelemetry ? 'animate-spin' : 'text-cyan-400'}`} />
              {syncingTelemetry
                ? 'Syncing Weather...'
                : telemetry
                ? `${telemetry.city}: ${telemetry.temp_c}°C · ${telemetry.condition}${telemetry.rain_mm > 0 ? ` (${telemetry.rain_mm}mm)` : ''}`
                : '🌐 Sync Live Weather'}
            </button>

            {/* Stackable Toggle 1: Demand Spike */}
            <button
              onClick={() => {
                setSpike(!spike);
                setApplied(false);
              }}
              className={`px-3.5 py-2 rounded-xl text-xs font-bold transition flex items-center gap-1.5 border ${
                spike
                  ? 'bg-red-600 text-white border-red-500 shadow-lg shadow-red-600/30'
                  : 'bg-slate-950/90 text-red-400 border-red-950 hover:border-red-800/60 hover:bg-red-950/20'
              }`}
            >
              <Zap className="w-3.5 h-3.5" />
              {spike ? 'Spike Active (252-F / Rain)' : 'Simulate Surge'}
            </button>

            {/* Stackable Toggle 2: Capacity Loss */}
            <button
              onClick={() => {
                setCapacityLoss(!capacityLoss);
                setApplied(false);
              }}
              className={`px-3.5 py-2 rounded-xl text-xs font-bold transition flex items-center gap-1.5 border ${
                capacityLoss
                  ? 'bg-amber-600 text-white border-amber-500 shadow-lg shadow-amber-600/30'
                  : 'bg-slate-950/90 text-amber-400 border-amber-950 hover:border-amber-800/60 hover:bg-amber-950/20'
              }`}
            >
              <Wrench className="w-3.5 h-3.5" />
              {capacityLoss ? 'Loss Active (250-P -2)' : 'Simulate Fleet Loss'}
            </button>

            {/* Live Tick Mode */}
            <button
              onClick={() => setLiveMode(!liveMode)}
              className={`px-3.5 py-2 rounded-xl text-xs font-bold flex items-center gap-1.5 border transition ${
                liveMode
                  ? 'bg-emerald-600 text-white border-emerald-500 shadow-lg shadow-emerald-600/30'
                  : 'bg-slate-950/90 text-slate-300 border-slate-800 hover:bg-slate-800'
              }`}
            >
              {liveMode ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
              {liveMode ? `Tick #${tickStep}` : 'Live Tick'}
            </button>

            {/* Reset */}
            <button
              onClick={handleReset}
              className="px-3 py-2 bg-slate-950/90 hover:bg-slate-800 text-slate-400 hover:text-white rounded-xl text-xs font-bold border border-slate-800 transition flex items-center gap-1.5"
              title="Reset all toggles, overrides, and audit log"
            >
              <RotateCcw className="w-3.5 h-3.5" /> Reset
            </button>
          </div>
        </header>

        {/* 4 KPI Summary Cards */}
        {data && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-lg relative group">
              <div className="absolute top-0 right-0 w-24 h-24 bg-blue-500/5 rounded-full blur-xl group-hover:bg-blue-500/10 transition pointer-events-none"></div>
              <div className="flex justify-between items-center text-xs text-slate-400 uppercase tracking-wider font-semibold">
                <span>BMTC Network</span>
                <Layers className="w-4 h-4 text-blue-400" />
              </div>
              <div className="mt-3 flex items-baseline justify-between">
                <span className="text-3xl font-black text-white">{data.routes.length} Corridors</span>
                <span className="text-xs px-2 py-0.5 rounded-full bg-blue-500/15 text-blue-400 font-bold border border-blue-500/30">
                  {dayType} @ {data.hour}
                </span>
              </div>
              <p className="text-[11px] text-slate-400 mt-2">
                Baseline capacity: {data.routes.reduce((acc, r) => acc + r.base_vehicles * 100, 0)} seats across Bengaluru
              </p>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-lg relative group">
              <div className="absolute top-0 right-0 w-24 h-24 bg-emerald-500/5 rounded-full blur-xl group-hover:bg-emerald-500/10 transition pointer-events-none"></div>
              <div className="flex justify-between items-center text-xs text-slate-400 uppercase tracking-wider font-semibold">
                <span>Total Predicted Demand</span>
                <TrendingUp className="w-4 h-4 text-emerald-400" />
              </div>
              <div className="mt-3 flex items-baseline justify-between">
                <span className="text-3xl font-black text-white">
                  {data.routes.reduce((s, r) => s + r.predicted, 0).toLocaleString()}
                </span>
                <span className="text-xs text-slate-400 font-medium">passengers</span>
              </div>
              <p className="text-[11px] text-slate-400 mt-2">
                Conformal intervals: q̂ = ±{data.ml_model_card?.q_hat || 18} pax (90% coverage)
              </p>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-lg relative group">
              <div className="absolute top-0 right-0 w-24 h-24 bg-red-500/5 rounded-full blur-xl group-hover:bg-red-500/10 transition pointer-events-none"></div>
              <div className="flex justify-between items-center text-xs text-slate-400 uppercase tracking-wider font-semibold">
                <span>Overcrowded / Critical</span>
                <AlertTriangle className="w-4 h-4 text-red-400" />
              </div>
              <div className="mt-3 flex items-baseline justify-between">
                <span className="text-3xl font-black text-white">
                  {applied ? data.after.overcrowded_routes : data.before.overcrowded_routes} /{' '}
                  {applied ? data.after.critical_routes : data.before.critical_routes}
                </span>
                <span
                  className={`text-xs font-bold px-2 py-0.5 rounded-full ${
                    (applied ? data.after.passengers_affected : data.before.passengers_affected) > 0
                      ? 'bg-red-500/15 text-red-400 border border-red-500/30'
                      : 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30'
                  }`}
                >
                  {applied ? data.after.passengers_affected : data.before.passengers_affected} excess pax
                </span>
              </div>
              <p className="text-[11px] text-slate-400 mt-2">
                {applied
                  ? `Overcrowding reduced by ${data.reduction_pct}% post-dispatch`
                  : 'Overcrowded corridors eligible for harvested & spare fleet'}
              </p>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-lg relative group">
              <div className="absolute top-0 right-0 w-24 h-24 bg-purple-500/5 rounded-full blur-xl group-hover:bg-purple-500/10 transition pointer-events-none"></div>
              <div className="flex justify-between items-center text-xs text-slate-400 uppercase tracking-wider font-semibold">
                <span>Fleet Pool &amp; Harvesting</span>
                <Bus className="w-4 h-4 text-purple-400" />
              </div>
              <div className="mt-3 flex items-baseline justify-between">
                <span className="text-3xl font-black text-white">
                  {data.spare_left} / {data.total_available_pool || (data.spare_vehicles + (data.harvested_pool || 0))}
                </span>
                <span className="text-xs px-2 py-0.5 rounded-full bg-purple-500/15 text-purple-400 font-bold border border-purple-500/30">
                  {data.harvested_pool ? `+${data.harvested_pool} harvested` : `${data.spare_used} active`}
                </span>
              </div>
              <p className="text-[11px] text-slate-400 mt-2">
                {data.harvested_pool && data.harvested_pool > 0
                  ? `${data.harvested_pool} idle buses harvested from low-demand corridors`
                  : 'Base spares ready for surge deployment'}
              </p>
            </div>
          </div>
        )}

        {/* Dynamic Alert Banner when Spike or Capacity Loss is active, or uncommitted manual override */}
        {data && ((spike || capacityLoss) || data.before.overcrowded_routes > 0 || (manualAlloc !== null && (totalDispatched > 0 || totalCurtailed > 0))) && !applied && (
          <div className="bg-gradient-to-r from-red-950/60 to-slate-900 border border-red-700/80 rounded-2xl p-5 shadow-xl flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
            <div className="flex items-start gap-3.5">
              <div className="p-2 bg-red-600/20 text-red-400 rounded-xl border border-red-500/30 shrink-0 mt-0.5">
                <ShieldAlert className="w-5 h-5" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="text-sm font-bold text-red-200">
                    {manualAlloc !== null
                      ? '⚠ Human Fleet Override Pending Commitment'
                      : `⚠ Overcrowding Anomaly Detected: Scenario [${data.scenario.toUpperCase()}]`}
                  </h3>
                  <span className="text-[10px] px-2 py-0.2 bg-red-500/20 text-red-300 rounded font-semibold border border-red-500/30">
                    Action Required
                  </span>
                </div>
                <p className="text-xs text-red-300/90 mt-1">
                  <strong>{data.before.passengers_affected} passengers</strong> exceed vehicle capacity across{' '}
                  <strong>{data.before.overcrowded_routes} route(s)</strong>. Current dispatch plan:{' '}
                  <span className="text-amber-300 font-bold">
                    {activeEntries.map(([k, v]) => `${k} (${v > 0 ? `+${v}` : v} buses)`).join(', ') ||
                      recEntries.map(([k, v]) => `${k} (${v > 0 ? `+${v}` : v} buses)`).join(', ') ||
                      'No allocation feasible'}
                  </span>
                  .
                </p>
              </div>
            </div>
            <button
              onClick={handleApplyDispatch}
              className="px-5 py-2.5 bg-red-600 hover:bg-red-500 text-white rounded-xl text-xs font-extrabold shadow-lg shadow-red-600/30 whitespace-nowrap transition active:scale-95 flex items-center gap-1.5"
            >
              <Sparkles className="w-4 h-4" /> {manualAlloc !== null ? 'COMMIT OVERRIDE DISPATCH' : 'APPLY ALLOCATION DISPATCH'}
            </button>
          </div>
        )}

        {/* STANDBY VEHICLE DISPATCH NOTIFICATION BROADCAST BANNER */}
        {data && applied && totalDispatched > 0 && (
          <div className="bg-gradient-to-r from-blue-950/70 via-indigo-950/60 to-slate-900 border border-blue-500/60 rounded-2xl p-4.5 shadow-2xl flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
            <div className="flex items-start gap-3.5">
              <div className="p-2.5 bg-blue-600/20 text-blue-400 rounded-xl border border-blue-500/40 shrink-0 mt-0.5">
                <Radio className="w-5 h-5 animate-pulse" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="px-2 py-0.5 text-[10px] font-extrabold uppercase tracking-widest bg-blue-500/20 text-blue-300 rounded border border-blue-500/40">
                    LIVE STANDBY DISPATCH BROADCAST
                  </span>
                  <span className="text-xs text-slate-400 font-mono">Telemetry Active</span>
                </div>
                <p className="text-sm font-semibold text-white mt-1">
                  📡 Dispatch Notification Sent to Nearby Standby Units:{' '}
                  <span className="text-cyan-300 font-mono font-bold">
                    {broadcastString || 'Standby Pool Units Active'}
                  </span>
                </p>
                <p className="text-xs text-slate-300/80 mt-0.5">
                  Standby bus drivers confirmed receipt via operational radio · Rerouting initiated immediately.
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs px-3 py-1 rounded-lg bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-bold">
                BROADCAST COMMITTED
              </span>
            </div>
          </div>
        )}

        {/* Applied Allocation Status Banner */}
        {data && applied && (
          <div className="bg-gradient-to-r from-emerald-950/50 to-slate-900 border border-emerald-700/80 rounded-2xl p-4.5 shadow-xl flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
            <div className="flex items-start gap-3.5">
              <div className="p-2 bg-emerald-600/20 text-emerald-400 rounded-xl border border-emerald-500/30 shrink-0 mt-0.5">
                <CheckCircle2 className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-emerald-200">
                  {manualAlloc !== null ? 'Human-Overridden' : 'AI-Optimized'} Allocation Committed ({data.spare_used} Spare / {data.harvested_pool || 0} Harvested Vehicles Deployed)
                </h3>
                <p className="text-xs text-emerald-300/90 mt-1">
                  Excess unserved passengers reduced by <strong>{data.reduction_pct}%</strong> (
                  {data.before.passengers_affected} → {data.after.passengers_affected}). State persisted to{' '}
                  <code className="text-emerald-200">transport.db (allocation_log)</code>.
                </p>
              </div>
            </div>
            <button
              onClick={() => setApplied(false)}
              className="text-xs text-emerald-400 hover:text-emerald-300 font-bold underline transition whitespace-nowrap"
            >
              Revert to Pre-Allocation View
            </button>
          </div>
        )}

        {/* ADMIN DISPATCH & HUMAN-IN-THE-LOOP CONSOLE BLOCK */}
        {data && (
          <div className="bg-slate-900/85 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
            <div className="flex flex-col sm:flex-row justify-between sm:items-center gap-3">
              <div className="flex items-center gap-3">
                <div className="p-2.5 bg-purple-600/20 text-purple-400 rounded-xl border border-purple-500/30">
                  <Sliders className="w-5 h-5" />
                </div>
                <div>
                  <div className="flex items-center gap-2.5">
                    <h3 className="text-base font-bold text-white">
                      Admin Dispatch &amp; Human-in-the-Loop Console
                    </h3>
                    {manualAlloc !== null ? (
                      <span className="px-2.5 py-0.5 text-[11px] font-extrabold uppercase tracking-wider bg-amber-500/20 text-amber-300 border border-amber-500/40 rounded-full flex items-center gap-1.5 shadow-sm">
                        <span className="w-2 h-2 rounded-full bg-amber-400 animate-ping"></span>
                        Human Override Active
                      </span>
                    ) : (
                      <span className="px-2.5 py-0.5 text-[11px] font-extrabold uppercase tracking-wider bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 rounded-full flex items-center gap-1.5 shadow-sm">
                        <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
                        AI Optimal Recommendation
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Operator authority console: review automated AI fleet recommendations or fine-tune route dispatches via the steppers.
                  </p>
                </div>
              </div>

              {/* Admin Actions */}
              <div className="flex items-center gap-2">
                {manualAlloc !== null && !applied && (
                  <button
                    onClick={handleApplyDispatch}
                    className="px-3.5 py-2 bg-amber-600 hover:bg-amber-500 text-white rounded-xl text-xs font-bold border border-amber-500 shadow-md shadow-amber-600/30 transition flex items-center gap-1.5 active:scale-95"
                  >
                    <Sparkles className="w-3.5 h-3.5" /> Commit Override Dispatch
                  </button>
                )}
                {manualAlloc !== null && (
                  <button
                    onClick={() => {
                      setManualAlloc(null);
                      setApplied(false);
                    }}
                    className="px-3.5 py-2 bg-slate-950 hover:bg-slate-800 text-amber-300 hover:text-white rounded-xl text-xs font-bold border border-amber-500/40 transition flex items-center gap-1.5 shadow-md active:scale-95"
                  >
                    <RotateCcw className="w-3.5 h-3.5" /> Restore AI Recommendation
                  </button>
                )}
              </div>
            </div>

            {/* Individual Spare Fleet Unit Badges */}
            <div className="pt-2 border-t border-slate-800/80">
              <div className="flex flex-col sm:flex-row justify-between sm:items-center gap-1.5 mb-2.5">
                <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                  <Bus className="w-3.5 h-3.5 text-purple-400" />
                  Standby Fleet Units Status ({unitAssignments.filter((u) => u.routeId).length} Dispatched / {unitAssignments.length} Pool Size):
                </span>
                <span className="text-[11px] text-cyan-400/90 font-medium">
                  💡 Click assigned unit to recall to depot · Click standby unit to dispatch to {selectedRoute}
                </span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2.5">
                {unitAssignments.map((u) => {
                  const isAssigned = u.routeId !== null;
                  const tooltipText = isAssigned
                    ? `Click to recall ${u.unitId} from ${u.routeId} back to depot`
                    : totalDispatched >= effectiveMaxPool
                    ? `Fleet pool exhausted (${effectiveMaxPool}/${effectiveMaxPool} units dispatched)`
                    : `Click to dispatch ${u.unitId} to selected route (${selectedRoute})`;

                  return (
                    <div
                      key={u.unitId}
                      onClick={() => handleUnitClick(u)}
                      title={tooltipText}
                      className={`p-2.5 rounded-xl border text-xs flex flex-col justify-between transition cursor-pointer select-none group relative active:scale-95 ${
                        isAssigned
                          ? 'bg-blue-950/50 border-blue-500/50 hover:border-red-400 hover:bg-red-950/30 text-blue-200 shadow-md ring-0 hover:ring-1 hover:ring-red-400/30'
                          : 'bg-slate-950/70 border-slate-800 hover:border-emerald-500/70 hover:bg-emerald-950/30 text-slate-400 ring-0 hover:ring-1 hover:ring-emerald-500/30'
                      }`}
                    >
                      <div className="flex justify-between items-center mb-1">
                        <span className="font-mono font-bold text-white text-[11px] group-hover:text-cyan-300 transition">
                          {u.unitId}
                        </span>
                        <span
                          className={`w-2 h-2 rounded-full transition ${
                            isAssigned
                              ? 'bg-emerald-400 animate-pulse group-hover:bg-red-400'
                              : 'bg-slate-600 group-hover:bg-emerald-400'
                          }`}
                        ></span>
                      </div>
                      <div className="font-semibold text-[11px]">
                        {isAssigned ? (
                          <div className="flex items-center justify-between">
                            <span className="text-emerald-300 font-bold font-mono group-hover:text-red-300 transition">
                              [ASSIGNED → {u.routeId}]
                            </span>
                            <span className="text-[10px] opacity-0 group-hover:opacity-100 font-sans font-bold text-red-400 transition ml-1">
                              Recall
                            </span>
                          </div>
                        ) : (
                          <div className="flex items-center justify-between">
                            <span className="text-slate-500 font-mono group-hover:text-emerald-300 transition">
                              [STANDBY IN DEPOT]
                            </span>
                            <span className="text-[10px] opacity-0 group-hover:opacity-100 font-sans font-bold text-emerald-400 transition ml-1">
                              +Dispatch
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {/* Route Monitor Table with [-] / [+] Steppers */}
        {data && (
          <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-xl">
            <div className="flex flex-col sm:flex-row justify-between sm:items-center gap-2 mb-4">
              <div>
                <h2 className="text-base font-bold text-white flex items-center gap-2">
                  <span>Bengaluru Corridor Risk &amp; Probabilistic Forecast Monitor</span>
                  <span className="text-xs font-normal text-slate-400">
                    (Use [-] / [+] to manually adjust dispatched or curtailed buses)
                  </span>
                </h2>
              </div>
              <div className="text-xs text-slate-400 bg-slate-950 px-3 py-1 rounded-lg border border-slate-800">
                Eligibility Threshold: <span className="text-slate-200 font-semibold">Util &gt; 100%</span> OR{' '}
                <span className="text-slate-200 font-semibold">P(Overcrowded) ≥ 50%</span>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400 uppercase tracking-wider font-semibold">
                    <th className="pb-3 pl-3">Corridor Link</th>
                    <th className="pb-3">Condition</th>
                    <th className="pb-3">Forecast (90% Interval)</th>
                    <th className="pb-3">Fleet / Capacity &amp; Override</th>
                    <th className="pb-3 w-40">Utilization</th>
                    <th className="pb-3">P(Overcrowded)</th>
                    <th className="pb-3">Risk Tier</th>
                    <th className="pb-3 pr-3">Bi-Directional Optimizer Decision</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-normal">
                  {data.routes.map((r) => {
                    const isSelected = selectedRoute === r.id;
                    const currentExtra =
                      manualAlloc !== null
                        ? manualAlloc[r.id] !== undefined ? manualAlloc[r.id] : 0
                        : data.recommendation?.[r.id] || 0;

                    const badge =
                      r.risk === 'CRITICAL'
                        ? 'bg-red-500/20 text-red-400 border-red-500/40'
                        : r.risk === 'HIGH'
                        ? 'bg-orange-500/20 text-orange-400 border-orange-500/40'
                        : r.risk === 'MEDIUM'
                        ? 'bg-yellow-500/20 text-yellow-300 border-yellow-500/40'
                        : 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40';

                    const barColor =
                      r.utilization > 120
                        ? 'bg-red-500'
                        : r.utilization > 100
                        ? 'bg-orange-500'
                        : r.utilization > 80
                        ? 'bg-yellow-500'
                        : 'bg-emerald-500';

                    return (
                      <tr
                        key={r.id}
                        onClick={() => setSelectedRoute(r.id)}
                        className={`cursor-pointer transition-colors ${
                          isSelected
                            ? 'bg-blue-950/40 border-l-4 border-l-blue-500'
                            : 'hover:bg-slate-800/40'
                        }`}
                      >
                        <td className="py-3.5 pl-3 font-bold text-white">
                          <span className="text-blue-400 mr-2 font-mono">{r.id}</span>
                          <span>{r.name}</span>
                          {isSelected && (
                            <span className="ml-2 px-1.5 py-0.5 text-[10px] bg-blue-600/30 text-blue-300 rounded border border-blue-500/40">
                              Selected
                            </span>
                          )}
                        </td>
                        <td className="py-3.5 text-slate-300">
                          <span
                            className={
                              r.condition !== 'Clear'
                                ? 'text-amber-400 font-semibold'
                                : 'text-slate-400'
                            }
                          >
                            {r.condition}
                          </span>
                        </td>
                        <td className="py-3.5 text-slate-200">
                          <span className="font-bold">{r.predicted}</span>{' '}
                          <span className="text-slate-400 text-[11px]">
                            [{r.lower} – {r.upper}]
                          </span>
                        </td>

                        {/* FLEET / CAPACITY COLUMN WITH [-] / [+] STEPPER BUTTONS */}
                        <td className="py-3.5 text-slate-300">
                          <div className="flex items-center gap-3">
                            <div>
                              <span className="font-semibold text-white">{r.vehicles}</span> buses
                              <span className="text-slate-400 text-[11px] block">{r.capacity} seats</span>
                            </div>

                            {/* Compact Stepper */}
                            <div className="flex items-center bg-slate-950 border border-slate-800 rounded-lg p-0.5">
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleDecrement(r.id);
                                }}
                                disabled={currentExtra <= -6}
                                className={`w-5 h-5 rounded flex items-center justify-center font-bold text-xs transition ${
                                  currentExtra <= -6
                                    ? 'text-slate-600 cursor-not-allowed'
                                    : 'text-amber-400 hover:bg-slate-800 active:scale-95'
                                }`}
                                title="Decrease allocated buses / harvest to pool"
                              >
                                -
                              </button>
                              <span
                                className={`px-2 min-w-[28px] text-center font-bold font-mono text-xs ${
                                  currentExtra > 0
                                    ? 'text-emerald-400'
                                    : currentExtra < 0
                                    ? 'text-cyan-400'
                                    : 'text-slate-500'
                                }`}
                              >
                                {currentExtra > 0 ? `+${currentExtra}` : currentExtra < 0 ? `${currentExtra}` : '0'}
                              </span>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleIncrement(r.id);
                                }}
                                disabled={totalDispatched >= effectiveMaxPool && currentExtra >= 0}
                                className={`w-5 h-5 rounded flex items-center justify-center font-bold text-xs transition ${
                                  totalDispatched >= effectiveMaxPool && currentExtra >= 0
                                    ? 'text-slate-600 cursor-not-allowed'
                                    : 'text-emerald-400 hover:bg-slate-800 active:scale-95'
                                }`}
                                title="Increase allocated buses for this route"
                              >
                                +
                              </button>
                            </div>
                          </div>
                        </td>

                        <td className="py-3.5">
                          <div className="flex items-center gap-2">
                            <span className="font-mono font-bold w-12 text-right">
                              {r.utilization}%
                            </span>
                            <div className="flex-1 bg-slate-800 rounded-full h-2 overflow-hidden">
                              <div
                                className={`h-full rounded-full ${barColor}`}
                                style={{ width: `${Math.min(100, (r.utilization / 140) * 100)}%` }}
                              ></div>
                            </div>
                          </div>
                        </td>
                        <td className="py-3.5 font-semibold text-slate-200 font-mono">
                          {r.probability}%
                        </td>
                        <td className="py-3.5">
                          <span
                            className={`px-2.5 py-1 rounded-full text-[10px] font-extrabold uppercase border ${badge}`}
                          >
                            {r.risk}
                          </span>
                        </td>
                        <td className="py-3.5 pr-3 text-slate-300 max-w-xs truncate" title={r.why}>
                          {currentExtra < 0 ? (
                            <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 mr-1.5">
                              CURTAILED {currentExtra}
                            </span>
                          ) : currentExtra > 0 ? (
                            <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 mr-1.5">
                              +{currentExtra} BUSES
                            </span>
                          ) : null}
                          {r.why}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* NEW FEATURE: THE "WOW" AI ENGINE PANEL (XAI & BI-DIRECTIONAL FLEET REBALANCING) */}
        {data && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* SUB-PANEL 1: MULTI-QUANTILE GBDT & XAI FEATURE ATTRIBUTION */}
            <div className="bg-slate-900/85 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
              <div className="flex justify-between items-start">
                <div className="flex items-center gap-3">
                  <div className="p-2.5 bg-indigo-600/20 text-indigo-400 rounded-xl border border-indigo-500/30">
                    <Cpu className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="text-base font-bold text-white flex items-center gap-2">
                      <span>Multi-Quantile GBDT &amp; Conformal XAI</span>
                    </h3>
                    <p className="text-xs text-slate-400 mt-0.5">
                      HistGradientBoostingRegressor (q05, q50, q95) with Split Conformal Calibration.
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <span className="px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold bg-indigo-500/15 text-indigo-300 border border-indigo-500/30 block">
                    q̂ Residual: ±{data.ml_model_card?.q_hat || 18} pax
                  </span>
                  <span className="text-[10px] text-slate-400 mt-1 block">
                    Conformal Coverage: {data.ml_model_card?.conformal_coverage_pct || 91.2}% (Target 90%)
                  </span>
                </div>
              </div>

              {/* Global Feature Importances Progress Bars */}
              <div className="pt-2 border-t border-slate-800/80 space-y-2">
                <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                  Global Feature Attribution Weights:
                </div>
                <div className="space-y-1.5">
                  {data.ml_model_card?.feature_importances.map((f, idx) => (
                    <div key={idx} className="space-y-0.5">
                      <div className="flex justify-between text-xs">
                        <span className="text-slate-300 font-medium">{f.feature}</span>
                        <span className="text-indigo-400 font-mono font-bold">{Math.round(f.importance * 100)}%</span>
                      </div>
                      <div className="w-full bg-slate-950 rounded-full h-1.5 overflow-hidden border border-slate-800">
                        <div
                          className="bg-indigo-500 h-full rounded-full"
                          style={{ width: `${Math.round(f.importance * 100)}%` }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Selected Route XAI Passenger Waterfall Breakdown */}
              <div className="pt-3 border-t border-slate-800/80 bg-slate-950/60 p-3.5 rounded-xl border border-slate-800/60">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-xs font-bold text-white flex items-center gap-1.5">
                    <span className="text-indigo-400 font-mono">{selectedRoute}</span> XAI Additive Passenger Drivers:
                  </span>
                  <span className="text-[11px] text-slate-400">
                    Total Predicted: <strong className="text-emerald-400">{selectedRouteObj?.predicted} pax</strong>
                  </span>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 text-center text-xs">
                  <div className="bg-slate-900/90 p-2 rounded-lg border border-slate-800">
                    <div className="text-[10px] text-slate-400 uppercase font-semibold">Base Schedule</div>
                    <div className="text-slate-200 font-mono font-bold mt-0.5">
                      +{selectedRouteObj?.xai_drivers?.base_schedule || 0}
                    </div>
                  </div>
                  <div className="bg-slate-900/90 p-2 rounded-lg border border-slate-800">
                    <div className="text-[10px] text-slate-400 uppercase font-semibold">Autoregressive</div>
                    <div className="text-blue-400 font-mono font-bold mt-0.5">
                      +{selectedRouteObj?.xai_drivers?.autoregressive_trend || 0}
                    </div>
                  </div>
                  <div className="bg-slate-900/90 p-2 rounded-lg border border-slate-800">
                    <div className="text-[10px] text-slate-400 uppercase font-semibold">Weekend Shift</div>
                    <div
                      className={`font-mono font-bold mt-0.5 ${
                        (selectedRouteObj?.xai_drivers?.weekend_land_use_effect || 0) < 0
                          ? 'text-cyan-400'
                          : (selectedRouteObj?.xai_drivers?.weekend_land_use_effect || 0) > 0
                          ? 'text-emerald-400'
                          : 'text-slate-400'
                      }`}
                    >
                      {(selectedRouteObj?.xai_drivers?.weekend_land_use_effect || 0) > 0
                        ? `+${selectedRouteObj?.xai_drivers?.weekend_land_use_effect}`
                        : selectedRouteObj?.xai_drivers?.weekend_land_use_effect || 0}
                    </div>
                  </div>
                  <div className="bg-slate-900/90 p-2 rounded-lg border border-slate-800">
                    <div className="text-[10px] text-slate-400 uppercase font-semibold">Rain / Weather</div>
                    <div className="text-amber-400 font-mono font-bold mt-0.5">
                      +{selectedRouteObj?.xai_drivers?.weather_rain_impact || 0}
                    </div>
                  </div>
                  <div className="bg-slate-900/90 p-2 rounded-lg border border-slate-800">
                    <div className="text-[10px] text-slate-400 uppercase font-semibold">Event/Metro</div>
                    <div className="text-purple-400 font-mono font-bold mt-0.5">
                      +{selectedRouteObj?.xai_drivers?.event_metro_spillover || 0}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* SUB-PANEL 2: BI-DIRECTIONAL FLEET REBALANCING & OPERATIONAL SAVINGS */}
            <div className="bg-slate-900/85 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4 flex flex-col justify-between">
              <div>
                <div className="flex justify-between items-start">
                  <div className="flex items-center gap-3">
                    <div className="p-2.5 bg-emerald-600/20 text-emerald-400 rounded-xl border border-emerald-500/30">
                      <GitBranch className="w-5 h-5" />
                    </div>
                    <div>
                      <h3 className="text-base font-bold text-white flex items-center gap-2">
                        <span>Bi-Directional Fleet Rebalancing Map</span>
                      </h3>
                      <p className="text-xs text-slate-400 mt-0.5">
                        Harvests idle buses on low-demand corridors (&lt;58% util) to feed surging transit hubs.
                      </p>
                    </div>
                  </div>
                  <span className="px-2.5 py-0.5 rounded-full text-[10px] font-extrabold uppercase tracking-wider bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
                    {data.harvested_pool && data.harvested_pool > 0 ? `${data.harvested_pool} BUSES HARVESTED` : 'NORMAL FLEET BALANCE'}
                  </span>
                </div>

                {/* Savings Metric Pills */}
                <div className="grid grid-cols-3 gap-2.5 mt-3 pt-3 border-t border-slate-800/80">
                  <div className="bg-slate-950/80 border border-slate-800 p-2.5 rounded-xl text-center">
                    <div className="text-[10px] text-slate-400 uppercase font-semibold flex items-center justify-center gap-1">
                      <Fuel className="w-3 h-3 text-amber-400" /> Fuel Saved
                    </div>
                    <div className="text-sm font-black text-amber-400 mt-0.5">
                      {data.savings?.fuel_liters_saved || 0} L
                    </div>
                    <div className="text-[9px] text-slate-500 mt-0.5">Diesel Conserved</div>
                  </div>

                  <div className="bg-slate-950/80 border border-slate-800 p-2.5 rounded-xl text-center">
                    <div className="text-[10px] text-slate-400 uppercase font-semibold flex items-center justify-center gap-1">
                      <IndianRupee className="w-3 h-3 text-emerald-400" /> INR Saved
                    </div>
                    <div className="text-sm font-black text-emerald-400 mt-0.5">
                      ₹{(data.savings?.cost_saved_inr || 0).toLocaleString()}
                    </div>
                    <div className="text-[9px] text-slate-500 mt-0.5">Operating Cost</div>
                  </div>

                  <div className="bg-slate-950/80 border border-slate-800 p-2.5 rounded-xl text-center">
                    <div className="text-[10px] text-slate-400 uppercase font-semibold flex items-center justify-center gap-1">
                      <Leaf className="w-3 h-3 text-teal-400" /> CO₂ Abated
                    </div>
                    <div className="text-sm font-black text-teal-400 mt-0.5">
                      {data.savings?.co2_kg_saved || 0} kg
                    </div>
                    <div className="text-[9px] text-slate-500 mt-0.5">Emissions Cut</div>
                  </div>
                </div>

                {/* Donor -> Receiver Transfer Pairs */}
                <div className="mt-3.5 space-y-2">
                  <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider flex items-center justify-between">
                    <span>Donor ➔ Receiver Fleet Allocations:</span>
                    <span className="text-[10px] text-cyan-400 lowercase font-normal">
                      {data.transfer_pairs?.length || 0} active cross-corridor transfers
                    </span>
                  </div>

                  {data.transfer_pairs && data.transfer_pairs.length > 0 ? (
                    <div className="space-y-2 max-h-44 overflow-y-auto pr-1">
                      {data.transfer_pairs.map((t, idx) => (
                        <div
                          key={idx}
                          className="p-3 bg-slate-950/90 border border-slate-800 rounded-xl flex items-center justify-between text-xs hover:border-slate-700 transition"
                        >
                          <div className="flex items-center gap-2">
                            <span className="px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300 font-mono font-bold text-[11px]">
                              {t.donor_id}
                            </span>
                            <ArrowRight className="w-3.5 h-3.5 text-slate-500 shrink-0" />
                            <span className="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-mono font-bold text-[11px]">
                              {t.receiver_id}
                            </span>
                          </div>
                          <div className="text-right">
                            <div className="font-bold text-white text-[11px]">
                              {t.buses} Bus(es) Transferred
                            </div>
                            <div className="text-[10px] text-slate-400 truncate max-w-xs mt-0.5">
                              {t.rationale}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="p-4 bg-slate-950/60 border border-dashed border-slate-800 rounded-xl text-center text-xs text-slate-400">
                      No corridor currently below 58% utilization threshold. Switch to{' '}
                      <strong className="text-purple-300 cursor-pointer underline" onClick={() => setDayType('WEEKEND')}>
                        Weekend mode
                      </strong>{' '}
                      to trigger automatic harvesting on the Silk Board ORR IT corridor.
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Charts & Before/After Panel Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Recharts Analytics Panel with Tab 1 and Tab 2 */}
          <div className="lg:col-span-2 bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col justify-between">
            <div>
              <div className="flex flex-col sm:flex-row justify-between sm:items-center gap-3 mb-4">
                <div>
                  <h3 className="text-base font-bold text-white flex items-center gap-2">
                    <span>Corridor Analytics:</span>
                    <span className="text-blue-400 font-mono">{selectedRoute}</span>
                    <span className="text-slate-300 font-normal">
                      ({selectedRouteObj?.name})
                    </span>
                  </h3>
                  <p className="text-xs text-slate-400 mt-0.5">
                    {activeChartTab === 'intraday'
                      ? `8-window intraday forecast curve (06:00 to 21:00) with 90% normal interval band vs capacity [${dayType}]`
                      : `30-day historical time-series at ${hour} with hold-out backtest split (Days 1–23 Train vs 24–30 Test)`}
                  </p>
                </div>

                {/* Tab Switcher */}
                <div className="flex bg-slate-950 p-1 rounded-xl border border-slate-800 self-start sm:self-auto">
                  <button
                    onClick={() => setActiveChartTab('intraday')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-bold transition flex items-center gap-1.5 ${
                      activeChartTab === 'intraday'
                        ? 'bg-blue-600 text-white shadow-md'
                        : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    <Clock className="w-3.5 h-3.5" /> 8-Window Intraday
                  </button>
                  <button
                    onClick={() => setActiveChartTab('history')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-bold transition flex items-center gap-1.5 ${
                      activeChartTab === 'history'
                        ? 'bg-blue-600 text-white shadow-md'
                        : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    <History className="w-3.5 h-3.5" /> 30-Day History &amp; Backtest
                  </button>
                </div>
              </div>

              {/* Chart Content */}
              <div className="h-72 w-full mt-2">
                {activeChartTab === 'intraday' ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={intraday} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="hour" stroke="#94a3b8" fontSize={12} tickLine={false} />
                      <YAxis stroke="#94a3b8" fontSize={12} tickLine={false} domain={['auto', 'auto']} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#090d16',
                          borderColor: '#334155',
                          borderRadius: '0.75rem',
                          fontSize: 12,
                        }}
                      />
                      <Legend wrapperStyle={{ fontSize: 12, paddingTop: '8px' }} />
                      <Area
                        type="monotone"
                        dataKey="upper"
                        name="90% Upper Bound"
                        fill="#3b82f6"
                        fillOpacity={0.12}
                        stroke="#60a5fa"
                        strokeDasharray="4 4"
                        isAnimationActive={false}
                      />
                      <Line
                        type="monotone"
                        dataKey="predicted"
                        name="Predicted Demand"
                        stroke="#38bdf8"
                        strokeWidth={2.5}
                        dot={{ r: 4, fill: '#0284c7' }}
                        activeDot={{ r: 6 }}
                        isAnimationActive={false}
                      />
                      <Line
                        type="stepAfter"
                        dataKey="capacity"
                        name="Effective Seat Capacity"
                        stroke="#10b981"
                        strokeWidth={2.2}
                        dot={false}
                        isAnimationActive={false}
                      />
                      <Line
                        type="monotone"
                        dataKey="baseline"
                        name="Clear Baseline Mean"
                        stroke="#94a3b8"
                        strokeWidth={1.5}
                        strokeDasharray="2 2"
                        dot={false}
                        isAnimationActive={false}
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={historyData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis
                        dataKey="date"
                        stroke="#94a3b8"
                        fontSize={11}
                        tickLine={false}
                        tickFormatter={(val) => val.slice(5)}
                      />
                      <YAxis stroke="#94a3b8" fontSize={12} tickLine={false} domain={['auto', 'auto']} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#090d16',
                          borderColor: '#334155',
                          borderRadius: '0.75rem',
                          fontSize: 12,
                        }}
                        formatter={(value: any, name: any, item: any) => {
                          const w = item.payload.weather;
                          const ev = item.payload.event;
                          const tag = ev !== 'None' ? ` (${ev})` : w === 'Rain' ? ' (Rain)' : '';
                          return [`${value} pax${tag}`, name];
                        }}
                      />
                      <Legend wrapperStyle={{ fontSize: 12, paddingTop: '8px' }} />
                      <ReferenceLine
                        x={historyData[22]?.date}
                        stroke="#f59e0b"
                        strokeDasharray="4 4"
                        label={{
                          value: 'Train/Test Split (Day 23)',
                          fill: '#f59e0b',
                          fontSize: 11,
                          position: 'top',
                        }}
                      />
                      <Line
                        type="monotone"
                        dataKey="passengers"
                        name="Observed History"
                        stroke="#60a5fa"
                        strokeWidth={2}
                        dot={{ r: 3, fill: '#3b82f6' }}
                        isAnimationActive={false}
                      />
                      <Line
                        type="monotone"
                        dataKey="capacity"
                        name="Seat Capacity"
                        stroke="#10b981"
                        strokeWidth={1.5}
                        strokeDasharray="3 3"
                        dot={false}
                        isAnimationActive={false}
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                )}
              </div>
            </div>

            <div className="mt-4 pt-3 border-t border-slate-800 text-xs text-slate-400 flex flex-wrap justify-between items-center gap-2">
              <span>
                Inspecting: <strong className="text-white">{selectedRouteObj?.name}</strong> | Baseline Clear Mean:{' '}
                <strong className="text-white">{selectedRouteObj?.baseline_mean} pax</strong>
              </span>
              <span>
                Conformal Sigma σ:{' '}
                <strong className="text-white">±{selectedRouteObj?.sigma} pax</strong>
              </span>
            </div>
          </div>

          {/* Before vs After Impact & Fairness Comparison Card */}
          {data && (
            <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col justify-between">
              <div>
                <div className="flex justify-between items-center mb-3">
                  <h3 className="text-base font-bold text-white flex items-center gap-2">
                    <BarChart2 className="w-4 h-4 text-emerald-400" />
                    <span>Before vs. After AI</span>
                  </h3>
                  <span className="text-[10px] px-2 py-0.5 bg-slate-800 text-slate-300 rounded font-semibold uppercase tracking-wider border border-slate-700">
                    {data.label}
                  </span>
                </div>

                {/* Overcrowding Reduction Hero Metric */}
                <div className="p-3.5 bg-emerald-950/40 border border-emerald-800/60 rounded-xl mb-4 text-center">
                  <div className="text-3xl font-black text-emerald-400">
                    {data.reduction_pct}% Overcrowding Reduction
                  </div>
                  <div className="text-xs text-slate-300 mt-1">
                    {data.before.passengers_affected === 0
                      ? 'Network operating normally. 0 spare vehicles needed.'
                      : `Excess passengers: ${data.before.passengers_affected} → ${data.after.passengers_affected} unserved`}
                  </div>
                </div>

                {/* Metrics Breakdown Table */}
                <div className="space-y-2.5 text-xs">
                  <div className="flex justify-between py-1.5 border-b border-slate-800/80">
                    <span className="text-slate-400">Overcrowded Routes (&gt;100%)</span>
                    <span className="font-mono font-bold">
                      <span className="text-red-400">{data.before.overcrowded_routes}</span>
                      <span className="text-slate-500 mx-1.5">→</span>
                      <span className="text-emerald-400">{data.after.overcrowded_routes}</span>
                    </span>
                  </div>

                  <div className="flex justify-between py-1.5 border-b border-slate-800/80">
                    <span className="text-slate-400">Critical Routes (&gt;120%)</span>
                    <span className="font-mono font-bold">
                      <span className="text-red-400">{data.before.critical_routes}</span>
                      <span className="text-slate-500 mx-1.5">→</span>
                      <span className="text-emerald-400">{data.after.critical_routes}</span>
                    </span>
                  </div>

                  <div className="flex justify-between py-1.5 border-b border-slate-800/80">
                    <span className="text-slate-400">Average Fleet Utilization</span>
                    <span className="font-mono font-bold">
                      <span className="text-slate-300">{data.before.avg_utilization}%</span>
                      <span className="text-slate-500 mx-1.5">→</span>
                      <span className="text-emerald-400">{data.after.avg_utilization}%</span>
                    </span>
                  </div>

                  <div className="flex justify-between py-1.5 border-b border-slate-800/80">
                    <span className="text-slate-400">Fairness: Service-Level Share</span>
                    <span className="font-mono font-bold text-blue-400">
                      <span>{data.before.service_level_share}%</span>
                      <span className="text-slate-500 mx-1.5">→</span>
                      <span className="text-emerald-400">{data.after.service_level_share}%</span>
                    </span>
                  </div>

                  <div className="flex justify-between py-1.5">
                    <span className="text-slate-400">Fairness: Utilization Spread</span>
                    <span className="font-mono font-bold text-blue-400">
                      <span>{data.before.utilization_spread} pts</span>
                      <span className="text-slate-500 mx-1.5">→</span>
                      <span className="text-emerald-400">{data.after.utilization_spread} pts</span>
                    </span>
                  </div>
                </div>
              </div>

              {/* Action Button */}
              <button
                onClick={() => {
                  if (applied) {
                    setApplied(false);
                  } else {
                    handleApplyDispatch();
                  }
                }}
                disabled={totalDispatched === 0 && recEntries.length === 0}
                className={`mt-4 w-full py-3 rounded-xl text-xs font-black uppercase tracking-wider transition ${
                  totalDispatched === 0 && recEntries.length === 0
                    ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                    : applied
                    ? 'bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700'
                    : 'bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white shadow-lg shadow-blue-600/30 active:scale-95'
                }`}
              >
                {totalDispatched === 0 && recEntries.length === 0
                  ? 'No Dispatch Intervention Needed'
                  : applied
                  ? 'Allocation Committed · Click to Preview Pre-State'
                  : `Commit Allocation (${totalDispatched || data.spare_used} Vehicles Deployed)`}
              </button>
            </div>
          )}
        </div>

        {/* Verification Section: Hold-Out Backtest & SQLite Audit Log */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Hold-Out Backtest Validation Table */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-xl">
            <div className="flex justify-between items-center mb-1">
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <Database className="w-4 h-4 text-blue-400" />
                <span>Hold-Out GBDT Error Backtest (Days 1–23 Train · Days 24–30 Test @ {hour})</span>
              </h3>
              <span className="text-[10px] px-2 py-0.5 bg-blue-500/10 text-blue-400 rounded border border-blue-500/20 font-bold">
                MAE &amp; MAPE
              </span>
            </div>
            <p className="text-[11px] text-slate-400 mb-3">
              Multi-quantile GBDT validation against held-out Bengaluru corridor history confirming 90% conformal interval coverage.
            </p>

            {backtest && (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400 uppercase tracking-wider font-semibold">
                      <th className="pb-2">Corridor</th>
                      <th className="pb-2">MAE (Pax)</th>
                      <th className="pb-2">MAPE (%)</th>
                      <th className="pb-2">90% CI Coverage</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono">
                    {backtest.routes.map((b) => (
                      <tr key={b.route_id} className="hover:bg-slate-800/30">
                        <td className="py-2.5 font-bold text-white font-sans">
                          <span className="text-blue-400 font-mono mr-1.5">{b.route_id}</span>
                          <span className="text-slate-300">{b.route_name}</span>
                        </td>
                        <td className="py-2.5 text-slate-200">±{b.mae}</td>
                        <td className="py-2.5 text-slate-200">{b.mape}%</td>
                        <td className="py-2.5 font-bold text-emerald-400">{b.coverage_90}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* SQLite Allocation Audit Trail (`allocation_log`) */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col justify-between">
            <div>
              <div className="flex justify-between items-center mb-1">
                <h3 className="text-sm font-bold text-white flex items-center gap-2">
                  <History className="w-4 h-4 text-purple-400" />
                  <span>SQLite Allocation Audit Trail (`allocation_log`)</span>
                </h3>
                <span className="text-[10px] px-2 py-0.5 bg-purple-500/10 text-purple-400 rounded border border-purple-500/20 font-bold">
                  Immutable Log
                </span>
              </div>
              <p className="text-[11px] text-slate-400 mb-3">
                Records operator-approved interventions and fleet curtailment with pre- and post-allocation utilization.
              </p>

              {logs.length === 0 ? (
                <div className="text-xs text-slate-500 py-10 text-center border border-dashed border-slate-800 rounded-xl">
                  No allocations committed yet. Activate Spike or Capacity Loss and click &quot;Apply Allocation Dispatch&quot;.
                </div>
              ) : (
                <div className="space-y-2 max-h-56 overflow-y-auto pr-1">
                  {logs.map((l) => (
                    <div
                      key={l.id}
                      className="p-3 bg-slate-950/90 border border-slate-800 rounded-xl text-xs flex justify-between items-center hover:border-slate-700 transition"
                    >
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="font-mono font-bold text-blue-400">{l.route_id}</span>
                          <span
                            className={`px-1.5 py-0.2 rounded text-[10px] font-bold ${
                              l.vehicles_added < 0
                                ? 'bg-cyan-500/20 text-cyan-300'
                                : 'bg-emerald-500/20 text-emerald-300'
                            }`}
                          >
                            {l.vehicles_added > 0 ? `+${l.vehicles_added}` : `${l.vehicles_added}`} buses
                          </span>
                          <span className="text-slate-400 font-mono">
                            {l.util_before}% → <strong className="text-emerald-400">{l.util_after}%</strong>
                          </span>
                        </div>
                        <div className="text-[11px] text-slate-400 mt-0.5 truncate max-w-sm">
                          {l.reason}
                        </div>
                      </div>
                      <div className="text-right text-[10px] text-slate-500">
                        <div>{l.hour}</div>
                        <div>{l.ts}</div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {logs.length > 0 && (
              <div className="mt-3 text-right">
                <button
                  onClick={handleReset}
                  className="text-[11px] text-slate-400 hover:text-red-400 underline"
                >
                  Clear Audit Trail
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
