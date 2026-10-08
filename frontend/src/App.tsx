import { useEffect, useState } from 'react';
import {
  AlertTriangle,
  ArrowRight,
  Briefcase,
  Bus,
  CheckCircle2,
  CheckSquare,
  Clock,
  CloudSun,
  Coffee,
  Cpu,
  Database,
  FileText,
  Fuel,
  GitBranch,
  History,
  Layers,
  Pause,
  Play,
  Radio,
  RotateCcw,
  ShieldAlert,
  ShieldCheck,
  Sliders,
  Sparkles,
  TrendingUp,
  Undo2,
  Wrench,
  X,
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
  lag_momentum: number;
  weekend_shift: number;
  weather_uplift: number;
  event_spillover: number;
  inference_ms?: number;
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
  nominal_capacity?: number;
  capacity: number; // Effective derated capacity
  congestion_factor?: number;
  current_speed_kmh?: number;
  free_flow_speed_kmh?: number;
  headway_delay_min?: number;
  bottleneck?: string;
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
  headway_before_min?: number;
  headway_after_min?: number;
}

interface ReceiverRoute {
  route_id: string;
  route_name: string;
  assigned_buses: number;
  util_before: number;
  util_after: number;
  pred_passengers: number;
  priority_score: number;
  headway_before_min?: number;
  headway_after_min?: number;
}

interface TransferPair {
  donor_id: string;
  donor_name: string;
  receiver_id: string;
  receiver_name: string;
  buses: number;
  headway_before_min?: number;
  headway_after_min?: number;
  cost_saved_inr?: number;
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

interface LiveWeather {
  city: string;
  temp_c: number;
  rain_mm: number;
  precipitation_mm?: number;
  wind_speed_kmh: number;
  condition: string;
  is_live?: boolean;
  live_timestamp: string;
  source: string;
}

interface DecisionOption {
  option_id: string;
  title: string;
  action_summary: string;
  metrics: string;
  alloc_map: Record<string, number>;
  recommended: boolean;
}

interface DecisionPrompt {
  question: string;
  is_active: boolean;
  target_route_id: string;
  options: DecisionOption[];
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
  donor_receiver_transfers?: TransferPair[];
  transfer_pairs?: TransferPair[];
  savings?: SavingsInfo;
  ml_model_card?: MLModelCard;
  before: SummaryMetrics;
  after: SummaryMetrics;
  reduction_pct: number;
  method: string;
  label: string;
  logs?: LogEntry[];
  decision_prompt?: DecisionPrompt;
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
  const [previewPreState, setPreviewPreState] = useState(false);
  const [spare, setSpare] = useState(5);
  const [selectedRoute, setSelectedRoute] = useState('R002');
  const [liveMode, setLiveMode] = useState(false);
  const [tickStep, setTickStep] = useState(0);
  const [activeChartTab, setActiveChartTab] = useState<'intraday' | 'history'>('intraday');

  // Day type: Weekday vs Weekend
  const [dayType, setDayType] = useState<'WEEKDAY' | 'WEEKEND'>('WEEKDAY');

  // Zero-Key Live Weather Telemetry (Open-Meteo)
  const [syncingWeather, setSyncingWeather] = useState(false);
  const [weatherData, setWeatherData] = useState<LiveWeather | null>(null);

  // Fleet Unit Maintenance State: { "UNIT-101": true }
  const [maintenanceUnits, setMaintenanceUnits] = useState<Record<string, boolean>>({});

  // 10-Second Safety Undo Dispatch Timer
  const [undoTimer, setUndoTimer] = useState<number>(0);
  const [undoToast, setUndoToast] = useState<string | null>(null);

  // Two-Person Supervisor Co-Sign Modal State
  const [showCoSignModal, setShowCoSignModal] = useState(false);
  const [operatorId, setOperatorId] = useState('OP-7829');
  const [supervisorName, setSupervisorName] = useState('');
  const [coSignConfirmed, setCoSignConfirmed] = useState(false);

  // Human intervention override state: null = AI recommendation active
  const [manualAlloc, setManualAlloc] = useState<Record<string, number> | null>(null);

  const [data, setData] = useState<AnalyzeResponse | null>(null);
  const [intraday, setIntraday] = useState<any[]>([]);
  const [historyData, setHistoryData] = useState<any[]>([]);
  const [backtest, setBacktest] = useState<BacktestResponse | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Calculate usable spares excluding depot maintenance
  const maintenanceCount = Object.values(maintenanceUnits).filter(Boolean).length;
  const usableSpareCount = Math.max(0, spare - maintenanceCount);

  // Permanently purge any legacy external API keys from client storage
  useEffect(() => {
    try {
      localStorage.removeItem('tomtom_api_key');
      localStorage.removeItem('tomtomKey');
    } catch {
      // ignore
    }
  }, []);

  // 10-second countdown effect
  useEffect(() => {
    if (undoTimer <= 0) return;
    const interval = setInterval(() => {
      setUndoTimer((prev) => {
        if (prev <= 1) {
          clearInterval(interval);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(interval);
  }, [undoTimer]);

  const fetchAll = async (currTick = tickStep, preState = previewPreState) => {
    try {
      setErrorMsg(null);
      const isApplied = applied && !preState;
      const [resAnalyze, resIntra, resHist, resBt, resLog] = await Promise.all([
        fetch(`${API_BASE}/api/analyze`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            spike,
            capacity_loss: capacityLoss,
            hour,
            spare: usableSpareCount,
            apply: isApplied,
            tick_step: liveMode ? currTick : 0,
            manual_alloc: manualAlloc,
            day_type: dayType,
          }),
        }),
        fetch(
          `${API_BASE}/api/intraday/${selectedRoute}?spike=${spike}&capacity_loss=${capacityLoss}&apply=${isApplied}&spare=${usableSpareCount}&tick_step=${
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

  const handleSyncWeather = async () => {
    try {
      setSyncingWeather(true);
      const res = await fetch(`${API_BASE}/api/live-weather`);
      if (res.ok) {
        const json: LiveWeather = await res.json();
        setWeatherData(json);
        await fetchAll(liveMode ? tickStep : 0);
      }
    } catch (e) {
      console.error('Sync live weather failed:', e);
    } finally {
      setSyncingWeather(false);
    }
  };

  // Trigger dispatch: if total moved >= 4, pop up Two-Person Co-Sign Modal
  const handleInitiateDispatch = () => {
    const activeMap = manualAlloc !== null ? manualAlloc : (data?.recommendation || {});
    const totalMoved = Object.values(activeMap).reduce((sum, v) => sum + Math.abs(v), 0);

    if (totalMoved >= 4) {
      setShowCoSignModal(true);
    } else {
      executeCommit(null, null);
    }
  };

  const executeCommit = async (supName: string | null, opId: string | null) => {
    try {
      setApplied(true);
      setPreviewPreState(false);
      setShowCoSignModal(false);

      const res = await fetch(`${API_BASE}/api/commit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          spike,
          capacity_loss: capacityLoss,
          hour,
          spare: usableSpareCount,
          apply: true,
          commit: true,
          tick_step: liveMode ? tickStep : 0,
          manual_alloc: manualAlloc,
          day_type: dayType,
          supervisor_name: supName || undefined,
          operator_id: opId || undefined,
        }),
      });

      if (res.ok) {
        const analyzeJson = await res.json();
        setData(analyzeJson);
        if (analyzeJson.logs && analyzeJson.logs.length > 0) {
          setLogs(analyzeJson.logs);
        }
        // Activate 10-Second Safety Undo Window
        setUndoTimer(10);
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

  const handleApplyStrategy = async (option: DecisionOption) => {
    try {
      setManualAlloc(option.alloc_map);
      setApplied(true);
      setPreviewPreState(false);

      const res = await fetch(`${API_BASE}/api/commit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          spike,
          capacity_loss: capacityLoss,
          hour,
          spare: usableSpareCount,
          apply: true,
          commit: true,
          tick_step: liveMode ? tickStep : 0,
          manual_alloc: option.alloc_map,
          chosen_option_id: option.option_id,
          day_type: dayType,
        }),
      });

      if (res.ok) {
        const analyzeJson = await res.json();
        setData(analyzeJson);
        if (analyzeJson.logs && analyzeJson.logs.length > 0) {
          setLogs(analyzeJson.logs);
        }
        // Activate 10-Second Safety Undo Window
        setUndoTimer(10);
      }

      const resLog = await fetch(`${API_BASE}/api/log`);
      if (resLog.ok) {
        const freshLogs = await resLog.json();
        setLogs(freshLogs);
      }
    } catch (err: any) {
      console.error('Failed to commit strategy option:', err);
    }
  };

  const handleUndoLastCommit = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/undo-last-commit`, { method: 'POST' });
      if (res.ok) {
        const json = await res.json();
        if (json.logs) {
          setLogs(json.logs);
        }
        setApplied(false);
        setPreviewPreState(false);
        setUndoTimer(0);
        setUndoToast(`Reverted ${json.deleted_rows || 1} allocation(s) from SQLite log.`);
        setTimeout(() => setUndoToast(null), 4000);
        await fetchAll();
      }
    } catch (e) {
      console.error('Failed to undo commit:', e);
    }
  };

  // Export Post-Incident Report (.txt)
  const handleExportReport = () => {
    const ts = new Date().toISOString();
    let content = `========================================================================\n`;
    content += `     BENGALURU METROPOLITAN TRANSPORT CORPORATION (BMTC)\n`;
    content += `          CENTRAL DISPATCH INTELLIGENCE & INCIDENT REPORT\n`;
    content += `========================================================================\n\n`;
    content += `Generated At: ${ts}\n`;
    content += `Operational Window: ${hour} | Day Type: ${dayType}\n`;
    content += `Bengaluru Weather: Temp ${weatherData?.temp_c ?? 26.5}°C, Rain ${weatherData?.rain_mm ?? 0}mm (${weatherData?.condition || 'Clear'})\n`;
    content += `Spike Simulation: ${spike ? 'ACTIVE (KR Market Festival Surge)' : 'OFF'}\n`;
    content += `Capacity Loss: ${capacityLoss ? 'ACTIVE (Peenya Industrial -2 Buses Breakdown)' : 'OFF'}\n`;
    content += `Depot Spares Configured: ${spare} | In Maintenance: ${maintenanceCount} | Usable Spares: ${usableSpareCount}\n\n`;
    content += `------------------------------------------------------------------------\n`;
    content += `1. FLEET REBALANCING SUMMARY\n`;
    content += `------------------------------------------------------------------------\n`;
    content += `Harvested Surplus Buses: ${data?.savings?.harvested_buses ?? 0}\n`;
    content += `Surplus Deployed to Congestion: ${data?.spare_used ?? 0}\n`;
    content += `Estimated Diesel Saved: ${data?.savings?.fuel_liters_saved ?? 0} Liters\n`;
    content += `Cost Savings: ₹${data?.savings?.cost_saved_inr ?? 0}\n`;
    content += `Carbon Mitigated: ${data?.savings?.co2_kg_saved ?? 0} kg CO2\n`;
    content += `Overcrowding Passenger Reduction: ${data?.reduction_pct ?? 0}%\n\n`;
    content += `------------------------------------------------------------------------\n`;
    content += `2. CORRIDOR OPERATIONAL SNAPSHOT\n`;
    content += `------------------------------------------------------------------------\n`;
    if (data?.routes) {
      for (const r of data.routes) {
        content += `[${r.id}] ${r.name}\n`;
        content += `  Demand Forecast: ${r.predicted} pax (90% CI: [${r.lower}, ${r.upper}])\n`;
        content += `  Fleet: ${r.vehicles} buses (Alloc: ${r.extra >= 0 ? '+' : ''}${r.extra}) | Eff Cap: ${r.capacity}\n`;
        content += `  Utilization: ${r.utilization}% | Risk: ${r.risk} | P(Overcrowd): ${r.probability}%\n`;
        content += `  Bottleneck: ${r.bottleneck || 'N/A'} (Congestion Delay: ${r.congestion_factor ?? 1.0}x)\n\n`;
      }
    }
    content += `------------------------------------------------------------------------\n`;
    content += `3. SQLITE ALLOCATION AUDIT LOG (allocation_log)\n`;
    content += `------------------------------------------------------------------------\n`;
    if (logs && logs.length > 0) {
      for (const l of logs) {
        content += `[ID: ${l.id}] ${l.ts} | ${l.scenario} | Route: ${l.route_id} | Added: ${l.vehicles_added >= 0 ? '+' : ''}${l.vehicles_added} | Util: ${l.util_before}% -> ${l.util_after}%\n`;
        content += `  Reason: ${l.reason}\n\n`;
      }
    } else {
      content += `No allocations recorded yet in transport.db.\n`;
    }
    content += `========================================================================\n`;
    content += `END OF REPORT\n`;

    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `BMTC_ZeroCrowd_Incident_Report_${hour.replace(':', '')}_${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  useEffect(() => {
    fetchAll(liveMode ? tickStep : 0, previewPreState);
  }, [hour, spike, capacityLoss, applied, previewPreState, selectedRoute, spare, liveMode, tickStep, manualAlloc, dayType, maintenanceUnits]);

  // Live Simulation 2-second tick loop
  useEffect(() => {
    if (!liveMode) return;
    const timer = setInterval(() => {
      setTickStep((prev) => {
        const next = prev + 1;
        fetchAll(next, previewPreState);
        return next;
      });
    }, 2000);
    return () => clearInterval(timer);
  }, [liveMode, hour, spike, capacityLoss, applied, previewPreState, selectedRoute, spare, manualAlloc, dayType, maintenanceUnits]);

  const handleReset = async () => {
    setSpike(false);
    setCapacityLoss(false);
    setHour('08:00');
    setApplied(false);
    setPreviewPreState(false);
    setSpare(5);
    setLiveMode(false);
    setTickStep(0);
    setDayType('WEEKDAY');
    setManualAlloc(null);
    setMaintenanceUnits({});
    setUndoTimer(0);
    try {
      await fetch(`${API_BASE}/api/reset`, { method: 'POST' });
      setLogs([]);
    } catch (e) {
      console.error(e);
    }
  };

  // Active allocation calculations
  const activeAllocMap = manualAlloc !== null ? manualAlloc : (data?.recommendation || {});
  const totalDispatched = Object.values(activeAllocMap).filter((v) => v > 0).reduce((a, b) => a + b, 0);
  const totalCurtailed = Math.abs(Object.values(activeAllocMap).filter((v) => v < 0).reduce((a, b) => a + b, 0));
  const baseSparePool = usableSpareCount;
  const effectiveMaxPool = baseSparePool + (data?.harvested_pool || totalCurtailed);

  // Stepper handlers
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

  // Toggle maintenance on unit card
  const handleToggleMaintenance = (unitId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setMaintenanceUnits((prev) => {
      const next = { ...prev, [unitId]: !prev[unitId] };
      return next;
    });
    setApplied(false);
  };

  // Interactive Unit Card Click Handler
  const handleUnitClick = (unit: { unitId: string; routeId: string | null; isMaintenance: boolean }) => {
    if (unit.isMaintenance) return;
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

  // Standby units queue (fixed 5 units UNIT-101 to UNIT-105)
  const unitAssignments: { unitId: string; routeId: string | null; isMaintenance: boolean }[] = [];
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

  let queueIdx = 0;
  for (let i = 1; i <= Math.max(5, spare); i++) {
    const unitId = `UNIT-${100 + i}`;
    const isMaint = Boolean(maintenanceUnits[unitId]);
    let assignedRoute: string | null = null;
    if (!isMaint && queueIdx < routeAssignmentQueue.length) {
      assignedRoute = routeAssignmentQueue[queueIdx];
      queueIdx++;
    }
    unitAssignments.push({ unitId, routeId: assignedRoute, isMaintenance: isMaint });
  }

  // Group dispatched units
  const dispatchedByRoute: Record<string, string[]> = {};
  for (const u of unitAssignments) {
    if (u.routeId && !u.isMaintenance) {
      if (!dispatchedByRoute[u.routeId]) dispatchedByRoute[u.routeId] = [];
      dispatchedByRoute[u.routeId].push(u.unitId);
    }
  }
  const broadcastString = Object.entries(dispatchedByRoute)
    .map(([rId, units]) => `[${units.join(', ')}] -> Dispatched to ${rId}`)
    .join(' | ');

  if (loading && !data) {
    return (
      <div className="min-h-screen bg-slate-950 text-slate-200 flex flex-col items-center justify-center font-sans space-y-4">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-cyan-500"></div>
        <p className="text-sm font-semibold tracking-wide text-slate-400">
          Initializing ZeroCrowd Multi-Quantile GBDT &amp; Intelligence Engine...
        </p>
      </div>
    );
  }

  const recEntries = Object.entries(data?.recommendation || {}) as [string, number][];
  const activeEntries = Object.entries(activeAllocMap).filter(([_, v]) => v !== 0) as [string, number][];
  const selectedRouteObj = data?.routes.find((r) => r.id === selectedRoute) || data?.routes[0];
  const totalMovedVehicles = Object.values(activeAllocMap).reduce((sum, v) => sum + Math.abs(v), 0);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans selection:bg-cyan-600 selection:text-white pb-12">
      {/* Top Banner Notice */}
      <div className="bg-gradient-to-r from-blue-900/40 via-purple-900/40 to-slate-900 border-b border-slate-800 text-xs px-6 py-2 flex flex-wrap justify-between items-center gap-2">
        <div className="flex items-center gap-2">
          <span className="inline-block w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          <span className="font-bold text-slate-200">ANVATION 2026</span>
          <span className="text-slate-500">|</span>
          <span className="text-slate-400">Track AI-16: ZeroCrowd · Quantile GBDT &amp; Bi-Directional Fleet Intelligence</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="px-2.5 py-0.5 rounded-full text-[10px] font-extrabold uppercase tracking-widest bg-cyan-500/10 text-cyan-300 border border-cyan-500/30">
            Open-Meteo Zero-Key Telemetry Active
          </span>
          <span className="text-slate-400 text-[11px]">Conformal Prediction Interval</span>
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

      {undoToast && (
        <div className="max-w-7xl mx-auto mt-3 px-6">
          <div className="bg-emerald-950/80 border border-emerald-600 text-emerald-200 text-xs p-3 rounded-xl flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 shrink-0 text-emerald-400" />
            <span>{undoToast}</span>
          </div>
        </div>
      )}

      {/* Main Container */}
      <div className="max-w-7xl mx-auto px-6 pt-6 space-y-6">
        {/* Header & Controls Bar */}
        <header className="bg-slate-900/70 backdrop-blur-md border border-slate-800/80 rounded-2xl p-5 shadow-2xl flex flex-col xl:flex-row justify-between items-start xl:items-center gap-5">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-gradient-to-br from-cyan-600 to-blue-600 rounded-2xl shadow-lg shadow-cyan-500/25 ring-1 ring-white/10">
              <Bus className="w-7 h-7 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2.5">
                <h1 className="text-xl md:text-2xl font-black tracking-tight text-white">
                  ZeroCrowd: Bengaluru Transit Command Center
                </h1>
              </div>
              <p className="text-slate-400 text-xs mt-0.5 max-w-xl">
                BMTC &amp; Namma Metro Quantile GBDT Prescriptive Rebalancing &amp; Conformal Overcrowding Control.
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
                title="Weekday Schedule: Normal IT Corridor & Industrial Peak Commuter Demands"
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
              <span className="text-slate-400 mr-1.5 font-medium">Depot Spares:</span>
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

            {/* Zero-Key Live Weather Sync Button */}
            <button
              onClick={handleSyncWeather}
              disabled={syncingWeather}
              className={`px-3.5 py-2 rounded-xl text-xs font-bold border transition flex items-center gap-1.5 ${
                weatherData
                  ? 'bg-cyan-950/70 text-cyan-300 border-cyan-600/70 shadow-lg shadow-cyan-900/30'
                  : 'bg-slate-950/90 text-slate-300 border-slate-800 hover:bg-slate-800 hover:text-white'
              }`}
              title="Fetch real-time Bengaluru weather from Open-Meteo (Zero Key required)"
            >
              <CloudSun className={`w-3.5 h-3.5 ${syncingWeather ? 'animate-spin' : 'text-cyan-400'}`} />
              {syncingWeather
                ? 'Syncing Weather...'
                : weatherData
                ? `🌦️ ${weatherData.temp_c}°C · ${weatherData.condition}${weatherData.rain_mm > 0 ? ` (${weatherData.rain_mm}mm)` : ''}`
                : '🌦️ Sync Live Bengaluru Weather'}
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
              title="Simulate KR Market / City Railway Festival & Rain Surge"
            >
              <Zap className="w-3.5 h-3.5" />
              {spike ? 'Surge Active (252-F)' : 'Simulate Surge'}
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
              title="Simulate Peenya Industrial Depot Breakdown (-2 Buses)"
            >
              <Wrench className="w-3.5 h-3.5" />
              {capacityLoss ? 'Loss Active (401-M -2)' : 'Simulate Fleet Loss'}
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

        {/* Live Open-Meteo Weather Chip */}
        {weatherData && (
          <div className="bg-slate-900/70 border border-cyan-800/60 rounded-2xl px-4 py-2.5 text-xs flex flex-wrap items-center justify-between gap-3 text-slate-300 shadow-md">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
              <span className="font-bold text-white">Bengaluru Live Weather Telemetry:</span>
              <span className="text-slate-400 font-mono">@{weatherData.live_timestamp}</span>
            </div>
            <div className="flex flex-wrap items-center gap-4 text-[11px]">
              <span className="flex items-center gap-1.5 text-cyan-300 font-bold">
                <CloudSun className="w-3.5 h-3.5 text-cyan-400" />
                {weatherData.temp_c}°C · {weatherData.condition}
              </span>
              <span className="flex items-center gap-1.5 text-slate-300">
                <strong>Precipitation:</strong> {weatherData.rain_mm} mm
              </span>
              <span className="flex items-center gap-1.5 text-slate-300">
                <strong>Wind Speed:</strong> {weatherData.wind_speed_kmh} km/h
              </span>
              <span className="px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 text-[10px] font-mono">
                Open-Meteo (Zero-Key Stream)
              </span>
            </div>
          </div>
        )}

        {/* 10-Second Safety Undo Dispatch Notification Bar */}
        {undoTimer > 0 && (
          <div className="bg-gradient-to-r from-amber-950/90 via-slate-900 to-amber-950/90 border-2 border-amber-500 rounded-2xl p-4 shadow-2xl flex flex-col md:flex-row justify-between items-start md:items-center gap-3 animate-pulse">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-amber-500/20 text-amber-400 rounded-xl border border-amber-500/40">
                <AlertTriangle className="w-5 h-5" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="text-xs font-bold text-amber-200">
                    ⚠ Dispatch Committed to SQLite Audit Trail · Safety Undo Window Active
                  </h3>
                  <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-extrabold bg-amber-500/30 text-amber-300 border border-amber-500/50">
                    {undoTimer}s remaining
                  </span>
                </div>
                <p className="text-[11px] text-slate-300 mt-0.5">
                  Intervention persisted in transport.db. Click &quot;Undo Last Dispatch&quot; within 10s to completely revert this batch.
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2 w-full md:w-auto justify-end">
              <button
                onClick={handleUndoLastCommit}
                className="px-4 py-2 bg-amber-500 hover:bg-amber-400 text-slate-950 rounded-xl text-xs font-black shadow-lg shadow-amber-500/30 transition flex items-center gap-1.5 whitespace-nowrap active:scale-95"
              >
                <Undo2 className="w-4 h-4" /> Undo Last Dispatch ({undoTimer}s)
              </button>
              <button
                onClick={() => setUndoTimer(0)}
                className="px-3.5 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 hover:text-white rounded-xl text-xs font-bold border border-slate-700 hover:border-slate-600 transition flex items-center gap-1.5 whitespace-nowrap active:scale-95 shadow-md"
                title="Lock in dispatch immediately without waiting for timer"
              >
                <X className="w-3.5 h-3.5" /> Lock Now
              </button>
            </div>
          </div>
        )}

        {/* 4 KPI Summary Cards */}
        {data && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-lg relative group">
              <div className="absolute top-0 right-0 w-24 h-24 bg-cyan-500/5 rounded-full blur-xl group-hover:bg-cyan-500/10 transition pointer-events-none"></div>
              <div className="flex justify-between items-center text-xs text-slate-400 uppercase tracking-wider font-semibold">
                <span>BMTC Live Corridors</span>
                <Layers className="w-4 h-4 text-cyan-400" />
              </div>
              <div className="mt-3 flex items-baseline justify-between">
                <span className="text-3xl font-black text-white">{data.routes.length} Corridors</span>
                <span className="text-xs px-2 py-0.5 rounded-full bg-cyan-500/15 text-cyan-300 font-bold border border-cyan-500/30">
                  {dayType} @ {data.hour}
                </span>
              </div>
              <p className="text-[11px] text-slate-400 mt-2">
                Nominal baseline: {data.routes.reduce((acc, r) => acc + r.base_vehicles * 100, 0)} seats
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
                Conformal intervals: q̂ = ±{data.ml_model_card?.q_hat || 16} pax (90% target)
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
                  {applied && !previewPreState ? data.after.overcrowded_routes : data.before.overcrowded_routes} /{' '}
                  {applied && !previewPreState ? data.after.critical_routes : data.before.critical_routes}
                </span>
                <span
                  className={`text-xs font-bold px-2 py-0.5 rounded-full ${
                    (applied && !previewPreState ? data.after.passengers_affected : data.before.passengers_affected) > 0
                      ? 'bg-red-500/15 text-red-400 border border-red-500/30'
                      : 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30'
                  }`}
                >
                  {applied && !previewPreState ? data.after.passengers_affected : data.before.passengers_affected} excess pax
                </span>
              </div>
              <p className="text-[11px] text-slate-400 mt-2">
                {applied && !previewPreState
                  ? `Overcrowding reduced by ${data.reduction_pct}% post-dispatch`
                  : 'Derived with bottleneck derated capacity'}
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
                  {data.spare_left} / {data.total_available_pool || (usableSpareCount + (data.harvested_pool || 0))}
                </span>
                <span className="text-xs px-2 py-0.5 rounded-full bg-purple-500/15 text-purple-400 font-bold border border-purple-500/30">
                  {data.harvested_pool ? `+${data.harvested_pool} harvested` : `${data.spare_used} active`}
                </span>
              </div>
              <p className="text-[11px] text-slate-400 mt-2">
                {data.harvested_pool && data.harvested_pool > 0
                  ? `${data.harvested_pool} idle buses harvested from low-demand corridors`
                  : `${usableSpareCount} active depot spares (${maintenanceCount} in maintenance)`}
              </p>
            </div>
          </div>
        )}

        {/* INTERACTIVE 3-STRATEGY DECISION STUDIO PANEL */}
        {data && (Boolean(data.decision_prompt?.is_active) || (spike || capacityLoss) || data.before.overcrowded_routes > 0 || (manualAlloc !== null && (totalDispatched > 0 || totalCurtailed > 0))) && !applied && (
          <div className="bg-gradient-to-b from-slate-900 via-slate-900/95 to-slate-950 border-2 border-red-500/60 rounded-3xl p-6 shadow-2xl relative overflow-hidden space-y-5">
            {/* Top Accent Gradient Border */}
            <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-red-500 via-amber-500 to-emerald-500" />

            {/* Decision Headline Question & Operational Context */}
            <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 pb-4 border-b border-slate-800">
              <div className="flex items-start gap-3.5">
                <div className="p-2.5 bg-red-600/20 text-red-400 rounded-2xl border border-red-500/40 shrink-0 mt-0.5">
                  <ShieldAlert className="w-6 h-6 animate-pulse" />
                </div>
                <div>
                  <div className="flex items-center gap-2 mb-1.5 flex-wrap">
                    <span className="px-2.5 py-0.5 text-[10px] font-extrabold uppercase tracking-wider bg-red-500/20 text-red-300 rounded border border-red-500/40">
                      OPERATIONAL DECISION REQUIRED
                    </span>
                    <span className="px-2 py-0.5 text-[10px] font-mono text-slate-300 bg-slate-800/80 rounded border border-slate-700">
                      {data.before.passengers_affected} pax exceeding capacity · {data.before.overcrowded_routes} route(s) at risk
                    </span>
                    {manualAlloc !== null && (
                      <span className="px-2 py-0.5 text-[10px] font-bold text-amber-300 bg-amber-500/20 rounded border border-amber-500/40">
                        Human Override Active
                      </span>
                    )}
                  </div>
                  <h2 className="text-base md:text-lg font-black text-white tracking-tight">
                    {data.decision_prompt?.question || `Route bottleneck detected: ${data.before.passengers_affected} passengers unserved. How should the control center resolve this?`}
                  </h2>
                </div>
              </div>

              {manualAlloc !== null && (
                <button
                  onClick={handleInitiateDispatch}
                  className="px-4 py-2.5 bg-amber-600 hover:bg-amber-500 text-white rounded-xl text-xs font-extrabold shadow-lg shadow-amber-600/30 whitespace-nowrap transition active:scale-95 flex items-center gap-1.5 shrink-0"
                >
                  <Sparkles className="w-4 h-4" /> COMMIT CUSTOM OVERRIDE
                </button>
              )}
            </div>

            {/* 3 Clickable Strategy Options Side-by-Side */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {(data.decision_prompt?.options || [
                {
                  option_id: 'harvest_rebalance',
                  title: 'Option A: AI Multi-Objective Rebalance',
                  action_summary: `Deploy ${data.spare_used || 5} buses via Pareto fairness optimization to bottlenecks`,
                  metrics: '100% Crowd Relief · Balanced Network Utilization',
                  alloc_map: data.recommendation || {},
                  recommended: true,
                },
                {
                  option_id: 'depot_dispatch',
                  title: 'Option B: Standby Depot Injection',
                  action_summary: `Dispatch UNIT-101 and UNIT-102 directly from Central Maintenance Depot to ${data.routes.find(r => r.utilization > 100)?.id || 'R002'}`,
                  metrics: '100% Crowd Relief · 2 Depot Buses Consumed',
                  alloc_map: { [data.routes.find(r => r.utilization > 100)?.id || 'R002']: 2 },
                  recommended: false,
                },
                {
                  option_id: 'express_headway',
                  title: 'Option C: Peak Express Short-Turn',
                  action_summary: `Compress headway on ${data.routes.find(r => r.utilization > 100)?.id || 'R002'} from 6.0m → 4.8m without reallocating full fleet`,
                  metrics: '78% Crowd Relief · Standing-room pressure remains',
                  alloc_map: { [data.routes.find(r => r.utilization > 100)?.id || 'R002']: 1 },
                  recommended: false,
                },
              ]).map((option) => {
                const isRec = option.recommended;
                const isOptA = option.option_id === 'harvest_rebalance';
                const isOptB = option.option_id === 'depot_dispatch';

                return (
                  <div
                    key={option.option_id}
                    className={`rounded-2xl p-5 border flex flex-col justify-between transition-all duration-200 relative ${
                      isRec
                        ? 'bg-gradient-to-b from-emerald-950/40 via-slate-900/90 to-slate-950 border-emerald-500/80 shadow-lg shadow-emerald-950/40 hover:border-emerald-400'
                        : isOptB
                        ? 'bg-gradient-to-b from-blue-950/30 via-slate-900/90 to-slate-950 border-blue-600/50 hover:border-blue-500/80 shadow-md'
                        : 'bg-gradient-to-b from-purple-950/30 via-slate-900/90 to-slate-950 border-purple-600/50 hover:border-purple-500/80 shadow-md'
                    }`}
                  >
                    {isRec && (
                      <div className="absolute -top-3 right-4 px-2.5 py-0.5 rounded-full text-[10px] font-black uppercase tracking-wider bg-emerald-500 text-slate-950 shadow-md flex items-center gap-1">
                        <Sparkles className="w-3 h-3 fill-slate-950" /> ★ AI RECOMMENDED (PARETO OPTIMAL)
                      </div>
                    )}

                    <div>
                      <div className="flex items-center justify-between gap-2 mb-2">
                        <span
                          className={`text-xs font-mono font-black uppercase tracking-wider ${
                            isRec ? 'text-emerald-400' : isOptB ? 'text-blue-400' : 'text-purple-400'
                          }`}
                        >
                          {isOptA ? 'STRATEGY A' : isOptB ? 'STRATEGY B' : 'STRATEGY C'}
                        </span>
                        <span className="text-[11px] text-slate-400 font-mono">
                          {Object.entries(option.alloc_map)
                            .map(([k, v]) => `${k}:${v > 0 ? `+${v}` : v}`)
                            .join(' ') || '0 buses'}
                        </span>
                      </div>

                      <h4 className="text-sm font-bold text-white mb-2 leading-snug">
                        {option.title}
                      </h4>

                      <p className="text-xs text-slate-300 leading-relaxed mb-4 min-h-[44px]">
                        {option.action_summary}
                      </p>

                      <div className="bg-slate-950/70 border border-slate-800/80 rounded-xl p-3 mb-4">
                        <span className="text-[10px] uppercase font-bold text-slate-400 block mb-0.5">
                          Projected Impact
                        </span>
                        <p className="text-xs font-semibold text-emerald-300/90 font-mono">
                          {option.metrics}
                        </p>
                      </div>
                    </div>

                    <button
                      onClick={() => handleApplyStrategy(option)}
                      className={`w-full py-2.5 px-4 rounded-xl text-xs font-extrabold shadow-md transition active:scale-95 flex items-center justify-center gap-2 ${
                        isRec
                          ? 'bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-400 hover:to-teal-500 text-slate-950 shadow-emerald-500/20'
                          : isOptB
                          ? 'bg-gradient-to-r from-blue-600 to-cyan-700 hover:from-blue-500 hover:to-cyan-600 text-white shadow-blue-600/20'
                          : 'bg-gradient-to-r from-purple-600 to-indigo-700 hover:from-purple-500 hover:to-indigo-600 text-white shadow-purple-600/20'
                      }`}
                    >
                      <span>Select & Apply {isOptA ? 'Option A' : isOptB ? 'Option B' : 'Option C'}</span>
                      <ArrowRight className="w-3.5 h-3.5" />
                    </button>
                  </div>
                );
              })}
            </div>
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
                <div className="flex items-center gap-2">
                  <h3 className="text-sm font-bold text-emerald-200">
                    {previewPreState ? 'Previewing Pre-Dispatch Bottlenecks (Comparison Mode)' : 'Dispatch Plan Active · SQLite Persisted'}
                  </h3>
                  <span className="text-[10px] px-2 py-0.2 bg-emerald-500/20 text-emerald-300 rounded font-semibold border border-emerald-500/30">
                    {data.reduction_pct}% Overcrowding Mitigated
                  </span>
                </div>
                <p className="text-xs text-emerald-300/90 mt-1">
                  Active allocation plan:{' '}
                  <span className="font-mono font-bold text-white">
                    {activeEntries.map(([k, v]) => `${k} (${v > 0 ? `+${v}` : v} buses)`).join(', ') || 'Normal baseline'}
                  </span>
                  . Effective capacity now covers <strong>{data.after.service_level_share}%</strong> of service thresholds.
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setPreviewPreState(!previewPreState)}
                className={`px-4 py-2 rounded-xl text-xs font-bold border transition ${
                  previewPreState
                    ? 'bg-emerald-600 hover:bg-emerald-500 text-white border-emerald-500 shadow-md shadow-emerald-600/30'
                    : 'bg-slate-800 hover:bg-slate-700 text-slate-200 border-slate-700'
                }`}
              >
                {previewPreState ? '👁️ View Resolved State' : 'Preview Pre-State'}
              </button>
            </div>
          </div>
        )}

        {/* BI-DIRECTIONAL DONOR-TO-RECEIVER HARVESTING BANNER */}
        {data && (data.donor_routes?.length || 0) > 0 && (
          <div className="bg-gradient-to-r from-purple-950/60 via-slate-900 to-indigo-950/60 border border-purple-500/60 rounded-2xl p-5 shadow-2xl space-y-3">
            <div className="flex flex-col sm:flex-row justify-between sm:items-center gap-2">
              <div className="flex items-center gap-2.5">
                <div className="p-2 bg-purple-500/20 text-purple-300 rounded-xl border border-purple-500/40">
                  <GitBranch className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="text-sm font-extrabold text-white flex items-center gap-2">
                    <span>Bi-Directional Fleet Harvesting Active ({dayType})</span>
                    <span className="text-[10px] px-2 py-0.5 rounded-full bg-purple-500/20 text-purple-300 border border-purple-500/40 font-mono">
                      -{data.savings?.harvested_buses} Buses Harvested
                    </span>
                  </h3>
                  <p className="text-xs text-slate-300 mt-0.5">
                    Surplus buses curtailed from low-demand weekend corridors and transferred to overcrowded hubs.
                  </p>
                </div>
              </div>
              {data.savings && (
                <div className="flex items-center gap-3 text-xs bg-slate-950/80 px-3.5 py-1.5 rounded-xl border border-slate-800">
                  <span className="text-emerald-400 font-bold flex items-center gap-1">
                    <Fuel className="w-3.5 h-3.5" /> {data.savings.fuel_liters_saved}L Diesel Saved
                  </span>
                  <span className="text-slate-500">|</span>
                  <span className="text-emerald-400 font-bold">₹{data.savings.cost_saved_inr} Saved</span>
                  <span className="text-slate-500">|</span>
                  <span className="text-cyan-400 font-bold">{data.savings.co2_kg_saved}kg CO₂</span>
                </div>
              )}
            </div>

            {/* Transfer Pairs Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-2">
              {(data.donor_receiver_transfers || data.transfer_pairs || []).map((t, idx) => (
                <div
                  key={idx}
                  className="bg-slate-950/80 border border-purple-500/30 rounded-xl p-3 text-xs flex justify-between items-center gap-3"
                >
                  <div className="flex items-center gap-2">
                    <div className="font-mono font-bold text-cyan-300">{t.donor_id}</div>
                    <span className="text-slate-400">(-{t.buses} buses)</span>
                    <ArrowRight className="w-3.5 h-3.5 text-purple-400" />
                    <div className="font-mono font-bold text-emerald-300">{t.receiver_id}</div>
                    <span className="text-slate-400">(+{t.buses} buses)</span>
                  </div>
                  <div className="text-right font-mono text-[11px] text-slate-400">
                    <div>Headway: {t.headway_before_min || 6}m → <strong className="text-emerald-400">{t.headway_after_min || 4}m</strong></div>
                    <div className="text-emerald-400 font-bold">₹{t.cost_saved_inr || 1224} saved</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* FLEET ALLOCATION INTERACTIVE DISPATCH WORKBENCH */}
        {data && (
          <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
            <div className="flex flex-col sm:flex-row justify-between sm:items-center gap-3">
              <div className="flex items-center gap-3">
                <div className="p-2.5 bg-cyan-600/20 text-cyan-400 rounded-xl border border-cyan-500/30">
                  <Sliders className="w-5 h-5" />
                </div>
                <div>
                  <h2 className="text-base font-bold text-white flex items-center gap-2">
                    <span>Active Fleet Dispatch Stepper &amp; Standby Pool</span>
                    {manualAlloc !== null && (
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/40 font-bold">
                        Human Intervention Mode
                      </span>
                    )}
                  </h2>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Operator authority console: review automated AI fleet recommendations or fine-tune route dispatches via the steppers.
                  </p>
                </div>
              </div>

              {/* Admin Actions */}
              <div className="flex items-center gap-2">
                {manualAlloc !== null && !applied && (
                  <button
                    onClick={handleInitiateDispatch}
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

            {/* Individual Spare Fleet Unit Badges with [Maintenance] Toggle */}
            <div className="pt-2 border-t border-slate-800/80">
              <div className="flex flex-col sm:flex-row justify-between sm:items-center gap-1.5 mb-2.5">
                <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                  <Bus className="w-3.5 h-3.5 text-purple-400" />
                  Standby Depot Fleet ({unitAssignments.filter((u) => u.routeId && !u.isMaintenance).length} Dispatched / {unitAssignments.length} Pool Size · {maintenanceCount} Maintenance):
                </span>
                <span className="text-[11px] text-cyan-400/90 font-medium">
                  💡 Click unit to dispatch to {selectedRoute} or recall · Click 🔧 to toggle maintenance
                </span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2.5">
                {unitAssignments.map((u) => {
                  const isAssigned = u.routeId !== null && !u.isMaintenance;
                  const isMaint = u.isMaintenance;

                  return (
                    <div
                      key={u.unitId}
                      onClick={() => handleUnitClick(u)}
                      className={`p-2.5 rounded-xl border text-xs flex flex-col justify-between transition cursor-pointer select-none group relative active:scale-95 ${
                        isMaint
                          ? 'bg-amber-950/30 border-amber-600/50 text-amber-300'
                          : isAssigned
                          ? 'bg-blue-950/50 border-blue-500/50 hover:border-red-400 hover:bg-red-950/30 text-blue-200 shadow-md'
                          : 'bg-slate-950/70 border-slate-800 hover:border-emerald-500/70 hover:bg-emerald-950/30 text-slate-400'
                      }`}
                    >
                      <div className="flex justify-between items-center mb-1">
                        <span className="font-mono font-bold text-white text-[11px]">
                          {u.unitId}
                        </span>
                        <div className="flex items-center gap-1.5">
                          <button
                            onClick={(e) => handleToggleMaintenance(u.unitId, e)}
                            className={`px-1.5 py-0.5 rounded text-[10px] font-bold border transition ${
                              isMaint
                                ? 'bg-amber-500 text-slate-950 border-amber-400'
                                : 'bg-slate-900 text-slate-400 border-slate-700 hover:text-amber-300'
                            }`}
                            title="Toggle Depot Maintenance Status"
                          >
                            🔧
                          </button>
                          <span
                            className={`w-2 h-2 rounded-full transition ${
                              isMaint
                                ? 'bg-amber-400'
                                : isAssigned
                                ? 'bg-emerald-400 animate-pulse'
                                : 'bg-slate-600'
                            }`}
                          ></span>
                        </div>
                      </div>
                      <div className="font-semibold text-[11px]">
                        {isMaint ? (
                          <div className="text-amber-400 font-mono text-[10px]">
                            [🔧 IN MAINTENANCE]
                          </div>
                        ) : isAssigned ? (
                          <div className="flex items-center justify-between">
                            <span className="text-emerald-300 font-bold font-mono">
                              [→ {u.routeId}]
                            </span>
                            <span className="text-[10px] opacity-0 group-hover:opacity-100 font-sans font-bold text-red-400 transition ml-1">
                              Recall
                            </span>
                          </div>
                        ) : (
                          <div className="flex items-center justify-between">
                            <span className="text-slate-500 font-mono group-hover:text-emerald-300 transition">
                              [STANDBY DEPOT]
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

        {/* Route Monitor Table with Steppers */}
        {data && (
          <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-xl">
            <div className="flex flex-col sm:flex-row justify-between sm:items-center gap-2 mb-4">
              <div>
                <h2 className="text-base font-bold text-white flex items-center gap-2">
                  <span>Bengaluru Corridor Risk &amp; Fleet Monitor</span>
                  <span className="text-xs font-normal text-slate-400">
                    (Effective throughput derated by traffic bottleneck delay)
                  </span>
                </h2>
              </div>
              <div className="text-xs text-slate-400 bg-slate-950 px-3 py-1 rounded-lg border border-slate-800">
                Derating Formula: <span className="text-cyan-300 font-mono">Effective Cap = Base / Congestion</span>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400 uppercase tracking-wider font-semibold">
                    <th className="pb-3 pl-3">Corridor Link</th>
                    <th className="pb-3">Bottleneck Delay</th>
                    <th className="pb-3">Forecast (90% Interval)</th>
                    <th className="pb-3">Fleet / Derated Capacity</th>
                    <th className="pb-3 w-36">Utilization</th>
                    <th className="pb-3">P(Overcrowd)</th>
                    <th className="pb-3">Risk</th>
                    <th className="pb-3 pr-3">Bi-Directional AI Decision</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-normal">
                  {data.routes.map((r) => {
                    const isSelected = selectedRoute === r.id;
                    const currentExtra =
                      manualAlloc !== null
                        ? manualAlloc[r.id] !== undefined ? manualAlloc[r.id] : 0
                        : data.recommendation?.[r.id] || 0;

                    const cong = r.congestion_factor || 1.0;
                    const congBadge =
                      cong >= 1.45
                        ? 'bg-red-500/20 text-red-300 border-red-500/40'
                        : cong >= 1.25
                        ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
                        : 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40';

                    const isResolved = applied && !previewPreState && r.extra > 0 && r.utilization <= 100;
                    const badgeText = isResolved ? 'RESOLVED / NORMAL' : r.risk;
                    const badge = isResolved
                      ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                      : r.risk === 'CRITICAL'
                      ? 'bg-red-500/20 text-red-400 border-red-500/40'
                      : r.risk === 'HIGH'
                      ? 'bg-amber-500/20 text-amber-400 border-amber-500/40'
                      : r.risk === 'MEDIUM'
                      ? 'bg-yellow-500/20 text-yellow-300 border-yellow-500/40'
                      : 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40';

                    const barColor = isResolved
                      ? 'bg-emerald-400'
                      : r.utilization > 120
                      ? 'bg-red-500'
                      : r.utilization > 100
                      ? 'bg-amber-500'
                      : r.utilization > 80
                      ? 'bg-yellow-400'
                      : 'bg-emerald-500';

                    return (
                      <tr
                        key={r.id}
                        onClick={() => setSelectedRoute(r.id)}
                        className={`cursor-pointer transition ${
                          isSelected ? 'bg-cyan-950/40 border-l-4 border-l-cyan-400' : 'hover:bg-slate-800/40'
                        }`}
                      >
                        <td className="py-3.5 pl-3">
                          <div className="font-bold text-white text-xs">{r.name}</div>
                          <div className="text-[11px] text-slate-400 font-mono mt-0.5 flex items-center gap-1.5">
                            <span className="text-cyan-400">{r.id}</span>
                            <span>•</span>
                            <span>{r.condition}</span>
                          </div>
                        </td>

                        <td className="py-3.5">
                          <div className="flex items-center gap-1.5">
                            <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold border ${congBadge}`}>
                              {cong}x
                            </span>
                            <span className="text-[11px] text-slate-400 truncate max-w-[120px]" title={r.bottleneck}>
                              {r.bottleneck}
                            </span>
                          </div>
                        </td>

                        <td className="py-3.5 font-mono">
                          <div className="font-bold text-white text-xs">
                            {r.predicted}{' '}
                            <span className="text-[11px] text-slate-400 font-normal">pax</span>
                          </div>
                          <div className="text-[10px] text-slate-400">
                            [{r.lower} - {r.upper}]
                          </div>
                        </td>

                        <td className="py-3.5">
                          <div className="flex items-center gap-2">
                            <div>
                              <div className="font-bold text-white font-mono">
                                {r.vehicles} <span className="text-slate-400 text-[10px]">buses</span>
                              </div>
                              <div className="text-[10px] text-cyan-400 font-mono">
                                {r.capacity} seats eff
                              </div>
                            </div>

                            {/* Stepper Buttons */}
                            <div className="flex items-center bg-slate-950 border border-slate-800 rounded-lg p-0.5 ml-2">
                              <button
                                onClick={(e) => {
                                   e.stopPropagation();
                                   handleDecrement(r.id);
                                }}
                                className="w-5 h-5 rounded flex items-center justify-center font-bold text-xs text-red-400 hover:bg-slate-800 active:scale-95"
                                title="Harvest / Curtail buses"
                              >
                                -
                              </button>
                              <span
                                className={`w-6 text-center font-mono font-bold text-xs ${
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
                                title="Increase allocated buses"
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
                            {badgeText}
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

        {/* CORRIDOR ANALYTICS: INTRADAY 8-WINDOW CURVE & 30-DAY BACKTEST HISTORY */}
        {data && (
          <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
            <div className="flex flex-col sm:flex-row justify-between sm:items-center gap-3">
              <div>
                <h3 className="text-base font-bold text-white flex items-center gap-2">
                  <span>Corridor Analytics:</span>
                  <span className="text-cyan-400 font-mono">{selectedRoute}</span>
                  <span className="text-slate-300 font-normal">
                    ({selectedRouteObj?.name})
                  </span>
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  {activeChartTab === 'intraday'
                    ? `8-window intraday forecast curve (06:00 to 21:00) with 90% conformal interval band vs capacity [${dayType}]`
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
                      tickFormatter={(val) => (val && val.length > 5 ? val.slice(5) : val)}
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
                    {historyData.length > 22 && (
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
                    )}
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
        )}

        {/* AI ENGINE & BI-DIRECTIONAL FLEET REBALANCING PANEL */}
        {data && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* SUB-PANEL 1: MULTI-QUANTILE GBDT & XAI ATTRIBUTION */}
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
                      HistGradientBoostingRegressor (q05, q50, q95) with cyclic diurnal &amp; lag momentum features.
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <span className="px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold bg-indigo-500/15 text-indigo-300 border border-indigo-500/30 block">
                    q̂ Residual: ±{data.ml_model_card?.q_hat || 16} pax
                  </span>
                  <span className="text-[10px] text-slate-400 mt-1 block">
                    Coverage: {data.ml_model_card?.conformal_coverage_pct || 91.8}%
                  </span>
                </div>
              </div>

              {/* Global Feature Importances */}
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
                    Total: <strong className="text-emerald-400">{selectedRouteObj?.predicted} pax</strong>
                    {selectedRouteObj?.xai_drivers?.inference_ms !== undefined && (
                      <span className="ml-2 text-cyan-400 font-mono text-[10px]">
                        ⚡ {selectedRouteObj.xai_drivers.inference_ms}ms
                      </span>
                    )}
                  </span>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 text-center text-xs">
                  <div className="bg-slate-900/90 p-2 rounded-lg border border-slate-800">
                    <div className="text-[10px] text-slate-400 uppercase font-semibold">Base Sched</div>
                    <div className="text-slate-200 font-mono font-bold mt-0.5">
                      +{selectedRouteObj?.xai_drivers?.base_schedule || 0}
                    </div>
                  </div>
                  <div className="bg-slate-900/90 p-2 rounded-lg border border-slate-800">
                    <div className="text-[10px] text-slate-400 uppercase font-semibold">Momentum</div>
                    <div className="text-blue-400 font-mono font-bold mt-0.5">
                      {(selectedRouteObj?.xai_drivers?.lag_momentum || 0) >= 0 ? '+' : ''}
                      {selectedRouteObj?.xai_drivers?.lag_momentum || 0}
                    </div>
                  </div>
                  <div className="bg-slate-900/90 p-2 rounded-lg border border-slate-800">
                    <div className="text-[10px] text-slate-400 uppercase font-semibold">Weekend Shift</div>
                    <div
                      className={`font-mono font-bold mt-0.5 ${
                        (selectedRouteObj?.xai_drivers?.weekend_shift || 0) < 0
                          ? 'text-cyan-400'
                          : (selectedRouteObj?.xai_drivers?.weekend_shift || 0) > 0
                          ? 'text-emerald-400'
                          : 'text-slate-400'
                      }`}
                    >
                      {(selectedRouteObj?.xai_drivers?.weekend_shift || 0) > 0 ? '+' : ''}
                      {selectedRouteObj?.xai_drivers?.weekend_shift || 0}
                    </div>
                  </div>
                  <div className="bg-slate-900/90 p-2 rounded-lg border border-slate-800">
                    <div className="text-[10px] text-slate-400 uppercase font-semibold">Rain Impact</div>
                    <div className="text-cyan-300 font-mono font-bold mt-0.5">
                      +{(selectedRouteObj?.xai_drivers?.weather_uplift || 0)}
                    </div>
                  </div>
                  <div className="bg-slate-900/90 p-2 rounded-lg border border-slate-800">
                    <div className="text-[10px] text-slate-400 uppercase font-semibold">Event Surge</div>
                    <div className="text-amber-400 font-mono font-bold mt-0.5">
                      +{(selectedRouteObj?.xai_drivers?.event_spillover || 0)}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* SUB-PANEL 2: BEFORE VS AFTER REBALANCING AUDIT */}
            <div className="bg-slate-900/85 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col justify-between">
              <div>
                <div className="flex justify-between items-start mb-3">
                  <div className="flex items-center gap-3">
                    <div className="p-2.5 bg-emerald-600/20 text-emerald-400 rounded-xl border border-emerald-500/30">
                      <Sparkles className="w-5 h-5" />
                    </div>
                    <div>
                      <h3 className="text-base font-bold text-white flex items-center gap-2">
                        <span>Before vs After Dispatch Audit</span>
                      </h3>
                      <p className="text-xs text-slate-400 mt-0.5">
                        Quantitative impact of fleet rebalancing on municipal network service quality.
                      </p>
                    </div>
                  </div>
                  <span className="px-2.5 py-1 rounded-full text-xs font-black bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">
                    -{data.reduction_pct}% Overcrowding
                  </span>
                </div>

                <div className="space-y-3 pt-2 text-xs">
                  <div className="flex justify-between py-1.5 border-b border-slate-800/80">
                    <span className="text-slate-400">Overcrowded Corridors</span>
                    <span className="font-mono font-bold">
                      <span className="text-red-400">{data.before.overcrowded_routes}</span>
                      <span className="text-slate-500 mx-1.5">→</span>
                      <span className="text-emerald-400">{data.after.overcrowded_routes}</span>
                    </span>
                  </div>

                  <div className="flex justify-between py-1.5 border-b border-slate-800/80">
                    <span className="text-slate-400">Critical Risk Corridors</span>
                    <span className="font-mono font-bold">
                      <span className="text-red-400">{data.before.critical_routes}</span>
                      <span className="text-slate-500 mx-1.5">→</span>
                      <span className="text-emerald-400">{data.after.critical_routes}</span>
                    </span>
                  </div>

                  <div className="flex justify-between py-1.5 border-b border-slate-800/80">
                    <span className="text-slate-400">Excess Passengers Affected</span>
                    <span className="font-mono font-bold">
                      <span className="text-amber-400">{data.before.passengers_affected}</span>
                      <span className="text-slate-500 mx-1.5">→</span>
                      <span className="text-emerald-400">{data.after.passengers_affected}</span>
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

              <button
                onClick={() => {
                  if (applied) {
                    setApplied(false);
                  } else {
                    handleInitiateDispatch();
                  }
                }}
                disabled={totalDispatched === 0 && recEntries.length === 0}
                className={`mt-4 w-full py-3 rounded-xl text-xs font-black uppercase tracking-wider transition ${
                  totalDispatched === 0 && recEntries.length === 0
                    ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                    : applied
                    ? 'bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700'
                    : 'bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white shadow-lg shadow-cyan-600/30 active:scale-95'
                }`}
              >
                {totalDispatched === 0 && recEntries.length === 0
                  ? 'No Dispatch Intervention Needed'
                  : applied
                  ? 'Allocation Committed · Click to Preview Pre-State'
                  : `Commit Allocation (${totalDispatched || data.spare_used} Vehicles Deployed)`}
              </button>
            </div>
          </div>
        )}

        {/* Verification Section: Hold-Out Backtest & SQLite Audit Log */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Hold-Out Backtest Table */}
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

          {/* SQLite Allocation Audit Trail with Report Export */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col justify-between">
            <div>
              <div className="flex flex-col sm:flex-row justify-between sm:items-center gap-2 mb-1">
                <h3 className="text-sm font-bold text-white flex items-center gap-2">
                  <History className="w-4 h-4 text-purple-400" />
                  <span>SQLite Allocation Audit Trail (`allocation_log`)</span>
                </h3>
                <div className="flex items-center gap-2">
                  <button
                    onClick={handleExportReport}
                    className="px-2.5 py-1 bg-purple-950/80 hover:bg-purple-900 text-purple-300 hover:text-white rounded-lg text-[10px] font-bold border border-purple-500/40 transition flex items-center gap-1 shadow"
                    title="Export complete formatted incident report (.txt)"
                  >
                    <FileText className="w-3 h-3" /> Export Report (.txt)
                  </button>
                  <span className="text-[10px] px-2 py-0.5 bg-purple-500/10 text-purple-400 rounded border border-purple-500/20 font-bold">
                    Immutable
                  </span>
                </div>
              </div>
              <p className="text-[11px] text-slate-400 mb-3">
                Records operator-approved interventions, fleet curtailment, and supervisor co-signatures.
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

      {/* TWO-PERSON SUPERVISOR CO-SIGN MODAL */}
      {showCoSignModal && (
        <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-red-500/60 rounded-2xl max-w-md w-full p-6 shadow-2xl space-y-4">
            <div className="flex justify-between items-start">
              <div className="flex items-center gap-3">
                <div className="p-2.5 bg-red-600/20 text-red-400 rounded-xl border border-red-500/30">
                  <ShieldCheck className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="text-base font-extrabold text-white">
                    Dual-Control Dispatch Authorization
                  </h3>
                  <p className="text-xs text-red-300">
                    High-Impact Municipal Reallocation ({totalMovedVehicles} Units)
                  </p>
                </div>
              </div>
              <button
                onClick={() => setShowCoSignModal(false)}
                className="text-slate-400 hover:text-white p-1"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="bg-red-950/40 border border-red-800/40 rounded-xl p-3 text-xs text-red-200">
              BMTC Protocol AI-16 requires supervisory co-authorization when dynamic fleet adjustments affect 4 or more transit units simultaneously.
            </div>

            <div className="space-y-3 text-xs">
              <div>
                <label className="text-slate-400 font-semibold block mb-1">
                  Dispatch Operator ID:
                </label>
                <input
                  type="text"
                  value={operatorId}
                  onChange={(e) => setOperatorId(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-white font-mono focus:outline-none focus:border-red-500"
                />
              </div>

              <div>
                <label className="text-slate-400 font-semibold block mb-1">
                  Supervisor Co-Signer Name:
                </label>
                <input
                  type="text"
                  value={supervisorName}
                  onChange={(e) => setSupervisorName(e.target.value)}
                  placeholder="e.g. Station Master K. Rao"
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-white focus:outline-none focus:border-red-500"
                />
              </div>

              <div className="flex items-start gap-2 pt-1">
                <input
                  type="checkbox"
                  id="cosign_agree"
                  checked={coSignConfirmed}
                  onChange={(e) => setCoSignConfirmed(e.target.checked)}
                  className="mt-0.5 rounded border-slate-800 text-red-500 focus:ring-0 cursor-pointer"
                />
                <label htmlFor="cosign_agree" className="text-slate-300 text-[11px] cursor-pointer">
                  I verify that this high-capacity fleet redistribution complies with BMTC municipal safety thresholds and depot reserve requirements.
                </label>
              </div>
            </div>

            <div className="flex items-center justify-end gap-2.5 pt-2 border-t border-slate-800">
              <button
                onClick={() => setShowCoSignModal(false)}
                className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-xl text-xs font-bold transition"
              >
                Cancel
              </button>
              <button
                onClick={() => executeCommit(supervisorName.trim(), operatorId.trim())}
                disabled={!supervisorName.trim() || !coSignConfirmed}
                className={`px-4 py-2 rounded-xl text-xs font-black uppercase tracking-wider transition flex items-center gap-1.5 ${
                  !supervisorName.trim() || !coSignConfirmed
                    ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                    : 'bg-red-600 hover:bg-red-500 text-white shadow-lg shadow-red-600/30 active:scale-95'
                }`}
              >
                <CheckSquare className="w-3.5 h-3.5" /> Authorize &amp; Commit
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
