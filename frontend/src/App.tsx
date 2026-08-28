import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  Antenna,
  ArrowDownRight,
  ArrowUpRight,
  BarChart3,
  BrainCircuit,
  Check,
  ChevronRight,
  CircleStop,
  Clock3,
  Cpu,
  Gauge,
  Layers3,
  LayoutDashboard,
  MapPinned,
  Pause,
  Play,
  Radio,
  RadioTower,
  RefreshCw,
  Server,
  ShieldAlert,
  ShieldCheck,
  Signal,
  SlidersHorizontal,
  Sparkles,
  Square,
  TowerControl,
  Users,
  Wifi,
  X,
  Zap,
} from "lucide-react";
import { api, openLiveSocket } from "./api";
import type {
  Cell,
  ComparisonRow,
  Decision,
  DecisionHistoryRow,
  DeliveryGroup,
  KPIRecord,
  Mode,
  SimConfig,
  SimulationStatus,
  TickMessage,
  UE,
} from "./types";

const MODE_COLOR: Record<string, string> = { unicast: "#f3a45d", broadcast: "#55d6c2", none: "#596575" };
const POLICY_LABEL: Record<string, string> = {
  "always-unicast": "Always Unicast",
  "always-broadcast": "Always Broadcast",
  "rule-by-group-size": "Rule-based",
  ai: "AI Controller",
};

function fmt(value: number | null | undefined, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: digits, minimumFractionDigits: digits }).format(value);
}
function pct(value: number | null | undefined, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${fmt(value * 100, digits)}%`;
}
function safeArray<T>(value: T[] | undefined | null): T[] { return Array.isArray(value) ? value : []; }
function parseModeBreakdown(value = "") {
  return value.split(",").reduce<Record<string, number>>((acc, part) => {
    const [mode, count] = part.split("=");
    if (mode && count) acc[mode] = Number(count) || 0;
    return acc;
  }, {});
}
function statusTone(status?: string) {
  if (status === "running") return "live";
  if (status === "paused") return "paused";
  if (status === "failed") return "danger";
  if (status === "completed" || status === "stopped") return "muted";
  return "idle";
}
function shortId(id?: string) { return id ? id.replace("sim_", "SIM-").slice(0, 15) : "—"; }
function formatPolicy(policy?: string) { return policy ? POLICY_LABEL[policy] || policy.replace(/-/g, " ") : "—"; }
function modeLabel(mode?: string) { return mode === "broadcast" ? "Broadcast" : mode === "unicast" ? "Unicast" : "None"; }

function SectionHeading({ eyebrow, title, copy, icon }: { eyebrow: string; title: string; copy?: string; icon: React.ReactNode }) {
  return <div className="section-heading"><div className="section-icon">{icon}</div><div><div className="eyebrow">{eyebrow}</div><h2>{title}</h2>{copy && <p>{copy}</p>}</div></div>;
}

function MetricCard({ label, value, unit, helper, accent = "teal", icon, trend }: { label: string; value: string; unit?: string; helper?: string; accent?: string; icon: React.ReactNode; trend?: "up" | "down" }) {
  return <div className={`metric-card accent-${accent}`}><div className="metric-top"><span>{label}</span><span className="metric-icon">{icon}</span></div><div className="metric-value">{value}<small>{unit}</small></div><div className="metric-bottom">{trend && <span className={`trend ${trend}`}>{trend === "up" ? <ArrowUpRight size={13} /> : <ArrowDownRight size={13} />}</span>}<span>{helper || "Backend measurement"}</span></div></div>;
}

function ProgressBar({ value, color = "teal", label, right }: { value: number; color?: string; label?: string; right?: string }) {
  const width = Math.max(0, Math.min(100, value * 100));
  return <div className="progress-wrap">{(label || right) && <div className="progress-label"><span>{label}</span><strong>{right ?? pct(value)}</strong></div>}<div className="progress-track"><div className={`progress-fill ${color}`} style={{ width: `${width}%` }} /></div></div>;
}

function Sparkline({ values, color = "#55d6c2", fill = true, height = 94 }: { values: number[]; color?: string; fill?: boolean; height?: number }) {
  if (!values.length) return <div className="empty-chart"><Activity size={16} />Waiting for live ticks</div>;
  const max = Math.max(...values, 0.0001);
  const min = Math.min(...values, 0);
  const spread = Math.max(max - min, 0.0001);
  const points = values.map((value, index) => `${(index / Math.max(values.length - 1, 1)) * 100},${height - 10 - ((value - min) / spread) * (height - 22)}`).join(" ");
  const area = `0,${height} ${points} 100,${height}`;
  return <svg className="sparkline" viewBox={`0 0 100 ${height}`} preserveAspectRatio="none" role="img" aria-label="Metric trend"><defs><linearGradient id={`fill-${color.replace("#", "")}`} x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor={color} stopOpacity=".25" /><stop offset="1" stopColor={color} stopOpacity="0" /></linearGradient></defs>{fill && <polygon points={area} fill={`url(#fill-${color.replace("#", "")})`} />}<polyline points={points} fill="none" stroke={color} strokeWidth="1.6" vectorEffect="non-scaling-stroke" /></svg>;
}

function DualSparkline({ primary, secondary }: { primary: number[]; secondary: number[] }) {
  if (!primary.length) return <div className="empty-chart"><Activity size={16} />Waiting for live ticks</div>;
  const max = Math.max(...primary, ...secondary, 1);
  const path = (values: number[]) => values.map((value, index) => `${(index / Math.max(values.length - 1, 1)) * 100},${94 - (value / max) * 78}`).join(" ");
  return <svg className="sparkline" viewBox="0 0 100 94" preserveAspectRatio="none" role="img" aria-label="Unicast and broadcast group trend"><polyline points={path(primary)} fill="none" stroke={MODE_COLOR.broadcast} strokeWidth="1.5" vectorEffect="non-scaling-stroke" /><polyline points={path(secondary)} fill="none" stroke={MODE_COLOR.unicast} strokeWidth="1.5" vectorEffect="non-scaling-stroke" /></svg>;
}

function Donut({ unicast, broadcast }: { unicast: number; broadcast: number }) {
  const total = unicast + broadcast;
  const ratio = total ? unicast / total : 0;
  const dash = `${ratio * 100} ${100 - ratio * 100}`;
  return <div className="donut-wrap"><svg viewBox="0 0 42 42" className="donut"><circle cx="21" cy="21" r="15.9" fill="none" stroke="#182636" strokeWidth="5" /><circle cx="21" cy="21" r="15.9" fill="none" stroke={MODE_COLOR.unicast} strokeWidth="5" strokeDasharray={dash} strokeDashoffset="25" pathLength="100" /><circle cx="21" cy="21" r="15.9" fill="none" stroke={MODE_COLOR.broadcast} strokeWidth="5" strokeDasharray={`${100 - ratio * 100} ${ratio * 100}`} strokeDashoffset={`${25 - ratio * 100}`} pathLength="100" /><text x="21" y="21" textAnchor="middle" dominantBaseline="central" className="donut-number">{total || "—"}</text><text x="21" y="26.3" textAnchor="middle" className="donut-caption">groups</text></svg></div>;
}

function LiveMap({ cells, ues, selectedCell, onSelectCell }: { cells: Cell[]; ues: UE[]; selectedCell: string; onSelectCell: (id: string) => void }) {
  const all = [...cells.map((c) => ({ x: c.center_x_km, y: c.center_y_km })), ...ues.map((u) => ({ x: u.x_km, y: u.y_km }))];
  const xs = all.map((item) => item.x); const ys = all.map((item) => item.y);
  const minX = xs.length ? Math.min(...xs) : 0; const maxX = xs.length ? Math.max(...xs) : 10; const minY = ys.length ? Math.min(...ys) : 0; const maxY = ys.length ? Math.max(...ys) : 10;
  const dx = Math.max(maxX - minX, 1); const dy = Math.max(maxY - minY, 1);
  const point = (x: number, y: number) => ({ left: `${8 + ((x - minX) / dx) * 84}%`, top: `${90 - ((y - minY) / dy) * 78}%` });
  const visibleUes = selectedCell === "all" ? ues : ues.filter((ue) => ue.cell_id === selectedCell);
  return <div className="map-card"><div className="map-grid" />{visibleUes.map((ue) => { const pos = point(ue.x_km, ue.y_km); return <div key={ue.ue_id} className={`ue-dot mode-${ue.delivery_mode} qoe-${ue.qoe_state}`} style={pos} title={`${ue.ue_id} · ${ue.delivery_mode} · ${fmt(ue.buffer_seconds)}s buffer`} />; })}{cells.map((cell) => { const pos = point(cell.center_x_km, cell.center_y_km); const active = selectedCell === cell.cell_id; return <button key={cell.cell_id} className={`cell-node ${active ? "selected" : ""}`} style={{ ...pos, borderColor: cell.congestion > .7 ? "#ff796d" : "#55d6c2" }} onClick={() => onSelectCell(active ? "all" : cell.cell_id)} title={`${cell.cell_id} · ${pct(cell.congestion)} congestion`}><span className="cell-ring" style={{ opacity: .2 + cell.congestion * .6 }} /><span className="cell-core"><TowerControl size={16} /></span><b>{cell.cell_id.replace("cell_", "C")}</b><small>{pct(cell.congestion)}</small></button>; })}<div className="map-scale"><span><i className="legend-dot unicast" />Unicast</span><span><i className="legend-dot broadcast" />Broadcast</span><span><i className="legend-dot starved" />Starved UE</span></div>{!cells.length && <div className="map-empty"><MapPinned size={24} /><span>Start a simulation to render live synthetic coordinates.</span></div>}</div>;
}

function DecisionPanel({ decisions, selected, onSelect, policy }: { decisions: Decision[]; selected?: Decision; onSelect: (decision: Decision) => void; policy?: string }) {
  if (policy !== "ai") return <div className="empty-state"><ShieldCheck size={21} /><strong>Decision stream is AI-only</strong><p>Fixed policies deliver their configured mode without an explainable AI decision record.</p></div>;
  return <div className="decision-panel"><div className="decision-list">{decisions.slice(0, 8).map((decision) => <button key={`${decision.cell_id}-${decision.group_id}`} className={`decision-row ${selected?.cell_id === decision.cell_id && selected?.group_id === decision.group_id ? "active" : ""}`} onClick={() => onSelect(decision)}><span className={`decision-badge ${decision.selected_mode}`}>{decision.selected_mode === "broadcast" ? "B" : "U"}</span><span className="decision-main"><strong>{decision.cell_id} / {decision.group_id}</strong><small>{decision.explanation}</small></span><span className="decision-confidence">{pct(decision.confidence)}<small>{decision.switched ? "handover" : "hold"}</small></span></button>)}{!decisions.length && <div className="empty-state compact"><BrainCircuit size={19} /><span>Waiting for the AI controller’s first tick.</span></div>}</div><div className="decision-detail">{selected ? <><div className="detail-top"><span className={`mode-pill ${selected.selected_mode}`}>{selected.selected_mode === "broadcast" ? <Radio size={14} /> : <Wifi size={14} />}{modeLabel(selected.selected_mode)}</span>{selected.switched && <span className="switch-pill"><Zap size={12} /> switched</span>}</div><div className="detail-copy">{selected.explanation}</div><div className="confidence-block"><div className="confidence-meta"><span>Controller confidence</span><strong>{pct(selected.confidence)}</strong></div><ProgressBar value={selected.confidence} color={selected.policy_compliant ? "teal" : "coral"} /></div><div className="detail-facts"><span><b>Final bearer</b>{selected.bearer}</span><span><b>PCF status</b>{selected.policy_compliant ? "Compliant" : "Policy override"}</span><span><b>Previous mode</b>{modeLabel(selected.previous_mode || undefined)}</span></div></> : <div className="detail-placeholder"><Sparkles size={18} /><span>Select a live group to inspect the final mode, confidence, and reason.</span></div>}</div></div>;
}

function ComparisonChart({ rows }: { rows: ComparisonRow[] }) {
  const metrics = [{ key: "resource_savings_vs_baseline", label: "Savings" }, { key: "service_continuity", label: "Continuity" }, { key: "mean_qoe", label: "Mean QoE" }];
  return <div className="comparison-chart">{metrics.map((metric) => <div className="comparison-group" key={metric.key}><div className="comparison-label">{metric.label}</div><div className="bars">{rows.map((row) => { const value = Number(row.summary[metric.key] ?? 0); return <div className="bar-item" key={row.policy} title={`${formatPolicy(row.policy)}: ${pct(value)}`}><div className={`bar ${row.policy === "ai" ? "ai" : row.policy === "always-unicast" ? "unicast" : row.policy === "always-broadcast" ? "broadcast" : "rule"}`} style={{ height: `${Math.max(3, Math.min(value * 100, 100))}%` }} /><span>{row.policy === "ai" ? "AI" : row.policy === "always-unicast" ? "U" : row.policy === "always-broadcast" ? "B" : "R"}</span></div>; })}</div></div>)}</div>;
}

function App() {
  const [backend, setBackend] = useState<"connecting" | "online" | "offline">("connecting");
  const [config, setConfig] = useState<SimConfig>({});
  const [status, setStatus] = useState<SimulationStatus | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [kpi, setKpi] = useState<KPIRecord | null>(null);
  const [kpiHistory, setKpiHistory] = useState<KPIRecord[]>([]);
  const [cells, setCells] = useState<Cell[]>([]);
  const [ues, setUes] = useState<UE[]>([]);
  const [groups, setGroups] = useState<DeliveryGroup[]>([]);
  const [decisions, setDecisions] = useState<Decision[]>([]);
  const [decisionHistory, setDecisionHistory] = useState<DecisionHistoryRow[]>([]);
  const [comparison, setComparison] = useState<ComparisonRow[]>([]);
  const [selectedDecision, setSelectedDecision] = useState<Decision | undefined>();
  const [selectedCell, setSelectedCell] = useState("all");
  const [scenario, setScenario] = useState("flash_crowd");
  const [selector, setSelector] = useState("ai");
  const [ticks, setTicks] = useState(120);
  const [interval, setIntervalValue] = useState(0.2);
  const [seed, setSeed] = useState(11);
  const [error, setError] = useState<string | null>(null);
  const [comparisonBusy, setComparisonBusy] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const ueTimerRef = useRef<number | undefined>(undefined);

  const closeSocket = useCallback(() => { socketRef.current?.close(); socketRef.current = null; }, []);

  const loadSnapshot = useCallback(async (id: string) => {
    const results = await Promise.allSettled([api.cells(id), api.ues(id), api.groups(id), api.decisionsLatest(id), api.decisionsHistory(id, 250), api.kpiHistory(id), api.latestKpi(id)]);
    const [cellResult, ueResult, groupResult, decisionResult, historyResult, kpiHistoryResult, latestResult] = results;
    if (cellResult.status === "fulfilled") setCells(cellResult.value);
    if (ueResult.status === "fulfilled") setUes(ueResult.value.items);
    if (groupResult.status === "fulfilled") setGroups(groupResult.value);
    if (decisionResult.status === "fulfilled") setDecisions(decisionResult.value);
    if (historyResult.status === "fulfilled") setDecisionHistory(historyResult.value);
    if (kpiHistoryResult.status === "fulfilled") setKpiHistory(kpiHistoryResult.value);
    if (latestResult.status === "fulfilled") setKpi(latestResult.value);
  }, []);

  const connectSocket = useCallback((id: string) => {
    closeSocket();
    const socket = openLiveSocket(id, (message) => {
      if (message.type === "tick") {
        const tick = message as TickMessage;
        setKpi(tick.kpi); setCells(tick.cells); setDecisions(tick.decisions); setGroups(tick.groups); setLastUpdate(new Date());
        setKpiHistory((previous) => [...previous.filter((item) => item.tick !== tick.kpi.tick), tick.kpi].slice(-90));
        setStatus((previous) => previous ? { ...previous, status: "running", current_tick: tick.tick } : previous);
        if (tick.decisions.length) setSelectedDecision((previous) => previous && tick.decisions.some((item) => item.cell_id === previous.cell_id && item.group_id === previous.group_id) ? tick.decisions.find((item) => item.cell_id === previous.cell_id && item.group_id === previous.group_id) : tick.decisions.find((item) => item.switched) || tick.decisions[0]);
      } else if (message.type === "status") {
        setStatus(message as SimulationStatus); setLastUpdate(new Date());
      } else if (message.type === "error") setError(message.detail);
    }, () => { socketRef.current = null; }, () => setError("Live WebSocket connection failed. Check that the API server is running."));
    socketRef.current = socket;
  }, [closeSocket]);

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const [health, activeConfig, runs] = await Promise.all([api.health(), api.config(), api.simulations()]);
        if (!mounted) return;
        setBackend(health.status === "ok" ? "online" : "online"); setConfig(activeConfig);
        const candidate = safeArray(runs).find((run) => run.status === "running" || run.status === "paused");
        if (candidate) { setStatus(candidate); setActiveId(candidate.simulation_id); setScenario(candidate.scenario); setSelector(candidate.mode_selector); connectSocket(candidate.simulation_id); void loadSnapshot(candidate.simulation_id); }
      } catch (caught) { if (mounted) { setBackend("offline"); setError(caught instanceof Error ? caught.message : "Unable to reach the FastAPI backend."); } }
    })();
    return () => { mounted = false; closeSocket(); };
  }, [closeSocket, connectSocket, loadSnapshot]);

  useEffect(() => {
    if (!activeId) return;
    const poll = () => void api.ues(activeId, selectedCell === "all" ? undefined : selectedCell).then((payload) => setUes(payload.items)).catch(() => undefined);
    poll();
    ueTimerRef.current = window.setInterval(poll, 1000);
    return () => { if (ueTimerRef.current) window.clearInterval(ueTimerRef.current); };
  }, [activeId, selectedCell]);

  useEffect(() => () => closeSocket(), [closeSocket]);

  const start = async () => {
    try {
      setError(null); closeSocket(); setKpi(null); setKpiHistory([]); setCells([]); setUes([]); setGroups([]); setDecisions([]); setDecisionHistory([]); setSelectedDecision(undefined);
      const next = await api.start({ scenario, mode_selector: selector, ticks, seed, tick_interval_s: interval });
      setStatus(next); setActiveId(next.simulation_id); setBackend("online"); connectSocket(next.simulation_id); void loadSnapshot(next.simulation_id);
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Unable to start simulation."); }
  };
  const control = async (action: "pause" | "resume" | "stop") => {
    if (!activeId) return;
    try { setError(null); const next = await api[action](activeId); setStatus(next); if (action === "stop") closeSocket(); } catch (caught) { setError(caught instanceof Error ? caught.message : `Unable to ${action} simulation.`); }
  };
  const runComparison = async () => {
    try { setComparisonBusy(true); setError(null); setComparison(await api.comparison(scenario, Math.min(ticks, 100), [seed, seed + 11])); } catch (caught) { setError(caught instanceof Error ? caught.message : "Comparison request failed."); } finally { setComparisonBusy(false); }
  };

  const modeBreakdown = useMemo(() => parseModeBreakdown(kpi?.mode_breakdown), [kpi?.mode_breakdown]);
  const modeTotal = (modeBreakdown.unicast || 0) + (modeBreakdown.broadcast || 0);
  const visibleGroups = useMemo(() => selectedCell === "all" ? groups : groups.filter((group) => group.cell_id === selectedCell), [groups, selectedCell]);
  const activeCell = selectedCell === "all" ? null : cells.find((cell) => cell.cell_id === selectedCell);
  const filteredHistory = useMemo(() => selectedCell === "all" ? decisionHistory : decisionHistory.filter((row) => row.cell_id === selectedCell), [decisionHistory, selectedCell]);
  const chartHistory = kpiHistory.slice(-36);
  const live = status?.status === "running";
  const canStart = !status || ["completed", "stopped", "failed"].includes(status.status);
  const canPause = status?.status === "running";
  const canResume = status?.status === "paused";
  const canStop = Boolean(status && ["running", "paused"].includes(status.status));

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark"><Antenna size={18} /></div><div><strong>SWITCHBOARD</strong><small>AI NETWORK CONTROL</small></div></div>
      <div className="side-status"><span className={`status-dot ${backend}`} /><span>{backend === "online" ? "API connected" : backend === "connecting" ? "Connecting" : "API offline"}</span><span className="status-pulse" /></div>
      <nav className="nav-list"><a className="nav-item active" href="#overview"><LayoutDashboard size={16} /><span>Command center</span><ChevronRight size={14} /></a><a className="nav-item" href="#network"><RadioTower size={16} /><span>Network fabric</span></a><a className="nav-item" href="#decisions"><BrainCircuit size={16} /><span>AI decisions</span></a><a className="nav-item" href="#experience"><Gauge size={16} /><span>User experience</span></a><a className="nav-item" href="#comparison"><BarChart3 size={16} /><span>Policy comparison</span></a></nav>
      <div className="sidebar-bottom"><div className="mini-label">SIMULATOR PROFILE</div><div className="profile-row"><span>cells</span><b>{config.num_cells ?? "—"}</b></div><div className="profile-row"><span>UE population</span><b>{config.num_ues ?? "—"}</b></div><div className="profile-row"><span>segments</span><b>{config.num_segments ?? "—"}</b></div><div className="sidebar-note"><Sparkles size={14} /><span>Synthetic 2D coordinates<br />No real-world geography</span></div></div>
    </aside>
    <main className="main-content">
      <header className="topbar"><div><div className="breadcrumb"><span>TELECOM LAB</span><ChevronRight size={13} /><span>SIMULATION CONTROL</span></div><h1>Smart traffic switching <span>/ live observatory</span></h1></div><div className="topbar-right"><div className="socket-chip"><span className={`status-dot ${live ? "live" : "idle"}`} />{live ? "LIVE STREAM" : status?.status?.toUpperCase() || "NO RUN"}</div><div className="time-chip"><Clock3 size={14} />{lastUpdate ? lastUpdate.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "awaiting tick"}</div></div></header>
      <section className="control-strip"><div className="control-intro"><div className="control-icon"><SlidersHorizontal size={17} /></div><div><strong>Run controls</strong><span>Launch a matched simulator run and watch the controller adapt in real time.</span></div></div><div className="control-fields"><label>SCENARIO<select value={scenario} onChange={(event) => setScenario(event.target.value)} disabled={!canStart}><option value="flash_crowd">Flash crowd</option><option value="uniform">Uniform traffic</option></select></label><label>POLICY<select value={selector} onChange={(event) => setSelector(event.target.value)} disabled={!canStart}><option value="ai">AI controller</option><option value="always-unicast">Always unicast</option><option value="always-broadcast">Always broadcast</option><option value="rule-by-group-size">Rule-based</option><option value="random">Random</option></select></label><label>TICKS<input type="number" min="1" max="500" value={ticks} onChange={(event) => setTicks(Number(event.target.value))} disabled={!canStart} /></label><label>INTERVAL<input type="number" min="0" max="10" step="0.1" value={interval} onChange={(event) => setIntervalValue(Number(event.target.value))} disabled={!canStart} /></label><label>SEED<input type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value))} disabled={!canStart} /></label></div><div className="control-actions"><button className="button primary" onClick={start} disabled={!canStart}><Play size={14} /> Start run</button><button className="button ghost" onClick={() => control("pause")} disabled={!canPause}><Pause size={14} /> Pause</button><button className="button ghost" onClick={() => control("resume")} disabled={!canResume}><Play size={14} /> Resume</button><button className="button danger-ghost" onClick={() => control("stop")} disabled={!canStop}><CircleStop size={14} /> Stop</button></div></section>
      {error && <div className="alert-banner"><ShieldAlert size={17} /><span>{error}</span><button onClick={() => setError(null)} aria-label="Dismiss error"><X size={15} /></button></div>}
      <section id="overview" className="overview-section"><div className="run-context"><div><span className="eyebrow">ACTIVE SIMULATION</span><strong>{activeId ? shortId(activeId) : "No active run"}</strong><span className="run-meta">{status ? `${formatPolicy(status.mode_selector)} · ${status.scenario.replace(/_/g, " ")}` : "Configure a run to populate the observatory"}</span></div><div className="tick-readout"><span>TICK</span><strong>{status?.current_tick ?? kpi?.tick ?? "—"}</strong><small>/ {status?.total_ticks ?? ticks}</small></div></div><div className="headline-grid"><MetricCard label="Service continuity" value={pct(kpi?.service_continuity)} helper="Actual playback continuity" accent="teal" icon={<ShieldCheck size={16} />} trend="up" /><MetricCard label="Resource savings" value={pct(kpi?.resource_savings_vs_baseline)} helper="Versus always-unicast" accent="orange" icon={<Zap size={16} />} trend="up" /><MetricCard label="Mean QoE" value={pct(kpi?.mean_qoe)} helper="Buffer health + no starvation" accent="violet" icon={<Sparkles size={16} />} /><MetricCard label="Mode split" value={modeTotal ? `${modeBreakdown.broadcast || 0} / ${modeBreakdown.unicast || 0}` : "—"} unit={modeTotal ? " B / U" : undefined} helper={modeTotal ? `${modeTotal} active groups this tick` : "Waiting for group delivery"} accent="blue" icon={<Layers3 size={16} />} /></div></section>
      <section className="hero-grid"><div className="panel timeline-panel"><div className="panel-header"><SectionHeading eyebrow="CLOSED-LOOP FEEDBACK" title="Switching timeline" copy="The controller’s final group modes over the active run." icon={<Activity size={17} />} /><div className="panel-legend"><span><i className="legend-line broadcast" />Broadcast</span><span><i className="legend-line unicast" />Unicast</span></div></div><div className="timeline-chart"><DualSparkline primary={chartHistory.map((item) => parseModeBreakdown(item.mode_breakdown).broadcast || 0)} secondary={chartHistory.map((item) => parseModeBreakdown(item.mode_breakdown).unicast || 0)} /><div className="chart-axis"><span>{chartHistory[0] ? `T${chartHistory[0].tick}` : "T—"}</span><span>broadcast groups</span><span>{chartHistory[chartHistory.length - 1] ? `T${chartHistory[chartHistory.length - 1].tick}` : "T—"}</span></div></div><div className="timeline-footer"><div><span className="metric-inline-label">SWITCHES THIS TICK</span><strong>{kpi?.switch_count ?? "—"}</strong></div><div><span className="metric-inline-label">AI LATENCY</span><strong>{kpi?.ai_inference_latency_ms == null ? "—" : `${fmt(kpi.ai_inference_latency_ms, 2)} ms`}</strong></div><div><span className="metric-inline-label">DELIVERY GROUPS</span><strong>{kpi?.num_groups ?? "—"}</strong></div></div></div><div className="panel live-snapshot"><div className="panel-header compact-header"><SectionHeading eyebrow="CURRENT TICK" title="Mode allocation" icon={<Radio size={17} />} /><span className="live-tag"><span className="status-dot live" />live</span></div><div className="snapshot-body"><Donut unicast={modeBreakdown.unicast || 0} broadcast={modeBreakdown.broadcast || 0} /><div className="snapshot-legend"><div><span className="legend-square broadcast" /><div><strong>{modeBreakdown.broadcast || 0}</strong><small>Broadcast groups</small></div></div><div><span className="legend-square unicast" /><div><strong>{modeBreakdown.unicast || 0}</strong><small>Unicast groups</small></div></div><div><span className="legend-square neutral" /><div><strong>{kpi?.num_ues_delivered ?? "—"}</strong><small>UEs delivered</small></div></div></div></div><div className="snapshot-caption"><span>Physical bearers are reconciled by MBOS</span><span className="signal-strength"><Signal size={13} /> {kpi ? `${fmt(kpi.mean_sinr_db)} dB mean SINR` : "—"}</span></div></div></section>
      <section id="network" className="section-block"><SectionHeading eyebrow="NETWORK FABRIC" title="Cells & user equipment" copy="Synthetic 2D positions show where the simulated demand is being served. Click a cell to filter the live UE layer." icon={<MapPinned size={17} />} /><div className="network-grid"><div className="panel map-panel"><div className="panel-toolbar"><div className="toolbar-title"><span className="live-pip" />LIVE TOPOLOGY <small>{ues.length ? `${ues.length} UEs rendered` : "No UE snapshot yet"}</small></div><div className="filter-pills"><button className={selectedCell === "all" ? "selected" : ""} onClick={() => setSelectedCell("all")}>All cells</button>{cells.map((cell) => <button key={cell.cell_id} className={selectedCell === cell.cell_id ? "selected" : ""} onClick={() => setSelectedCell(cell.cell_id)}>{cell.cell_id.replace("cell_", "C")}</button>)}</div></div><LiveMap cells={cells} ues={ues} selectedCell={selectedCell} onSelectCell={setSelectedCell} /></div><div className="panel cell-panel"><div className="panel-header compact-header"><div><div className="eyebrow">CELL INVENTORY</div><h3>{activeCell ? activeCell.cell_id : "All serving cells"}</h3></div><Server size={17} className="muted-icon" /></div><div className="cell-list">{cells.map((cell) => <button className={`cell-row ${selectedCell === cell.cell_id ? "active" : ""}`} key={cell.cell_id} onClick={() => setSelectedCell(selectedCell === cell.cell_id ? "all" : cell.cell_id)}><span className="cell-id"><span className="cell-status" />{cell.cell_id}</span><span className="cell-load"><strong>{fmt(cell.current_load_mbps)} <small>Mbps</small></strong><ProgressBar value={cell.congestion} color={cell.congestion > .7 ? "coral" : "teal"} right={pct(cell.congestion)} /></span><span className="cell-users"><Users size={13} />{cell.num_connected_ues}</span></button>)}{!cells.length && <div className="empty-state compact"><RadioTower size={18} /><span>Cells appear after the first simulation tick.</span></div>}</div><div className="capacity-footer"><span>Broadcast pool</span><strong>{activeCell ? `${fmt(activeCell.broadcast_capacity_mbps)} Mbps` : config.broadcast_cell_capacity_mbps ? `${fmt(config.broadcast_cell_capacity_mbps)} Mbps / cell` : "—"}</strong></div></div></div></section>
      <section id="decisions" className="section-block"><SectionHeading eyebrow="INTELLIGENCE LAYER" title="AI decisions & policy guardrails" copy="Only the backend’s final, post-MBOS / post-PCF mode is shown here." icon={<BrainCircuit size={17} />} /><div className="decisions-grid"><div className="panel decision-feed-panel"><div className="panel-header compact-header"><div><div className="eyebrow">LIVE DECISION FEED</div><h3>{decisions.length ? `${decisions.length} groups evaluated` : "Awaiting AI decisions"}</h3></div><span className={`policy-chip ${selector === "ai" ? "ai" : "neutral"}`}>{formatPolicy(selector)}</span></div><DecisionPanel decisions={decisions} selected={selectedDecision} onSelect={setSelectedDecision} policy={selector} /></div><div className="panel guardrail-panel"><div className="panel-header compact-header"><div><div className="eyebrow">MBOS / PCF</div><h3>Orchestration status</h3></div><ShieldCheck size={17} className="teal-icon" /></div><div className="guardrail-stack"><div className="guardrail-row"><span className="guardrail-icon mbos"><TowerControl size={15} /></span><div><strong>MBOS</strong><small>Bearer coordination</small></div><span className="guardrail-status">{selector === "ai" && decisions.length ? "active" : "ready"}<i /></span></div><div className="guardrail-row"><span className="guardrail-icon pcf"><ShieldCheck size={15} /></span><div><strong>PCF</strong><small>Policy enforcement</small></div><span className="guardrail-status">{selector === "ai" && decisions.some((item) => !item.policy_compliant) ? "override" : "compliant"}<i /></span></div></div><div className="guardrail-explainer"><div className="mini-label">CONTROL PATH</div><div className="path-row"><span>AI rank</span><ChevronRight size={12} /><span>MBOS bearer</span><ChevronRight size={12} /><span>PCF gate</span><ChevronRight size={12} /><span>Deliver</span></div><p>MBOS makes the physical bearer coherent; PCF can revert an unsafe or low-confidence switch.</p></div><div className="config-facts"><span><small>Min confidence</small><b>{config.min_decision_confidence != null ? pct(config.min_decision_confidence) : "—"}</b></span><span><small>Hysteresis</small><b>{config.hysteresis_margin != null ? fmt(config.hysteresis_margin, 2) : "—"}</b></span><span><small>Cooldown</small><b>{config.cooldown_period_s != null ? `${fmt(config.cooldown_period_s)}s` : "—"}</b></span></div></div></div></section>
      <section id="experience" className="section-block"><SectionHeading eyebrow="RADIO + EXPERIENCE" title="What the network delivered" copy="Live aggregates come directly from the simulator’s KPI record; UE detail is refreshed over REST." icon={<Gauge size={17} />} /><div className="kpi-grid"><div className="panel kpi-panel"><div className="panel-header compact-header"><div><div className="eyebrow">RADIO CONDITIONS</div><h3>Signal & throughput</h3></div><Signal size={17} className="muted-icon" /></div><div className="stat-stack"><div className="stack-stat"><span><Signal size={14} /> Mean SINR</span><strong>{kpi ? <>{fmt(kpi.mean_sinr_db)} <small>dB</small></> : "—"}</strong><div className="stat-spark"><Sparkline values={chartHistory.map((item) => item.mean_sinr_db)} color="#7ea7ff" height={42} /></div></div><div className="stack-stat"><span><Wifi size={14} /> Achievable rate</span><strong>{kpi ? <>{fmt(kpi.mean_achievable_mbps)} <small>Mbps</small></> : "—"}</strong><div className="stat-spark"><Sparkline values={chartHistory.map((item) => item.mean_achievable_mbps)} color="#55d6c2" height={42} /></div></div><div className="stack-stat"><span><Cpu size={14} /> Spectrum efficiency</span><strong>{kpi ? <>{fmt(kpi.mean_spectrum_efficiency, 3)} <small>bits/s/Hz</small></> : "—"}</strong><ProgressBar value={Math.min((kpi?.mean_spectrum_efficiency || 0) / 10, 1)} color="blue" /></div></div></div><div className="panel kpi-panel"><div className="panel-header compact-header"><div><div className="eyebrow">BUFFER + PLAYBACK</div><h3>Experience health</h3></div><Gauge size={17} className="muted-icon" /></div><div className="experience-stats"><div className="big-inline"><span>Mean buffer</span><strong>{kpi ? <>{fmt(kpi.mean_buffer_seconds)}<small>s</small></> : "—"}</strong><Sparkline values={chartHistory.map((item) => item.mean_buffer_seconds)} color="#c08cff" height={46} /></div><div className="experience-row"><span>Minimum group buffer</span><strong>{kpi ? `${fmt(kpi.min_group_buffer_seconds)}s` : "—"}</strong></div><div className="experience-row"><span>Buffer starvation</span><strong className={kpi && kpi.buffer_starvation_rate > .2 ? "danger-text" : ""}>{pct(kpi?.buffer_starvation_rate)}</strong></div><div className="experience-row"><span>Successful delivery</span><strong>{pct(kpi?.successful_delivery_ratio)}</strong></div><div className="experience-row"><span>Rebuffering events</span><strong>{kpi?.rebuffering_events ?? "—"}</strong></div></div></div><div className="panel kpi-panel ue-detail-panel"><div className="panel-header compact-header"><div><div className="eyebrow">UE DETAIL</div><h3>{selectedCell === "all" ? "All devices" : selectedCell}</h3></div><span className="ue-count">{ues.length} / {config.num_ues ?? "—"}</span></div><div className="ue-summary"><div><span className="summary-number teal-text">{ues.filter((ue) => ue.qoe_state === "good").length}</span><small>good QoE</small></div><div><span className="summary-number orange-text">{ues.filter((ue) => ue.delivery_mode === "unicast").length}</span><small>unicast</small></div><div><span className="summary-number teal-text">{ues.filter((ue) => ue.delivery_mode === "broadcast").length}</span><small>broadcast</small></div></div><div className="ue-table"><div className="ue-table-head"><span>UE</span><span>MODE</span><span>BUFFER</span><span>RADIO</span></div>{ues.slice(0, 7).map((ue) => <div className="ue-table-row" key={ue.ue_id}><span>{ue.ue_id}</span><span className={`mode-text ${ue.delivery_mode}`}>{ue.delivery_mode}</span><span>{fmt(ue.buffer_seconds)}s</span><span className={`radio-text ${ue.radio_state}`}>{ue.radio_state}</span></div>)}{!ues.length && <div className="empty-state compact">No UE snapshot yet.</div>}</div></div></div></section>
      <section className="section-block history-section"><SectionHeading eyebrow="AUDIT TRAIL" title="AI switching history" copy="Persisted decision records from the backend rolling window. Explanation leads; reason codes stay secondary." icon={<RefreshCw size={17} />} /><div className="panel history-panel"><div className="history-toolbar"><div><strong>{filteredHistory.length || "No"} recorded decisions</strong><span>{selectedCell === "all" ? "All cells" : selectedCell}</span></div><div className="history-key"><span><i className="mode-dot broadcast" />Broadcast</span><span><i className="mode-dot unicast" />Unicast</span></div></div><div className="history-table"><div className="history-head"><span>TICK</span><span>GROUP</span><span>FINAL MODE</span><span>CONFIDENCE</span><span>EXPLANATION</span><span>PCF</span></div>{filteredHistory.slice().reverse().slice(0, 14).map((row) => <div className="history-row" key={`${row.tick}-${row.cell_id}-${row.group_id}`}><span className="mono">{row.tick}</span><span><b>{row.cell_id}</b><small>{row.group_id}</small></span><span><span className={`mode-pill mini ${row.selected_mode}`}>{row.selected_mode}</span></span><span><strong>{pct(row.confidence)}</strong><ProgressBar value={row.confidence} color="teal" /></span><span className="explanation-cell">{row.explanation}</span><span className={row.pcf_result?.startsWith("reverted") ? "danger-text" : "teal-text"}>{row.pcf_result?.startsWith("reverted") ? "override" : "compliant"}</span></div>)}{!filteredHistory.length && <div className="empty-state"><RefreshCw size={20} /><strong>History will fill as the AI observes each tick.</strong><p>Start an AI run with a non-zero interval to watch the audit trail build.</p></div>}</div></div></section>
      <section id="comparison" className="section-block comparison-section"><div className="comparison-heading"><SectionHeading eyebrow="POLICY BENCHMARK" title="AI vs. fixed baselines" copy="The same simulator is evaluated under matched seeds to make the policy trade-off visible." icon={<BarChart3 size={17} />} /><button className="button outline" onClick={runComparison} disabled={comparisonBusy}><RefreshCw size={14} className={comparisonBusy ? "spin" : ""} />{comparisonBusy ? "Running comparison" : "Run comparison"}</button></div><div className="comparison-grid"><div className="panel comparison-visual"><div className="panel-header compact-header"><div><div className="eyebrow">THESIS VIEW</div><h3>Efficiency × continuity</h3></div><span className="comparison-seed">{comparison.length ? `${comparison.length} policies` : "not run"}</span></div>{comparison.length ? <><ComparisonChart rows={comparison} /><div className="bar-legend"><span><i className="bar-key ai" />AI</span><span><i className="bar-key unicast" />Always U</span><span><i className="bar-key broadcast" />Always B</span><span><i className="bar-key rule" />Rule</span></div></> : <div className="comparison-empty"><BarChart3 size={24} /><strong>Compare the live policy design</strong><span>Run a modest matched-seed batch to populate real baseline results.</span></div>}</div><div className="panel comparison-table-panel"><div className="panel-header compact-header"><div><div className="eyebrow">REPORTED KPIS</div><h3>Policy scorecard</h3></div><Layers3 size={17} className="muted-icon" /></div><div className="score-table"><div className="score-head"><span>POLICY</span><span>SAVINGS</span><span>CONTINUITY</span><span>QOE</span><span>CONGESTION</span></div>{comparison.map((row) => <div className={`score-row ${row.policy === "ai" ? "featured" : ""}`} key={row.policy}><span><i className={`policy-dot ${row.policy}`} />{formatPolicy(row.policy)}</span><span>{pct(Number(row.summary.resource_savings_vs_baseline))}</span><span>{pct(Number(row.summary.service_continuity))}</span><span>{pct(Number(row.summary.mean_qoe))}</span><span>{pct(Number(row.summary.mean_congestion))}</span></div>)}{!comparison.length && <div className="empty-state compact">No comparison has been requested.</div>}</div></div></div></section>
      <footer className="footer"><span><Antenna size={13} /> SWITCHBOARD / AI-ENABLED SMART DATA TRAFFIC SWITCHING</span><span>REST snapshots + WebSocket tick stream · backend fields only</span></footer>
    </main>
  </div>;
}

export default App;
