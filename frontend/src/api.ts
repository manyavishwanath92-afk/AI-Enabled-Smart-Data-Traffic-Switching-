import type {
  Cell,
  ComparisonRow,
  CreateSimulationRequest,
  Decision,
  DecisionHistoryRow,
  DeliveryGroup,
  KPIRecord,
  SimConfig,
  SimulationStatus,
  TickMessage,
  UeResponse,
} from "./types";

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "") || "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch {
      // Preserve the HTTP status when the API does not return JSON.
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string }>("/api/health"),
  config: () => request<SimConfig>("/api/config"),
  scenarios: () => request<unknown>("/api/scenarios"),
  simulations: () => request<SimulationStatus[]>("/api/simulations"),
  start: (body: CreateSimulationRequest) =>
    request<SimulationStatus>("/api/simulations", { method: "POST", body: JSON.stringify(body) }),
  status: (id: string) => request<SimulationStatus>(`/api/simulations/${id}/status`),
  pause: (id: string) => request<SimulationStatus>(`/api/simulations/${id}/pause`, { method: "POST" }),
  resume: (id: string) => request<SimulationStatus>(`/api/simulations/${id}/resume`, { method: "POST" }),
  stop: (id: string) => request<SimulationStatus>(`/api/simulations/${id}/stop`, { method: "POST" }),
  latestKpi: (id: string) => request<KPIRecord>(`/api/simulations/${id}/kpis/latest`),
  kpiHistory: (id: string) => request<KPIRecord[]>(`/api/simulations/${id}/kpis/history`),
  kpiSummary: (id: string) => request<Record<string, number>>( `/api/simulations/${id}/kpis/summary`),
  cells: (id: string) => request<Cell[]>(`/api/simulations/${id}/cells`),
  ues: (id: string, cellId?: string) => {
    const query = new URLSearchParams({ limit: "1000", offset: "0" });
    if (cellId) query.set("cell_id", cellId);
    return request<UeResponse>(`/api/simulations/${id}/ues?${query.toString()}`);
  },
  groups: (id: string) => request<DeliveryGroup[]>(`/api/simulations/${id}/groups/latest`),
  decisionsLatest: (id: string) => request<Decision[]>(`/api/simulations/${id}/decisions/latest`),
  decisionsHistory: (id: string, limit = 250) =>
    request<DecisionHistoryRow[]>(`/api/simulations/${id}/decisions/history?limit=${limit}`),
  comparison: (scenario: string, ticks: number, seeds: number[]) => {
    const query = new URLSearchParams({ scenario, ticks: String(ticks) });
    seeds.forEach((seed) => query.append("seeds", String(seed)));
    return request<ComparisonRow[]>(`/api/comparison?${query.toString()}`);
  },
};

export function openLiveSocket(
  simulationId: string,
  onMessage: (message: TickMessage | SimulationStatus & { type: "status" } | { type: "error"; detail: string }) => void,
  onClose: () => void,
  onError: () => void,
): WebSocket {
  const configured = (import.meta.env.VITE_WS_BASE_URL as string | undefined)?.replace(/\/$/, "");
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = configured || `${protocol}//${window.location.host}`;
  const socket = new WebSocket(`${host}/api/simulations/${simulationId}/live`);
  socket.onmessage = (event) => {
    try {
      onMessage(JSON.parse(event.data));
    } catch {
      onError();
    }
  };
  socket.onclose = onClose;
  socket.onerror = onError;
  return socket;
}
