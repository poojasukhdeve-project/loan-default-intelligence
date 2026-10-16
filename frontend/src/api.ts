import type { LoanInput, ScoreResponse } from './types';
const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';
export async function requestScore(input: LoanInput, exposure: number, lgd: number): Promise<ScoreResponse> {
  const response = await fetch(`${API_BASE}/score`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...input, exposure, lgd }),
  });
  if (!response.ok) {
    let detail = `API returned HTTP ${response.status}`;
    try { const payload = await response.json(); detail = payload.detail || detail; } catch { /* response may not be JSON */ }
    throw new Error(detail);
  }
  return response.json() as Promise<ScoreResponse>;
}
