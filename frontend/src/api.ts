
import type { LoanInput, ScoreResponse } from './types';

const API_BASE: string =
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

export async function requestScore(
  input: LoanInput,
  _exposure: number,
  _lgd: number,
): Promise<ScoreResponse> {
  let response: Response;

  try {
    response = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(input),
    });
  } catch {
    throw new Error(
      'Cannot connect to FastAPI. Check the backend and CORS settings.',
    );
  }

  if (!response.ok) {
    let detail = `API returned HTTP ${response.status}`;

    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch {
      // Response may not contain JSON.
    }

    throw new Error(detail);
  }

  return (await response.json()) as ScoreResponse;
}
