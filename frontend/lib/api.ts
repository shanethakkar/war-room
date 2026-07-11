export const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Player = {
  board_rank: number;
  model_rank: number;
  position_rank: number;
  position_tier: number;
  player_id: string;
  player_name: string;
  position: string;
  position_group: string;
  team: string | null;
  is_rookie: boolean;
  projected_games: number;
  projected_points: number;
  points_low: number;
  points_high: number;
  vor: number;
  adp: number | null;
  adp_market_rank: number | null;
  model_tilt: number | null;
};

export type BoardResponse = {
  season: number;
  format: string;
  format_name: string;
  model: string;
  players: Player[];
};

export async function fetchBoard(
  season: number,
  format: string,
): Promise<BoardResponse> {
  const res = await fetch(`${API_URL}/board?season=${season}&format=${format}`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`);
  return res.json();
}

export async function fetchSeasons(): Promise<number[]> {
  const res = await fetch(`${API_URL}/seasons`, { cache: "no-store" });
  if (!res.ok) throw new Error(`API ${res.status}`);
  return (await res.json()).seasons;
}
