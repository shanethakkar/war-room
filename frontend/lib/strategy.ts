// Positional cliff math (mirrors src/decision/strategy.py).
//
// Descriptive insight, not an auto-pilot: our six-season simulations showed
// that explicit position-sequencing policies do NOT beat disciplined
// best-available drafting - so cliffs are shown as information for a human
// weighing a reach, never as an override of the board.

import type { Player } from "./api";
import { survivalProb } from "./draft";

export type Cliff = {
  pos: string;
  bestNowName: string;
  bestNow: number;
  expectedAtNext: number;
  expectedAfter: number;
  drop: number; // bestNow - expectedAtNext: what waiting one turn costs
};

/** E[best projected points among survivors], ordered-inclusion by points. */
function expectedBest(
  players: { points: number; surv: number }[],
): number {
  const sorted = [...players].sort((a, b) => b.points - a.points);
  let noneBetter = 1;
  let expected = 0;
  for (const p of sorted) {
    expected += p.points * p.surv * noneBetter;
    noneBetter *= 1 - p.surv;
  }
  if (sorted.length > 0) expected += noneBetter * sorted[sorted.length - 1].points;
  return expected;
}

export function positionalCliffs(
  available: Player[],
  currentPick: number,
  nextPick: number | null,
  afterPick: number | null,
  positions: string[] = ["QB", "RB", "WR", "TE"],
): Cliff[] {
  const cliffs: Cliff[] = [];
  for (const pos of positions) {
    const group = available.filter((p) => p.position_group === pos);
    if (group.length === 0) continue;
    const best = group.reduce((a, b) =>
      b.projected_points > a.projected_points ? b : a,
    );
    const at = (pick: number | null): number => {
      if (pick === null) return 0;
      return expectedBest(
        group.map((p) => ({
          points: p.projected_points,
          // No ADP -> nobody is racing you to him; assume he survives.
          surv:
            p.adp === null
              ? 1
              : (survivalProb(p.adp, p.adp_stdev, pick, currentPick) ?? 1),
        })),
      );
    };
    const eNext = at(nextPick);
    cliffs.push({
      pos,
      bestNowName: best.player_name,
      bestNow: best.projected_points,
      expectedAtNext: eNext,
      expectedAfter: at(afterPick),
      drop: best.projected_points - eNext,
    });
  }
  return cliffs.sort((a, b) => b.drop - a.drop);
}
