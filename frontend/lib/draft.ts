// Draft-day math: snake pick numbers, survival odds, roster slot assignment.

export type PickState = "taken" | "mine";
export type DraftPicks = Record<string, PickState>;

/** Overall pick numbers for a slot in a snake draft (1-indexed). */
export function myPickNumbers(slot: number, teams: number, rounds = 16): number[] {
  const picks: number[] = [];
  for (let r = 0; r < rounds; r++) {
    picks.push(r % 2 === 0 ? r * teams + slot : (r + 1) * teams - slot + 1);
  }
  return picks;
}

/** My next overall pick at/after `currentOverall`, or null if the draft is over. */
export function nextMyPick(
  currentOverall: number,
  slot: number,
  teams: number,
): number | null {
  return myPickNumbers(slot, teams).find((p) => p >= currentOverall) ?? null;
}

/**
 * P(player is still on the board at overall pick `pick`), modeling his actual
 * draft position as Normal(adp, stdev). FFC's observed per-player stdev feeds
 * this; floored so nothing reads as a false certainty.
 */
export function survivalProb(
  adp: number | null,
  stdev: number | null | undefined,
  pick: number,
): number | null {
  if (adp === null || adp === undefined) return null;
  const sd = Math.max(stdev ?? 8, 2);
  return normCdf((adp - pick) / sd);
}

function normCdf(z: number): number {
  // Abramowitz & Stegun 26.2.17; plenty accurate for display purposes.
  const t = 1 / (1 + 0.2316419 * Math.abs(z));
  const d = 0.3989423 * Math.exp((-z * z) / 2);
  const p =
    d *
    t *
    (0.3193815 +
      t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return z > 0 ? 1 - p : p;
}

export type RosterSlot = { label: string; eligible: string[]; player?: string };

/** Starting slots for a format key (bench rendered separately). */
export function starterSlots(format: string): RosterSlot[] {
  const slots: RosterSlot[] = [
    { label: "QB", eligible: ["QB"] },
    { label: "RB", eligible: ["RB"] },
    { label: "RB", eligible: ["RB"] },
    { label: "WR", eligible: ["WR"] },
    { label: "WR", eligible: ["WR"] },
    { label: "TE", eligible: ["TE"] },
    { label: "FLX", eligible: ["RB", "WR", "TE"] },
  ];
  if (format === "superflex")
    slots.push({ label: "SFX", eligible: ["QB", "RB", "WR", "TE"] });
  if (format === "two_qb") slots.push({ label: "QB", eligible: ["QB"] });
  slots.push({ label: "DST", eligible: ["DST"] }, { label: "K", eligible: ["K"] });
  return slots;
}

/** Greedily place my players (draft order) into starting slots; rest to bench. */
export function fillRoster(
  format: string,
  minePlayers: { name: string; pos: string }[],
): { slots: RosterSlot[]; bench: { name: string; pos: string }[] } {
  const slots = starterSlots(format);
  const bench: { name: string; pos: string }[] = [];
  for (const p of minePlayers) {
    const open = slots.find((s) => !s.player && s.eligible.includes(p.pos));
    if (open) open.player = p.name;
    else bench.push(p);
  }
  return { slots, bench };
}
