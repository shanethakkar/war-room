"use client";

import clsx from "clsx";
import { fillRoster } from "@/lib/draft";
import { POS_BAR } from "./atoms";

export function RosterPanel({
  format,
  mine,
  nextPick,
  round,
}: {
  format: string;
  mine: { name: string; pos: string }[];
  nextPick: number | null;
  round: number | null;
}) {
  const { slots, bench } = fillRoster(format, mine);
  const needs = slots.filter((s) => !s.player).map((s) => s.label);

  return (
    <aside className="w-60 shrink-0">
      <div className="sticky top-6 rounded-xl border border-border bg-neutral-bg2/60 p-4">
        <div className="flex items-baseline justify-between">
          <h2 className="text-sm font-semibold text-text-primary">My roster</h2>
          {nextPick !== null ? (
            <span className="num text-xs tabular-nums text-text-secondary">
              next pick <span className="font-semibold text-brand-light">#{nextPick}</span>
              {round !== null && <span className="text-text-muted"> · R{round}</span>}
            </span>
          ) : (
            <span className="text-xs text-text-muted">draft done</span>
          )}
        </div>

        <ul className="mt-3 space-y-1">
          {slots.map((s, i) => (
            <li key={i} className="flex items-center gap-2 text-sm">
              <span
                className={clsx(
                  "num w-9 shrink-0 rounded px-1 py-0.5 text-center text-[10px] font-semibold",
                  s.player
                    ? "bg-neutral-bg4 text-text-secondary"
                    : "bg-brand-subtle text-brand-light",
                )}
              >
                {s.label}
              </span>
              {s.player ? (
                <span className="truncate text-text-primary">{s.player}</span>
              ) : (
                <span className="text-text-muted">—</span>
              )}
            </li>
          ))}
        </ul>

        {bench.length > 0 && (
          <>
            <div className="mt-3 text-[10px] font-medium uppercase tracking-wider text-text-muted">
              Bench
            </div>
            <ul className="mt-1 space-y-1">
              {bench.map((p, i) => (
                <li key={i} className="flex items-center gap-2 text-sm">
                  <span className={clsx("h-4 w-[3px] rounded-full", POS_BAR[p.pos] ?? "bg-white/10")} />
                  <span className="truncate text-text-secondary">{p.name}</span>
                </li>
              ))}
            </ul>
          </>
        )}

        {needs.length > 0 && (
          <p className="mt-3 text-xs leading-relaxed text-text-muted">
            Still need:{" "}
            <span className="text-text-secondary">{needs.join(", ")}</span>
          </p>
        )}
      </div>
    </aside>
  );
}
