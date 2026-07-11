"use client";

import clsx from "clsx";

const POS_STYLE: Record<string, string> = {
  QB: "text-pos-qb bg-pos-qb/10 ring-pos-qb/25",
  RB: "text-pos-rb bg-pos-rb/10 ring-pos-rb/25",
  WR: "text-pos-wr bg-pos-wr/10 ring-pos-wr/25",
  TE: "text-pos-te bg-pos-te/10 ring-pos-te/25",
};

export const POS_BAR: Record<string, string> = {
  QB: "bg-pos-qb",
  RB: "bg-pos-rb",
  WR: "bg-pos-wr",
  TE: "bg-pos-te",
};

export function PositionBadge({ pos, tier }: { pos: string; tier: number }) {
  return (
    <span
      className={clsx(
        "num inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-xs font-semibold ring-1 ring-inset",
        POS_STYLE[pos] ?? "text-text-secondary bg-white/5 ring-white/10",
      )}
    >
      {pos}
      <span className="text-[10px] font-medium opacity-60">T{tier}</span>
    </span>
  );
}

export function IntervalBar({
  low,
  mid,
  high,
  max,
}: {
  low: number;
  mid: number;
  high: number;
  max: number;
}) {
  const pct = (v: number) => Math.max(0, Math.min(100, (v / max) * 100));
  return (
    <div className="group/bar relative h-1.5 w-full rounded-full bg-neutral-bg4">
      <div
        className="absolute h-1.5 rounded-full bg-brand/25"
        style={{ left: `${pct(low)}%`, width: `${pct(high) - pct(low)}%` }}
      />
      <div
        className="absolute top-1/2 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-brand ring-2 ring-neutral-bg1"
        style={{ left: `${pct(mid)}%` }}
      />
    </div>
  );
}

export function TiltCell({ delta }: { delta: number | null }) {
  if (delta === null)
    return <span className="text-text-muted">·</span>;
  const up = delta > 0;
  const big = Math.abs(delta) >= 8;
  return (
    <span
      className={clsx(
        "num inline-flex items-center gap-1 tabular-nums",
        delta === 0 ? "text-text-muted" : up ? "text-good" : "text-bad",
        big && "font-semibold",
      )}
      title={
        up
          ? "Model tilts this player up from market ADP"
          : "Model tilts this player down from market ADP"
      }
    >
      {delta === 0 ? "—" : up ? "▲" : "▼"}
      {delta !== 0 && Math.abs(delta)}
    </span>
  );
}

export function Segmented<T extends string | number>({
  options,
  value,
  onChange,
  size = "md",
}: {
  options: { value: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
  size?: "sm" | "md";
}) {
  return (
    <div className="inline-flex rounded-lg bg-neutral-bg2 p-0.5 ring-1 ring-inset ring-border">
      {options.map((o) => (
        <button
          key={String(o.value)}
          onClick={() => onChange(o.value)}
          className={clsx(
            "rounded-md font-medium transition-colors",
            size === "sm" ? "px-2.5 py-1 text-xs" : "px-3 py-1.5 text-sm",
            value === o.value
              ? "bg-neutral-bg4 text-text-primary shadow-sm"
              : "text-text-secondary hover:text-text-primary",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
