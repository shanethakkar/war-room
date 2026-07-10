"use client";

import { useEffect, useMemo, useState } from "react";
import { ArbCell, IntervalBar, POS_BAR, PositionBadge, Segmented } from "@/components/atoms";
import { type BoardResponse, type Player, fetchBoard } from "@/lib/api";

const FORMATS = [
  { value: "redraft_ppr", label: "Redraft PPR" },
  { value: "superflex", label: "Superflex" },
];
const SEASONS = [2026, 2025, 2024, 2023];
const POSITIONS = ["ALL", "QB", "RB", "WR", "TE"];
const SORTS = [
  { value: "vor", label: "Value" },
  { value: "adp", label: "Market (ADP)" },
  { value: "arb", label: "Arbitrage" },
];

export default function Page() {
  const [season, setSeason] = useState(2026);
  const [format, setFormat] = useState("redraft_ppr");
  const [pos, setPos] = useState("ALL");
  const [sort, setSort] = useState("vor");
  const [query, setQuery] = useState("");
  const [data, setData] = useState<BoardResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setError(null);
    fetchBoard(season, format)
      .then((d) => alive && setData(d))
      .catch((e) => alive && setError(String(e)))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [season, format]);

  const players = data?.players ?? [];
  const scaleMax = useMemo(
    () => Math.max(100, ...players.map((p) => p.points_high)),
    [players],
  );

  const rows = useMemo(() => {
    let r = players;
    if (pos !== "ALL") r = r.filter((p) => p.position_group === pos);
    if (query.trim())
      r = r.filter((p) =>
        p.player_name.toLowerCase().includes(query.trim().toLowerCase()),
      );
    const s = [...r];
    if (sort === "adp")
      s.sort((a, b) => (a.adp ?? 9999) - (b.adp ?? 9999));
    else if (sort === "arb")
      s.sort(
        (a, b) => (b.arbitrage_delta ?? -9999) - (a.arbitrage_delta ?? -9999),
      );
    else s.sort((a, b) => a.overall_rank - b.overall_rank);
    return s.slice(0, 250);
  }, [players, pos, query, sort]);

  return (
    <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
      <Header season={season} format={format} data={data} />

      <div className="mt-6 flex flex-wrap items-center gap-3">
        <Segmented options={SEASONS.map((s) => ({ value: s, label: String(s) }))} value={season} onChange={setSeason} size="sm" />
        <Segmented options={FORMATS} value={format} onChange={setFormat} />
        <div className="mx-1 h-6 w-px bg-border" />
        <Segmented options={POSITIONS.map((p) => ({ value: p, label: p }))} value={pos} onChange={setPos} size="sm" />
        <div className="ml-auto flex items-center gap-3">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search player…"
            className="w-44 rounded-lg bg-neutral-bg2 px-3 py-1.5 text-sm text-text-primary ring-1 ring-inset ring-border placeholder:text-text-muted focus:ring-brand/50"
          />
          <Segmented options={SORTS} value={sort} onChange={setSort} size="sm" />
        </div>
      </div>

      <div className="mt-4 overflow-hidden rounded-xl border border-border bg-neutral-bg2/60 shadow-2xl shadow-black/40">
        <TableHeader />
        <div className="max-h-[calc(100vh-19rem)] overflow-y-auto">
          {loading && <Message text="Loading board…" />}
          {error && <Message text={`Could not reach the API. Is it running on :8000?  (${error})`} bad />}
          {!loading && !error && rows.length === 0 && <Message text="No players match." />}
          {!loading &&
            !error &&
            rows.map((p) => <Row key={p.player_id} p={p} scaleMax={scaleMax} />)}
        </div>
      </div>

      <p className="mt-4 max-w-3xl text-xs leading-relaxed text-text-muted">
        <span className="font-medium text-text-secondary">Honest framing.</span>{" "}
        Projections derive entirely from open nflverse data; the bar shows the 80%
        prediction interval (floor–ceiling), and <span className="text-good">Arb</span>{" "}
        flags where our value diverges from market ADP. In backtests our board is
        roughly at parity with ADP — treat this as a calibrated decision aid, not a
        promise to beat the market.
      </p>
    </main>
  );
}

function Header({
  season,
  format,
  data,
}: {
  season: number;
  format: string;
  data: BoardResponse | null;
}) {
  return (
    <header className="flex flex-wrap items-end justify-between gap-4">
      <div>
        <div className="flex items-center gap-2.5">
          <span className="h-6 w-6 rounded-lg bg-gradient-to-br from-brand to-brand-light shadow-glow" />
          <h1 className="text-2xl font-semibold tracking-tight">War Room</h1>
        </div>
        <p className="mt-1 text-sm text-text-secondary">
          Pre-draft value board · calibrated projections + ADP arbitrage radar
        </p>
      </div>
      <div className="text-right">
        <div className="num text-3xl font-semibold tabular-nums">{season}</div>
        <div className="text-xs uppercase tracking-wider text-text-muted">
          {data?.format_name ?? format}
        </div>
      </div>
    </header>
  );
}

const GRID =
  "grid grid-cols-[3px_2.5rem_4rem_minmax(0,1fr)_15rem_4.5rem_4rem_4.5rem] items-center gap-x-3";

function TableHeader() {
  return (
    <div
      className={`${GRID} border-b border-border bg-neutral-bg3/80 px-3 py-2.5 text-[11px] font-medium uppercase tracking-wider text-text-muted backdrop-blur`}
    >
      <div />
      <div className="text-right">#</div>
      <div>Pos</div>
      <div>Player</div>
      <div>Projection · floor–ceiling</div>
      <div className="text-right">VOR</div>
      <div className="text-right">ADP</div>
      <div className="text-right">Arb</div>
    </div>
  );
}

function Row({ p, scaleMax }: { p: Player; scaleMax: number }) {
  return (
    <div className={`${GRID} border-b border-border-subtle px-3 py-2 transition-colors hover:bg-white/[0.025]`}>
      <div className={`h-6 w-[3px] rounded-full ${POS_BAR[p.position_group] ?? "bg-white/10"}`} />
      <div className="num text-right text-sm tabular-nums text-text-muted">{p.overall_rank}</div>
      <div>
        <PositionBadge pos={p.position_group} tier={p.position_tier} />
      </div>
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <span className="truncate text-sm font-medium text-text-primary">{p.player_name}</span>
          {p.is_rookie && (
            <span className="rounded bg-brand-subtle px-1 text-[10px] font-semibold text-brand-light">R</span>
          )}
        </div>
        <div className="text-xs text-text-muted">{p.team ?? "FA"}</div>
      </div>
      <div className="flex items-center gap-3">
        <span className="num w-10 text-right text-sm font-semibold tabular-nums">
          {Math.round(p.projected_points)}
        </span>
        <div className="flex-1">
          <IntervalBar low={p.points_low} mid={p.projected_points} high={p.points_high} max={scaleMax} />
        </div>
        <span className="num w-20 text-right text-[11px] tabular-nums text-text-muted">
          {Math.round(p.points_low)}–{Math.round(p.points_high)}
        </span>
      </div>
      <div className="num text-right text-sm tabular-nums text-text-secondary">{p.vor.toFixed(0)}</div>
      <div className="num text-right text-sm tabular-nums text-text-muted">
        {p.adp !== null ? p.adp.toFixed(1) : "·"}
      </div>
      <div className="text-right">
        <ArbCell delta={p.arbitrage_delta} />
      </div>
    </div>
  );
}

function Message({ text, bad }: { text: string; bad?: boolean }) {
  return (
    <div className={`px-4 py-16 text-center text-sm ${bad ? "text-bad" : "text-text-muted"}`}>
      {text}
    </div>
  );
}
