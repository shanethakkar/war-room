"use client";

import { useEffect, useMemo, useState } from "react";
import {
  IntervalBar,
  POS_BAR,
  PositionBadge,
  Segmented,
  TiltCell,
} from "@/components/atoms";
import { RosterPanel } from "@/components/roster";
import { type BoardResponse, type Player, fetchBoard } from "@/lib/api";
import { type DraftPicks, nextMyPick, survivalProb } from "@/lib/draft";

const FORMATS = [
  { value: "redraft_ppr", label: "PPR" },
  { value: "redraft_half", label: "Half" },
  { value: "redraft_standard", label: "Std" },
  { value: "superflex", label: "SFLX" },
  { value: "two_qb", label: "2QB" },
];
const SEASONS = [2026, 2025, 2024, 2023];
const POSITIONS = ["ALL", "QB", "RB", "WR", "TE", "DST", "K"];
const SORTS = [
  { value: "board", label: "Board" },
  { value: "model", label: "Model" },
  { value: "adp", label: "Market" },
  { value: "tilt", label: "Tilt" },
];
const TEAMS = 12;

export default function Page() {
  const [season, setSeason] = useState(2026);
  const [format, setFormat] = useState("redraft_ppr");
  const [pos, setPos] = useState("ALL");
  const [sort, setSort] = useState("board");
  const [query, setQuery] = useState("");
  const [data, setData] = useState<BoardResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  // Draft-day state, persisted per (season, format).
  const [draft, setDraft] = useState(false);
  const [slot, setSlot] = useState(6);
  const [hideDrafted, setHideDrafted] = useState(true);
  const [picks, setPicks] = useState<DraftPicks>({});
  const [history, setHistory] = useState<string[]>([]);
  const storageKey = `warroom-draft-${season}-${format}`;

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

  // Load persisted draft on board switch. Saving happens imperatively in the
  // pick handlers (a single writer avoids StrictMode double-mount races that a
  // reactive save-effect would lose data to).
  useEffect(() => {
    try {
      const raw = localStorage.getItem(storageKey);
      const s = raw ? JSON.parse(raw) : null;
      setPicks(s?.picks ?? {});
      setHistory(s?.history ?? []);
      if (s?.slot) setSlot(s.slot);
    } catch {
      setPicks({});
      setHistory([]);
    }
  }, [storageKey]);

  const persist = (p: DraftPicks, h: string[], sl: number) => {
    try {
      localStorage.setItem(storageKey, JSON.stringify({ picks: p, history: h, slot: sl }));
    } catch {
      /* storage blocked: draft mode still works in-memory */
    }
  };

  const players = data?.players ?? [];
  const scaleMax = useMemo(
    () => Math.max(100, ...players.map((p) => p.points_high)),
    [players],
  );

  const currentOverall = Object.keys(picks).length + 1;
  const nextPick = draft ? nextMyPick(currentOverall, slot, TEAMS) : null;
  const nextRound = nextPick !== null ? Math.ceil(nextPick / TEAMS) : null;
  const mine = useMemo(
    () =>
      history
        .filter((id) => picks[id] === "mine")
        .map((id) => players.find((p) => p.player_id === id))
        .filter((p): p is Player => Boolean(p))
        .map((p) => ({ name: p.player_name, pos: p.position_group })),
    [history, picks, players],
  );

  const rows = useMemo(() => {
    let r = players;
    if (draft && hideDrafted) r = r.filter((p) => !picks[p.player_id]);
    if (pos !== "ALL") r = r.filter((p) => p.position_group === pos);
    if (query.trim())
      r = r.filter((p) =>
        p.player_name.toLowerCase().includes(query.trim().toLowerCase()),
      );
    const s = [...r];
    if (sort === "adp") s.sort((a, b) => (a.adp ?? 9999) - (b.adp ?? 9999));
    else if (sort === "model") s.sort((a, b) => a.model_rank - b.model_rank);
    else if (sort === "tilt")
      s.sort((a, b) => (b.model_tilt ?? -9999) - (a.model_tilt ?? -9999));
    else s.sort((a, b) => a.board_rank - b.board_rank);
    return s.slice(0, 250);
  }, [players, pos, query, sort, draft, hideDrafted, picks]);

  const mark = (id: string, state: "taken" | "mine") => {
    if (picks[id]) return;
    const nextPicks = { ...picks, [id]: state };
    const nextHistory = [...history, id];
    setPicks(nextPicks);
    setHistory(nextHistory);
    persist(nextPicks, nextHistory, slot);
  };
  const undo = () => {
    const last = history[history.length - 1];
    if (!last) return;
    const nextPicks = { ...picks };
    delete nextPicks[last];
    const nextHistory = history.slice(0, -1);
    setPicks(nextPicks);
    setHistory(nextHistory);
    persist(nextPicks, nextHistory, slot);
  };
  const reset = () => {
    setPicks({});
    setHistory([]);
    persist({}, [], slot);
  };
  const changeSlot = (n: number) => {
    setSlot(n);
    persist(picks, history, n);
  };

  return (
    <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
      <Header season={season} format={format} data={data} />

      <div className="mt-6 flex flex-wrap items-center gap-3">
        <Segmented
          options={SEASONS.map((s) => ({ value: s, label: String(s) }))}
          value={season}
          onChange={setSeason}
          size="sm"
        />
        <Segmented options={FORMATS} value={format} onChange={setFormat} size="sm" />
        <div className="mx-1 h-6 w-px bg-border" />
        <Segmented
          options={POSITIONS.map((p) => ({ value: p, label: p }))}
          value={pos}
          onChange={setPos}
          size="sm"
        />
        <div className="ml-auto flex items-center gap-3">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search player…"
            className="w-40 rounded-lg bg-neutral-bg2 px-3 py-1.5 text-sm text-text-primary ring-1 ring-inset ring-border placeholder:text-text-muted focus:ring-brand/50"
          />
          <Segmented options={SORTS} value={sort} onChange={setSort} size="sm" />
          <button
            onClick={() => setDraft(!draft)}
            className={`rounded-lg px-3 py-1.5 text-sm font-semibold transition-colors ${
              draft
                ? "bg-brand text-white shadow-glow"
                : "bg-neutral-bg2 text-text-secondary ring-1 ring-inset ring-border hover:text-text-primary"
            }`}
          >
            {draft ? "Drafting…" : "Draft mode"}
          </button>
        </div>
      </div>

      {draft && (
        <div className="mt-3 flex flex-wrap items-center gap-3 rounded-lg border border-brand/25 bg-brand-subtle/40 px-3 py-2 text-sm">
          <span className="num tabular-nums text-text-secondary">
            Pick <span className="font-semibold text-text-primary">#{currentOverall}</span>
            <span className="text-text-muted"> · R{Math.ceil(currentOverall / TEAMS)}</span>
          </span>
          <label className="flex items-center gap-1.5 text-text-secondary">
            my slot
            <select
              value={slot}
              onChange={(e) => changeSlot(Number(e.target.value))}
              className="rounded bg-neutral-bg2 px-1.5 py-0.5 text-sm text-text-primary ring-1 ring-inset ring-border"
            >
              {Array.from({ length: TEAMS }, (_, i) => i + 1).map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
          <span className="text-text-muted">
            click a row = drafted by someone ·{" "}
            <span className="text-brand-light">＋ = my pick</span>
          </span>
          <div className="ml-auto flex items-center gap-2">
            <button
              onClick={() => setHideDrafted(!hideDrafted)}
              className="rounded px-2 py-1 text-xs text-text-secondary ring-1 ring-inset ring-border hover:text-text-primary"
            >
              {hideDrafted ? "show drafted" : "hide drafted"}
            </button>
            <button
              onClick={undo}
              disabled={history.length === 0}
              className="rounded px-2 py-1 text-xs text-text-secondary ring-1 ring-inset ring-border hover:text-text-primary disabled:opacity-40"
            >
              undo
            </button>
            <button
              onClick={reset}
              className="rounded px-2 py-1 text-xs text-bad/80 ring-1 ring-inset ring-border hover:text-bad"
            >
              reset
            </button>
          </div>
        </div>
      )}

      <div className="mt-4 flex items-start gap-5">
        <div className="min-w-0 flex-1 overflow-hidden rounded-xl border border-border bg-neutral-bg2/60 shadow-2xl shadow-black/40">
          <TableHeader draft={draft} />
          <div className="max-h-[calc(100vh-21rem)] overflow-y-auto">
            {loading && <Message text="Loading board…" />}
            {error && (
              <Message text={`Could not reach the API. Is it running?  (${error})`} bad />
            )}
            {!loading && !error && rows.length === 0 && <Message text="No players match." />}
            {!loading &&
              !error &&
              rows.map((p) => (
                <Row
                  key={p.player_id}
                  p={p}
                  scaleMax={scaleMax}
                  draft={draft}
                  state={picks[p.player_id]}
                  nextPick={nextPick}
                  onTake={() => mark(p.player_id, "taken")}
                  onMine={() => mark(p.player_id, "mine")}
                />
              ))}
          </div>
        </div>
        {draft && (
          <RosterPanel format={format} mine={mine} nextPick={nextPick} round={nextRound} />
        )}
      </div>

      <p className="mt-4 max-w-3xl text-xs leading-relaxed text-text-muted">
        <span className="font-medium text-text-secondary">How this board works.</span>{" "}
        The ranking anchors to market ADP with a small validated tilt from our
        open-data models (~80% market / 20% model ensemble; DST and K rank purely
        by market — our signal there is noise, and we measured it). In realistic
        six-season draft simulations vs a full league of ADP drafters this board
        averaged <span className="text-text-secondary">+33 points a season (~2/week)</span>,
        lifting points-based playoff odds from 50% to ~57% — but the edge is
        year-dependent: it won clearly in 2021–2023 and{" "}
        <span className="text-bad">underperformed the market in 2019 and 2024</span>.
        With six seasons of history we estimate an ~83% chance the edge is real —
        a multi-season tilt, not a guarantee. <span className="text-good">Tilt</span>{" "}
        shows where the models nudge a player from market; in draft mode,{" "}
        <span className="text-text-secondary">Avail</span> is the chance a player
        survives to your next pick. Bars show the calibrated 80% floor–ceiling
        interval.
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
          Pre-draft board · market-anchored ranking + calibrated model tilt
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

const gridCols = (draft: boolean) =>
  draft
    ? "grid grid-cols-[1.8rem_3px_2.5rem_4rem_minmax(0,1fr)_13rem_4rem_4rem_4rem_4rem] items-center gap-x-3"
    : "grid grid-cols-[3px_2.5rem_4rem_minmax(0,1fr)_15rem_4.5rem_4rem_4.5rem] items-center gap-x-3";

function TableHeader({ draft }: { draft: boolean }) {
  return (
    <div
      className={`${gridCols(draft)} border-b border-border bg-neutral-bg3/80 px-3 py-2.5 text-[11px] font-medium uppercase tracking-wider text-text-muted backdrop-blur`}
    >
      {draft && <div />}
      <div />
      <div className="text-right">#</div>
      <div>Pos</div>
      <div>Player</div>
      <div>Projection · floor–ceiling</div>
      <div className="text-right">VOR</div>
      <div className="text-right">ADP</div>
      <div className="text-right">Tilt</div>
      {draft && <div className="text-right">Avail</div>}
    </div>
  );
}

function AvailCell({ prob }: { prob: number | null }) {
  if (prob === null) return <span className="text-text-muted">·</span>;
  const pct = Math.round(prob * 100);
  const tone = pct >= 75 ? "text-good" : pct >= 40 ? "text-pos-te" : "text-bad";
  return <span className={`num tabular-nums ${tone}`}>{pct}%</span>;
}

function Row({
  p,
  scaleMax,
  draft,
  state,
  nextPick,
  onTake,
  onMine,
}: {
  p: Player;
  scaleMax: number;
  draft: boolean;
  state?: "taken" | "mine";
  nextPick: number | null;
  onTake: () => void;
  onMine: () => void;
}) {
  const dimmed = draft && state === "taken";
  const isMine = draft && state === "mine";
  return (
    <div
      onClick={draft && !state ? onTake : undefined}
      className={`${gridCols(draft)} border-b border-border-subtle px-3 py-2 transition-colors ${
        draft && !state ? "cursor-pointer hover:bg-white/[0.03]" : "hover:bg-white/[0.02]"
      } ${dimmed ? "opacity-35" : ""} ${isMine ? "bg-brand-subtle/40" : ""}`}
    >
      {draft && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            if (!state) onMine();
          }}
          disabled={Boolean(state)}
          title="My pick"
          className="num rounded text-center text-sm text-brand-light ring-1 ring-inset ring-brand/30 hover:bg-brand-subtle disabled:opacity-30"
        >
          {isMine ? "✓" : "＋"}
        </button>
      )}
      <div className={`h-6 w-[3px] rounded-full ${POS_BAR[p.position_group] ?? "bg-white/10"}`} />
      <div className="num text-right text-sm tabular-nums text-text-muted">{p.board_rank}</div>
      <div>
        <PositionBadge pos={p.position_group} tier={p.position_tier} />
      </div>
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <span
            className={`truncate text-sm font-medium ${dimmed ? "line-through" : ""} text-text-primary`}
          >
            {p.player_name}
          </span>
          {p.is_rookie && (
            <span className="rounded bg-brand-subtle px-1 text-[10px] font-semibold text-brand-light">
              R
            </span>
          )}
        </div>
        <div className="text-xs text-text-muted">{p.team ?? "FA"}</div>
      </div>
      <div className="flex items-center gap-3">
        <span className="num w-10 text-right text-sm font-semibold tabular-nums">
          {Math.round(p.projected_points)}
        </span>
        <div className="flex-1">
          <IntervalBar
            low={p.points_low}
            mid={p.projected_points}
            high={p.points_high}
            max={scaleMax}
          />
        </div>
        <span className="num w-20 text-right text-[11px] tabular-nums text-text-muted">
          {Math.round(p.points_low)}–{Math.round(p.points_high)}
        </span>
      </div>
      <div className="num text-right text-sm tabular-nums text-text-secondary">
        {p.vor.toFixed(0)}
      </div>
      <div className="num text-right text-sm tabular-nums text-text-muted">
        {p.adp !== null ? p.adp.toFixed(1) : "·"}
      </div>
      <div className="text-right">
        <TiltCell delta={p.model_tilt} />
      </div>
      {draft && (
        <div className="text-right">
          <AvailCell
            prob={nextPick !== null ? survivalProb(p.adp, p.adp_stdev, nextPick) : null}
          />
        </div>
      )}
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
