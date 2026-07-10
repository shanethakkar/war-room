# War Room — frontend

Dark, data-dense **pre-draft board** (Next.js App Router + Tailwind). Renders the
value board from the FastAPI backend: calibrated projections with 80% intervals,
format-aware VOR + tiers, and the **ADP arbitrage radar** (where our value diverges
from the market).

## Run

Start the API first (from the repo root):

```bash
uv run uvicorn src.api.main:app --reload   # serves http://localhost:8000
```

Then the frontend:

```bash
cd frontend
npm install
npm run dev                                 # http://localhost:3000
```

The API base URL defaults to `http://localhost:8000`; override with
`NEXT_PUBLIC_API_URL`.

> Honest framing: in backtests the board is roughly at parity with ADP. It's a
> calibrated decision aid (floor/ceiling + market disagreement), not a promise to
> beat the market.
