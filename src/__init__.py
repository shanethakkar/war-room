"""War Room — open-source fantasy football draft & analysis engine.

Layered pipeline (see design.md §3):

    Data (ingest, features) → Projection → Decision → Interface (api, frontend)

Each layer consumes the one above. Projections derive entirely from the open
nflverse ecosystem; Sleeper is used only for ADP and live-draft sync.
"""

__version__ = "0.1.0"
