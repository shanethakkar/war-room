"""Data layer — feature panel.

Assemble the clean **player-season panel** (one row per player-season) plus the
weekly detail needed to derive opportunity features: volume (targets, carries,
routes, snaps), efficiency, and opportunity. Pure ``panel → panel`` transforms;
I/O stays at the edges. See design.md §3.1 and §4.
"""
