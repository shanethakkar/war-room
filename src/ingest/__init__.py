"""Data layer — ingestion.

Pull nflverse tables via ``nflreadpy`` and cache them to local Parquet. Output is
deterministic and offline-reproducible; nothing downstream reads the network.
See design.md §3.1 and §9.
"""
