---
name: data-wrangling
description: Transform CSV, JSON, INI and log files reliably with Python's standard library.
---
- Prefer a short Python script over long shell pipelines for anything with quoting, headers or types.
- CSV: `csv.DictReader` / `csv.DictWriter`; never split lines on commas by hand (quoted fields contain commas).
- JSON: `json.load` / `json.dump(obj, f, indent=2, sort_keys=True)` when output order matters.
- INI: `configparser`; values are strings, so convert types explicitly ("true" -> True, "8080" -> 8080).
- Logs: `re` with a compiled pattern; count with `collections.Counter`.
- SQLite: `sqlite3.connect(path)`; inspect tables with `SELECT name FROM sqlite_master`.
- Before writing the answer, print a few rows of the result and check them against the task's exact format
  (sorting, header row, trailing newline, number formatting).
