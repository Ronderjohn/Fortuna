# Recommended workspace location

Fortuna runs best from a **local disk path** (not OneDrive) to avoid:

- Shell/path issues with synced folders
- Accidental sync of large caches (`.venv`, Parquet, DuckDB)
- Slower I/O during backtests

**Canonical path:** `C:\dev\Fortuna`

Open this folder in Cursor/VS Code as your workspace root.

If you still have a copy under OneDrive (`Documents\Abhay's Projects\Fortuna`), close the editor, delete that folder, and use only `C:\dev\Fortuna`.
