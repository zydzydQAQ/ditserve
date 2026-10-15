"""Best-effort Excel exports after reports, without changing experiment evidence."""

import json
from pathlib import Path
import warnings

from .util import CONFIGS


def export_report(root, *, ledger=None):
    """Refresh the lightweight workbook; export failures never fail a report.

    Raw files are working inputs until a separate verified cleanup is possible.
    Import the optional workbook exporter only after accounting is settled.
    """
    root = Path(root).resolve()
    try:
        policy = json.loads((CONFIGS / "result_storage.json").read_text(encoding="utf-8"))
        if not policy.get("auto_export_reports", True):
            return dict(status="disabled")
        directory = policy.get("export_directory", "excel")
        if not isinstance(directory, str) or directory in ("", ".", "..") or Path(directory).name != directory:
            raise ValueError("Excel export directory must be one directory name")
        ledgers = {Path(ledger).resolve()} if ledger is not None else set()
        for name, field in (("protocol.json", "ledger"), ("report.json", "budget_path")):
            path = root / name
            if path.is_file():
                metadata = json.loads(path.read_text(encoding="utf-8"))
                reference = metadata.get(field)
                if reference:
                    reference = Path(reference)
                    ledgers.add(reference if reference.is_absolute() else root / reference)
        for path in ledgers:
            if not path.exists():
                # A missing ledger may simply mean that GPU work never started.
                continue
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("active") or value.get("active_session"):
                return dict(status="protected", reason="GPU accounting is still active", ledger=str(path))
        from .compact_results import export
        return export(root, root / directory, replace_existing=True)
    except Exception as error:
        # Reporting must remain available with missing optional dependencies,
        # changing active evidence, or a previously finalized archive.
        warnings.warn("Excel export deferred for %s: %s" % (root, error), RuntimeWarning, stacklevel=2)
        return dict(status="deferred", reason=str(error))
