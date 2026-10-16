# Fixed legacy archive fixture

Created once from synthetic CPU inputs with the former `ditserve.results.legacy.finalize_run` writer (before its removal).
Writer source SHA256: `cf3b1f157ea6ce18364bdd486f7f9159443906e0d24a0b66593d65093af66476`.
The archived absolute source path identifies a removed temporary fixture directory; it is not a live experiment dependency.

This freezes `ditserve.archive.v1` independently of the current reader. The payload contains one failed `edf` case,
zero completed requests, one parity failure, a precise float, a false boolean, a large integer, a literal formula-like
string and a missing value. The raw artifact is represented only by its size/hash. Tests copy this archive into
private temporary directories to verify unchanged reads and reject payload, workbook type/value and report corruption.
No GPU experiment or real result is represented here. Do not regenerate it using the current reader.

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `INDEX.md` | 335 | `a628644e52c2f87070c19e99ebfe65e22814bbc05ebf9b2e0ccf00c27431523c` |
| `archive.json` | 382 | `2d25edcdf65aac0e366c8ddd478cd0a2eb08eb4a227ee6c4fa89e7652237ec8a` |
| `series/step/REPORT.md` | 1360 | `705fbb36796588769d59caf9ff2ec19a4a67b85a25ab6171592f497895e7d12e` |
| `series/step/data.json.gz` | 747 | `21b9c82804b9b337331aaecdc8b52f7b4ef66eef44841c5890dab7009130f8f2` |
| `series/step/results.xlsx` | 8045 | `f8288c34351dcacac6033190cd5ef15bfbf6d7a5ef79dbbd7d267afbf6ec631e` |
