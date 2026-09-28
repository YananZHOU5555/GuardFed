# Source package and provenance

This directory preserves the runtime layout used by the 2026-09-28 image gates and validation-screen workers. Source files and jobs were copied byte-for-byte; `PUBLISH_COPY_INVENTORY.json` records 128 selected files including the three adjacent deployment entrypoints. `PUBLISH_VALIDATION.json` records syntax, import-dependent offline checks and the credential-pattern scan performed on this copied tree. It does not replace the separately recorded real-data/GPU gates.

No `.git` directory, Python cache, dataset, archive or model binary is included. The four Python files under `integration_20260928/source_snapshot/` are the minimum reference CNN/core/data-loader files required by its offline synthetic-image check, not a complete nested checkout. Runtime workers continue loading the explicit `--repo` core and verifying its frozen source/data identities.

## Upstream source and licensing status

- FedAA: https://github.com/Gp1g/FedAA at commit `1fba884934cbec1b3506812d9d612e6a181b1d79`. The pinned `DDPG/DDPG.py`, upstream README and dependency list are retained alongside the adapter. The existing audit reports that no LICENSE/COPYING file was found in that tracked repository. This package does not assign a new license or claim an upstream redistribution grant. See `fedaa/ADAPTER_REPORT.md` for the original provenance and adaptation boundaries.
- LASA: https://github.com/JiiahaoXU/LASA at commit `8477367a4e8708cde264f7572805040c650af59f`. The three already retrieved reference source files remain under `lasa_20260928/upstream/`; their hashes and exact-output comparisons remain in its audit/check records. The existing retrieved subset does not contain a LICENSE file, and these records do not establish a license grant. No license is invented here. See `lasa_20260928/AUDIT.md`.

## Path and byte-identity constraints

Preserve the relative `fedaa/`, `integration_20260928/`, `lasa_20260928/` and `screen_20260928/{fedaa,lasa}` hierarchy. FedAA loads its sibling official wrapper and pinned DDPG source; its cross-version comparison loads the retained pilot comparator. LASA offline checks read the sibling pilot and its retained upstream reference files. These paths were exercised by the copied-tree checks.

The supervised entrypoints and shared-calibration tools contain the original `/workspace/GuardFed-celeba-expanded` paths. Historical records also retain Windows project paths. These are deployment/audit locations, not credentials; this package does not claim arbitrary-directory portability for the fixed entrypoints. A new deployment location requires a separately recorded path/config amendment rather than silently editing a frozen worker.

Some executed upstream/source files and frozen JSON jobs contain CRLF. The local `.gitattributes` disables text conversion in this subtree, and `deployment/.gitattributes` does the same for the three adjacent entrypoints, preserving their recorded byte-level SHA256 despite the repository-wide LF preference. Checks after copying found no changed source/job/report hashes and no matches for the credential patterns scanned.

The parent task owns the final parent queue manifest, protocol and `run_screen.py`; they were deliberately excluded from this copying subtask. No commit, push or server operation was performed by it.
