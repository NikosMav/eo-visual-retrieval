# BigEarthNet S2 acquisition

The downloader consumes the original compressed stream in bounded HTTP ranges, discards
non-selected members, and retains native GeoTIFFs only for frozen IDs. It never saves the S2
archive or a decompressed tar. This is acquisition infrastructure; it produces no embeddings,
model inputs, relevance manifests, or retrieval scores.

## Current gate

The [executed diagnostic](results/bigearthnet-s2-pilot-diagnostic.json) reached two of the planned
30 pilot patches within 64 MiB. Their 24 bands agreed with the frozen footprints, but neither the
three-partition pilot nor whole-source integrity passed. Full acquisition remains blocked.

[Zenodo](https://zenodo.org/records/10891137) publishes a checksum for the complete
63,251,710,377-byte S2 archive. No independently checksummed patch delivery has been established.
A prefix hash cannot verify that checksum. With this source, authenticating even a 30-patch pilot
requires consuming the complete compressed stream. Acquiring the full selection afterward needs
another pass because the pilot discards other patches. Neither large pass has been executed.
Proceeding with those costs needs a separate decision; the default stays at 64 MiB.

## Frozen inputs and pilot

`docs/results/bigearthnet-selection-audit.json` pins the SHA-256 of the ignored local
`acquisition-selection.json` and `footprints.parquet`. The downloader verifies both, reads only
IDs and geometry, and checks the exact 4,000 / 500 / 500 partition membership. It does not select
new partitions. The pilot uses ten existing IDs from distinct spatial cells in each partition,
chosen deterministically by sorted ID without labels or model outputs. Resume state binds the
audit, selection, inventory, pilot IDs, and all selected IDs.

The explicit native Level-2A band order is:

```text
B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12
```

| Resolution | Bands | Native dimensions |
|---|---|---|
| 10 m | B02, B03, B04, B08 | 120 x 120 |
| 20 m | B05, B06, B07, B8A, B11, B12 | 60 x 60 |
| 60 m | B01, B09 | 20 x 20 |

B10 is absent; this is not the 13-band EuroSAT layout. Each file must have one uint16 band.
Every retained band is fully decoded and checked for exact bounds and CRS against its frozen
footprint. Its affine transform must use that footprint's upper-left origin, its band's native
resolution, north-up orientation, and no rotation. Coarser bands share the 1,200 m footprint;
they do not share the reference map's 10 m pixel spacing. Nothing is resampled or adjusted.
Any geometry disagreement is terminal and records expected and observed values.

## Running

Install the locked `bigearthnet` and `geo` extras in the project's short virtual environment.
From the repository, this command only prints the plan and existing storage count:

```powershell
python scripts/acquire_bigearthnet_s2.py `
  --selection data/bigearthnet-v2/acquisition-selection.json `
  --inventory data/bigearthnet-v2/footprints.parquet `
  --source-dir data/downloads/bigearthnet-v2 `
  --root data/bigearthnet-v2/s2-acquisition/pilot
```

Add `--download` to execute. Defaults are `--phase pilot`, `--network-budget 67108864`, and
`--max-seconds 300`. The deadline is checked between members and HTTP reads; a blocking socket
read can last up to its timeout. The command exits nonzero on an incomplete or failed attempt.

`--phase full` additionally requires `--pilot-root` pointing to a verified pilot with matching
inputs, complete receipts, and unchanged band hashes. It fails before network access otherwise.
The CLI never raises the network/time allowance automatically, even in full mode.

## Storage, trust, and restart

The hard ceiling is **2,147,483,648 logical file bytes**, including existing metadata/reference
files, inventory, selection, sibling acquisition directories, retained bands, receipts, state,
and temporary atomic writes. `--source-dir` must point to the actual retained source directory.
Keep acquisition files together and do not modify ancillary files during a run. This ceiling
counts file contents, not filesystem allocation units, journals, or directory metadata.

Writes reserve their complete temporary payload before touching disk, then flush and atomically
replace. A process lock prevents competing writers to the same acquisition root. Links,
unexpected member types/paths, oversized members/extensions, duplicate selected bands, and
incomplete band groups fail closed. The tar parser buffers at most one 256 KiB member, HTTP reads
are capped at 64 KiB, and decompression has an explicit window limit.

`files/<patch>/Bxx.tif` and `patch.json` are **untrusted staging** until `COMPLETE.json` exists
and `require_complete()` validates it. Only the complete published MD5 and byte count, a full
stream SHA-256, every required band, and every geometry comparison can unlock that marker.
The marker is written last. Local file hashes identify cached bytes; they do not independently
authenticate a prefix against the publisher. No consumer should infer completion from TIFF counts
or from `state.json` alone.

Use `--resume` after interruption. In-process transport retries continue at the last delivered
compressed offset, preserving decoder and digest state. A process restart must replay from byte
zero because neither the Zstandard decoder nor hash state is saved. Matching TIFFs are reused
without rewriting; all bytes are rehashed and geometry is rechecked. Corrupt cached data fails.
Stale temporary files are removed only inside the dedicated acquisition root while holding its
lock. An incomplete run can never promote a partial selection.

The cumulative network ledger reserves each complete range response **before** opening it. Failed
or interrupted reservations remain charged, including across process restarts. Thus a retry may
need an explicit larger cumulative `--network-budget`; the tool never resets that ledger.
`network_received_bytes` counts compressed bytes delivered to the application. Reservations are
conservative and can exceed received bytes; neither count includes HTTP/TLS overhead.

`geometry_mismatch` and `integrity_failure` are terminal. Do not rewrite footprints or erase state
to bypass them. Investigate and obtain a decision before another attempt.
