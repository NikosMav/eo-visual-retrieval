# BigEarthNet S2 acquisition

The downloader consumes the original compressed stream in bounded HTTP ranges, discards
non-selected members, and retains native GeoTIFFs only for frozen IDs. It never saves the S2
archive or a decompressed tar. This is acquisition infrastructure; it produces no embeddings,
model inputs, relevance manifests, or retrieval scores.

## Single-pass gate

One sequential pass computes the compressed-source MD5 while extracting all 5,000 frozen patches
to staging. Each selected GeoTIFF is decoded and checked immediately. The first geometry mismatch
aborts at its observed compressed-stream offset and records the patch and band. If all patches
agree, only the exact 63,251,710,377-byte stream with Zenodo's published MD5 can create the final
completion marker. There is no separate pilot pass.

The initial [64 MiB diagnostic](results/bigearthnet-s2-pilot-diagnostic.json) already supplied the
early pilot signal: 24 bands from two selected patches matched. A subsequent
[two-minute sample](results/bigearthnet-s2-throughput-sample.json) measured 1.18 MiB/s and projects
about 14.2 hours for the source if that rate persists. Both prefixes remain untrusted staging.

## Frozen inputs

`docs/results/bigearthnet-selection-audit.json` pins the SHA-256 of the ignored local
`acquisition-selection.json` and `footprints.parquet`. The downloader verifies both, reads only
IDs and geometry, and checks the exact 4,000 / 500 / 500 partition membership. It does not select
new partitions. Resume state binds the independently verified reference archive MD5, audit,
selection, inventory, and all selected IDs.

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
  --root data/bigearthnet-v2/s2-acquisition/full
```

Add `--download` to execute the single real pass. Acquisition defaults to the exact published
source byte count as its cumulative network budget and a 24-hour attempt deadline. The deadline is
checked between members and HTTP reads; a blocking socket read can last up to its timeout. The
command exits nonzero on an incomplete or failed attempt.

`--mode sample` selects a 120-second diagnostic with a 512 MiB network budget. It uses the same
stream/parser/geometry path but can never promote staging. A sample is not an acquisition gate.

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
and `require_complete()` validates it. Per-patch verification is performed first: each band must
parse, match the independently checksum-verified reference geometry, and have a valid local hash.
At stream end, the complete published MD5 and byte count plus a full-stream SHA-256 must also pass.
Only all of these checks together can unlock the marker.
The marker is written last. Local file hashes identify cached bytes; they do not independently
authenticate a prefix against the publisher. No consumer should infer completion from TIFF counts
or from `state.json` alone.

In-process transport retries continue at the exact last delivered compressed offset, preserving
the live decoder and digest state. After process or machine restart, use `--resume` with an
explicit cumulative `--network-budget`. The restart replays from byte zero because the one-frame
Zstandard decoder state is not serializable. That replay preserves a byte-exact whole-source MD5
for the final attempt; it may retransfer the prefix consumed by the interrupted attempt. Matching
TIFFs are reused without rewriting, while all bytes are rehashed and geometry is rechecked.
Corrupt cached data fails.
Stale temporary files are removed only inside the dedicated acquisition root while holding its
lock. An incomplete run can never promote a partial selection.

A process-restart mode that starts the compressed source at offset `N` and promotes from
per-patch checks alone is deliberately unavailable. The source's first frame declares the entire
102,406,983,680-byte decompressed tar, and the source has no seek table. The
[Zstandard format](https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md)
does not provide random access; blocks can depend on prior decoded data, offsets, Huffman trees,
and FSE tables. The installed decoder exposes
no serializable state. Starting a fresh decoder at `N` therefore cannot decode or verify later tar
members. Labeling that behavior as resume would create partial data that could appear complete.
Supporting such a mode requires an independently seekable source, publisher-supplied per-patch
objects, or a retained/repacked archive outside the 2 GiB ceiling.

The cumulative network ledger reserves each complete range response **before** opening it. Failed
or interrupted reservations remain charged, including across process restarts. Thus a retry may
need an explicit larger cumulative `--network-budget`; the tool never resets that ledger.
`network_received_bytes` counts compressed bytes delivered to the application. Reservations are
conservative and can exceed received bytes; neither count includes HTTP/TLS overhead.

`geometry_mismatch` and `integrity_failure` are terminal. Do not rewrite footprints or erase state
to bypass them. Investigate and obtain a decision before another attempt.
