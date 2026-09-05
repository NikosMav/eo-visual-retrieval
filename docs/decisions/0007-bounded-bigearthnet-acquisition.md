# ADR 0007: Bounded BigEarthNet imagery acquisition

- Status: accepted; streaming implemented and tested, full acquisition not yet executed
- Date: 2026-09-03
- Scope: reduce local disk use while preserving source integrity

## Evidence

The executed [metadata audit](../results/bigearthnet-metadata-audit.json) establishes usable
labels and dates. The subsequent [footprint inventory](../results/bigearthnet-footprints.json) and
[selection audit](../results/bigearthnet-selection-audit.json) establish a source-georeferenced
acquisition selection under [ADR 0008](0008-bigearthnet-selection-protocol.md). Usable image
partitions still depend on the full S2 stream and per-band validation.

The [official Zenodo record](https://zenodo.org/records/10891137) reports an S2 archive of
63,251,710,377 bytes and a reference-map archive of 282,391,301 bytes. An executed pair of 32-byte
range requests for the start and end of the S2 archive returned HTTP 206 with matching content
ranges. Its final four bytes do not match the mandatory footer of the
[Zstandard seekable format](https://github.com/facebook/zstd/blob/dev/contrib/seekable_format/zstd_seekable_compression_format.md).
No external member/seek index is advertised by the source. We have not established random patch
access without a sequential pass.

A 1 MiB prefix probe successfully streamed through Zstandard and tar to decode a complete
12-band patch in memory. Its bands are uint16, have the expected 120/60/20-pixel dimensions, and
share a georeferenced 1,200 m square footprint. The native GeoTIFF payload totals 164,560 bytes.
At that single-patch size, 5,000 patches would occupy 822,800,000 bytes (about 0.77 GiB). This is a
planning estimate, not a measured full-subset size. The sample was neither scored nor installed
as benchmark data. Its ID is recorded in the [probe evidence](evidence/bigearthnet-access-2026-09-03.json)
and should be excluded from the future final-query set.

The first compressed frame advertises 102,406,983,680 decompressed bytes and a 2 MiB window.
This was read from its header, not established by decoding the full archive. Avoiding full
decompression onto disk would therefore save substantially more space than just the compressed
archive. The successful probe used `zstandard==0.25.0` and Rasterio in memory. An earlier probe
failed because its decoder memory limit was too low; neither probe persisted image payloads.

## Alternative sources checked

| Source | Finding | Implication |
|---|---|---|
| [TorchGeo mirror](https://huggingface.co/datasets/torchgeo/bigearthnet/tree/3cf3a5910a5302d449fdb8e570e5b78de24fe07f/V2) | S2 is distributed as `.tar.gzaa` and `.tar.gzab`, about 48.3 and 15.2 GB | These are large archive pieces; independent patch access was not established. |
| [Community LMDB conversion](https://huggingface.co/datasets/hackelle/BigEarthNetV2-LMDB/tree/118d1b6285c080ba8e4078414e1b8a243b18c9bd) | About 155 GB; explicitly unofficial; geospatial TIFF metadata is not retained as original files | Does not meet the local-storage or original-georeferencing goals. |
| [BIFOLD BigEarthNet.txt](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/blob/72d865f2146f0a85b720f7f3ca1cdbaeafc3d316/README.md) | Contains text/patch metadata; directs users to obtain imagery separately | Useful contextual metadata, not an alternate image delivery service. |

These checks are bounded to the named sources and revisions. They do not establish that no other
valid source could exist.

## Proposed acquisition sequence and budgets

1. Build a source-georeferenced footprint inventory before fixing the subset. **Executed:** the
   checksum-verified 282,391,301-byte reference archive supplies 549,488 map footprints in an
   11,606,074-byte local Parquet inventory. No maps were extracted to disk. The first S2 probe
   footprint agrees; all selected S2 bands must still be checked after acquisition.
2. Audit geography and dates, then freeze exactly 4,000 index, 500 development, and 500 final IDs
   inside their official splits. **Executed:** every partition retains all 19 labels, disjoint
   50 km cells, a 7 km centre guard, and the chronological windows in ADR 0008. All 5,000 selected
   reference footprints were independently reopened with Rasterio and matched the inventory.
3. Stream the original S2 archive once, hashing all compressed bytes while reading tar members
   sequentially. Retain only the frozen IDs' 12 native bands and the required small manifests.
4. Apply a **2 GiB (2,147,483,648-byte) ceiling** to acquisition files, including staging, selected
   imagery, any retained reference archive, and the footprint inventory. Reject unexpected member
   types/paths, duplicate bands, oversized members, and incomplete patch groups. Bounded buffers
   must prevent the decompressed tar from accumulating in memory or on disk.
5. Consume the complete compressed stream and verify the published MD5 and exact byte count before
   promoting the subset. Reaching the last selected member is insufficient for source integrity.
   Record the full-stream SHA-256 and per-file identities. Recheck selected S2 georeferencing
   against the inventory before marking the split usable.

The S2 network ceiling is **63,251,710,377 bytes per complete pass**. Including one reference-map
pass gives 63,534,101,678 bytes, excluding the already completed small metadata/probe requests.
A failed S2 pass may require a restart: HTTP range support does not, by itself, preserve decoder
state. A retry policy must expose cumulative transferred bytes and must not silently multiply
this budget. No reliable full-transfer time has been measured from the small probes.

Streaming reduces retained storage, not the bytes needed to verify the complete source archive.
Later embeddings, checkpoints, and model-ready derived arrays require their own budget; they are
not included in the 2 GiB acquisition ceiling. Retaining the complete archive for fast reruns is
an optional alternative that would require approximately 59 GiB extra disk space.

## Relationship to ADR 0006

ADR 0006 currently specifies a checksum-verified archive under `data/downloads/`. This proposal
would allow complete-stream checksum verification followed by promotion of a bounded subset,
without keeping the full archive. The alternative must be implemented and validated before this
proposal can replace that acquisition requirement. It changes neither relevance metrics nor the
holdout sizes and makes no benchmark-quality claim.

## Remaining gates

- [x] Validate a compact, source-georeferenced footprint inventory.
- [x] Freeze and audit the spatial/temporal acquisition selection with sufficient label coverage.
- [x] Implement and test bounded streaming, complete-source integrity, failure cleanup, and budgets.
      `datasets/bigearthnet_s2.py` provides ranged streaming, an acquisition lock, budget
      checkpointing, `--resume`, integrity verification, and a completion marker, under 18 tests
      in `tests/test_bigearthnet_s2.py`. A [120-second sample](../validation.md) executed the path
      against the real source at 1.1802 MiB/s.
- [ ] Execute full acquisition only after these gates; then audit all selected S2 files.
      Not started. Paused by operator decision, not blocked by missing code.

### Correcting this record

Until 2026-09-05 the status line above read "full S2 streaming remains unimplemented" and the
third gate was unchecked, both of which the code had already overtaken. That stale text was read
as current and produced a recommendation against acquisition on the grounds that its restart
strategy did not exist. It does: resume replays the compressed stream from zero, which is costly
after an interruption but is a strategy, and the implementation says so about itself.

A plan document that outlives its own implementation is a hazard, because it is quoted with the
authority of a decision record. The gate list is the thing to check against the code, not the
status line.
