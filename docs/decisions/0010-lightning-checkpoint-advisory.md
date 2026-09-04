# ADR 0010: Lightning checkpoint advisory reachability

- Status: accepted; not reachable in any executed path, and no patched release exists
- Date: 2026-09-04
- Scope: decide how to hold Dependabot alert #4 while the upstream fix is unreleased

## The advisory

Read from the [GitHub advisory record](https://github.com/advisories/GHSA-qqmf-gpg7-g8gw) on
2026-09-04. It is `GHSA-qqmf-gpg7-g8gw` / `CVE-2026-58659`, severity high, published 2026-07-15,
last updated 2026-09-03, not withdrawn.

Its description states that PyTorch Lightning **through 2.6.5**, fixed in commit `d710d68`,
executes attacker-controlled module names taken from a checkpoint's `_instantiator`
hyperparameter, and that this happens when `LightningModule.load_from_checkpoint` is called.

Its structured version data disagrees with that description. The affected range is recorded as
`< 2022.6.15` for the `lightning` pip package, with `2022.6.15` as the first patched version.
That is a calendar-numbered 2022 development release. PEP 440 orders `2.6.5` below `2022.6.15`,
which is why the alert fires against the installed version, but the range's remedy would install
a package four years older than the one the description calls vulnerable.

**No date-numbered release may be installed to satisfy this range.** Doing so would replace a
current dependency with a 2022 development build in order to silence an alert.

Both [`lightning`](https://pypi.org/project/lightning/) and
[`pytorch-lightning`](https://pypi.org/project/pytorch-lightning/) reported 2.6.5 as their latest
release. The advisory names the fix as a commit, not a release. There was nothing to upgrade to.

## Where Lightning enters this project

It is transitive and optional. Nothing in `pyproject.toml` declares it. The `foundation` extra
installs `terratorch`, which requires `lightning`, `torchgeo`, and `lightly`; `torchgeo` requires
`lightning` and `lightly` requires `pytorch-lightning`.

| Profile | Installed extras | Lightning present |
|---|---|---|
| Container image | `app` | no |
| CI test job | `dev app stac geo search pca bigearthnet` | no |
| CI browser job | `app browser` | no |
| Local TerraMind experiment | `foundation` | yes |

The served product surface therefore never installs it.

## Reachability

The vulnerable path, read in the installed `lightning` 2.6.5:

```text
LightningModule.load_from_checkpoint      lightning/pytorch/core/module.py:1796
LightningDataModule.load_from_checkpoint  lightning/pytorch/core/datamodule.py:245
  -> _load_from_checkpoint                lightning/pytorch/core/saving.py
     -> _load_state                       lightning/pytorch/core/saving.py:119
        -> __import__(module_path, ...)   lightning/pytorch/core/saving.py:157-161
```

`_load_state` pops `_instantiator` from the checkpoint's hyperparameters, imports that dotted path,
and then calls the imported object to build the model. Loading the file with `weights_only=True`
does not prevent this: the instantiator path is an ordinary string inside an already-deserialized
hyperparameter dictionary, and the import happens afterwards.

Executed checks on 2026-09-04:

| Check | Result |
|---|---|
| This repository's `src/`, `tests/`, `scripts/` | No reference to `lightning` or `load_from_checkpoint` |
| terratorch 1.2.11 | No file references `load_from_checkpoint` |
| torchgeo 0.8.1 | No file references `load_from_checkpoint` |
| lightly 1.5.22 | No file references `load_from_checkpoint` |
| The project's only checkpoint deserialization | `embeddings/terramind.py` verifies SHA-256 against a pinned digest, builds the backbone with `pretrained=False`, and calls `torch.load(..., weights_only=True)` directly |

The project never constructs a `LightningModule` or `LightningDataModule`, so the entry point to
the vulnerable function is never reached. The protection is the absence of that call, not the
`weights_only=True` used on our own path: the advisory describes a bypass of exactly that flag
inside Lightning's loader.

## Decision

1. Keep `lightning` 2.6.5. No patched release exists, so there is no upgrade to take.
2. Never install a date-numbered release to satisfy the advisory's version range.
3. Treat the alert as assessed rather than as noise. This record, not a silent dismissal, is what
   resolves it; anyone dismissing the alert should cite this ADR as the reason.
4. Keep Lightning out of the serving profile, so the dependency stays confined to the optional
   local experiment that needs TerraTorch.

## Re-check triggers

This assessment is void, and must be redone, if any of the following happens:

- A `lightning` or `pytorch-lightning` release after 2.6.5 contains commit `d710d68`. Upgrade and
  re-lock instead of relying on this record.
- Any code in this repository calls `load_from_checkpoint`, constructs a `LightningModule`, or
  loads a checkpoint that this project did not produce and verify.
- `terratorch`, `torchgeo`, or `lightly` is upgraded. Re-run the source checks above; a new
  version may introduce the call.
- The advisory is updated, withdrawn, or its version range is corrected.

## Boundary

This is source inspection of the versions installed on one machine on 2026-09-04, not an
exhaustive audit. It establishes that the specific function named by this advisory is not called
in this project's dependency set. It says nothing about other vulnerabilities in Lightning, and it
does not apply to any environment where a user runs the Lightning trainer or CLI. Machine-readable
evidence is in [`evidence/lightning-advisory-2026-09-04.json`](evidence/lightning-advisory-2026-09-04.json).
