---
name: bisect-triton-release
description: Bisecting triton releases to figure out which commit causes regressions.
---

TokenSpeed depends on `tokenspeed-triton`, a vendor release of Triton, for
faster access to latest features.
If a new release causes a regression, we need to switch to upstream Triton and
bisect which commits caused the issue.

## Overall Steps

Always activate the local venv in the `tokenspeed/` directory so that `triton`
is installed into it.

1. Record the version of the currently installed `triton` (or `triton-rocm` for
   AMD) pip package, which is pulled in as a dependency of `torch`, so we can
   restore it later.
2. Patch `_triton.py` to switch `tokenspeed_triton` imports to direct `triton`
   imports.
3. Ask which test in TokenSpeed shows regression.
4. Ask which upstream Triton commit is the last known good, and which upstream
   Triton commit shows regression.
5. Build and install Triton at the two commits from the previous step to
   confirm that the regression comes from an upstream commit.
   If not, then ask where the downstream triton repo is to inspect downstream
   changes.
6. Once confirmed an upstream commit causes regression, start `git bisect` flow
   to identify the exact root cause.
7. Once identified, reinstall the recorded `triton` (or `triton-rocm` package)
   to the original version and revert local changes.

## Build and Install Triton

In the Triton codebase, do the following:

```shell
pip install -r python/requirements.txt
pip install .
```
