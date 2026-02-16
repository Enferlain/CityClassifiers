# Wrapper Removal Checklist

This checklist governs removal of root compatibility wrappers and legacy root modules.

## Current status
Root wrappers have been removed.

## Preconditions
All items below should be complete before removal:
1. Quality gate is stable on package paths:
   - `scripts/quality/run_quality.sh` passes consistently.
2. Usage migration is confirmed:
   - internal automation/jobs no longer call root wrappers directly, or
   - wrapper calls are intentionally retained and documented as public API.
3. Docs/config examples are package-first:
   - no new examples rely on root wrappers as primary path.
4. Release communication is prepared:
   - deprecation note in `CHANGELOG.md`
   - migration note in `README.md`

## Checkpoint Execution Log
### February 15, 2026 (baseline execution)
1. Quality gate stability: met (`scripts/quality/run_quality.sh` passing).
2. Docs/config package-first migration: met for training commands and root-script references.
3. Usage migration confirmation:
   - internal code/docs enforced by `scripts/quality/check_wrapper_references.py`.
   - repo owner confirmed no external dependency constraints for this repo.
4. Release communication: met (changelog + README migration notes present).

Result: wrappers removed; repo now uses package entrypoints only.

## Timeline outcome
1. **February 15, 2026**: Checklist created; root-surface enforcement active.
2. **February 15, 2026**: Preconditions satisfied and wrappers removed immediately.

## Removal sequence (completed)
1. Removed wrapper scripts and related compatibility-only tests/docs assumptions.
2. Updated allowlist in `scripts/quality/root_surface_allowlist.txt`.
3. Ran full quality gate and smoke tests.
4. Published migration notes with exact replacement commands.
