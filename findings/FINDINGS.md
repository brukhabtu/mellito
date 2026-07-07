# Findings log

Append-only. This file is the project's memory: paste it (or its tail) into
fresh Claude sessions to restore context. Every entry cites run IDs.

## Entry schema

```
## 2026-07-XX · <phase> · <entry-type: run | mutation | admission | incident | decision>
- variant: vNNN (parent vNNN) — hypothesis: "..."
- run: <run_id> · tasks: N dev · trials: T
- result: pass 62.5% ±4.1 (paired vs parent: +6 tasks / -1 / 31 tie) · $/solved: 0.41 · s/task: 312
- by provenance: public 68% · own-repo 55% · (divergence note if any)
- verdict: kept | rejected | inconclusive — reason
- notes: anything surprising, one line each
```

Decision-rule states (from PLAN.md) to evaluate at each cycle end:
- [ ] Minimum detectable difference respected (no keeps below +5 paired)
- [ ] Dev/holdout gap check (G4 only)
- [ ] Kill criterion: evaluated? met?

---

(no entries yet)
