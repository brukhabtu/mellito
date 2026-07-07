---
paths: experiments/variants/**
---
# Variant discipline

Applies when creating or editing scaffold variants.

- One mutation per variant: exactly one conceptual change from the parent, stated as a falsifiable hypothesis in manifest.yaml before any config edit.
- Every variant has a complete manifest.yaml: id, parent, hypothesis, date, status (proposed | evaluated | kept | rejected).
- Variant config lives in claude-config/ and is data, never active config. It is materialized into task workspaces at run time only; nothing here is ever copied into this repo's own .claude/.
- Variants are immutable once evaluated; a fix or follow-up is a new child variant.
- Lineage is git: one commit per variant, message `variant: vNNN <hypothesis summary>`.
