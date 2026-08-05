# Architecture Decision Records

One file per decision that was expensive to make and would be expensive to
reverse. Each records the options that were actually on the table, what was
chosen, and — the part most ADRs skip — what it costs us.

An ADR is never edited to reflect a change of mind. It is superseded by a new
one, and the old one keeps its `Status` updated to say so. The value is in the
trail, not in the current state; the current state is the code.

| # | Decision | Status |
| :-- | :-- | :-- |
| [0001](0001-litellm-as-provider-abstraction.md) | Adopt the LiteLLM SDK as the provider abstraction (embedded, not Proxy) | Accepted |
| [0002](0002-hexagonal-layout.md) | Hexagonal package layout with a mechanically enforced dependency rule | Accepted |

## Format

```markdown
# ADR-NNNN: Title in the imperative

- **Status:** Proposed | Accepted | Superseded by ADR-MMMM
- **Deciders:**
- **Related:**

## Context      what forced a decision, and what the options were
## Decision     what we chose, and the reasoning that is not obvious
## Consequences what we now have to live with — including the bad parts
## Compliance   how anyone can check the decision is still being honoured
```

That last section is the one that keeps these documents honest. A decision with
no way to verify it is being followed is a preference.
