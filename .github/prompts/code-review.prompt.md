---
description: "Production-readiness code review of src/ — checks correctness, reliability, maintainability, performance, type safety, testability, and architecture. Ignores formatting and style covered by Ruff or PyRight."
agent: agent
tools: [read, search]
---

Act as a senior software engineer performing a production-readiness review.

Review **all Python files under `src/`**.

## Focus On

1. Correctness
2. Reliability
3. Maintainability
4. Performance
5. Type Safety
6. Architecture

## Ignore

- Formatting
- Import ordering
- Naming preferences
- Minor style issues already covered by Ruff and PyRight

## Prioritize Findings In This Order

1. Bugs
2. Runtime failures
3. Data corruption risks
4. Maintainability concerns

## Output Format

Only report actionable findings. Avoid praise and unnecessary commentary.

```
## Code Review — src/

### Critical
**`src/xxx/yyy.py` L42**
- **Description**: <what the issue is>
- **Impact**: <what goes wrong at runtime or in production>
- **Recommended Fix**: <concrete code-level fix>

### High
...

### Medium
...

### Low
...

---
### Summary
| Severity | Count |
|----------|-------|
| Critical | N |
| High     | N |
| Medium   | N |
| Low      | N |
```
