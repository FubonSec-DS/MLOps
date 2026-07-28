---
description: "Enterprise Python coding standards for MLOps project. Use when writing or reviewing Python code under src/."
applyTo: "src/**/*.py"
---

# Python Coding Standards

Applies to every `.py` file under `src/`.

## 1. Python Version & Annotations

- Python ≥ 3.12
- Use built-in generics: `list[str]`, `dict[str, int]` (no `List`, `Dict`)
- Use `X | Y`, `X | None` (no `Union`, `Optional`)
- All functions (public + private) must be fully typed

## 2. Naming Conventions

- Module: `snake_case`
- Class: `PascalCase`
- Function: `snake_case`
- Constant: `UPPER_SNAKE_CASE`
- Private: `_name`
- Type alias: `PascalCase`
- Avoid single-letter variables (except loop indices)

## 3. Docstrings (Google Style)

- Public modules, classes, and functions must have Google-style docstrings.
- Private helpers may use a single-line docstring.

## 4. Configuration — Config Files & Environment Variables

- **Never** hardcode endpoints, API keys, model names, file paths, or numeric thresholds.
- Load runtime config from module-level `config.yaml` files or environment variables (`.env`).

## 5. File Paths — `pathlib.Path`

- **Never** construct paths with hard-coded `"./relative/path"` strings or `os.path.join(os.getcwd(), ...)`.
- Use `pathlib.Path` throughout; do not use `os.path` string manipulation.
- Derive the project root dynamically: `Path(__file__).resolve().parents[n]`.

## 6. Database

- Use parameterized SQL (bind variables).
- **Never** build SQL with string formatting (f-string, `%`, `.format()`, or concatenation).
