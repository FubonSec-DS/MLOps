## Getting Started

### Prerequisites

Make sure you have the following installed

|                                            | Version |
| ------------------------------------------ | ------- |
| Python                                     | 3.12 +  |
| uv                                         | 0.9 +   |
| Oracle Instant Client for Windows (64-bit) | 21.3 +  |

### Create Virtual Environment

All required dependencies are listed in the `pyproject.toml` file. To install them, create a virtual environment and run the following command:

```bash
uv sync
```

### Configuration and Environment Variables

- Configuration `config.json` file are located in the `config` folder. Customize the configuration file according to your needs.

  ```yaml
  # Oracle
  oracle:
      instant_client_path: "The Dir Path of Oracle Instant Client"

  # Data pipeline
  data_pipeline:
      test_1: 1000
      ...
  ```

- Environment variables are defined in the `.env` file. Customize the environment variables according to your needs. Make sure to name the variables with distinguishable prefixes to avoid conflicts .

  ```bash
  # Oracle
  ORACLE_USER=your_oracle_user
  ORACLE_PASSWORD=your_oracle_password
  ORACLE_DSN=your_oracle_dsn

  # Other environment variables
  ...
  ```

- After modifying the configuration file, updating `src/common/config.py` BaseModel class to include the new configuration parameters is necessary. This ensures that the application can correctly load and validate the updated configuration settings.

## Project Structure

```
Fubon_MLOps/
├── .github/                # Instructions and Prompts for GitHub Copilot
├── pyproject.toml         # Python project configuration file
├── README.md
├── configs/
│   ├── .env               # Environment variables (sensitive information, not to be committed)
│   └── config.yaml        # Configuration file for the project
├── instantclient-23.26/   # Oracle Instant Client directory (for Windows)
├── legacy/                # Old code or scripts
└── src/                   # Core source code of the project
    ├── common/            # Tools and utilities used across the project
    │   ├── config.py      # Validation and loading of configuration files
    │   ├── database.py    # Database connection and query execution
    │   └── logger.py      # Logging setup and configuration
    ├── data_pipeline/     # Design your modules or pipeline here ...
    ├── training/          # Design your modules or pipeline here ...
    └── sql/               # SQL sripts
```

## Before Commit

### Linting and Type Checking

#### 1. Ruff

Use following command to check for linting errors:

```powershell
uv run ruff check .  # check all files
uv run ruff check src\common\config.py  # check specific file
uv run ruff check . --fix  # check then fix all files

# output
All checks passed!

# output with errors
D101 Missing docstring in public class
 --> src\common\database.py:7:7
7 | class OracleDB:
  |       ^^^^^^^^

Found 2 errors (1 fixed, 1 remaining).
```

Use following command to check for formatting:

```powershell
uv run ruff format --check --diff .\src # check unformatted code in src folder
uv run ruff format .\src # format unformatted code in src folder

# output
--- src\common\config.py
+++ src\common\config.py
@@ -51,7 +51,8 @@

-    with YAML_CONFIG_PATH.open(encoding="utf-8") as file: cfg = yaml.safe_load(file)
+    with YAML_CONFIG_PATH.open(encoding="utf-8") as file:
+        cfg = yaml.safe_load(file)

1 file would be reformatted, 2 files already formatted
```

#### 2. Pyright

Use following command to check for type errors in the project:

```powershell
uv run pyright

# output
0 errors, 0 warnings, 0 informations

# output with errors
test.py
  test.py:14:14 - error: Argument of type "Literal['2']" cannot be assigned to parameter "int_2" of type "int" in function "add" ...
1 error, 0 warnings, 0 informations
```

### AI Code Review

#### 1. Code Review

Prompt `.github/prompts/code-review.prompt.md` aims to perform a production-readiness review of all Python files under `src/`. You can trigger the code review process using the slash command `/code-review`.

```markdown
## Code Review — src/

...

### Medium

database.py

- Description: `query()` doesn’t rollback on failure (only closes connection).
- Impact: If a query is part of a larger transactional workflow (or session state changes), failures may leave the session in an unexpected state.
- Recommended Fix: ...

...

### Summary

| Severity | Count |
| -------- | ----- |
| Critical | 1     |
| High     | 2     |
| Medium   | 2     |
| Low      | 1     |
```

#### 2. Security Review

Prompt `.github/prompts/security-review.prompt.md` aims to perform an additional security review of all Python files under `src/`. You can trigger the security review process using the slash command `/security-review`. You can also trigger the security review process with slash command `/security-review` after the code review process is completed.

These prompts are provided primarily for GitHub Copilot, but can be freely adapted into skills, agents, workflows, or custom instructions for other coding assistants such as Claude, Codex, Cursor, or similar tools.
