---
description: "Security review of src/ — checks for secrets, SQL injection, path traversal, unsafe subprocess, credential exposure, and unsafe deserialization in Python MLOps code"
agent: agent
tools: [read, search]
---

Act as an application security engineer.

Review **all Python files under `src/`** for realistic security risks in a Python-based data engineering and MLOps system.

Do **not** report style, formatting, or type annotation issues.
Do **not** report theoretical risks — only realistic attack scenarios given the codebase context.

---

## Checklist

### 1. Secrets Management

- Hardcoded passwords, API keys, access tokens, database credentials in source code or config files checked into version control

### 2. Database Security

- SQL injection via dynamic SQL generation (f-string, `%`, `.format()`, concatenation)
- Missing parameter binding (bind variables not used)

### 3. File System Security

- Path traversal (user-controlled input in file paths without sanitization)
- Unsafe file access or overwriting sensitive files
- `open()` calls with user-supplied paths

### 4. Command Execution

- Unsafe `subprocess` usage with `shell=True`
- Shell injection via untrusted command arguments
- `os.system()` calls

### 5. Data Protection

- Sensitive information (passwords, tokens, PII) written to logs or exception messages
- Credentials exposed in tracebacks or error outputs

### 6. Configuration Security

- Missing environment variable validation (e.g., empty string accepted as valid credential)
- Insecure defaults that could be used in production

### 7. Dependency & Deserialization Risks

- Unsafe use of `pickle.load()` / `pickle.loads()` on untrusted data
- `yaml.load()` without `Loader=yaml.SafeLoader`
- Untrusted or unverified package usage

---

## Output Format

Report each finding using the following structure. If a category has no issues, write `✅ No issues found.`

```markdown
## Security Review — src/

### 1. Secrets Management

**[SEVERITY: Critical/High/Medium/Low]**

- **File**: `src/xxx/yyy.py` L42
- **Issue**: <one-line description>
- **Attack Scenario**: <realistic scenario>
- **Impact**: <what an attacker gains>
- **Recommended Fix**: <concrete code-level fix>

✅ No issues found. (if clean)

...（repeat for each category）

---

### Summary

| Severity | Count |
| -------- | ----- |
| Critical | N     |
| High     | N     |
| Medium   | N     |
| Low      | N     |

**Priority fixes**: <list the most urgent items>
```
