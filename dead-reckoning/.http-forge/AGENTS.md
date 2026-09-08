# HTTP Forge — AI Agent Guide

This folder is already the **HTTP Forge workspace root**. Do not create a second nested `.http-forge/` directory inside it.

Every `*.md` file under `.http-forge/knowledge/` is loaded and included in AI
prompts, along with the workspace `README.md` and `AGENTS.md`. Prefer short,
dense notes — the knowledge is bounded to keep token cost predictable. There is
no need to paste Confluence/Jira content into collection files themselves.

> **GitHub Copilot tip:** To keep this guide in every Copilot conversation, add one line to
> `.github/copilot-instructions.md`: `See .http-forge/AGENTS.md for the HTTP Forge AI guide.`

---

## Decision Tree — What to Use and When

HTTP Forge gives AI agents three ways to interact with a workspace.
**Choose the lowest-cost option that can complete the task:**

```
Task
 │
 ├─ Discover structure / read or edit collections, requests, environments, suites?
 │    └─ ✅ Read / write the JSON files directly  (zero token cost, always available)
 │
 ├─ Execute a request, collection, folder, or suite?
 │    ├─ Is @http-forge/cli installed?  (check: http-forge --version)
 │    │    ├─ YES → ✅ CLI  (lower token cost — no tool schema preloaded)
 │    │    │            http-forge run collection <name> --env <env> --json
 │    │    │            http-forge run suite <name> --json
 │    │    │            http-forge run request <name> --collection <ref> --json
 │    │    └─ NO  → ✅ MCP  run_collection / run_suite / run_request
 │    │
 │    └─ Need async execution or real-time polling?
 │         └─ ✅ MCP  run_collection --async, then get_run_status / get_run_summary
 │
 └─ Diagnose failures, suggest assertions, explain errors?
      └─ ✅ MCP  explain_failure / suggest_assertions / analyze-test-failure prompt

 └─ Design a NEW API from a plain-English intent (endpoints + DTOs + auth)?
      ├─ Is @http-forge/cli installed?  (check: http-forge --version)
      │    └─ YES → ✅ CLI  http-forge architect "I need a shopping cart"
      └─ NO  → ✅ MCP  design_api_from_intent  (then review; apply:true to approve)
```

---

## Why This Ordering?

| Method | Token cost | Always available | Can execute | Best for |
|--------|:----------:|:----------------:|:-----------:|---------|
| Direct file access | **Zero** | ✅ (if file system access) | ❌ | Discover, read, create, edit |
| CLI `http-forge run` | **Low** (no tool schema) | ❌ (must be installed + shell access) | ✅ | Execution when CLI is present |
| MCP tools | **Medium** (schemas preloaded) | ✅ (when extension runs) | ✅ | Execution fallback; AI analysis |

**Key rule:** Never use MCP or CLI to discover structure — just read the JSON files.
`list_collections`, `list_requests`, `get_request` are redundant when you have file access.

---

## JSON Schemas

Every file contains a `$schema` field — your editor and AI can validate files automatically.

| File | Schema URL |
|------|-----------|
| `collection.json` | `https://raw.githubusercontent.com/hsl1230/http-forge/main/resources/collection.schema.json` |
| `folder.json`     | `https://raw.githubusercontent.com/hsl1230/http-forge/main/resources/folder.schema.json` |
| `request.json`    | `https://raw.githubusercontent.com/hsl1230/http-forge/main/resources/request.schema.json` |
| `{env}.json`      | `https://raw.githubusercontent.com/hsl1230/http-forge/main/resources/environment.schema.json` |
| `_global.json`    | `https://raw.githubusercontent.com/hsl1230/http-forge/main/resources/global-environment.schema.json` |
| `*.suite.json`    | `https://raw.githubusercontent.com/hsl1230/http-forge/main/resources/suite.schema.json` |
