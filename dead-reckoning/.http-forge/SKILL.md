# HTTP Forge — Skill Guide

This folder contains procedural guides and local rules that supplement the kernel reference in AGENTS.md.

## Authoring by Direct File Editing

### Create a new request
1. Create directory: `assets/collections/{collection-slug}/{optional-folder-slug}/{request-slug}/`
2. Write `request.json` (use the schema in AGENTS.md for all valid fields)
3. Add the slug to `order` in the parent `collection.json` or `folder.json`
4. Optionally add `pre-request.js` and/or `post-response.js` for scripts/assertions

### Edit a request
Read `request.json`, modify fields, write it back.
Variables: `{{variableName}}`. Filters: `{{value | upper}}`, `{{date | date:'YYYY-MM-DD'}}`.

### Bulk update (e.g. change baseUrl across all requests)
Find-and-replace across the `assets/collections/` tree — no MCP needed.

### Add / update environment variables
Edit `assets/environments/{env}.json` directly.

### Create or edit a test suite
Write or edit `assets/suites/{name}.suite.json`.
Supports: `request`, `block`, `if/elseif/else`, `for`, `while`, `switch`, `script` nodes.

## Execution: CLI Commands (when installed)

```bash
# Check if CLI is available
http-forge --version

# Run a collection (outputs JSON to stdout)
http-forge run collection <name> --env <env> --json

# Run a specific folder within a collection
http-forge run folder <folder-path> --collection <name> --json

# Run a single request
http-forge run request <request-name> --collection <name> --json

# Run a test suite
http-forge run suite <name> --json

# Design a new API from an intent (persists a collection; add --apply to
# persist the suite + write flow/docs/OpenAPI byproducts)
http-forge architect "I need a shopping cart"

# Pipe output to filter results (saves tokens — AI only sees what it needs)
http-forge run collection auth --json | jq '.failedRequests'
http-forge run suite checkout --json | jq '.summary'
```

Output always contains: `summary` (total/passed/failed) and `failedRequests` (when failures exist).
For collection/suite/folder runs, add `--include report` to generate an HTML report (`report.uri`) with full response details.

## Execution: MCP Tools (always available when extension runs)

Use MCP when:
- CLI is not installed or no shell access is available
- Async / long-running execution with real-time polling is needed
- AI analysis tools are needed (failure diagnosis, assertion suggestions)

```
run_request      → execute one request
run_folder       → execute a folder within a collection
run_collection   → execute an entire collection
run_suite        → execute a test suite
  └─ add async:true for background execution, then poll with get_run_status

get_run_summary  → summary + failed requests after a run completes
get_failed_requests → paginated failed request details
explain_failure  → AI-powered root cause analysis
suggest_assertions → generate pm.test() assertions for a request
```

> **Token tip:** Response bodies are truncated at 4 KB by default.
> Pass `include: ["fullBody"]` to get the complete body.
> For collection / suite / folder runs, pass `include: ["report"]` to generate an HTML report
> (`report.uri`) — open it in a browser to inspect full response details without consuming tokens.
> Single `run_request` calls do not generate a report unless `include: ["report"]` is also passed.