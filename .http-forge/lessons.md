# Lessons Learned

## API Design with HTTP Forge Architect

Example: Design a todo list API from plain-English intent.

```bash
http-forge architect "a todo list" --apply --flow-out ./todo.flow.js --docs-out ./todo.md
```

This persists a collection, writes a test suite, and generates flow/docs/OpenAPI byproducts.