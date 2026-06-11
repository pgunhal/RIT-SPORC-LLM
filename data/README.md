# Data

Place task inputs here as JSONL.

Expected row shape:

```json
{
  "id": "row-001",
  "dataset": "custom",
  "document": "const email = \"person@example.com\";",
  "instruction": "Focus only on email addresses.",
  "question": "Does this document contain PII? Cite the supporting evidence."
}
```

Minimum fields:
- `id`
- `document`
- `question`

Optional fields:
- `dataset`
- `instruction`
- any metadata you want copied through to outputs
