# Spend control for paid model calls

`agent_harness.spend` and `agent_harness.spend_proxy` enforce a dollar cap per provider and record exactly
what each role (executor, editor, judge, ...) cost. Nothing upstream (Harness², OpenCode, Codex) tracks dollars.

```bash
python -m agent_harness.spend_proxy --cap openai=20 --cap kimi=20 --cap xai=20 --ledger runs/spend.jsonl --port 8765
# point a client at  http://127.0.0.1:8765/<provider>/<label>/v1  with any dummy API key
python -m agent_harness.spend_proxy --cap openai=20 --ledger runs/spend.jsonl --report
```

- The proxy reads the real keys (`OPENAI_API_KEY`, `XAI_API_KEY`, `KIMI_API_KEY`) from its own environment and
  injects them; clients and their subprocesses never see them. It binds to 127.0.0.1 only.
- It meters `POST .../chat/completions` and `.../responses` (streaming too) and `GET .../models`. Every other
  endpoint is refused.
- Once a provider is within `--headroom` USD (default 1.00) of its cap, new calls get HTTP 402. The headroom
  absorbs requests already in flight, so a cap can be overshot by roughly that amount, not more.
- Fail closed: unpriced models, unknown providers and priority/fast service tiers are refused. A successful
  response without usage is charged a conservative estimate and flagged `unmetered`.
- Prices live in `agent_harness/data/prices.json` (read from the providers' pages on 2026-10-19; standard
  tier). Costs are multiplied by 1.10. Cache-write fees and provider rounding are not modelled, so treat
  ledger totals as estimates, and check them against the provider dashboards.
- Request bodies, headers and keys are never logged.
