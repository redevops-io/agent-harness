# Contributing

Issues and pull requests are welcome on GitHub: <https://github.com/redevops-io/agent-harness/issues>.

```bash
git clone https://github.com/redevops-io/agent-harness && cd agent-harness
pip install -e ".[dev]"
python -m pytest -q
```

- Install from git or a clone only. Do not `pip install agent-harness` from PyPI: that is an unrelated third-party project.
- Keep tests offline and deterministic (the LLM client is a stub; no network, no real credentials).
- Never commit credentials; CI runs a secret scan (`.github/workflows/secret-scan.yml`).
- Contributions are accepted under the repository's licence (`LICENSE`, AGPL-3.0 text).

Security reports: see [SECURITY.md](SECURITY.md).
