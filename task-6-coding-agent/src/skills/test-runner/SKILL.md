---
name: test-runner
description: Run pytest, diagnose a failing test, and use a passing test suite as completion evidence.
---
Run `python -m pytest -q` in the target repository. Treat a nonzero return code as an observation to diagnose; stop only after it passes or the task cannot proceed.