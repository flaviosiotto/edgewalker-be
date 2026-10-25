---
name: local-test-env-gotchas
description: Local test env quirks for studio-svc/studio-runner/jupyterhub (missing deps in runtime .venv, tracked .pyc in devops)
metadata:
  type: reference
---

- Runtime venv `/home/flavio/playground/edgewalker-runtime/.venv` has pandas/pyarrow/httpx but NOT `nbformat`, `docker`, `matplotlib`, `jupyterhub`. studio-runner tests (`runner.execute` -> publication -> nbformat) need `pip install --target <scratch>/pydeps nbformat` + that dir on PYTHONPATH; do not install into the shared venv.
- studio-svc tests: `PYTHONPATH=../../edgewalker ../.venv/bin/python -m pytest tests` from `studio-svc/` (sqlite files in /tmp, DATABASE_URL via setdefault: the first imported test module wins).
- edgewalker-devops TRACKS `jupyterhub/__pycache__/edgewalker_authenticator.cpython-312.pyc`: importing the authenticator locally (smoke tests with stubbed `jupyterhub.*` modules) rewrites it; restore with `git show HEAD:<path> > <path>` and delete any new `lab-helpers/__pycache__/`.
