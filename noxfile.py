"""Test sessions using nox.

With pixi managing environments, nox sessions delegate to
``pixi run`` so that they use the same locked dependencies as CI.
"""

import nox

PYTHON_VERSIONS = ["3.11", "3.12", "3.13"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session):
    """Run py.test via the matching pixi environment."""
    env_map = {"3.11": "py311", "3.12": "py312", "3.13": "py313"}
    pixi_env = env_map[session.python]
    session.run("pixi", "run", "-e", pixi_env, "test", external=True)
