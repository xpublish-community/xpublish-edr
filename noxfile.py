"""Test against the same matrix as Github Actions"""

import nox
import yaml

nox.options.default_venv_backend = "uv"

with open("./.github/workflows/tests.yml") as f:
    workflow = yaml.safe_load(f)

python_versions = workflow["jobs"]["run"]["strategy"]["matrix"]["python-version"]


@nox.session(python=python_versions)
def tests(session: nox.Session):
    """Run py.test against Github Actions matrix"""
    session.install("-r", "requirements-dev.txt")
    session.install(".")
    session.run("pytest", "--verbose")
