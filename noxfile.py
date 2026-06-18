#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["nox", "pyyaml"]
# ///
"""Test against the same matrix as Github Actions"""

import nox
import yaml

nox.needs_version = ">= 2025.10.14"
nox.options.default_venv_backend = "uv|virtualenv"

with open("./.github/workflows/tests.yml") as f:
    workflow = yaml.safe_load(f)

python_versions = workflow["jobs"]["run"]["strategy"]["matrix"]["python-version"]


@nox.session(python=python_versions, default=True)
def tests(session: nox.Session):
    """Run py.test against Github Actions matrix"""
    session.install("--group", "dev")
    session.install(".")
    session.run("pytest", "--verbose")

@nox.session
def wheel(session: nox.Session):
    """Build a wheel."""
    session.install("build", "check-manifest", "twine")
    session.run("python", "-m", "build", "--wheel", ".", "--outdir", "dist")
    session.run("check-manifest", "--verbose")
    session.run("twine", "check", "dist/*")


@nox.session
def pre_commit(session: nox.Session):
    """Run pre-commit with prek."""
    session.install("prek")
    session.run("prek", "run")

if __name__ == "__main__":
    nox.main()