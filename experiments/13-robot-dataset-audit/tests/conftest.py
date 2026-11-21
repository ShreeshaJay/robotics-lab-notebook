import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pytest_addoption(parser):
    parser.addoption("--dataset", default="libero_10", help="profile name: libero_10, fineart, droid")
    parser.addoption("--root", default=None, help="dataset folder to judge (default: the profile's data folder)")
    parser.addoption("--spec", default=None, help="spec name under specs/ (default: dataset name)")


@pytest.fixture(scope="session")
def gate(request):
    from lerobot_audit.acceptance import run_gate
    ds, root, spec = (request.config.getoption(k) for k in ("--dataset", "--root", "--spec"))
    if root:
        ds = ds if ds.endswith("_corrupt") else ds + "_corrupt"
    _, checks = run_gate(ds, root, spec_name=spec, log=lambda *a: None)
    return checks
