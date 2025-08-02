import pytest
import os
from tqdm import tqdm

def pytest_addoption(parser):
    parser.addoption(
        "--crypto", action="store", default=None,
        help="Testmodus: a=alle, s=schnell, ss=sehr schnell"
    )
    parser.addoption(
        "--progress", action="store_true", default=False,
        help="Fortschrittsbalken für große Testbatches"
    )
    parser.addoption(
        "--pytest-verbose", action="store", default="v",
        help="Pytest Output: v=verbose, s=short, n=normal"
    )

def pytest_configure(config):
    crypto_opt = config.getoption("--crypto")
    if crypto_opt:
        if crypto_opt == "a":
            os.environ["CRYPTO_TESTMODE"] = "0"
            os.environ["CRYPTO_TESTFAST"] = "0"
        elif crypto_opt == "s":
            os.environ["CRYPTO_TESTMODE"] = "1"
            os.environ["CRYPTO_TESTFAST"] = "0"
        elif crypto_opt == "ss":
            os.environ["CRYPTO_TESTMODE"] = "1"
            os.environ["CRYPTO_TESTFAST"] = "1"
    # Verbosity/Output: Passe pytest addopts an
    output_level = config.getoption("--pytest-verbose")
    # Manipuliere die pytest-Optionen nach Wunsch (geht in der Regel via command line besser!)
    # Optional: hier noch weiter ausbauen

@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(session, config, items):
    # Fortschrittsbalken bei --progress aktivieren
    if config.getoption("--progress"):
        for item in tqdm(items, desc="Pytest Testcases", unit="test"):
            pass  # tqdm zeigt die Fortschrittsanzeige an

