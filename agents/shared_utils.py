"""Shared utilities for all bots."""
import json, time, logging, requests, sys, os
from pathlib import Path

# Project root = parent of agents/
PROJECT_ROOT = Path(__file__).parent.parent
SHARED = PROJECT_ROOT / "shared"
AGENTS = PROJECT_ROOT / "agents"
LOGS = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"
RESULTS = PROJECT_ROOT / "results"
CACHE = PROJECT_ROOT / "cache"
SALARY_CAP = 50000
ROSTER_SIZE = 6

DG_BASE = "https://feeds.datagolf.com"


def get_api_key():
    """Get DataGolf API key from config.py or config file."""
    # First try importing from existing config.py
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from config import DATAGOLF_API_KEY
        if DATAGOLF_API_KEY and DATAGOLF_API_KEY != "YOUR_DATAGOLF_API_KEY_HERE":
            return DATAGOLF_API_KEY
    except ImportError:
        pass
    # Fallback to config file
    key_file = CONFIG_DIR / "datagolf_api_key.txt"
    if key_file.exists():
        k = key_file.read_text().strip()
        if k and k != "YOUR_DATAGOLF_API_KEY_HERE":
            return k
    return None


def dg_api(endpoint, params=None, timeout=30):
    """Call DataGolf API endpoint."""
    key = get_api_key()
    if not key:
        raise ValueError("No DataGolf API key found in config.py or config/")
    if params is None:
        params = {}
    params['key'] = key
    params['file_format'] = 'json'
    r = requests.get(f"{DG_BASE}/{endpoint}", params=params, timeout=timeout)
    r.raise_for_status()
    time.sleep(0.5)
    return r.json()


def setup_logger(name, filename):
    """Create a logger that writes to both file and stdout."""
    LOGS.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # Prevent duplicate handlers on re-import
    if not logger.handlers:
        fh = logging.FileHandler(str(LOGS / filename))
        fh.setFormatter(logging.Formatter('%(asctime)s [%(name)s] %(message)s'))
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)
    return logger
