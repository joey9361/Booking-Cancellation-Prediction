import logging
import os
import sys
from datetime import datetime

try:
    from config.settings import PROJECT_ROOT
except ImportError:
    from src.config.settings import PROJECT_ROOT

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

LOG_FORMAT = "[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d - %(message)s"

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format=LOG_FORMAT,
    level=logging.INFO,
)

# Also print logs to the terminal (useful when running CLI commands).
_console = logging.StreamHandler(sys.stderr)
_console.setFormatter(logging.Formatter(LOG_FORMAT))
logging.getLogger().addHandler(_console)
