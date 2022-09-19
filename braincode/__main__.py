import logging
import sys
from datetime import datetime

from braincode.interface import CLI

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    start = datetime.now()
    CLI().run_main()
    elapsed = datetime.now() - start
    logging.getLogger(__name__).info(f"Completed successfully in {elapsed}.")
