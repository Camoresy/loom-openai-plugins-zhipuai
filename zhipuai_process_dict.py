from multiprocessing import Process
from typing import Dict
import logging

logger = logging.getLogger(__name__)
mp_manager = None
processes: Dict[str, Process] = {}


def stop():
    global processes
    for process in processes.values():
        logger.warning("Sending SIGKILL to %s", process)
        try:

            process.kill()
        except Exception as e:
            logger.info("Failed to kill process %s", process, exc_info=True)

    for process in processes.items():
        logger.info("Process status: %s", process)

    del processes
