import logging
import sys

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False


def get_logger(name: str = 'exam_merger') -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def progress(iterable, description: str = 'Processing', total: int = None):
    """
    Wrap an iterable with a tqdm progress bar.

    Handles iterables without __len__ (e.g. as_completed futures) by
    accepting an explicit `total` count, or leaving it as None so tqdm
    shows a spinner instead of a percentage bar.
    """
    if _TQDM_AVAILABLE:
        # Try to infer total from the iterable if not supplied
        if total is None:
            try:
                total = len(iterable)
            except TypeError:
                total = None  # spinner mode for generators/futures
        return tqdm(iterable, desc=description, total=total, leave=False,
                    unit='file', dynamic_ncols=True)
    # Fallback: plain passthrough
    return iterable
