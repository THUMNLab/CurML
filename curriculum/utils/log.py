import logging



def get_logger(log_file, log_name=None):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    log_format = logging.Formatter(
        fmt='%(asctime)s\t%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    fh = logging.FileHandler(log_file)
    fh.setFormatter(log_format)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(log_format)
    logger.addHandler(ch)

    return logger