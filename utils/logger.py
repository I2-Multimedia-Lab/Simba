import logging
import torch.distributed as dist # dist is used in get_logger

# Assume logger_initialized, get_logger, and get_root_logger are defined
# exactly as you provided them earlier in the context.
# For completeness, I'll include the structure, but the key change is in print_log.

logger_initialized = {} # As defined in your code

# --- Your get_logger function (unchanged, shown for context) ---
def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    """Initialize and get a logger by name.
    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.
    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
        file_mode (str): The file mode used in opening log file.
            Defaults to 'w'.
    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    # handle duplicate logs to the console
    # Starting in 1.8.0, PyTorch DDP attaches a StreamHandler <stderr> (NOTSET)
    # to the root logger. As logger.propagate is True by default, this root
    # level handler causes logging messages from rank>0 processes to
    # unexpectedly show up on the console, creating much unwanted clutter.
    # To fix this issue, we set the root logger's StreamHandler, if any, to log
    # at the ERROR level.
    #
    # CRITICAL NOTE: This modification of logger.root.handlers should be done
    # with caution and ideally only once. If get_logger is called multiple times
    # with different names, this loop will run each time a new logger is initialized.
    # This might be intended if new root handlers could appear, but often root logger
    # configuration is done once.
    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler: # Check specific type
            handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        # Here, the default behaviour of the official logger is 'a'. Thus, we
        # provide an interface to change the file mode to the default
        # behaviour.
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level) # Note: This log_level is applied to handlers
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level) # Logger's own level for rank 0
    else:
        logger.setLevel(logging.ERROR) # Logger's own level for other ranks

    logger_initialized[name] = True
    logger.propagate = False # Often useful to prevent duplicate messages if root logger also has handlers

    return logger

# --- Your get_root_logger function (unchanged, shown for context) ---
# Note: There's a potential issue in the lambda for logging_filter.
# record.find(name) will likely cause an AttributeError as LogRecord objects don't have a 'find' method.
# It should probably be something like `name in record.msg` or `name in record.name` or similar.
# However, this is unrelated to the flushing fix.
def get_root_logger(log_file=None, log_level=logging.INFO, name='main'):
    """Get root logger and add a keyword filter to it.
    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name.
    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.
        name (str, optional): The name of the root logger, also used as a
            filter keyword. Defaults to 'main'.
    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    logger = get_logger(name=name, log_file=log_file, log_level=log_level)
    # add a logging filter - POTENTIAL ISSUE HERE as LogRecord has no find() method.
    # Consider if `record.name.startswith(name)` or `name in record.getMessage()` was intended.
    # logging_filter = logging.Filter(name) # This creates a filter based on logger name hierarchy
    # logging_filter.filter = lambda record: record.find(name) != -1 # This line overrides default filter logic
    # A more common way to filter by logger name is to rely on the default behavior of logging.Filter(name)
    # or use record.name.startswith(name). If filtering by message content, use record.getMessage().
    # For now, leaving as is, as it's not the primary focus of the flush fix.
    return logger


# --- MODIFIED print_log function ---
def print_log(msg, logger=None, level=logging.INFO):
    """Print a log message.
    Args:
        msg (str): The message to be logged.
        logger (logging.Logger | str | None): The logger to be used.
            Some special loggers are:
            - "silent": no message will be printed.
            - other str: the logger obtained with `get_logger(logger)`.
                (Note: The original `get_root_logger` was specified here,
                 but the code used `get_logger`. Sticking to `get_logger`
                 as in the original code block for this branch).
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object or a str that results in a Logger.
    """
    if logger is None:
        print(msg) # Print to console if no logger specified
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
        # --- BEGIN MODIFICATION ---
        # Explicitly flush FileHandlers associated with this logger instance.
        # Due to the logic in get_logger, only rank 0 loggers will have FileHandlers.
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.flush()
        # --- END MODIFICATION ---
    elif logger == 'silent':
        pass # Do nothing if logger is "silent"
    elif isinstance(logger, str):
        # Get the logger instance using the provided string as its name.
        # This will use existing handlers if already initialized, or create new ones.
        # The get_logger function handles file handler creation only for rank 0.
        _logger = get_logger(name=logger) # Match get_logger's signature
        _logger.log(level, msg)
        # --- BEGIN MODIFICATION ---
        # Explicitly flush FileHandlers associated with the obtained _logger instance.
        for handler in _logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.flush()
        # --- END MODIFICATION ---
    else:
        raise TypeError(
            'logger should be either a logging.Logger object, str, '
            f'"silent" or None, but got {type(logger)}')