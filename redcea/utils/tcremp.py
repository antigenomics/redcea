import logging

from tcremp.utils import (
    configure_logging as _configure_logging,
    generate_output_prefix,
    get_representations_df,
    load_analysis_repertoire,
    load_prototype_repertoire,
    log_memory_usage,
    prepare_output_path,
    resolve_input_file,
    subsample_repertoire,
)


def configure_logging(input_path, output_path, output_prefix):
    """Keep debug logs in file, but show only info+ in console."""
    _configure_logging(input_path, output_path, output_prefix)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(logging.INFO)

__all__ = [
    "configure_logging",
    "generate_output_prefix",
    "get_representations_df",
    "load_analysis_repertoire",
    "load_prototype_repertoire",
    "log_memory_usage",
    "prepare_output_path",
    "resolve_input_file",
    "subsample_repertoire",
]
