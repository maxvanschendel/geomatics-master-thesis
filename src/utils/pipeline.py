import logging
from typing import Callable


class PipelineException(Exception):
    pass

def process_step(create: bool, write: bool, visualize: bool, analyse: bool,
                 create_func: Callable, write_func: Callable, read_func: Callable, visualize_func: Callable, analyse_func: Callable,
                 kwargs):

    # Step in processing timeline, in each step either create some data and optionally write it to disk, or read the data from disk.
    # After loading or creating data the data can be visualized and analysed. These actions are applicable to every step in the timeline
    # processing pipeline.

    step_failed = False

    if create:
        try:
            logging.info(f"Executing {create_func}")
            created = create_func(kwargs)
        except Exception as e:
            step_failed = True
            raise e

        if write:
            try:
                logging.info(f"Executing {write_func}")
                write_func(created, kwargs)
            except Exception as e:
                step_failed = True
                raise e
    else:
        try:
            logging.info(f"Executing {read_func}")
            created = read_func(kwargs)
        except Exception as e:
            step_failed = True
            raise e

    if analyse:
        try:
            logging.info(f"Executing {analyse_func}")
            analyse_func(created, kwargs)
        except Exception as e:
            step_failed = True
            raise e

    if visualize:
        try:
            logging.info(f"Executing {created}")
            visualize_func(created, kwargs)
        except Exception as e:
            step_failed = True
            raise e

    if step_failed:
        raise PipelineException("Pipeline step failed")
    else:
        return created