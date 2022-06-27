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
            created = create_func(kwargs)
        except Exception as e:
            step_failed = True
            raise e

        if write:
            try:
                write_func(created, kwargs)
            except Exception as e:
                step_failed = True
                raise e
    else:
        try:
            created = read_func(kwargs)
        except Exception as e:
            step_failed = True
            raise e

    if analyse:
        try:
            analyse_func(created, kwargs)
        except Exception as e:
            step_failed = True
            raise e

    if visualize:
        try:
            visualize_func(created, kwargs)
        except Exception as e:
            step_failed = True
            raise e

    if step_failed:
        raise PipelineException("Pipeline step failed")
    else:
        return created