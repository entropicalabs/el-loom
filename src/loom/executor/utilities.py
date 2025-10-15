"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import re


def format_channel_label_to_tuple(channel_label: str) -> tuple[int, ...]:
    """
    Convert a channel label string to a tuple of integers.
    The label is expected to be in the format '(x_y_z)' or 'c_(x_y_z)_i', the return
    value would be (x, y, z) in both cases.

    Parameters
    ----------
    channel_label : str
        The channel label string.

    Returns
    -------
    tuple[int, ...]
        A tuple of integers representing the coordinates.
    """
    return tuple(
        int(coord) for coord in re.sub("[( )c_]", "", channel_label).split(",")
    )
