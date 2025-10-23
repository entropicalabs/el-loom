"""
Copyright 2024 Entropica Labs Pte Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

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
