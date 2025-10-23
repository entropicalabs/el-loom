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

import json
from dataclasses import asdict


def findall(in_list, value):
    """
    Find all the indices of a value in a list.
    """
    return [i for i, x in enumerate(in_list) if x == value]


def apply_to_nested(nested_list, func, param=None):
    """
    Apply a function to all elements of a nested list.
    """
    if isinstance(nested_list, list):
        return [apply_to_nested(elem, func, param) for elem in nested_list]
    if param is not None:
        return func(nested_list, param)
    return func(nested_list)  # if neither of the above two conditions is satisfied


def dumps(data) -> str:
    """
    Return the pydantic dataclass as a JSON string. If the dataclass defines its own
    custom dumps() method, the custom method is used. Otherwise, the dataclass object is
    converted to a dictionary using the `asdict` method of the `dataclasses` module and
    then converted to a JSON string.

    Parameters
    ----------
    data : dataclass
        Pydantic dataclass to be saved as JSON

    Returns
    -------
    str
        JSON string representation of the Eka
    """

    # Check if the dataclass defines its own custom dumps() method and call it if it is
    # defined
    if hasattr(data, "asdict") and callable(data.asdict):
        return json.dumps(data.asdict())

    # Otherwise use default methods
    return json.dumps(asdict(data))


def dump(data, file: str):
    """
    Write a pydantic dataclass as JSON to a file.

    Parameters
    ----------
    data : dataclass
        Pydantic dataclass to be saved as JSON
    file : str
        Filename to write to
    """

    with open(file, "w", encoding="utf-8") as f:
        f.write(dumps(data))


def loads(cls, data_json: str):
    """
    Load the a pydantic dataclass from a JSON string. If the dataclass defines its own
    custom loads() method, the custom method is used, giving the loaded dictionary as
    input.

    Parameters
    ----------
    cls : dataclass
        The dataclass to be loaded from the JSON string
    data_json : str
        Dataclass saved as JSON string

    Returns
    -------
    An instance of cls containing the data from the JSON string
    """
    if hasattr(cls, "fromdict") and callable(cls.fromdict):
        return cls.fromdict(json.loads(data_json))

    return cls(**json.loads(data_json))


def load(cls, file: str):
    """
    Load a pydantic dataclass from a JSON file.

    Parameters
    ----------
    cls : dataclass
        The dataclass to be loaded from the JSON file
    file : str
        Filename to read from

    Returns
    -------
    An instance of cls containing the data from the JSON file
    """

    with open(file, "w", encoding="utf-8") as f:
        return cls(**json.loads(f.read()))
