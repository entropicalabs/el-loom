"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from typing import Union
import uuid

from pydantic import ValidationInfo

dataclass_params = {
    "frozen": True,
    "config": {"extra": "forbid"},
}


def coordinate_length_error(
    list_of_coordinates: tuple[tuple, ...],
) -> tuple[tuple, ...]:
    """
    Check if the length of the lists in the list are consistent.
    """
    if len(list_of_coordinates) != 0:
        length = len(list_of_coordinates[0])
    for item in list_of_coordinates:
        if len(item) != length:
            raise ValueError("Length of coordinates must be consistent.")

    return list_of_coordinates


def distinct_error(value: Union[tuple, list]):
    """
    Check if the elements in a list or tuple are distinct. Throw an error otherwise.
    """
    if all(isinstance(i, Union[tuple, list]) for i in value):
        value = tuple(map(tuple, value))  # Ensure we are dealing with a tuple of tuples
    else:
        value = tuple(value)
    if len(value) != len(set(value)):
        raise ValueError(f"List of {type(value[0])} must have distinct elements.")

    return value


def empty_list_error(list_obj: list):
    """
    Check if the list is empty. Throw an error if that is the case.
    """
    if len(list_obj) == 0:
        raise ValueError("List cannot be empty.")

    return list_obj


def ensure_tuple(list_obj):
    """
    Adjusts lists to tuples to ensure immutability of the data structures.
    """
    try:
        len(list_obj)
        return list_obj
    except TypeError:
        return (list_obj,)


def larger_than_zero_error(value: int, arg_name: str):
    """
    Check if the value is larger than zero.
    """
    if value <= 0:
        raise ValueError(f"{arg_name} has to be larger than 0.")
    return value


def no_name_error(name: str) -> str:
    """
    Check if the name is not empty.
    """
    if len(name) == 0:
        raise ValueError("Names of Circuit objects need to have at least one letter.")
    return name.lower()


def nr_of_qubits_error(qubits: tuple[tuple[int, ...], ...], values: ValidationInfo):
    """
    Check if the number of qubits is consistent with the length of the pauli string.

    This `field_validator` is used for objects that include a `pauli` field.
    """

    pauli = retrieve_field("pauli", values)

    if len(qubits) != len(pauli):
        raise ValueError(
            f"Number of qubits: {len(qubits)} does not match the length of the"
            f" pauli string: {len(pauli)}."
        )

    return qubits


def pauli_error(pauli_str: str) -> str:
    """
    Check if the pauli is a valid pauli string.
    """
    valid_pauli_chars = ["X", "Y", "Z"]

    def error_message(input_pauli: str) -> ValueError:
        return ValueError(
            f"Invalid pauli: {input_pauli}. Must be one of {valid_pauli_chars}."
        )

    for p in pauli_str:
        if p not in valid_pauli_chars:
            raise error_message(p)
    return pauli_str


def retrieve_field(name: str, info: ValidationInfo):
    """
    Retrieve a field from the info object.
    """
    message = (
        "This is most probably due to a previous validation error. Check, that the "
        "other fields in the data class are set correctly."
    )

    try:
        result = info.data[name]
        if result is None:
            raise ValueError(f"Field {name} is None. " + message)
        return result
    except Exception as exc:
        raise ValueError(
            f"Field {name} not found in {info.config['title']} object. " + message
        ) from exc


def uuid_error(uuid_str: str) -> str:
    """
    Check if the uuid is a valid uuid.
    """
    try:
        uuid_obj = uuid.UUID(uuid_str)
    except ValueError as exc:
        raise ValueError(f"Invalid uuid: {uuid_str}. UUID must be version 4.") from exc

    if uuid_obj.version != 4:
        raise ValueError(f"Invalid uuid: {uuid_str}. UUID must be version 4.")

    return uuid_str
