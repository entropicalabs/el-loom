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

from dataclasses import asdict
from pydantic.dataclasses import dataclass

from loom.eka.utilities import dataclass_config

from ..stabilizer import Stabilizer
from ..pauli_operator import PauliOperator


@dataclass(config=dataclass_config)
class Operation:
    """
    Parent class for all operations in the EKA.
    """

    def asdict(self):
        """Serialize the operation to a dictionary."""
        # Use the dataclass's asdict method to convert the instance to a dictionary
        class_dict = asdict(self)
        # Add the class name to the dictionary for deserialization
        class_dict["__name__"] = self.__class__.__name__
        return class_dict

    @classmethod
    def fromdict(cls, data_dict: dict):
        """Deserialize the operation from a dictionary."""
        # The class name should be included in the dictionary
        # (not needed for the dataclass initialization, only for deserialization)
        if not (cls_name := data_dict.pop("__name__")):
            raise ValueError(
                "No Operation name found in the data, the Operation cannot be loaded."
            )
        # If the right class is given, we can just instantiate it
        if cls_name == cls.__name__:
            return cls(**data_dict)

        # For abstract classes, we need to look through the subclasses
        operation_class = next(
            (
                subsubclass
                for subclass in cls.__subclasses__()  # Base, Code or Logical (abstract)
                for subsubclass in subclass.__subclasses__()  # Subsequent subclasses
                if subsubclass.__name__ == cls_name
            ),
            None,
        )
        if operation_class is None:
            raise ValueError(
                f"Operation {cls_name} was not found in the Operation subclasses."
            )

        return operation_class(**data_dict)


@dataclass(config=dataclass_config)
class BaseOperation(Operation):
    """
    Base class for all operations in the EKA.

    NOTE: Base Operations are Operations that can be generalized to any setup of the
    EKA. They are not specific to a particular code or setup. This could range from
    single qubit gates on specific qubits to measuring specific stabilizers of the code.
    """


@dataclass(config=dataclass_config)
class MeasureStabilizerSyndrome(BaseOperation):
    """
    Measure the syndrome of a single stabilizer.

    Parameters
    ----------

    stabilizer : Stabilizer
        Stabilizer to measure.
    """

    stabilizer: Stabilizer


@dataclass(config=dataclass_config)
class MeasureObservable(BaseOperation):
    """
    Measure an observable.

    Parameters
    ----------

    observable : PauliOperator
        Observable to measure.
    """

    observable: PauliOperator
