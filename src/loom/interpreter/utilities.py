"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from typing import Literal

#: A Cbit denotes the location of a classical bit by a tuple of a string and an integer.
#: str: Name/id of classical register, int: measurement index inside this register
Cbit = tuple[str, int] | Literal[1, 0]
