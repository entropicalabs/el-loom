"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import numpy as np
from .base_moment import MomentInterface, Moment


class BigMoment(MomentInterface):  # pylint: disable=too-few-public-methods
    """A Moment that will contain more Moment(s)."""

    def __init__(self, input_moments: tuple[Moment]) -> None:
        self.internal_moments = input_moments

    def transform_tab(self, input_tableau: np.ndarray) -> np.ndarray:
        pass
