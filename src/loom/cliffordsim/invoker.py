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

from typing import List, Dict
import traceback

from .tableau import Tableau
from .pauli_frame import PauliFrame
from .data_store import DataStore
from .moments.base_moment import Moment


class Invoker:
    """
    The main role of the Invoker is to handle the interactions between an input
    Moment, the Tableau, PauliFrames and DataStore.
    """

    def __init__(
        self,
        input_tableau: Tableau,
        pauli_frames_forward: List[PauliFrame],
        pauli_frames_backward: List[PauliFrame],
        data_store: DataStore,
        registry: Dict,
    ):  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self.input_tableau = input_tableau
        self.pauli_frames_forward = pauli_frames_forward
        self.pauli_frames_backward = pauli_frames_backward
        self.data_store = data_store
        self.registry = registry

    def transform_tab(self, input_moment: Moment) -> bool:
        """
        Transform the input_tableau according to the input moment.
        The data_store and registry are also passed to the moment's transform_tab
        method to record any potential output measurement and allow for classical control.
        """
        try:
            self.data_store.set_time_step(input_moment.time_step)
            input_moment.transform_tab(
                self.input_tableau, self.data_store, registry=self.registry
            )
            return True
        except Exception:  # pylint: disable=broad-exception-caught
            print(traceback.format_exc())
            return False

    def transform_pf(self, input_moment: Moment) -> bool:
        """
        Transform the pauli frames according to the input moment.
        The data_store is also passed to the moment's transform_pf method to record any
        potential output pauli frame state.
        """
        try:
            self.data_store.set_time_step(input_moment.time_step)
            # Apply transformation to all pauli frames
            input_moment.transform_pf(self.pauli_frames_forward, self.data_store)
            return True
        except Exception:  # pylint: disable=broad-exception-caught
            print(traceback.format_exc())
            return False

    def transform_pf_back(self, input_moment: Moment) -> bool:
        """
        Transform the pauli frames backwards according to the input moment.
        The data_store is also passed to the moment's transform_pf_back method to record any
        potential output pauli frame state.
        """
        try:
            self.data_store.set_time_step(input_moment.time_step)
            # Apply transformation to all pauli frames
            input_moment.transform_pf_back(self.pauli_frames_backward, self.data_store)
            return True
        except Exception:  # pylint: disable=broad-exception-caught
            print(traceback.format_exc())
            return False
