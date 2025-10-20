"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from copy import deepcopy

from .operations.base_operation import Operation
from .operations.datamanipulation_operation import CreatePauliFrame, RecordPauliFrame
from .operations.resize_operation import AddQubit, DeleteQubit
from .classicalreg import ClassicalRegister
from .moment_queue import MomentQueue
from .tableau import Tableau
from .data_store import DataStore
from .invoker import Invoker
from .exceptions import EngineRunError


class Engine:  # pylint: disable=too-many-instance-attributes
    """
    The Engine Object contains all the components that make up the Stabilizer
    Simulator. These include the MomentQueue, the Tableau and the
    DataStore.

    This Object facilitates the interaction between all of these internal
    components with each other.
    """

    def __init__(
        self,
        input_operations: list[Operation],
        nqubits: int,
        seed: int = None,
        shots: int = 1,
        parallelize: bool = False,
    ):  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self.input_operations = input_operations
        self._validate_operations(input_operations, nqubits)
        self.moment_queue = MomentQueue(input_operations, parallelize)
        self.nqubits = nqubits
        self.seed = seed
        self.shots = shots

        # initialize clifford simulator
        self._setup_clifford_simulator()

    # pylint: disable=too-many-branches
    def _validate_operations(self, input_operations: list[Operation], nqubits: int):
        """Validate input operations."""
        created_pfs = []
        recorded_pfs = []
        for operation in input_operations:
            if not isinstance(operation, Operation):
                raise TypeError(
                    "Input operations must be of type Operation or a subclass of Operation."
                )
            if isinstance(operation, CreatePauliFrame):
                created_pfs.append(operation.pauli_frame)
            if isinstance(operation, RecordPauliFrame):
                recorded_pfs.append(operation.pauli_frame)

        # Check that all recorded PF have been created first
        for pauli_frame in recorded_pfs:
            if pauli_frame not in created_pfs:
                raise ValueError(
                    "RecordPauliFrame operations must be preceded by a CreatePauliFrame operation."
                )

        # Check that all created PF have a unique id
        compare_pfs = deepcopy(created_pfs)
        for pauli_frame in created_pfs:
            compare_pfs.pop(0)  # remove index 0 to avoid comparing with itself
            for other_pauli_frame in compare_pfs:
                if pauli_frame.id == other_pauli_frame.id:
                    raise ValueError(
                        "CreatePauliFrame operations must have a unique PauliFrame id."
                    )

        # Check that PauliFrames have the right size
        compare_nqubits = nqubits
        for input_operation in input_operations:
            if isinstance(input_operation, AddQubit):
                compare_nqubits += 1
            if isinstance(input_operation, DeleteQubit):
                compare_nqubits -= 1
            if isinstance(input_operation, CreatePauliFrame):
                if len(input_operation.pauli_frame.x) != compare_nqubits:
                    raise ValueError(
                        f"Wrong size for the PauliFrame {input_operation.pauli_frame.id}. "
                        f"It has size {len(input_operation.pauli_frame.x)}. It must have "
                        f"the same length as the number of qubits in the system "
                        f"({compare_nqubits}). Make sure that you take into "
                        f"account resize operations."
                    )

    def _setup_clifford_simulator(self):
        """Set up the clifford simulator."""
        # Setup tableau
        # Initialize using nqubits attribute
        self.tableau_w_scratch = Tableau(nqubits=self.nqubits, seed=self.seed)

        # Setup PauliFrame lists, default empty
        self.pauli_frames_forward = []
        self.pauli_frames_backward = []

        # Setup Classical Register, default empty.
        self.registry: dict[str, ClassicalRegister] = {}

        # Initialize DataStore and Invoker objects
        self.data_store = DataStore()
        self.invoker = Invoker(
            self.tableau_w_scratch,
            self.pauli_frames_forward,
            self.pauli_frames_backward,
            self.data_store,
            self.registry,
        )

    def run(self):
        """Run the clifford simulator."""
        # get one by one all the moments in both directions
        for forward_moment, backward_moment in self.moment_queue.moment_generators:
            # Forward propagation
            output_bool_tab = self.invoker.transform_tab(forward_moment)
            output_bool_pf = self.invoker.transform_pf(forward_moment)
            # Check for errors when applying the forward transformations
            if not output_bool_tab:
                raise EngineRunError(
                    """An exception has occured while trying to apply a
                    transformation to the Tableau."""
                )
            if not output_bool_pf:
                raise EngineRunError(
                    """An exception has occured while trying to apply a forward
                    transformation to the PauliFrame."""
                )
            # Backward propagation
            output_bool_pf_back = self.invoker.transform_pf_back(backward_moment)
            # Check for errors when applying the backward transformations
            if not output_bool_pf_back:
                raise EngineRunError(
                    """An exception has occured while trying to apply a backward
                    transformation to the PauliFrame."""
                )

        # reset the moment queue
        self.moment_queue.reset_queue()

    @property
    def stabilizer_set(self) -> set[str]:
        """
        Returns the stabilizer set in string format.
        """
        return self.tableau_w_scratch.stabilizer_set

    @property
    def stabilizer_set_sparse_format(self) -> list[dict]:
        """
        Returns the stabilizer set in sparse format.
        """
        return self.tableau_w_scratch.stabilizer_set_sparse_format
