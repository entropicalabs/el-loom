"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

# pylint: disable=duplicate-code, too-many-lines
from __future__ import annotations
from uuid import uuid4
from functools import cached_property
from itertools import combinations
from pydantic.dataclasses import dataclass

from loom.eka import (
    Block,
    Lattice,
    LatticeType,
    PauliOperator,
    Stabilizer,
    SyndromeCircuit,
    Circuit,
    Channel,
)
from loom.eka.utilities import (
    Direction,
    Orientation,
    DiagonalDirection,
    dataclass_params,
)

from loom_rotated_surface_code.utilities import FourBodySchedule


# pylint: disable=duplicate-code
@dataclass(**dataclass_params)
class RotatedSurfaceCode(Block):  # pylint: disable=too-many-public-methods
    """
    A sub-class of ``Block`` that represents a rotated surface code block.
    Contains methods to create a rotated surface code block along with
    properties to access the block's size, upper left qubit, stabilizers,
    logical operators, and other relevant information.
    """

    @classmethod
    def create(
        # pylint: disable=too-many-branches,too-many-statements, too-many-locals
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        cls,
        dx: int,
        dz: int,
        lattice: Lattice,
        unique_label: str | None = None,
        position: tuple[int, ...] = (0, 0),
        x_boundary: Orientation = Orientation.HORIZONTAL,
        weight_2_stab_is_first_row: bool = True,
        weight_4_x_schedule: FourBodySchedule | None = None,
        logical_x_operator: PauliOperator | None = None,
        logical_z_operator: PauliOperator | None = None,
        skip_validation: bool = False,
    ) -> RotatedSurfaceCode:
        """
        Create a ``Block`` object for a rotated surface code block. The orientation of
        the block (i.e. where which boundaries are) and where the weight-2 stabilizers
        are can be controlled with the ``x_boundary`` and ``weight_2_stab_is_first_row``
        argument. By default, the top row and the left column are chosen as the logical
        operators. Their pauli string and whether they belong to the logical Z or X
        operator depends on the orientation of the boundaries.

        # TODO: Add parameters for the logical operators if a user wants logical
        operators which are different from the first row and column

        The coordinates used for data qubits are the following (here as an example for a
        d=3 rotated surface code, with ``x_boundary=Orientation.H`` and
        ``weight_2_stab_is_first_row=True``):

        .. code-block::

                               Z
              (0,0) --- (1,0) --- (2,0)
                |         |         |
            X   |    Z    |    X    |
                |         |         |
              (0,1) --- (1,1) --- (2,1)
                |         |         |
                |    X    |    Z    |  X
                |         |         |
              (0,2) --- (1,2) --- (2,2)
                     Z

        Parameters
        ----------
        dx : int
            Size of the block in the horizontal direction
        dz : int
            Size of the block in the vertical direction
        lattice : Lattice
            Lattice on which the block is defined. The qubit indices depend on the type
            of lattice.
        unique_label : str, optional
            Label for the block. It must be unique among all blocks in the initial Eka.
            If no label is provided, a unique label is generated automatically using the
            uuid module.
        position : tuple[int, ...], optional
            Position of the top left corner of the block on the lattice, by default
            (0, 0)
        x_boundary : Orientation, optional
            Specifies whether the X boundaries are horizontal (Orientation.HORIZONTAL),
            i.e. going from left to right, or vertical (Orientation.V), i.e. going from
            top to bottom.
            The X boundary is the boundary that exhibits X Pauli charges. In other words
            it is the boundary with 2-body Z stabilizers. By default
            Orientation.HORIZONTAL.
        weight_2_stab_is_first_row : bool, optional
            Specifies whether the top most weight-2 stabilizer at the left boundary is
            in the first row (if True) or in the second row (if False), by default True
        weight_4_x_schedule : FourBodySchedule, optional
            Schedule for measuring the XXXX stabilizer, by default None. If None is
            provided, the schedule is calculated from the orientation of the X
            boundary. E.g. if ``x_boundary`` is Orientation.HORIZONTAL, the schedule is
            set to FourBodySchedule.N.
            The default scheme is described in https://arxiv.org/abs/1404.3747 III, B.
            In the example above, ``weight_4_x_schedule`` is ``FourBodySchedule.N``,
            this is equivalent to measure the upper right XXXX stabilizer in the order
            (2,0) -> (2,1) -> (1,0) -> (1,1).
        logical_x_operator: PauliOperator | None, optional
            Logical X operator. If None is provided, by default the top row or the left
            column is chosen (depending on the orientation of the block as specified by
            the ``x_boundary`` and ``weight_2_stab_is_first_row`` parameter)
        logical_z_operator: PauliOperator | None, optional
            Logical Z operator. If None is provided, by default the top row or the left
            column is chosen (depending on the orientation of the block as specified by
            the ``x_boundary`` and ``weight_2_stab_is_first_row`` parameter)
        skip_validation : bool, optional
            Skip validation of the block object, by default False.

        Returns
        -------
        Block
            Block object for a rotated surface code block
        """

        # Input validation
        if lattice.lattice_type != LatticeType.SQUARE_2D:
            raise ValueError(
                "The creation of rotated surface code blocks is "
                "currently only supported for 2D square lattices. Instead "
                f"the lattice is of type {lattice.lattice_type}."
            )

        if not isinstance(position, tuple) or any(
            not isinstance(x, int) for x in position
        ):
            raise ValueError(
                f"`position` must be a tuple of integers. Got '{position}' instead."
            )

        # Check that lattice co-ordinate of the qubit == system dimension
        # The unit vector is not included in the lattice co-ordinate.
        if len(position) != lattice.n_dimensions:
            raise ValueError(
                f"`position` has length {len(position)} while length "
                f"{lattice.n_dimensions} is required to match the lattice dimension."
            )

        if unique_label is None:
            unique_label = str(uuid4())

        if isinstance(x_boundary, Orientation) is False:
            x_boundary = Orientation(x_boundary)

        # Create stabilizers
        top_left_is_xxxx = (
            x_boundary == Orientation.HORIZONTAL
        ) != weight_2_stab_is_first_row
        # Create the schedule for stabilizers
        if weight_4_x_schedule is None:
            weight_4_x_schedule = (
                FourBodySchedule.N
                if x_boundary == Orientation.HORIZONTAL
                else FourBodySchedule.Z
            )
        elif isinstance(weight_4_x_schedule, str):
            weight_4_x_schedule = FourBodySchedule(weight_4_x_schedule)
        weight_4_z_schedule = weight_4_x_schedule.opposite_schedule()

        # Generate weight-4 stabilizers covering half of the block in a checkerboard
        # pattern starting in the top left corner
        if top_left_is_xxxx:
            weight4_stabs_top_left = cls.generate_weight4_stabs(
                dx, dz, "XXXX", weight_4_x_schedule, True
            )
        else:
            weight4_stabs_top_left = cls.generate_weight4_stabs(
                dx, dz, "ZZZZ", weight_4_z_schedule, True
            )
        # Generate the remaining weight-4 stabilizers
        if top_left_is_xxxx:
            weight4_stabs_others = cls.generate_weight4_stabs(
                dx, dz, "ZZZZ", weight_4_z_schedule, False
            )
        else:
            weight4_stabs_others = cls.generate_weight4_stabs(
                dx, dz, "XXXX", weight_4_x_schedule, False
            )

        stab_left_right_is_x = top_left_is_xxxx != weight_2_stab_is_first_row
        stab_left_right = "XX" if stab_left_right_is_x else "ZZ"
        stab_top_bottom = "ZZ" if stab_left_right_is_x else "XX"

        # Left boundary
        if dz % 2 == 1:
            num_weight2_stabs = (dz - 1) / 2
        else:
            num_weight2_stabs = dz / 2 - (not weight_2_stab_is_first_row)
        stabs_left = cls.generate_weight2_stabs(
            pauli=stab_left_right,
            initial_position=(0, (not weight_2_stab_is_first_row)),
            num_stabs=num_weight2_stabs,
            orientation=Orientation.VERTICAL,
            is_bottom_or_right=False,
        )

        # Right boundary
        right_first_row = weight_2_stab_is_first_row != (dx % 2)
        if dz % 2 == 1:
            num_weight2_stabs = (dz - 1) / 2
        else:
            num_weight2_stabs = dz / 2 - (not right_first_row)
        stabs_right = cls.generate_weight2_stabs(
            pauli=stab_left_right,
            initial_position=(dx - 1, (not right_first_row)),
            num_stabs=num_weight2_stabs,
            orientation=Orientation.VERTICAL,
            is_bottom_or_right=True,
        )

        # Top boundary
        num_weight2_stabs = dx // 2
        if dx % 2 == 1:
            num_weight2_stabs = (dx - 1) / 2
        else:
            num_weight2_stabs = dx / 2 - weight_2_stab_is_first_row
        stabs_top = cls.generate_weight2_stabs(
            pauli=stab_top_bottom,
            initial_position=(weight_2_stab_is_first_row, 0),
            num_stabs=num_weight2_stabs,
            orientation=Orientation.HORIZONTAL,
            is_bottom_or_right=False,
        )

        # Bottom boundary
        bottom_first_col = weight_2_stab_is_first_row == (dz % 2)
        if dx % 2 == 1:
            num_weight2_stabs = (dx - 1) / 2
        else:
            num_weight2_stabs = dx / 2 - (not bottom_first_col)
        stabs_bottom = cls.generate_weight2_stabs(
            pauli=stab_top_bottom,
            initial_position=((not bottom_first_col), dz - 1),
            num_stabs=num_weight2_stabs,
            orientation=Orientation.HORIZONTAL,
            is_bottom_or_right=True,
        )

        # Combine all stabilizers
        stabilizers = (
            weight4_stabs_top_left
            + weight4_stabs_others
            + stabs_left
            + stabs_right
            + stabs_top
            + stabs_bottom
        )

        # Define syndrome circuits
        xxxx_syndrome_circuit = cls.generate_syndrome_circuit(
            pauli="XXXX", padding=None, name="XXXX"
        )
        zzzz_syndrome_circuit = cls.generate_syndrome_circuit(
            pauli="ZZZZ", padding=None, name="ZZZZ"
        )
        # Get left padding and generate left syndrome circuit
        left_padding = cls.find_padding(
            boundary="left",
            schedule=(
                weight_4_x_schedule if stab_left_right_is_x else weight_4_z_schedule
            ),
        )
        left_syndrome_circuit = cls.generate_syndrome_circuit(
            pauli=stab_left_right,
            padding=left_padding,
            name=f"left-{stab_left_right}",
        )
        # Get right padding and generate right syndrome circuit
        right_padding = cls.find_padding(
            boundary="right",
            schedule=(
                weight_4_x_schedule if stab_left_right_is_x else weight_4_z_schedule
            ),
        )
        right_syndrome_circuit = cls.generate_syndrome_circuit(
            pauli=stab_left_right,
            padding=right_padding,
            name=f"right-{stab_left_right}",
        )
        # Get top padding and generate top syndrome circuit
        top_padding = cls.find_padding(
            boundary="top",
            schedule=(
                weight_4_z_schedule if stab_left_right_is_x else weight_4_x_schedule
            ),
        )
        top_syndrome_circuit = cls.generate_syndrome_circuit(
            pauli=stab_top_bottom,
            padding=top_padding,
            name=f"top-{stab_top_bottom}",
        )
        # Get bottom padding and generate bottom syndrome circuit
        bottom_padding = cls.find_padding(
            boundary="bottom",
            schedule=(
                weight_4_z_schedule if stab_left_right_is_x else weight_4_x_schedule
            ),
        )
        bottom_syndrome_circuit = cls.generate_syndrome_circuit(
            pauli=stab_top_bottom,
            padding=bottom_padding,
            name=f"bottom-{stab_top_bottom}",
        )
        # Construct the list of all syndrome circuits
        syndrome_circuits = [
            xxxx_syndrome_circuit,
            zzzz_syndrome_circuit,
            top_syndrome_circuit,
            bottom_syndrome_circuit,
            left_syndrome_circuit,
            right_syndrome_circuit,
        ]

        # Create stabilizer_to_circuit mapping
        stabilizer_to_circuit = (
            {
                stab.uuid: (
                    xxxx_syndrome_circuit.uuid
                    if top_left_is_xxxx
                    else zzzz_syndrome_circuit.uuid
                )
                for stab in weight4_stabs_top_left
            }
            | {
                stab.uuid: (
                    zzzz_syndrome_circuit.uuid
                    if top_left_is_xxxx
                    else xxxx_syndrome_circuit.uuid
                )
                for stab in weight4_stabs_others
            }
            | {stab.uuid: left_syndrome_circuit.uuid for stab in stabs_left}
            | {stab.uuid: right_syndrome_circuit.uuid for stab in stabs_right}
            | {stab.uuid: top_syndrome_circuit.uuid for stab in stabs_top}
            | {stab.uuid: bottom_syndrome_circuit.uuid for stab in stabs_bottom}
        )

        # Create logical operators
        if logical_x_operator is None:
            qubits = (
                [(dx_i, 0, 0) for dx_i in range(dx)]
                if x_boundary == Orientation.HORIZONTAL
                else [(0, dz_i, 0) for dz_i in range(dz)]
            )
            logical_x_operator = PauliOperator(
                pauli="X" * len(qubits), data_qubits=qubits
            )

        if logical_z_operator is None:
            qubits = (
                [(0, dz_i, 0) for dz_i in range(dz)]
                if x_boundary == Orientation.HORIZONTAL
                else [(dx_i, 0, 0) for dx_i in range(dx)]
            )
            logical_z_operator = PauliOperator(
                pauli="Z" * len(qubits), data_qubits=qubits
            )

        block = cls(
            unique_label=unique_label,
            stabilizers=stabilizers,
            logical_x_operators=[logical_x_operator],
            logical_z_operators=[logical_z_operator],
            syndrome_circuits=syndrome_circuits,
            stabilizer_to_circuit=stabilizer_to_circuit,
            skip_validation=skip_validation,
        )
        if position == (0, 0):
            return block

        return block.shift(position)

    @staticmethod
    def find_padding(
        boundary: Direction, schedule: FourBodySchedule
    ) -> tuple[int, int]:
        """
        Finds the padding indices for the two body stabilizers. Padding indices
        are used to indicate empty time steps in the syndrome circuits. This allows
        to standardize the size of syndrome circuits for the rotated surface code.

        Parameters
        ----------
        boundary : Direction
            Type of boundary for the surface code can be LEFT, RIGHT, TOP, or
                BOTTOM
        schedule : FourBodySchedule
            Schedule for measuring the four body stabilizers, we can deduce the two
                body schedule from this.

        Returns
        -------
        tuple[int, int]
            Padding indices.
        """
        match (schedule, boundary):
            case (FourBodySchedule.N, Direction.LEFT) | (
                FourBodySchedule.Z,
                Direction.BOTTOM,
            ):
                return (2, 3)
            case (FourBodySchedule.N, Direction.RIGHT) | (
                FourBodySchedule.Z,
                Direction.TOP,
            ):
                return (0, 1)
            case (FourBodySchedule.N, Direction.TOP) | (
                FourBodySchedule.Z,
                Direction.RIGHT,
            ):
                return (0, 2)
            case (FourBodySchedule.N, Direction.BOTTOM) | (
                FourBodySchedule.Z,
                Direction.LEFT,
            ):
                return (1, 3)
            case _:
                return ValueError(
                    f"The boundary {boundary} and schedule {schedule} are"
                    f" not compatible. Only N and Z schedules are "
                    "supported."
                )

    @staticmethod
    def generate_syndrome_circuit(
        pauli: str, padding: tuple[int, ...] | None, name: str
    ) -> SyndromeCircuit:
        """
        Generates a syndrome circuit for a given Pauli string. The syndrome
        circuits generated are all the same size, where padding indices indicate
        where to add empty time steps.

        Parameters
        ----------
        pauli : str
            Pauli string associated the stabilizer
        padding : tuple[int, int] | None
            Padding indices for the two body stabilizers. Padding indices are used
            to locate empty spaces in the syndrome circuits.
        name : str
            Name of the syndrome circuit

        Returns
        -------
        SyndromeCircuit
            Syndrome circuit for the Pauli string
        """
        weight = len(pauli)
        data_channels = [Channel(type="quantum", label=f"d{i}") for i in range(weight)]
        cbit_channel = Channel(type="classical", label="c0")
        ancilla_channel = Channel(type="quantum", label="a0")

        reset = [Circuit("Reset_0", channels=[ancilla_channel])]
        hadamard1 = [Circuit("H", channels=[ancilla_channel])]
        hadamard2 = [Circuit("H", channels=[ancilla_channel])]
        # If there is no padding, there are no extra empty time steps
        if padding is None:
            entangle_ancilla = [
                [Circuit(f"C{p}", channels=[ancilla_channel, data_channels[i]])]
                for i, p in enumerate(pauli)
            ]
        # If there is padding, we add empty time steps to the circuit
        else:
            # We create lists of single items to insert None at the padding indices
            padded_pauli = list(pauli)
            padded_data_channels = list(data_channels)
            for i in padding:
                padded_pauli.insert(i, None)
                padded_data_channels.insert(i, None)

            entangle_ancilla = [
                (
                    [
                        Circuit(
                            f"C{p}",
                            channels=[ancilla_channel, padded_data_channels[i]],
                        )
                    ]
                    if p is not None  # If not None, we add the empty time step
                    else []
                )
                for i, p in enumerate(padded_pauli)
            ]
        measurement = [Circuit("Measurement", channels=[ancilla_channel, cbit_channel])]

        circuit_list = [reset, hadamard1] + entangle_ancilla + [hadamard2, measurement]
        return SyndromeCircuit(
            pauli=pauli,
            name=name,
            circuit=Circuit(
                name=name,
                circuit=circuit_list,
                channels=data_channels + [ancilla_channel, cbit_channel],
            ),
        )

    # Properties

    @property
    def size(self) -> tuple[int, int]:
        """
        Return the size of the block in the horizontal and vertical direction.
        """
        size_0 = max(qb[0] for qb in self.data_qubits) - min(
            qb[0] for qb in self.data_qubits
        )
        size_1 = max(qb[1] for qb in self.data_qubits) - min(
            qb[1] for qb in self.data_qubits
        )
        return (size_0 + 1, size_1 + 1)

    @property
    def upper_left_qubit(self) -> tuple[int, ...]:
        """
        Return the qubit with the smallest coordinates in the block.
        """
        return min(self.data_qubits, key=lambda x: x[0] + x[1])

    @property
    def upper_left_4body_stabilizer(self) -> Stabilizer:
        """
        Return the 4-body stabilizer associated with the upper left qubit.
        """
        return [
            stab
            for stab in self.stabilizers
            if len(stab.data_qubits) == 4 and self.upper_left_qubit in stab.data_qubits
        ][0]

    # NOTE these four properties should be deleted and included as fields at the
    # creation of the RotatedSurfaceCode instance
    @property
    def weight_4_x_schedule(self) -> FourBodySchedule:
        """
        Return the schedule for measuring the XXXX stabilizer.
        """
        for stab in self.stabilizers:
            if stab.pauli == "XXXX":
                schedule = (
                    FourBodySchedule.N
                    if stab.data_qubits[1][1] > stab.data_qubits[0][1]
                    else FourBodySchedule.Z
                )
                return schedule
        raise ValueError("XXXX stabilizer not found in the stabilizers.")

    @property
    def weight_4_z_schedule(self) -> FourBodySchedule:
        """
        Return the schedule for measuring the ZZZZ stabilizer.
        """
        return (
            FourBodySchedule.N
            if self.weight_4_x_schedule == FourBodySchedule.Z
            else FourBodySchedule.Z
        )

    @property
    def x_boundary(self) -> Orientation:
        """
        Return the orientation of the X boundary.
        """
        if self.boundary_type(Direction.TOP) == "X":
            return Orientation.HORIZONTAL

        # if the top boundary is Z, then the X boundary is vertical
        return Orientation.VERTICAL

    @property
    def weight_2_stab_is_first_row(self) -> bool:
        """
        Return whether the top most weight-2 stabilizer at the left boundary is
        in the first row.
        """
        first_qubit = self.upper_left_qubit
        second_qubit = tuple(
            q if i != 1 else q + 1 for i, q in enumerate(self.upper_left_qubit)
        )
        return any(
            # first_qubit in stab.data_qubits and second_qubit in stab.data_qubits
            # and len(stab.data_qubits) == 2
            set([first_qubit, second_qubit]) == set(stab.data_qubits)
            for stab in self.stabilizers
        )

    @property
    def topological_corners(self) -> tuple[tuple[int, ...], ...]:
        """
        Return the coordinates of the topological corners of the block.
        """
        return tuple(q for q, p in self.pauli_charges.items() if p == "Y")

    @property
    def geometric_corners(self) -> tuple[tuple[int, ...], ...]:
        """
        Return the coordinates of the geometric corners of the block. The geometric
        corners that can be detected are qubits that may be the single most:
        - top-left qubit
        - top-right qubit
        - bottom-left qubit
        - bottom-right qubit
        - right-qubit
        - bottom-qubit
        - left-qubit
        - top-qubit
        """

        def get_sole_max(lambda_expression):
            """
            Find and return the qubit with the maximum value of the lambda expression
            only if it is the only qubit with that value. If there are multiple qubits
            with the same value, return None.
            """
            max_value = max(lambda_expression(q) for q in self.data_qubits)
            max_qubits = [
                q for q in self.data_qubits if lambda_expression(q) == max_value
            ]
            return max_qubits[0] if len(max_qubits) == 1 else None

        lambdas_to_maximize = [
            lambda x: x[0] + x[1],  # bottom right corner
            lambda x: x[0] - x[1],  # top right corner
            lambda x: -x[0] + x[1],  # bottom left corner
            lambda x: -x[0] - x[1],  # top left corner
            lambda x: x[0],  # right corner
            lambda x: x[1],  # bottom corner
            lambda x: -x[0],  # left corner
            lambda x: -x[1],  # top corner
        ]

        return tuple(
            get_sole_max(lambda_expression)
            for lambda_expression in lambdas_to_maximize
            if get_sole_max(lambda_expression) is not None
        )

    @property
    def all_boundary_stabilizers(self) -> tuple[Stabilizer, ...]:
        """
        Return the stabilizers associated with any boundary.
        """
        return (
            self.boundary_stabilizers(direction=Direction.TOP)
            + self.boundary_stabilizers(direction=Direction.BOTTOM)
            + self.boundary_stabilizers(direction=Direction.LEFT)
            + self.boundary_stabilizers(direction=Direction.RIGHT)
        )

    @property
    def bulk_stabilizers(self) -> tuple[Stabilizer, ...]:
        """
        Return the stabilizers not associated with any boundary.
        """
        return tuple(
            stab
            for stab in self.stabilizers
            if stab not in self.all_boundary_stabilizers
        )

    # Static methods
    @staticmethod
    def generate_weight4_stabs(
        dx: int,
        dz: int,
        pauli: str,
        schedule: FourBodySchedule,
        start_in_top_left_corner: bool,
        initial_position: tuple[int, int] = (0, 0),
    ) -> list[Stabilizer]:
        """
        Generate the list of all weight-4 stabilizers of a rotated surface code
        block of the given type (= pauli string).

        Parameters
        ----------
        dx: int
            Distance in the horizontal direction. If dx=3, there will be 2 weight-4
            stabilizers created in the horizontal direction.
        dz: int
            Distance in the vertical direction. If dz=3, there will be 2 weight-4
            stabilizers created in the vertical direction.
        pauli : str
            Pauli string of the stabilizers
        start_in_top_left_corner : bool
            If True, the first stabilizer will start in the top left corner and then
            follow the alternating checkerboard pattern. If False, it will be
            exactly the opposite covering of the checkerboard pattern.
        schedule : FourBodySchedule
            Schedule for measuring the four body stabilizers, see
            https://arxiv.org/abs/1404.3747, III, B. for more details.
        initial_position : tuple[int, int], optional
            Initial position where the chain of weight-4 stabilizers should start,
            by default (0, 0)

        Returns
        -------
        list[Stabilizer]
            List of weight-4 stabilizers of the specified type for the rotated
            surface code
        """
        if len(initial_position) > 2:
            initial_position = initial_position[:2]
        elif len(initial_position) < 2:
            raise ValueError(
                "Initial position must be a tuple of length >= 2. Got "
                f"{initial_position}."
            )
        x, z = initial_position
        is_n_pattern = schedule == FourBodySchedule.N
        return [
            Stabilizer(
                pauli=pauli,
                data_qubits=[
                    (x + dx_i + 1, z + dz_i, 0),
                    (x + dx_i + is_n_pattern, z + dz_i + is_n_pattern, 0),
                    (x + dx_i + (not is_n_pattern), z + dz_i + (not is_n_pattern), 0),
                    (x + dx_i, z + dz_i + 1, 0),
                ],
                ancilla_qubits=[(x + dx_i + 1, z + dz_i + 1, 1)],
            )
            for dx_i in range(dx - 1)
            for dz_i in range(dz - 1)
            if (dx_i + dz_i) % 2 != start_in_top_left_corner
        ]

    @staticmethod
    def generate_weight2_stabs(
        pauli: str,
        initial_position: tuple[int, int],
        num_stabs: int,
        orientation: Orientation,
        is_bottom_or_right: bool,
    ) -> list[Stabilizer]:
        """
        Generate the list of all weight-2 stabilizers along one of the four
        boundaries. Note that the schedule for measuring the weight-4 stabilizers
        does not change the order in which we specify the weight-2 stabilizers.
        For stabilizers along the right and bottom boundaries, the stabilizers are
        generated differently. The stabilizers along the right and bottom boundary have
        the coordinates of the ancilla qubits shifted by (1, 0) and (0, 1) respectively.

        Parameters
        ----------
        pauli : str
            Pauli string of the stabilizers
        initial_position : tuple[int, int]
            Initial position where the chain of weight-2 stabilizers should start
        num_stabs : int
            Number of weight-2 stabilizers to generate along this boundary
        orientation : Orientation
            Orientation of the boundary, either HORIZONTAL or VERTICAL
        is_bottom_or_right : bool
            If True, the stabilizers are along the bottom or right boundaries, this
            means their ancilla qubit is 'outside' of the block geometry. If False, the
            stabilizers are along the top or left boundaries.

        Returns
        -------
        list[Stabilizer]
            List of weight-2 stabilizers along the specified boundary
        """
        is_horizontal = orientation == Orientation.HORIZONTAL
        return [
            Stabilizer(
                pauli=pauli,
                data_qubits=[
                    (
                        initial_position[0] + is_horizontal * (2 * i + 1),
                        initial_position[1] + (not is_horizontal) * 2 * i,
                        0,
                    ),
                    (
                        initial_position[0] + is_horizontal * (2 * i),
                        initial_position[1] + (not is_horizontal) * (2 * i + 1),
                        0,
                    ),
                ],
                ancilla_qubits=[
                    (
                        initial_position[0]
                        + 1 * (not is_horizontal) * is_bottom_or_right
                        + is_horizontal * (2 * i + 1),
                        initial_position[1]
                        + 1 * is_horizontal * is_bottom_or_right
                        + (not is_horizontal) * (2 * i + 1),
                        1,
                    )
                ],
            )
            for i in range(0, int(num_stabs))
        ]

    # Instance methods

    def boundary_qubits(self, direction: Direction | str) -> list[tuple[int, ...]]:
        """
        Return the data qubits that are part of the specified boundary.

        Parameters
        ----------
        direction : Direction | str
            Boundary (top, bottom, left, or right) for which the data qubits should be
            returned. If a string is provided, it is converted to a Direction enum.

        Returns
        -------
        list[tuple[int, ...]]
            Data qubits that are part of the specified boundary
        """
        # Input validation: cast direction to Direction enum if it is not already
        if not isinstance(direction, Direction):
            direction = Direction(direction)

        axis = 1 if direction in [Direction.TOP, Direction.BOTTOM] else 0
        selector_function = (
            max if direction in [Direction.BOTTOM, Direction.RIGHT] else min
        )
        min_or_max_value = selector_function(qb[axis] for qb in self.data_qubits)
        return [qb for qb in self.data_qubits if qb[axis] == min_or_max_value]

    def boundary_type(self, direction: Direction | str) -> str:
        """
        Return the type of the specified boundary, either X or Z. This assumes that
        the block is a standard rotated surface code block which is a square block with
        Y charges at the four corners and X and Z charges at the boundaries.

        Note that there are different conventions about when to call a boundary X or Z.
        We call a boundary X type if it exhibits X Pauli charges. In other words, it is
        of X type if the stabilizers along the boundary are Z stabilizers.

        Parameters
        ----------
        direction : Direction | str
            Boundary (top, bottom, left, or right) for which the data qubits should be
            returned. If a string is provided, it is converted to a Direction enum.

        Returns
        -------
        str
            Type of the boundary, either X or Z
        """
        # Input validation: cast direction to Direction if it is not already
        if not isinstance(direction, Direction):
            direction = Direction(direction)

        # Input validation: check that block size is > 2 in the direction of the
        # boundary
        if direction in [Direction.TOP, Direction.BOTTOM] and self.size[1] <= 2:
            raise ValueError(
                "The block is too small to have a top or bottom boundary of well "
                "defined type."
            )
        if direction in [Direction.LEFT, Direction.RIGHT] and self.size[0] <= 2:
            raise ValueError(
                "The block is too small to have a left or right boundary of well "
                "defined type."
            )

        # Get the Pauli charges of the boundary qubits,
        # transform them into a set to remove duplicates,
        # and remove the "Y" charge to be left with the X and Z charges.
        # For the standard rotated surface code block, there should be only one charge
        # left.
        boundary_pauli_charges = list(
            set(self.pauli_charges[qb] for qb in self.boundary_qubits(direction))
            - set("Y")
        )
        if len(boundary_pauli_charges) > 1:
            raise RuntimeError(
                "Boundary has multiple Pauli charges. This should not happen for the "
                "standard rotated surface code block. The boundary pauli charges "
                f"(excluding Y charges) are {boundary_pauli_charges}."
            )

        return boundary_pauli_charges[0]

    def boundary_stabilizers(self, direction: Direction) -> tuple[Stabilizer, ...]:
        """
        Return the stabilizers associated with the given boundary direction.
        """
        boundary_qubits = self.boundary_qubits(direction)
        return tuple(
            stab
            for stab in self.stabilizers
            if all(q in boundary_qubits for q in stab.data_qubits)
        )

    def get_corner_from_direction(
        self, block: RotatedSurfaceCode, which_corner: DiagonalDirection
    ) -> tuple[int, int, int]:
        """
        Get the coordinates of the qubit at the specified corner of the block.
        """
        return (
            set(block.boundary_qubits(which_corner.components[0]))
            .intersection(block.boundary_qubits(which_corner.components[1]))
            .pop()
        )

    @cached_property
    def config_and_pivot_corners(
        self,
    ) -> tuple[int, tuple[tuple[int, int, int], ...]]:
        """
        Classify the Block based on the geometry of its topological corners. Return the
        config and the topological corners ordered in a specific way that reflects their
        geometric arrangement.

        Type 1: Rectangular config
                Four topological corners coincide with 4 geometric corners
                The list of corners is returned in the order:
                (top-left, bottom-left, bottom-right, top-right)
                Visualized example::

                1-------4
                |       |
                |       |
                |       |
                2-------3

        Type 2: U-config
                Block has rectangular shape with size (d, 2d-1) or (2d-1, d)
                Three of four topological corners coincide with three geometric corners,
                the last topological corner resides on the middle of the long edge whose
                ends occupy only one topological corner
                The list of corners is returned in the order:
                (long_end, middle_edge, angle, short_end)
                Visualized example (can be rotated)::

                1-------|
                |       |
                |       2
                |       |
                3-------4

        Type 3: L-config
                Block has rectangular shape with size (d, 2d-1) or (2d-1, d)
                Three of four topological corners coincide with three geometric corners,
                the last topological corner resides on the middle of the long edge whose
                ends occupy two topological corners.
                The list of corners is returned in the order as seen below:
                (long_end, middle_edge, angle, short_end)
                Visualized example (can be rotated)::

                1-------|
                |       |
                2       |
                |       |
                3-------4

        Type 0: Other configs

        Returns
        -------
        int:
            The configuration type of the block.
        tuple[tuple[int, int, int], ...]:
            The list of corners in the order specified by the configuration type.
        """
        # Find topological corners that are also geometric corners
        dx, dz = self.size
        is_block_horizontal = dx >= dz
        topological_geometric_corners = set(self.topological_corners) & set(
            self.geometric_corners
        )

        if len(topological_geometric_corners) == 4:
            # Type 1 (rectangle)
            return 1, sorted(topological_geometric_corners)

        is_almost_two_squares = dx == 2 * dz - 1 or dz == 2 * dx - 1
        is_ul_config = len(topological_geometric_corners) == 3 and is_almost_two_squares
        if not is_ul_config:
            # Type 0
            return 0, self.topological_corners

        # Find the last topological corner
        topological_non_geometric_corner = (
            set(self.topological_corners) - set(topological_geometric_corners)
        ).pop()

        # Based on the orientation of the block, deduce the long and short edge indices
        long_edge_idx, short_edge_idx = (0, 1) if is_block_horizontal else (1, 0)

        # Find two topological-geometric corners that reside on the same long edge
        long_edge_geometric_corners = next(
            (corner_1, corner_2)
            for (corner_1, corner_2) in combinations(topological_geometric_corners, 2)
            if corner_1[short_edge_idx] == corner_2[short_edge_idx]
        )

        # Find the third topological-geometric corner
        short_edge_geometric_corner = next(
            corner
            for corner in topological_geometric_corners
            if corner not in long_edge_geometric_corners
        )

        # Find the middle point of the long edge and verify that it is where the last
        # topological corner is.
        # If it's not, the config is not U or L
        mid_point_long_edge = (
            long_edge_geometric_corners[0][long_edge_idx]
            + long_edge_geometric_corners[1][long_edge_idx]
        ) / 2
        if mid_point_long_edge != topological_non_geometric_corner[long_edge_idx]:
            # Type 0
            return 0, self.topological_corners

        # If the short edge index of the topological non-geometric corner matches:
        # - the short edge geometric corner, then it is a U shape
        # - one of the long edge geometric corners, then it is a L shape
        if (
            topological_non_geometric_corner[short_edge_idx]
            == short_edge_geometric_corner[short_edge_idx]
        ):
            # Type 2 (U shape)
            config = 2
        elif (
            topological_non_geometric_corner[short_edge_idx]
            == long_edge_geometric_corners[0][short_edge_idx]
        ):
            # Type 3 (L shape)
            config = 3
        else:
            raise RuntimeError("Something went wrong with the configuration detection.")

        # Sort the long_edge_geometric_corners so that the first returned corner is
        # the long_end.
        long_edge_geometric_corners = sorted(
            long_edge_geometric_corners,
            reverse=topological_non_geometric_corner[long_edge_idx]
            > short_edge_geometric_corner[long_edge_idx],
        )

        # Put all of the corners together
        corners = (
            long_edge_geometric_corners[0],  # long_end
            topological_non_geometric_corner,  # middle_edge
            long_edge_geometric_corners[1],  # angle
            short_edge_geometric_corner,  # short_edge
        )

        return config, corners

    def get_shifted_equivalent_logical_operator(
        self, initial_operator: PauliOperator, new_upleft_qubit: tuple[int, ...]
    ) -> tuple[PauliOperator, tuple[Stabilizer, ...]]:
        """
        Shifts the initial operator to a valid operator in the block that contains
        new_upleft_qubit.
        NOTE: this function currently assumes that the block is a square rotated surface
        code block with a single logical X and Z operator.

        Parameters
        ----------
        initial_operator : PauliOperator
            Initial logical operator for which the equivalent logical operator in the
            block should be found.
        new_upleft_qubit : tuple[int, ...]
            New top left qubit of the operator in the block.

        Returns
        -------
        tuple[PauliOperator, tuple[Stabilizer, ...]]
            Equivalent logical operator in the block and the stabilizers that are
            required to go from the initial to the new operator.

        Raises
        ------
        NotImplementedError
            If the block has more than one logical X or Z operator.
        ValueError
            If the new upleft qubit is not part of the data qubits of the block.
        ValueError
            If the initial operator is not one of the logical operators of the block.
        ValueError
            If the new upleft qubit is not part of the correct boundary.
        ValueError
            If the shift vector has more than a single non-zero dimension (ill defined
            logical operators to begin with).
        """
        if len(self.logical_x_operators) != 1 or len(self.logical_z_operators) != 1:
            raise NotImplementedError(
                "This function currently only supports blocks with a single logical "
                "X and Z operator."
            )

        if new_upleft_qubit not in self.data_qubits:
            raise ValueError(
                f"The new upleft qubit {new_upleft_qubit} is not part of the data "
                "qubits of the block."
            )

        if initial_operator not in [
            self.logical_x_operators[0],
            self.logical_z_operators[0],
        ]:
            raise ValueError(
                "The initial operator must be one of the logical operators of the "
                "block."
            )

        # We assume all Xs or all Zs pauli operators
        # The new upper left qubit needs to be part of the right boundary, Y is valid.
        # E.g. Z logical has an X or Y at it upper left most qubit.
        if self.pauli_charges[new_upleft_qubit] == initial_operator.pauli[0]:
            raise ValueError(
                f"The new upleft qubit {new_upleft_qubit} is not part of the correct "
                "boundary."
            )

        initial_upleft_qubit = min(
            initial_operator.data_qubits, key=lambda x: x[0] + x[1]
        )
        shift_vector = tuple(
            coord1 - coord2
            for coord1, coord2 in zip(
                new_upleft_qubit, initial_upleft_qubit, strict=True
            )
        )

        new_logical_operator = PauliOperator(
            pauli=initial_operator.pauli,
            data_qubits=[
                tuple(
                    coord1 + shift
                    for coord1, shift in zip(qb, shift_vector, strict=True)
                )
                for qb in initial_operator.data_qubits
            ],
        )

        # Find all stabilizers in the block that are included between the initial and
        # new operator
        def vector_range(vector: tuple[int, ...]) -> list[tuple[int, ...]]:
            """
            Generate vectors from (0, ..., 0) to (0, ..., k, ..., 0) where k is the only
            non-zero element in the input vector. The position of k in the tuple may
            vary. If k is negative, the range goes from k to 0.
            E.g.:
            vector_range((0, 5)) -> [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
            vector_range((-3, 0, 0)) -> [(-3, 0, 0), (-2, 0, 0), (-1, 0, 0), (0, 0, 0)]
            """
            # Find the position and value of the non-zero element in the vector
            k_position = next(i for i, x in enumerate(vector) if x != 0)
            k = vector[k_position]

            # Determine the range based on the sign of k
            if k > 0:
                range_values = range(k + 1)
            else:
                range_values = range(k, 1)

            return [
                tuple(i if j == k_position else 0 for j in range(len(vector)))
                for i in range_values
            ]

        # Test that shift vector has at most a single non-zero dimension
        n_non_zero_dims = sum(1 for x in shift_vector if x != 0)
        match n_non_zero_dims:
            # The two qubits have the same coordinates
            case 0:
                return new_logical_operator, ()
            # The two qubits have a single non-zero dimension, they are aligned
            case 1:
                qubits_in_between = [
                    tuple(
                        qubit_coord + shift_coord
                        for (qubit_coord, shift_coord) in zip(q, v, strict=True)
                    )
                    for q in initial_operator.data_qubits
                    for v in vector_range(shift_vector)
                ]
                # Here we assume all Xs or all Zs stabilizers
                stabilizers_in_between = [
                    stab
                    for stab in self.stabilizers
                    if all(q in qubits_in_between for q in stab.data_qubits)
                    and stab.pauli[0] == initial_operator.pauli[0]
                ]
                return new_logical_operator, tuple(stabilizers_in_between)
            # The two qubits have more than a single non-zero dimension,
            # they are not aligned.
            case _:
                raise ValueError(
                    "The shift vector must have at most a single non-zero dimension."
                )

    def rename(self, name: str) -> RotatedSurfaceCode:
        return super().rename(name)

    def shift(
        self, position: tuple[int, ...], new_label: str | None = None
    ) -> RotatedSurfaceCode:
        return super().shift(position, new_label)

    def __eq__(self, other) -> bool:  # pylint: disable=useless-parent-delegation
        return super().__eq__(other)

    @cached_property
    def stabilizers_labels(self) -> dict[str, dict[str, tuple[int, ...]]]:
        return super().stabilizers_labels
