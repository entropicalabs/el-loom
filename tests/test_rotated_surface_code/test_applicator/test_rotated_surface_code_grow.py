"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest
from itertools import product

from loom.eka import Circuit, Channel, Lattice, PauliOperator
from loom.eka.utilities import Direction, Orientation
from loom.eka.operations import Grow
from loom.interpreter import InterpretationStep, Syndrome

from loom_rotated_surface_code.utilities import FourBodySchedule
from loom_rotated_surface_code.code_factory import RotatedSurfaceCode
from loom_rotated_surface_code.applicator import grow


class TestRotatedSurfaceCodeGrow(unittest.TestCase):
    """
    Test the applicator for the grow operation of RotatedSurfaceCode blocks.
    """

    def test_applicator_grow_simple_example(self):
        """
        Test that the applicator modifies the block correctly when applying a simple
        grow operation.
        """
        lattice = Lattice.square_2d((4, 3))
        direction = Direction.RIGHT
        grow_op = Grow("q1", direction, length=1)
        rsc = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            lattice=lattice,
            unique_label="q1",
            weight_2_stab_is_first_row=False,
        )

        int_step = InterpretationStep(
            block_history=[(rsc,)],
            logical_x_operator_updates={
                rsc.logical_x_operators[0].uuid: (("dummy_X", 0),)
            },
            logical_z_operator_updates={
                rsc.logical_z_operators[0].uuid: (("dummy_Z", 0),)
            },
        )
        output_step = grow(int_step, grow_op, same_timeslice=False, debug_mode=True)
        new_block = output_step.block_history[-1][0]

        expected_block = RotatedSurfaceCode.create(
            dx=4,
            dz=3,
            lattice=lattice,
            unique_label="q1",
            weight_2_stab_is_first_row=False,
        )
        # Test that the blocks are the same
        self.assertEqual(expected_block, new_block)

        # Test that the circuit is correctly generated
        expected_circuit = Circuit(
            name="grow q1 by 1 to the right",
            circuit=[
                [
                    Circuit(
                        "reset_+",
                        channels=(Channel(label=f"{(x, y, 0)}", type="quantum"),),
                    )
                    for (x, y) in product(range(3, 4), range(3))
                ]
            ],
        )
        self.assertEqual(
            output_step.intermediate_circuit_sequence[0][0], expected_circuit
        )
        # Test that the syndrome creation is correct
        expected_syndromes = tuple(
            Syndrome(
                stab.uuid,
                measurements=(),
                block=rsc.uuid,
                round=0,
            )
            for stab in set(new_block.stabilizers).difference(set(rsc.stabilizers))
            if set(stab.pauli) == {"X"}
            and not all(a in rsc.ancilla_qubits for a in stab.ancilla_qubits)
        )
        self.assertEqual(
            set(
                (synd.stabilizer, synd.measurements, synd.block, synd.round)
                for synd in output_step.syndromes
            ),
            set(
                (synd.stabilizer, synd.measurements, synd.block, synd.round)
                for synd in expected_syndromes
            ),
        )

        # XX(2,0,0)(2,1,0) is gonna be morphed into XXXX(3,0,0)(2,0,0)(3,1,0)(2,1,0)
        stabs_to_be_changed = [
            stab
            for stab in rsc.stabilizers
            if set(stab.data_qubits).issubset(set(rsc.boundary_qubits(direction)))
            and stab.pauli == "XX"
        ]
        stabs_uuid_to_be_changed = [(stab.uuid,) for stab in stabs_to_be_changed]
        modified_stabs_uuid = [
            stab.uuid
            for stab in output_step.get_block("q1").stabilizers
            for old_stab in stabs_to_be_changed
            if set(old_stab.data_qubits).issubset(set(stab.data_qubits))
            and stab.pauli == "XXXX"
        ]
        expected_stab_evolution = dict(
            zip(modified_stabs_uuid, stabs_uuid_to_be_changed, strict=True)
        )
        # Test that the stabilizer evolution is correct
        self.assertEqual(expected_stab_evolution, output_step.stabilizer_evolution)

        initial_logical_x = rsc.logical_x_operators[0]
        final_logical_x = new_block.logical_x_operators[0]
        expected_logical_x_evolution = {final_logical_x.uuid: (initial_logical_x.uuid,)}
        # Test that the logical evolution is correct
        self.assertEqual(expected_logical_x_evolution, output_step.logical_x_evolution)

        # Test that the logical updates are correct
        # Note that X is modified and thus the update propagates to the new operator
        expected_x_updates = {
            rsc.logical_x_operators[0].uuid: (("dummy_X", 0),),
            new_block.logical_x_operators[0].uuid: (("dummy_X", 0),),
        }
        expected_z_updates = {
            rsc.logical_z_operators[0].uuid: (("dummy_Z", 0),),
        }
        self.assertEqual(expected_x_updates, output_step.logical_x_operator_updates)
        self.assertEqual(expected_z_updates, output_step.logical_z_operator_updates)

    def test_applicator_grow_all_directions(
        self,
    ):  # pylint: disable=too-many-statements, too-many-locals
        """
        Test that the applicator modifies the block correctly when applying grow. The
        stabilizer_evolution dictionary should also be updated accordingly. This is
        tested for grow in all four directions.
        """
        lattice = Lattice.square_2d((11, 11))
        weight_2_stab_is_first_row = False
        weight_4_x_schedule = FourBodySchedule.N
        initial_position = (3, 3)
        initial_block = RotatedSurfaceCode.create(
            dx=4,
            dz=4,
            position=initial_position,
            lattice=lattice,
            unique_label="q1",
            weight_4_x_schedule=weight_4_x_schedule,
            weight_2_stab_is_first_row=weight_2_stab_is_first_row,
        )

        for direction in (
            Direction.RIGHT,
            Direction.LEFT,
            Direction.TOP,
            Direction.BOTTOM,
        ):
            grow_op = Grow("q1", direction, length=3)
            base_step = InterpretationStep(
                block_history=((initial_block,),),
                logical_x_operator_updates={
                    initial_block.logical_x_operators[0].uuid: (("dummy_X", 0),)
                },
                logical_z_operator_updates={
                    initial_block.logical_z_operators[0].uuid: (("dummy_Z", 0),)
                },
            )
            output_step = grow(
                base_step, grow_op, same_timeslice=False, debug_mode=True
            )
            # Get the new block from the output InterpretationStep
            new_block = output_step.get_block(initial_block.unique_label)

            # We need to use modified logical operators to compare the blocks
            match direction:
                case Direction.LEFT:
                    new_log_z = PauliOperator(
                        pauli="ZZZZ", data_qubits=tuple((3, i, 0) for i in range(4))
                    )
                    new_log_x = None
                case Direction.RIGHT:
                    new_log_x = None
                    new_log_z = None
                case Direction.BOTTOM:
                    new_log_x = None
                    new_log_z = None
                case Direction.TOP:
                    new_log_x = PauliOperator(
                        pauli="XXXX", data_qubits=tuple((i, 3, 0) for i in range(4))
                    )
                    new_log_z = None

            new_weight_2_stab_is_first_row = new_block.weight_2_stab_is_first_row
            expected_block = RotatedSurfaceCode.create(
                dx=4
                + (direction in (Direction.LEFT, Direction.RIGHT)) * grow_op.length,
                dz=4
                + (direction in (Direction.TOP, Direction.BOTTOM)) * grow_op.length,
                lattice=lattice,
                unique_label="q1",
                # position=(3, 3),
                x_boundary=Orientation.HORIZONTAL,
                weight_2_stab_is_first_row=new_weight_2_stab_is_first_row,
                weight_4_x_schedule=weight_4_x_schedule,
                logical_x_operator=new_log_x,
                logical_z_operator=new_log_z,
            )
            new_init_position = (
                initial_position[0] - (direction == Direction.LEFT) * grow_op.length,
                initial_position[1] - (direction == Direction.TOP) * grow_op.length,
            )
            expected_block = expected_block.shift(new_init_position)

            # Test that the blocks are the same
            self.assertEqual(expected_block, new_block)

            # Test that the circuit is correctly generated
            new_qubits = set(expected_block.data_qubits).difference(
                initial_block.data_qubits
            )
            reset_type = "+" if direction in (Direction.RIGHT, Direction.LEFT) else "0"
            expected_circuit = Circuit(
                name=f"grow q1 by 3 to the {direction.value}",
                circuit=[
                    [
                        Circuit(
                            f"reset_{reset_type}",
                            channels=(Channel(label=f"{q}", type="quantum"),),
                        )
                        for q in new_qubits
                    ]
                ],
            )
            self.assertEqual(
                output_step.intermediate_circuit_sequence[0][0], expected_circuit
            )

            # Test that the syndrome creation is correct
            deterministic_stab_pauli = "X" if reset_type == "+" else "Z"
            expected_syndromes = tuple(
                Syndrome(
                    stab.uuid,
                    measurements=(),
                    block=initial_block.uuid,
                    round=0,
                )
                for stab in set(new_block.stabilizers).difference(
                    set(initial_block.stabilizers)
                )
                if set(stab.pauli) == {deterministic_stab_pauli}
                and not all(
                    a in initial_block.ancilla_qubits for a in stab.ancilla_qubits
                )
            )
            self.assertEqual(
                set(
                    (synd.stabilizer, synd.measurements, synd.block, synd.round)
                    for synd in output_step.syndromes
                ),
                set(
                    (synd.stabilizer, synd.measurements, synd.block, synd.round)
                    for synd in expected_syndromes
                ),
            )

            # Test that the stabilizer evolution is correct
            stabs_to_be_changed = [
                stab
                for stab in initial_block.stabilizers
                if set(stab.data_qubits).issubset(
                    set(initial_block.boundary_qubits(direction))
                )
                and len(stab.data_qubits) == 2
            ]
            stabs_uuid_to_be_changed = [(stab.uuid,) for stab in stabs_to_be_changed]
            modified_stabs_uuid = [
                stab.uuid
                for stab in output_step.get_block("q1").stabilizers
                for old_stab in stabs_to_be_changed
                if set(old_stab.data_qubits).issubset(set(stab.data_qubits))
                and old_stab.pauli in stab.pauli
            ]
            expected_stab_evolution = dict(
                zip(modified_stabs_uuid, stabs_uuid_to_be_changed, strict=True)
            )
            self.assertEqual(expected_stab_evolution, output_step.stabilizer_evolution)

            # Test that the logical evolution and updates are correct
            if direction in (Direction.LEFT, Direction.RIGHT):
                initial_logical_x = initial_block.logical_x_operators[0]
                final_logical_x = new_block.logical_x_operators[0]
                # test the logical evolution
                expected_logical_x_evolution = {
                    final_logical_x.uuid: (initial_logical_x.uuid,)
                }
                self.assertEqual(
                    expected_logical_x_evolution, output_step.logical_x_evolution
                )
                self.assertEqual({}, output_step.logical_z_evolution)
                # test the logical updates
                expected_x_updates = {
                    initial_block.logical_x_operators[0].uuid: (("dummy_X", 0),),
                    final_logical_x.uuid: (("dummy_X", 0),),
                }
                expected_z_updates = {
                    initial_block.logical_z_operators[0].uuid: (("dummy_Z", 0),),
                }
                self.assertEqual(
                    expected_x_updates, output_step.logical_x_operator_updates
                )
                self.assertEqual(
                    expected_z_updates, output_step.logical_z_operator_updates
                )

            elif direction in (Direction.TOP, Direction.BOTTOM):
                initial_logical_z = initial_block.logical_z_operators[0]
                final_logical_z = new_block.logical_z_operators[0]
                # test the logical evolution
                expected_logical_z_evolution = {
                    final_logical_z.uuid: (initial_logical_z.uuid,)
                }
                self.assertEqual(
                    expected_logical_z_evolution, output_step.logical_z_evolution
                )
                self.assertEqual({}, output_step.logical_x_evolution)
                # test the logical updates
                expected_x_updates = {
                    initial_block.logical_x_operators[0].uuid: (("dummy_X", 0),),
                }
                expected_z_updates = {
                    initial_block.logical_z_operators[0].uuid: (("dummy_Z", 0),),
                    final_logical_z.uuid: (("dummy_Z", 0),),
                }
                self.assertEqual(
                    expected_x_updates, output_step.logical_x_operator_updates
                )
                self.assertEqual(
                    expected_z_updates, output_step.logical_z_operator_updates
                )


if __name__ == "__main__":
    unittest.main()
