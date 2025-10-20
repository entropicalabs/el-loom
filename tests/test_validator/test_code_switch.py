"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest

from loom.eka import (
    Circuit,
    Channel,
    ChannelType,
    LogicalState,
    Lattice,
    Block,
    Stabilizer,
    PauliOperator,
)
from loom.validator import is_circuit_valid


# pylint: disable=duplicate-code
class TestCodeSwitchValidator(unittest.TestCase):
    """
    Test cases for validating grow circuits using the Validator module.
    """

    def test_probabilistic_grow_operation_stabilizers(self):
        """
        Test the validation of the grow operation from a distance 3 to distance 5
        RepetitionCode. The main aspect here is checking that the probabilistic
        stabilizers are correctly handled.
        """
        # pylint: disable=unnecessary-lambda-assignment
        rep_code = lambda d: Block(
            unique_label="q1",
            stabilizers=tuple(
                Stabilizer(
                    pauli="ZZ",
                    data_qubits=(
                        (i, 0),
                        (i + 1, 0),
                    ),
                    ancilla_qubits=((i, 1),),
                )
                for i in range(d - 1)
            ),
            logical_x_operators=(PauliOperator("Z", ((0, 0),)),),
            logical_z_operators=(
                PauliOperator("X" * d, tuple((i, 0) for i in range(d))),
            ),
        )

        rc_3 = rep_code(3)
        rc_5 = rep_code(5)

        channels_dict = {
            qub: Channel(type=ChannelType.QUANTUM, label=str(qub))
            for qub in rc_3.qubits + rc_5.qubits
        }

        # Find the new data qubits
        new_data_qubits = list(set(rc_5.data_qubits) - set(rc_3.data_qubits))

        # Find new stabilizers
        new_stabilizers = list(set(rc_5.stabilizers) - set(rc_3.stabilizers))
        new_stabilizer_c_channels = [
            Channel(
                type=ChannelType.CLASSICAL, label=str(f"c_{stab.ancilla_qubits[0]}_0")
            )
            for stab in new_stabilizers
        ]

        grow_circuit_seq = []

        # reset the new data qubits and set them to |+> state
        grow_circuit_seq += [
            Circuit("reset_+", channels=[channels_dict[qub]]) for qub in new_data_qubits
        ]

        # for every new stabilizer, measure it
        for new_stab, c_chan in zip(
            new_stabilizers, new_stabilizer_c_channels, strict=True
        ):
            m_stab_seq = []

            # reset the ancilla qubit
            m_stab_seq += [
                Circuit("reset", channels=[channels_dict[new_stab.ancilla_qubits[0]]])
            ]
            # apply CNOT gates to the ancilla qubit
            m_stab_seq += [
                Circuit(
                    "cx",
                    channels=[
                        channels_dict[qub],
                        channels_dict[new_stab.ancilla_qubits[0]],
                    ],
                )
                for qub in new_stab.data_qubits
            ]
            # measure the ancilla qubit
            m_stab_seq += [
                Circuit(
                    "measurement",
                    channels=[channels_dict[new_stab.ancilla_qubits[0]], c_chan],
                )
            ]
            grow_circuit_seq += m_stab_seq

        circuit = Circuit("grow_operation", circuit=grow_circuit_seq)

        # Check the circuit with output stabilizers with any value
        _ = is_circuit_valid(
            circuit=circuit,
            input_block=rc_3,
            output_block=rc_5,
            output_stabilizers_parity={},
            output_stabilizers_with_any_value=new_stabilizers,
            logical_state_transformations_with_parity={},
            logical_state_transformations={},
            measurement_to_input_stabilizer_map={},
        )

        # Check the circuit with output stabilizers with the updates specified
        debug_data = is_circuit_valid(
            circuit=circuit,
            input_block=rc_3,
            output_block=rc_5,
            output_stabilizers_parity={
                new_stabilizers[i]: [new_stabilizer_c_channels[i].label]
                for i in range(len(new_stabilizers))
            },
            output_stabilizers_with_any_value=[],
            logical_state_transformations_with_parity={},
            logical_state_transformations={},
            measurement_to_input_stabilizer_map={},
        )
        self.assertTrue(debug_data.valid)


class TestCodeSwitchValidatorSplit(unittest.TestCase):
    """
    Test cases for validating split circuits using the Validator module.
    """

    def setUp(self):
        """
        Define necessary objects for the split operation tests.
        """
        # self.lattice = Lattice.square_2d()
        # self.block_big = RotatedSurfaceCode.create(
        #     7, 3, self.lattice, "q_big", position=(0, 0)
        # )
        # self.block_left = RotatedSurfaceCode.create(
        #     3, 3, self.lattice, "q_left", position=(0, 0)
        # )
        # self.block_right = RotatedSurfaceCode.create(
        #     3, 3, self.lattice, "q_right", position=(4, 0)
        # )
        self.lattice = Lattice.square_2d()
        self.block_big = Block(
            stabilizers=(
                Stabilizer(
                    "ZZZZ",
                    ((1, 0, 0), (0, 0, 0), (1, 1, 0), (0, 1, 0)),
                    ancilla_qubits=((1, 1, 1),),
                ),
                Stabilizer(
                    "ZZZZ",
                    ((2, 1, 0), (1, 1, 0), (2, 2, 0), (1, 2, 0)),
                    ancilla_qubits=((2, 2, 1),),
                ),
                Stabilizer(
                    "ZZZZ",
                    ((3, 0, 0), (2, 0, 0), (3, 1, 0), (2, 1, 0)),
                    ancilla_qubits=((3, 1, 1),),
                ),
                Stabilizer(
                    "ZZZZ",
                    ((4, 1, 0), (3, 1, 0), (4, 2, 0), (3, 2, 0)),
                    ancilla_qubits=((4, 2, 1),),
                ),
                Stabilizer(
                    "ZZZZ",
                    ((5, 0, 0), (4, 0, 0), (5, 1, 0), (4, 1, 0)),
                    ancilla_qubits=((5, 1, 1),),
                ),
                Stabilizer(
                    "ZZZZ",
                    ((6, 1, 0), (5, 1, 0), (6, 2, 0), (5, 2, 0)),
                    ancilla_qubits=((6, 2, 1),),
                ),
                Stabilizer(
                    "XXXX",
                    ((1, 1, 0), (1, 2, 0), (0, 1, 0), (0, 2, 0)),
                    ancilla_qubits=((1, 2, 1),),
                ),
                Stabilizer(
                    "XXXX",
                    ((2, 0, 0), (2, 1, 0), (1, 0, 0), (1, 1, 0)),
                    ancilla_qubits=((2, 1, 1),),
                ),
                Stabilizer(
                    "XXXX",
                    ((3, 1, 0), (3, 2, 0), (2, 1, 0), (2, 2, 0)),
                    ancilla_qubits=((3, 2, 1),),
                ),
                Stabilizer(
                    "XXXX",
                    ((4, 0, 0), (4, 1, 0), (3, 0, 0), (3, 1, 0)),
                    ancilla_qubits=((4, 1, 1),),
                ),
                Stabilizer(
                    "XXXX",
                    ((5, 1, 0), (5, 2, 0), (4, 1, 0), (4, 2, 0)),
                    ancilla_qubits=((5, 2, 1),),
                ),
                Stabilizer(
                    "XXXX",
                    ((6, 0, 0), (6, 1, 0), (5, 0, 0), (5, 1, 0)),
                    ancilla_qubits=((6, 1, 1),),
                ),
                Stabilizer("XX", ((0, 0, 0), (0, 1, 0)), ancilla_qubits=((0, 1, 1),)),
                Stabilizer("XX", ((6, 1, 0), (6, 2, 0)), ancilla_qubits=((7, 2, 1),)),
                Stabilizer("ZZ", ((2, 0, 0), (1, 0, 0)), ancilla_qubits=((2, 0, 1),)),
                Stabilizer("ZZ", ((4, 0, 0), (3, 0, 0)), ancilla_qubits=((4, 0, 1),)),
                Stabilizer("ZZ", ((6, 0, 0), (5, 0, 0)), ancilla_qubits=((6, 0, 1),)),
                Stabilizer("ZZ", ((1, 2, 0), (0, 2, 0)), ancilla_qubits=((1, 3, 1),)),
                Stabilizer("ZZ", ((3, 2, 0), (2, 2, 0)), ancilla_qubits=((3, 3, 1),)),
                Stabilizer("ZZ", ((5, 2, 0), (4, 2, 0)), ancilla_qubits=((5, 3, 1),)),
            ),
            logical_x_operators=[
                PauliOperator(
                    pauli="XXXXXXX", data_qubits=tuple((i, 0, 0) for i in range(7))
                )
            ],
            logical_z_operators=[
                PauliOperator(
                    pauli="ZZZ", data_qubits=((0, 0, 0), (0, 1, 0), (0, 2, 0))
                )
            ],
            unique_label="q_big",
        )
        self.block_left = Block(
            stabilizers=(
                Stabilizer(
                    "ZZZZ",
                    ((1, 0, 0), (0, 0, 0), (1, 1, 0), (0, 1, 0)),
                    ancilla_qubits=((1, 1, 1),),
                ),
                Stabilizer(
                    "ZZZZ",
                    ((2, 1, 0), (1, 1, 0), (2, 2, 0), (1, 2, 0)),
                    ancilla_qubits=((2, 2, 1),),
                ),
                Stabilizer(
                    "XXXX",
                    ((1, 1, 0), (1, 2, 0), (0, 1, 0), (0, 2, 0)),
                    ancilla_qubits=((1, 2, 1),),
                ),
                Stabilizer(
                    "XXXX",
                    ((2, 0, 0), (2, 1, 0), (1, 0, 0), (1, 1, 0)),
                    ancilla_qubits=((2, 1, 1),),
                ),
                Stabilizer("XX", ((0, 0, 0), (0, 1, 0)), ancilla_qubits=((0, 1, 1),)),
                Stabilizer("XX", ((2, 1, 0), (2, 2, 0)), ancilla_qubits=((3, 2, 1),)),
                Stabilizer("ZZ", ((2, 0, 0), (1, 0, 0)), ancilla_qubits=((2, 0, 1),)),
                Stabilizer("ZZ", ((1, 2, 0), (0, 2, 0)), ancilla_qubits=((1, 3, 1),)),
            ),
            logical_x_operators=[
                PauliOperator(
                    pauli="XXX", data_qubits=tuple((i, 0, 0) for i in range(3))
                )
            ],
            logical_z_operators=[
                PauliOperator(
                    pauli="ZZZ", data_qubits=((0, 0, 0), (0, 1, 0), (0, 2, 0))
                )
            ],
            unique_label="q_left",
        )
        self.block_right = Block(
            stabilizers=(
                Stabilizer(
                    "ZZZZ",
                    ((5, 0, 0), (4, 0, 0), (5, 1, 0), (4, 1, 0)),
                    ancilla_qubits=((5, 1, 1),),
                ),
                Stabilizer(
                    "ZZZZ",
                    ((6, 1, 0), (5, 1, 0), (6, 2, 0), (5, 2, 0)),
                    ancilla_qubits=((6, 2, 1),),
                ),
                Stabilizer(
                    "XXXX",
                    ((5, 1, 0), (5, 2, 0), (4, 1, 0), (4, 2, 0)),
                    ancilla_qubits=((5, 2, 1),),
                ),
                Stabilizer(
                    "XXXX",
                    ((6, 0, 0), (6, 1, 0), (5, 0, 0), (5, 1, 0)),
                    ancilla_qubits=((6, 1, 1),),
                ),
                Stabilizer("XX", ((4, 0, 0), (4, 1, 0)), ancilla_qubits=((4, 1, 1),)),
                Stabilizer("XX", ((6, 1, 0), (6, 2, 0)), ancilla_qubits=((7, 2, 1),)),
                Stabilizer("ZZ", ((6, 0, 0), (5, 0, 0)), ancilla_qubits=((6, 0, 1),)),
                Stabilizer("ZZ", ((5, 2, 0), (4, 2, 0)), ancilla_qubits=((5, 3, 1),)),
            ),
            logical_x_operators=[
                PauliOperator(
                    pauli="XXX", data_qubits=((4, 0, 0), (5, 0, 0), (6, 0, 0))
                )
            ],
            logical_z_operators=[
                PauliOperator(
                    pauli="ZZZ", data_qubits=((4, 0, 0), (4, 1, 0), (4, 2, 0))
                )
            ],
            unique_label="q_right",
        )
        self.dqubits = {
            q: Channel(type=ChannelType.QUANTUM, label=str(q))
            for q in self.block_big.data_qubits
        }
        self.c_channels = [
            Channel(type=ChannelType.CLASSICAL, label=str(f"c_(3, {i}, 0)"))
            for i in range(3)
        ]

    def test_split_operation_log_state_transformation(self):
        """
        Test the validation of the split operation with corrections to get rid of
        the probabilistic behavior. The main aspect here is checking that the logical
        state transformations are correctly handled.
        """
        # Define the split circuit with corrections
        split_circ = Circuit(
            "split_circuit",
            circuit=[Circuit("h", channels=[self.dqubits[(3, j, 0)]]) for j in range(3)]
            + [
                Circuit(
                    "measurement",
                    channels=[
                        self.dqubits[(3, j, 0)],
                        self.c_channels[j],
                    ],
                )
                for j in range(3)
            ]
            # Fix the logical state to be in the +1 eigenstate by applying a Z logical operation
            # if the middle qubit is in the -1 eigenstate
            + [
                Circuit(
                    "cz", channels=[self.dqubits[(3, 1, 0)], self.dqubits[(4, j, 0)]]
                )
                for j in range(3)
            ]
            # Fix the stabilizer X_(4,0,0)X_(4,1,0) to be in the +1 eigenstate
            + [
                Circuit(
                    "cz", channels=[self.dqubits[(3, 0, 0)], self.dqubits[(4, 0, 0)]]
                ),
                Circuit(
                    "cz", channels=[self.dqubits[(3, 1, 0)], self.dqubits[(4, 0, 0)]]
                ),
            ]
            # Fix the stabilizer X_(2,1,0)X_(2,2,0) to be in the +1 eigenstate
            + [
                Circuit(
                    "cz", channels=[self.dqubits[(3, 1, 0)], self.dqubits[(2, 2, 0)]]
                ),
                Circuit(
                    "cz", channels=[self.dqubits[(3, 2, 0)], self.dqubits[(2, 2, 0)]]
                ),
            ],
        )

        # Define the logical state transformations which should happen
        logical_states_transformations_split = [
            (LogicalState("+Z0"), (LogicalState(["+Z0", "+Z1"]),)),
            (LogicalState("+X0"), (LogicalState(["+X0X1", "+Z0Z1"]),)),
        ]

        # Run Validator
        split_debug_data = is_circuit_valid(
            circuit=split_circ,
            input_block=self.block_big,
            output_block=[self.block_left, self.block_right],
            output_stabilizers_parity={},
            output_stabilizers_with_any_value=[],
            logical_state_transformations_with_parity={},
            logical_state_transformations=logical_states_transformations_split,
            measurement_to_input_stabilizer_map={},
        )

        self.assertTrue(split_debug_data.valid)

    def test_split_without_corrections_log_state_transformation(self):
        """
        Test the validation of the split operation without corrections. The main aspect
        here is checking that the logical state transformations are correctly handled.
        """

        split_circ_no_corrections = Circuit(
            "split_circuit_no_corrections",
            circuit=[Circuit("h", channels=[self.dqubits[(3, j, 0)]]) for j in range(3)]
            + [
                Circuit(
                    "measurement",
                    channels=[self.dqubits[(3, j, 0)], self.c_channels[j]],
                )
                for j in range(3)
            ],
        )

        # Create the output stabilizers parity
        output_stabilizers_parity = {}
        # Find X_(4,0,0)X_(4,1,0)
        x_stab_1 = next(
            stab
            for stab in self.block_right.stabilizers
            if (4, 0, 0) in stab.data_qubits
            and (4, 1, 0) in stab.data_qubits
            and stab.pauli[0] == "X"
        )
        output_stabilizers_parity[x_stab_1] = [
            self.c_channels[0].label,
            self.c_channels[1].label,
        ]
        # Find X_(2,1,0)X_(2,2,0)
        x_stab_2 = next(
            stab
            for stab in self.block_left.stabilizers
            if (2, 1, 0) in stab.data_qubits
            and (2, 2, 0) in stab.data_qubits
            and stab.pauli[0] == "X"
        )
        output_stabilizers_parity[x_stab_2] = [
            self.c_channels[1].label,
            self.c_channels[2].label,
        ]

        # Define the logical state transformations which should happen
        logical_states_transformations_w_parity = {
            LogicalState("+Z0"): (LogicalState(["+Z0", "+Z1"]), {}),
            LogicalState("+X0"): (
                LogicalState(["+X0X1", "+Z0Z1"]),
                {0: (self.c_channels[0].label,)},
            ),
        }

        # Run Validator
        split_debug_data_no_corrections = is_circuit_valid(
            circuit=split_circ_no_corrections,
            input_block=self.block_big,
            output_block=[self.block_left, self.block_right],
            output_stabilizers_parity=output_stabilizers_parity,
            output_stabilizers_with_any_value=[],
            logical_state_transformations_with_parity=logical_states_transformations_w_parity,
            logical_state_transformations=[],
            measurement_to_input_stabilizer_map={},
        )

        self.assertTrue(split_debug_data_no_corrections.valid)


if __name__ == "__main__":
    unittest.main()
