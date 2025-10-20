"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

# pylint: disable= duplicate-code
import unittest

from loom.eka import Lattice, Stabilizer, Block, PauliOperator
from loom.interpreter import InterpretationStep, Syndrome, Detector
from loom.interpreter.applicator import generate_detectors


# pylint: disable=too-many-instance-attributes
class TestGenerateDetectors(unittest.TestCase):
    """Tests for the generate_detectors function."""

    def setUp(self):
        self.square_2d_lattice = Lattice.square_2d((10, 20))
        self.rot_surf_code_1 = Block(
            stabilizers=(
                Stabilizer(
                    pauli="ZZZZ",
                    data_qubits=((1, 0, 0), (1, 1, 0), (0, 0, 0), (0, 1, 0)),
                    ancilla_qubits=((1, 1, 1),),
                ),
                Stabilizer(
                    pauli="ZZZZ",
                    data_qubits=((2, 1, 0), (2, 2, 0), (1, 1, 0), (1, 2, 0)),
                    ancilla_qubits=((2, 2, 1),),
                ),
                Stabilizer(
                    pauli="XXXX",
                    data_qubits=((1, 1, 0), (0, 1, 0), (1, 2, 0), (0, 2, 0)),
                    ancilla_qubits=((1, 2, 1),),
                ),
                Stabilizer(
                    pauli="XXXX",
                    data_qubits=((2, 0, 0), (1, 0, 0), (2, 1, 0), (1, 1, 0)),
                    ancilla_qubits=((2, 1, 1),),
                ),
                Stabilizer(
                    pauli="XX",
                    data_qubits=((0, 0, 0), (0, 1, 0)),
                    ancilla_qubits=((0, 1, 1),),
                ),
                Stabilizer(
                    pauli="XX",
                    data_qubits=((2, 1, 0), (2, 2, 0)),
                    ancilla_qubits=((3, 2, 1),),
                ),
                Stabilizer(
                    pauli="ZZ",
                    data_qubits=((2, 0, 0), (1, 0, 0)),
                    ancilla_qubits=((2, 0, 1),),
                ),
                Stabilizer(
                    pauli="ZZ",
                    data_qubits=((1, 2, 0), (0, 2, 0)),
                    ancilla_qubits=((1, 3, 1),),
                ),
            ),
            logical_x_operators=[
                PauliOperator(
                    pauli="XXX", data_qubits=((0, 0, 0), (1, 0, 0), (2, 0, 0))
                )
            ],
            logical_z_operators=[
                PauliOperator(
                    pauli="ZZZ", data_qubits=((0, 0, 0), (0, 1, 0), (0, 2, 0))
                )
            ],
            unique_label="q1",
        )
        self.rot_surf_code_2 = self.rot_surf_code_1.shift((4, 0))
        # We create artificial syndromes to test the detector generation
        self.general_syndromes = tuple(
            Syndrome(
                stabilizer=stab.uuid,
                measurements=[(f"c_{q}", i) for q in stab.ancilla_qubits],
                round=i - 2,
                block=self.rot_surf_code_1.uuid,
            )
            for stab in self.rot_surf_code_1.stabilizers
            for i in range(4, 6)
        )
        self.max_general_syndromes_round = 3  # The maximum round of syndromes created
        self.int_step_general_syndromes = InterpretationStep(
            block_history=((self.rot_surf_code_1,),),
            syndromes=self.general_syndromes,
        )

        # Create artificial syndromes for a (+)reset-like operation
        self.reset_syndromes = tuple(
            Syndrome(
                stabilizer=stab.uuid,
                measurements=tuple(),
                round=-1,
                block=self.rot_surf_code_1.uuid,
            )
            for stab in self.rot_surf_code_1.stabilizers
            if set(stab.pauli) == {"X"}
        )
        self.int_step_reset_syndromes = InterpretationStep(
            block_history=((self.rot_surf_code_1,),),
            syndromes=self.reset_syndromes,
        )

        # Examples
        new_syndromes = tuple(
            Syndrome(
                stabilizer=stab.uuid,
                measurements=[(f"c_{q}", 10) for q in stab.ancilla_qubits],
                round=6,
                block=self.rot_surf_code_1.uuid,
            )
            for stab in self.rot_surf_code_1.stabilizers
        )
        new_x_syndromes = tuple(
            syndrome
            for syndrome in new_syndromes
            if set(
                self.int_step_reset_syndromes.stabilizers_dict[
                    syndrome.stabilizer
                ].pauli
            )
            == {"X"}
        )

        # Create the final round of syndromes (Z)measurement like
        final_round_syndromes = tuple(
            Syndrome(
                stabilizer=stab.uuid,
                measurements=[(f"c_{q}", 0) for q in stab.data_qubits],
                round=0,
                block=self.rot_surf_code_1.uuid,
            )
            for stab in self.rot_surf_code_1.stabilizers
            if set(stab.pauli) == {"Z"}
        )
        general_z_syndromes = tuple(
            syndrome
            for syndrome in self.int_step_general_syndromes.syndromes[
                1::2
            ]  # Only latest syndromes
            if set(
                self.int_step_general_syndromes.stabilizers_dict[
                    syndrome.stabilizer
                ].pauli
            )
            == {"Z"}
        )

        self.args_and_expected_detectors = (
            (  # Simple pairing in time (regular syndrome measurements)
                {
                    "old_syndromes": tuple(
                        self.int_step_general_syndromes.get_prev_syndrome(
                            syndrome.stabilizer, syndrome.block
                        )
                        for syndrome in new_syndromes
                    ),
                    "new_syndromes": new_syndromes,
                    "interpretation_step": self.int_step_general_syndromes,
                },
                tuple(
                    Detector(
                        syndromes=(
                            past_syndrome,
                            new_syndrome,
                        ),
                    )
                    # Only the latest syndromes matter
                    for past_syndrome, new_syndrome in zip(
                        self.general_syndromes[1::2], new_syndromes, strict=True
                    )
                ),
            ),
            (  # Pairing in space with empty syndromes (reset-like operation)
                {
                    "old_syndromes": tuple(
                        self.int_step_reset_syndromes.get_prev_syndrome(
                            syndrome.stabilizer, syndrome.block
                        )
                        for syndrome in new_syndromes
                    ),
                    "new_syndromes": new_syndromes,
                    "interpretation_step": self.int_step_reset_syndromes,
                },
                tuple(
                    Detector(
                        syndromes=(
                            past_syndrome,
                            new_syndrome,
                        ),
                    )
                    # Only the latest syndromes matter
                    for past_syndrome, new_syndrome in zip(
                        self.reset_syndromes, new_x_syndromes, strict=True
                    )
                ),
            ),
            (
                # Pairing in space for a (Z)measurement-like operation
                {
                    "old_syndromes": tuple(
                        self.int_step_general_syndromes.get_prev_syndrome(
                            syndrome.stabilizer, syndrome.block
                        )
                        for syndrome in final_round_syndromes
                    ),
                    "new_syndromes": final_round_syndromes,
                    "interpretation_step": self.int_step_general_syndromes,
                },
                tuple(
                    Detector(
                        syndromes=(
                            past_syndrome,
                            new_syndrome,
                        ),
                    )
                    # Only the latest z syndromes matter
                    for past_syndrome, new_syndrome in zip(
                        general_z_syndromes, final_round_syndromes, strict=True
                    )
                ),
            ),
        )

    def test_generate_detectors(self):
        """Test the generation of detectors."""

        for args, expected_detectors in self.args_and_expected_detectors:
            int_step = args["interpretation_step"]
            new_syndromes = args["new_syndromes"]
            new_detectors = generate_detectors(int_step, new_syndromes)
            self.assertEqual(new_detectors, expected_detectors)

    def test_detector_generation_for_evolved_stabilizers(self):
        """Test detector generation for multiple stabilizers evolving into another
        stabilizer in a block evolution.
        """
        n_stabilizers_evolved = 5
        int_step = InterpretationStep(
            # Block 1 gets transformed into block 2
            block_history=((self.rot_surf_code_1,), (self.rot_surf_code_2,)),
            block_evolution={
                self.rot_surf_code_2.uuid: (self.rot_surf_code_1.uuid,),
            },
            # The first 5 stabilizers of the first block evolve into the first
            # n_stabilizers_evolved stabilizers of the second block
            stabilizer_evolution={
                self.rot_surf_code_2.stabilizers[0].uuid: tuple(
                    self.rot_surf_code_1.stabilizers[s_idx].uuid
                    for s_idx in range(n_stabilizers_evolved)
                )
            },
            syndromes=self.general_syndromes,
        )

        # Create a new syndrome for the first stabilizer of the second block
        new_syndrome = Syndrome(
            stabilizer=self.rot_surf_code_2.stabilizers[0].uuid,
            measurements=[
                (f"c_{self.rot_surf_code_2.stabilizers[0].ancilla_qubits[0]}", 0)
            ],
            round=0,
            block=self.rot_surf_code_2.uuid,
        )

        # Generate the detectors for the new syndrome
        detectors = generate_detectors(int_step, (new_syndrome,))

        # Ensure that only one detector is created
        # and its size should be n_stabilizers_evolved + 1, containing the new syndrome
        # along with the last round syndromes of the evolved stabilizers
        self.assertEqual(len(detectors), 1)
        self.assertEqual(len(detectors[0].syndromes), n_stabilizers_evolved + 1)
        expected_syndromes = [
            synd
            for synd in self.general_syndromes
            if synd.stabilizer
            in int_step.stabilizer_evolution[self.rot_surf_code_2.stabilizers[0].uuid]
            and synd.round == self.max_general_syndromes_round
        ] + [new_syndrome]
        self.assertEqual(set(detectors[0].syndromes), set(expected_syndromes))


if __name__ == "__main__":
    unittest.main()
