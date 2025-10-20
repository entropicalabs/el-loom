"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

# pylint: disable= duplicate-code
import unittest
from copy import deepcopy

from loom.eka import Lattice, Block, Stabilizer, PauliOperator
from loom.interpreter import InterpretationStep, Syndrome
from loom.interpreter.applicator import generate_syndromes


class TestGenerateSyndromes(unittest.TestCase):
    """Tests for the generate_syndromes function."""

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
        self.grown_rot_surf_code_1 = Block(
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
                    pauli="ZZZZ",
                    data_qubits=((3, 0, 0), (3, 1, 0), (2, 0, 0), (2, 1, 0)),
                    ancilla_qubits=((3, 1, 1),),
                ),
                Stabilizer(
                    pauli="ZZZZ",
                    data_qubits=((4, 1, 0), (4, 2, 0), (3, 1, 0), (3, 2, 0)),
                    ancilla_qubits=((4, 2, 1),),
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
                    pauli="XXXX",
                    data_qubits=((3, 1, 0), (2, 1, 0), (3, 2, 0), (2, 2, 0)),
                    ancilla_qubits=((3, 2, 1),),
                ),
                Stabilizer(
                    pauli="XXXX",
                    data_qubits=((4, 0, 0), (3, 0, 0), (4, 1, 0), (3, 1, 0)),
                    ancilla_qubits=((4, 1, 1),),
                ),
                Stabilizer(
                    pauli="XX",
                    data_qubits=((0, 0, 0), (0, 1, 0)),
                    ancilla_qubits=((0, 1, 1),),
                ),
                Stabilizer(
                    pauli="XX",
                    data_qubits=((4, 1, 0), (4, 2, 0)),
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
                Stabilizer(
                    pauli="ZZ",
                    data_qubits=((3, 0, 0), (4, 0, 0)),
                    ancilla_qubits=((4, 0, 1),),
                ),
                Stabilizer(
                    pauli="ZZ",
                    data_qubits=((3, 2, 0), (2, 2, 0)),
                    ancilla_qubits=((3, 3, 1),),
                ),
            ),
            logical_x_operators=[
                PauliOperator(
                    pauli="XXXXX",
                    data_qubits=((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)),
                )
            ],
            logical_z_operators=[
                PauliOperator(
                    pauli="ZZZ", data_qubits=((0, 0, 0), (0, 1, 0), (0, 2, 0))
                )
            ],
            unique_label="q1",
        )
        self.base_step = InterpretationStep(
            block_history=((self.rot_surf_code_1,),),
        )
        self.step_with_syndromes = InterpretationStep(
            block_history=((self.rot_surf_code_1,),),
            block_qec_rounds={
                self.rot_surf_code_1.uuid: 7
            },  # Manually set the round (round 6 was already performed)
            syndromes=tuple(
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=tuple((f"c_{q}", 3) for q in stab.ancilla_qubits),
                    round=6,
                    block=self.rot_surf_code_1.uuid,
                )
                for stab in self.rot_surf_code_1.stabilizers
            ),
        )

        # Only for rsc plugin
        self.step_with_stab_updates = InterpretationStep(
            block_history=((self.rot_surf_code_1,),),
            stabilizer_updates={  # Dummy updates
                # stab.uuid: ((f"c_(10, {i}, 0)", 5),)
                # for i, stab in enumerate(
                #     self.rot_surf_code_1.boundary_stabilizers("right")
                # )
            },
        )

    def test_generate_syndromes(self):
        """Test the generation of syndromes for different cases:
        1. Regular syndrome generation for all stabilizers
        2. Syndrome generation for stabilizers with previous round
        3. Syndrome generation for stabilizers with previous round and data_qubits
           measurements of a single type (logical measurement case)
        4. Syndrome generation with empty measurements for a single type of stabilizers
           (reset case)
        5. Syndrome generation with existing stabilizer updates (e.g. shrink)
        """
        args_and_expected_syndromes = (
            (  # Regular Syndrome generation for all stabs
                {
                    "stabilizers": self.rot_surf_code_1.stabilizers,
                    "block": self.rot_surf_code_1,
                    "stab_measurements": [
                        tuple((f"c_{q}", 3) for q in stab.ancilla_qubits)
                        for stab in self.rot_surf_code_1.stabilizers
                    ],
                    "interpretation_step": self.base_step,
                },
                tuple(
                    Syndrome(
                        stabilizer=stab.uuid,
                        measurements=tuple((f"c_{q}", 3) for q in stab.ancilla_qubits),
                        round=0,
                        block=self.rot_surf_code_1.uuid,
                    )
                    for stab in self.rot_surf_code_1.stabilizers
                ),
            ),
            (  # Syndrome generation for stabilizers with previous round
                {
                    "stabilizers": self.rot_surf_code_1.stabilizers,
                    "block": self.rot_surf_code_1,
                    "stab_measurements": [
                        tuple((f"c_{q}", 4) for q in stab.ancilla_qubits)
                        for stab in self.rot_surf_code_1.stabilizers
                    ],
                    "interpretation_step": deepcopy(self.step_with_syndromes),
                },
                tuple(
                    Syndrome(
                        stabilizer=stab.uuid,
                        measurements=tuple((f"c_{q}", 4) for q in stab.ancilla_qubits),
                        round=7,  # Is correctly incremented
                        block=self.rot_surf_code_1.uuid,
                    )
                    for stab in self.rot_surf_code_1.stabilizers
                ),
            ),
            (  # Syndrome generation for stabilizers with previous round and
                # data_qubits measurements of a single type (logical measurement case)
                {
                    "stabilizers": (
                        z_stabs := [
                            stab
                            for stab in self.rot_surf_code_1.stabilizers
                            if set(stab.pauli) == {"Z"}
                        ]
                    ),
                    "block": self.rot_surf_code_1,
                    "stab_measurements": [
                        tuple((f"c_{q}", 0) for q in stab.data_qubits)
                        for stab in z_stabs
                    ],
                    "interpretation_step": deepcopy(self.step_with_syndromes),
                },
                tuple(
                    Syndrome(
                        stabilizer=stab.uuid,
                        measurements=tuple((f"c_{q}", 0) for q in stab.data_qubits),
                        round=7,  # Is correctly incremented
                        block=self.rot_surf_code_1.uuid,
                    )
                    for stab in z_stabs
                ),
            ),
            (  # Syndrome generation with empty measurements for a single type of
                # stabilizers (reset case)
                {
                    "stabilizers": (
                        x_stabs := [
                            stab
                            for stab in self.rot_surf_code_1.stabilizers
                            if set(stab.pauli) == {"X"}
                        ]
                    ),
                    "block": self.rot_surf_code_1,
                    "stab_measurements": [tuple() for _ in x_stabs],
                    "interpretation_step": deepcopy(self.base_step),
                },
                tuple(
                    Syndrome(
                        stabilizer=stab.uuid,
                        measurements=tuple(),
                        round=0,  # Is correctly incremented
                        block=self.rot_surf_code_1.uuid,
                    )
                    for stab in x_stabs
                ),
            ),
            (  # Syndrome generation with existing stabilizer updates (e.g. shrink)
                {
                    "stabilizers": self.rot_surf_code_1.stabilizers,
                    "block": self.rot_surf_code_1,
                    "stab_measurements": [
                        tuple((f"c_{q}", 3) for q in stab.ancilla_qubits)
                        for stab in self.rot_surf_code_1.stabilizers
                    ],
                    "interpretation_step": self.step_with_stab_updates,
                },
                tuple(
                    Syndrome(
                        stabilizer=stab.uuid,
                        measurements=tuple((f"c_{q}", 3) for q in stab.ancilla_qubits),
                        corrections=self.step_with_stab_updates.stabilizer_updates.get(
                            stab.uuid, ()
                        ),
                        round=0,
                        block=self.rot_surf_code_1.uuid,
                    )
                    for stab in self.rot_surf_code_1.stabilizers
                ),
            ),
        )

        for args, expected_syndromes in args_and_expected_syndromes:
            new_syndromes = generate_syndromes(**args)
            self.assertEqual(new_syndromes, expected_syndromes)


if __name__ == "__main__":
    unittest.main()
