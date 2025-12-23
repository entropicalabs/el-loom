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

from loom.interpreter import InterpretationStep, Syndrome, Detector
from loom.interpreter.applicator import generate_detectors


# pylint: disable=too-many-instance-attributes
class TestGenerateDetectors:
    """Tests for the generate_detectors function."""

    # We create artificial syndromes to test the detector generation

    def test_generate_detectors(self, n_rsc_block_factory, subtests):
        """Test the generation of detectors."""

        rsc_block = n_rsc_block_factory(1)[0]

        initial_syndromes = tuple(
            Syndrome(
                stabilizer=stab.uuid,
                measurements=[(f"c_{q}", i) for q in stab.ancilla_qubits],
                round=i - 2,
                block=rsc_block.uuid,
            )
            for stab in rsc_block.stabilizers
            for i in range(4, 6)
        )

        int_step_general_syndromes = InterpretationStep.create(
            (rsc_block,),
            syndromes=initial_syndromes,
        )

        new_syndromes = tuple(
            Syndrome(
                stabilizer=stab.uuid,
                measurements=[(f"c_{q}", 10) for q in stab.ancilla_qubits],
                round=6,
                block=rsc_block.uuid,
            )
            for stab in rsc_block.stabilizers
        )

        # Create artificial syndromes for a (+)reset-like operation
        reset_syndromes = tuple(
            Syndrome(
                stabilizer=stab.uuid,
                measurements=tuple(),
                round=-1,
                block=rsc_block.uuid,
            )
            for stab in rsc_block.stabilizers
            if set(stab.pauli) == {"X"}
        )

        int_step_reset_syndromes = InterpretationStep.create(
            (rsc_block,),
            syndromes=reset_syndromes,
        )

        new_x_syndromes = tuple(
            syndrome
            for syndrome in new_syndromes
            if set(int_step_reset_syndromes.stabilizers_dict[syndrome.stabilizer].pauli)
            == {"X"}
        )

        # Create the final round of syndromes (Z)measurement like
        final_round_syndromes = tuple(
            Syndrome(
                stabilizer=stab.uuid,
                measurements=[(f"c_{q}", 0) for q in stab.data_qubits],
                round=0,
                block=rsc_block.uuid,
            )
            for stab in rsc_block.stabilizers
            if set(stab.pauli) == {"Z"}
        )

        general_z_syndromes = tuple(
            syndrome
            for syndrome in int_step_general_syndromes.syndromes[
                1::2
            ]  # Only latest syndromes
            if set(
                int_step_general_syndromes.stabilizers_dict[syndrome.stabilizer].pauli
            )
            == {"Z"}
        )

        test_cases = [
            (
                "regular_syndrome_measurement",
                {
                    "old_syndromes": tuple(
                        int_step_general_syndromes.get_prev_syndrome(
                            syndrome.stabilizer, syndrome.block
                        )
                        for syndrome in new_syndromes
                    ),
                    "new_syndromes": new_syndromes,
                    "interpretation_step": int_step_general_syndromes,
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
                        initial_syndromes[1::2], new_syndromes, strict=True
                    )
                ),
            ),
            (
                "reset_like_syndrome_measurement",
                {
                    "old_syndromes": tuple(
                        int_step_reset_syndromes.get_prev_syndrome(
                            syndrome.stabilizer, syndrome.block
                        )
                        for syndrome in new_syndromes
                    ),
                    "new_syndromes": new_syndromes,
                    "interpretation_step": int_step_reset_syndromes,
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
                        reset_syndromes, new_x_syndromes, strict=True
                    )
                ),
            ),
            (
                "z_measurement_like_operation",
                {
                    "old_syndromes": tuple(
                        int_step_general_syndromes.get_prev_syndrome(
                            syndrome.stabilizer, syndrome.block
                        )
                        for syndrome in final_round_syndromes
                    ),
                    "new_syndromes": final_round_syndromes,
                    "interpretation_step": int_step_general_syndromes,
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
        ]

        for name, args, expected_detectors in test_cases:
            with subtests.test(msg=name):
                int_step = args["interpretation_step"]
                new_syndromes = args["new_syndromes"]
                new_detectors = generate_detectors(int_step, new_syndromes)
                assert new_detectors == expected_detectors

    def test_detector_generation_for_evolved_stabilizers(self, n_rsc_block_factory):
        """Test detector generation for multiple stabilizers evolving into another
        stabilizer in a block evolution.
        """
        rsc_blocks = n_rsc_block_factory(2)

        initial_syndromes = tuple(
            Syndrome(
                stabilizer=stab.uuid,
                measurements=[(f"c_{q}", i) for q in stab.ancilla_qubits],
                round=i - 2,
                block=rsc_blocks[0].uuid,
            )
            for stab in rsc_blocks[0].stabilizers
            for i in range(4, 6)
        )

        n_stabilizers_evolved = 5
        # Create with first block, then update to second block to set up evolution
        int_step = InterpretationStep.create(
            (rsc_blocks[0],),
            syndromes=initial_syndromes,
        )
        # Update block history to include evolution from block 0 to block 1
        int_step.update_block_history_and_evolution_MUT(
            new_blocks=(rsc_blocks[1],),
            old_blocks=(rsc_blocks[0],),
            update_evolution=True,
        )
        # Set up stabilizer evolution for testing
        int_step.stabilizer_evolution[rsc_blocks[1].stabilizers[0].uuid] = tuple(
            rsc_blocks[0].stabilizers[s_idx].uuid
            for s_idx in range(n_stabilizers_evolved)
        )

        # Create a new syndrome for the first stabilizer of the second block
        new_syndrome = Syndrome(
            stabilizer=rsc_blocks[1].stabilizers[0].uuid,
            measurements=[(f"c_{rsc_blocks[1].stabilizers[0].ancilla_qubits[0]}", 0)],
            round=0,
            block=rsc_blocks[1].uuid,
        )

        # Generate the detectors for the new syndrome
        detectors = generate_detectors(int_step, (new_syndrome,))

        # Ensure that only one detector is created
        # and its size should be n_stabilizers_evolved + 1, containing the new syndrome
        # along with the last round syndromes of the evolved stabilizers
        assert len(detectors) == 1
        assert len(detectors[0].syndromes) == n_stabilizers_evolved + 1
        expected_syndromes = [
            synd
            for synd in initial_syndromes
            if synd.stabilizer
            in int_step.stabilizer_evolution[rsc_blocks[1].stabilizers[0].uuid]
            and synd.round == 3
        ] + [new_syndrome]
        assert set(detectors[0].syndromes) == set(expected_syndromes)
