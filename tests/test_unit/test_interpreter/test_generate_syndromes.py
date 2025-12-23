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

from loom.interpreter import InterpretationStep, Syndrome
from loom.interpreter.applicator import generate_syndromes


def test_generate_syndromes_subtests(n_rsc_block_factory, subtests):
    """Run generate_syndromes with multiple scenarios using subtests."""

    rsc_block = n_rsc_block_factory(1)[0]

    x_stabs = [stab for stab in rsc_block.stabilizers if set(stab.pauli) == {"X"}]
    z_stabs = [stab for stab in rsc_block.stabilizers if set(stab.pauli) == {"Z"}]

    test_cases = [
        (
            "regular_syndrome",
            {
                "stabilizers": rsc_block.stabilizers,
                "block": rsc_block,
                "stab_measurements": [
                    tuple((f"c_{q}", 3) for q in stab.ancilla_qubits)
                    for stab in rsc_block.stabilizers
                ],
                "interpretation_step": InterpretationStep.create((rsc_block,)),
            },
            tuple(
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=tuple((f"c_{q}", 3) for q in stab.ancilla_qubits),
                    round=0,
                    block=rsc_block.uuid,
                )
                for stab in rsc_block.stabilizers
            ),
        ),
        (
            "previous_round",
            {
                "stabilizers": rsc_block.stabilizers,
                "block": rsc_block,
                "stab_measurements": [
                    tuple((f"c_{q}", 4) for q in stab.ancilla_qubits)
                    for stab in rsc_block.stabilizers
                ],
                "interpretation_step": InterpretationStep.create(
                    (rsc_block,),
                    block_qec_rounds={rsc_block.uuid: 7},
                    syndromes=tuple(
                        Syndrome(
                            stabilizer=stab.uuid,
                            measurements=tuple(
                                (f"c_{q}", 3) for q in stab.ancilla_qubits
                            ),
                            round=6,
                            block=rsc_block.uuid,
                        )
                        for stab in rsc_block.stabilizers
                    ),
                ),
            },
            tuple(
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=tuple((f"c_{q}", 4) for q in stab.ancilla_qubits),
                    round=7,
                    block=rsc_block.uuid,
                )
                for stab in rsc_block.stabilizers
            ),
        ),
        (
            "logical_measurement",
            {
                "stabilizers": z_stabs,
                "block": rsc_block,
                "stab_measurements": [
                    tuple((f"c_{q}", 0) for q in stab.data_qubits) for stab in z_stabs
                ],
                "interpretation_step": InterpretationStep.create(
                    (rsc_block,),
                    block_qec_rounds={rsc_block.uuid: 7},
                    syndromes=tuple(
                        Syndrome(
                            stabilizer=stab.uuid,
                            measurements=tuple(
                                (f"c_{q}", 3) for q in stab.ancilla_qubits
                            ),
                            round=6,
                            block=rsc_block.uuid,
                        )
                        for stab in rsc_block.stabilizers
                    ),
                ),
            },
            tuple(
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=tuple((f"c_{q}", 0) for q in stab.data_qubits),
                    round=7,
                    block=rsc_block.uuid,
                )
                for stab in z_stabs
            ),
        ),
        (
            "reset_case",
            {
                "stabilizers": x_stabs,
                "block": rsc_block,
                "stab_measurements": [tuple() for _ in x_stabs],
                "interpretation_step": InterpretationStep.create((rsc_block,)),
            },
            tuple(
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=tuple(),
                    round=0,
                    block=rsc_block.uuid,
                )
                for stab in x_stabs
            ),
        ),
        (
            "existing_stabilizer_updates",
            {
                "stabilizers": rsc_block.stabilizers,
                "block": rsc_block,
                "stab_measurements": [
                    tuple((f"c_{q}", 3) for q in stab.ancilla_qubits)
                    for stab in rsc_block.stabilizers
                ],
                "interpretation_step": InterpretationStep.create((rsc_block,)),
            },
            tuple(
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=tuple((f"c_{q}", 3) for q in stab.ancilla_qubits),
                    corrections=InterpretationStep.create(
                        (rsc_block,)
                    ).stabilizer_updates.get(stab.uuid, ()),
                    round=0,
                    block=rsc_block.uuid,
                )
                for stab in rsc_block.stabilizers
            ),
        ),
    ]

    for name, args, expected in test_cases:
        with subtests.test(msg=name):
            new_syndromes = generate_syndromes(**args)
            assert new_syndromes == expected
