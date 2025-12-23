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

import builtins
from functools import cached_property
import pytest
from hugr.qsystem.result import QsysResult, QsysShot

from loom.eka import Circuit, ChannelType
from loom.executor import EkaToGuppylangConverter


class TestEkaToGuppylangConverter:
    "Test the EKA to Guppylang converter"

    convert_expected_json = "convert_guppylang.json"
    emit_expected_json = "emit_guppylang.json"

    SUPP_CIRCUIT_FIXTURES = [
        "empty_circuit",
        "simple_circuit",
        "bell_state_circuit",
        "circuit_with_nested_circuits",
        "circuit_with_multiple_nested_levels",
        "circuit_w_all_single_qubit_ops",
        "circuit_w_all_two_qubit_ops",
        "circuit_w_all_measurement_ops",
        "circuit_w_all_reset_ops",
        "circuit_with_simple_if_else",
        "circuit_with_boolean_logic_condition",
        "circuit_w_nested_if_else",
        "circuit_with_boolean_logic_condition_multibit",
        "circuit_surface_code_experiment",
    ]

    @cached_property
    def converter(self) -> EkaToGuppylangConverter:
        """Return the EKA to Guppy Hugr converter instance."""
        return EkaToGuppylangConverter()

    @pytest.mark.parametrize("input_fixture", SUPP_CIRCUIT_FIXTURES, indirect=True)
    @pytest.mark.parametrize(
        "load_expected_data", [convert_expected_json], indirect=True
    )
    def test_using_generic_cases(self, input_fixture, load_expected_data):
        """Test the converter using generic circuit fixtures."""

        fixture_content, fixture_name = input_fixture

        # Convert the circuit using the converter
        result = self.converter.convert_circuit(fixture_content)

        circuit, q_map, c_map = result

        assert isinstance(circuit, str)
        assert isinstance(c_map, dict)
        assert isinstance(q_map, dict)

        assert set(c.id for c in fixture_content.channels if c.is_classical()).issubset(
            set(c_map.keys())
        ), "Not all classical channels mapped to a outcome register"

        assert set(c.id for c in fixture_content.channels if c.is_quantum()).issubset(
            set(q_map.keys())
        ), "Not all quantum channels mapped to a stim qubit register"

        # If we have expected output data for this fixture, perform detailed assertions
        if fixture_name not in load_expected_data:
            pytest.fail(
                f"Expected data for fixture '{fixture_name}' not found in expected "
                "data."
            )
        expected = load_expected_data[fixture_name]
        circuit_program = circuit.splitlines()

        assert len(circuit_program) == len(expected["program"]), (
            f"Program length mismatch for fixture '{fixture_name}': "
            f"expected {len(expected['program'])}, got {len(circuit_program)}"
        )
        for i, (got_line, expected_line) in enumerate(
            zip(circuit_program, expected["program"], strict=True)
        ):
            assert repr(got_line) == repr(expected_line), (
                f"Line {i} mismatch for fixture '{fixture_name}': expected"
                f" {expected_line}, got {got_line}"
            )

    def test_parsing_function(self):
        """Test the parsing function of the converter."""
        single_shot = QsysShot(entries=[("c0", 1)])
        # Parse the run outcome
        run_output = QsysResult(results=[single_shot for _ in range(5)])

        parsed_shot_outcome = self.converter.parse_target_run_outcome(single_shot)
        parsed_multi_shot_outcome = self.converter.parse_target_run_outcome(run_output)

        # Assert that the parsed outcome is as expected
        assert isinstance(parsed_shot_outcome, dict)
        assert isinstance(parsed_multi_shot_outcome, dict)

        assert parsed_shot_outcome == {
            "c0": [True]
        }, f"Unexpected parsed outcome: {parsed_shot_outcome}"

        assert parsed_multi_shot_outcome == {
            "c0": [True, True, True, True, True]
        }, f"Unexpected parsed outcome: {parsed_multi_shot_outcome}"

    @pytest.mark.parametrize("load_expected_data", [emit_expected_json], indirect=True)
    def test_emit_functions(
        self, circuit_to_init_and_instructions_to_append, load_expected_data, subtests
    ):
        """Test the emit functions of the converter."""
        # If we have expected output data for this fixture, perform detailed assertions
        circuit, instructions_to_append = circuit_to_init_and_instructions_to_append

        q_chan = [c.id for c in circuit.channels if c.type == ChannelType.QUANTUM]
        c_chan = [c.id for c in circuit.channels if c.type == ChannelType.CLASSICAL]

        expected = load_expected_data

        with subtests.test(msg="Checking emit initialisation", case_id="init"):
            str_init, qreg_map, creg_map = self.converter.emit_init_instructions(
                circuit
            )
            str_mismatch_msg = (
                f"Initialization string does not match:\n"
                f"    - got      : {str_init}\n"
                f"    - expected : {expected['str_init']}"
            )
            assert str_init == expected["str_init"], str_mismatch_msg
            assert all(
                k in qreg_map for k in q_chan
            ), "Not all channels are present in the quantum map"
            assert all(
                k in creg_map for k in c_chan
            ), "Not all channels are present in the classical map"

        for new_instruction_name, new_instruction in instructions_to_append.items():
            if not isinstance(new_instruction, Circuit):
                pytest.fail(
                    "Instruction to append must be a "
                    f"Circuit, got {type(new_instruction)}"
                )
            with subtests.test(
                msg=f"Checking emit of {new_instruction_name} instruction",
                case_id=new_instruction_name,
            ):
                instruction_expected = expected[new_instruction_name]
                if instruction_expected["success"]:
                    str_inst = self.converter.emit_leaf_circuit_instruction(
                        new_instruction, qreg_map, creg_map
                    )
                    inst_mismatch_msg = (
                        "Instruction string does"
                        f" not match for {new_instruction_name}:\n"
                        f"    - got      : {str_inst}\n"
                        f"    - expected : {instruction_expected['str_inst']}"
                    )
                    assert (
                        str_inst == instruction_expected["str_inst"]
                    ), inst_mismatch_msg
                else:
                    with pytest.raises(
                        getattr(builtins, instruction_expected["expected_exception"])
                    ):
                        self.converter.emit_leaf_circuit_instruction(
                            new_instruction, qreg_map, creg_map
                        )
