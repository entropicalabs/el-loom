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

import pytest

from loom.eka.channel import Channel, ChannelType
from loom.eka.circuit import Circuit
from loom.executor import (
    EkaToStimConverter,
    EkaToGuppylangConverter,
    EkaToPennylaneConverter,
    EkaToQasmConverter,
    EkaToMimiqConverter,
    EkaToCudaqConverter,
)


class TestConverters:
    """Test basic error handling of all Eka to target language converters."""

    converters = [
        EkaToPennylaneConverter(is_catalyst=False),
        EkaToPennylaneConverter(is_catalyst=True),
        EkaToQasmConverter(),
        EkaToGuppylangConverter(),
        EkaToStimConverter(),
        EkaToMimiqConverter(),
        EkaToCudaqConverter(),
    ]

    def test_emit_circuit_wrong_input(self):
        """Test that emit raises TypeError for wrong input type."""
        for converter in self.converters:
            try:
                converter.emit_circuit_program("not_a_circuit", {}, {})
            except TypeError as e:
                assert str(e) == "Input must be a Circuit instance."
            else:
                assert False, "TypeError was not raised"

    def test_emit_init_wrong_input(self):
        """Test that _emit_init raises TypeError for wrong input type."""
        for converter in self.converters:
            try:
                converter.emit_init_instructions("not_a_circuit")
            except TypeError as e:
                assert str(e) == "Input must be a Circuit instance."
            else:
                assert False, "TypeError was not raised"

    def test_emit_leaf_instruction_wrong_input(self):
        """
        Test that emit_leaf_circuit_instruction raises TypeError for wrong input type.
        """
        for converter in self.converters:
            try:
                converter.emit_leaf_circuit_instruction("not_a_circuit", {}, {})
            except TypeError as e:
                assert str(e) == "Input must be a Circuit instance."
            else:
                assert False, "TypeError was not raised"

    def test_emit_wrong_instruction(self):
        """
        Test that emit_leaf_circuit_instruction raises ValueError for non-leaf circuit.
        """
        for converter in self.converters:
            chan = Channel(label="foo", type=ChannelType.CLASSICAL)
            with pytest.raises(ValueError):
                if isinstance(
                    converter,
                    (EkaToQasmConverter, EkaToMimiqConverter, EkaToCudaqConverter),
                ):
                    converter.emit_leaf_circuit_instruction(
                        Circuit(name="wrong", channels=[chan]),
                        {},
                        {chan: "boo"},
                    )
                else:
                    converter.emit_leaf_circuit_instruction(
                        Circuit(name="wrong", channels=[chan]),
                        {},
                        {chan.id: "boo"},
                    )
