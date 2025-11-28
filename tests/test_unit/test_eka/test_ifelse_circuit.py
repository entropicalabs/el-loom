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

from loom.eka import IfElseCircuit, Circuit, Channel, ChannelType
from loom.eka.utilities import BoolOp


# pylint: disable=unnecessary-lambda-assignment,redefined-outer-name


@pytest.fixture(scope="module")
def _example_circuits():
    """Fixture to provide example circuits for testing. This is a private method that is
    used together with the circ fixture. The scope is broadened to 'session' to avoid
    re-creation of circuits for each test.

    To add a new example:
        1. Decide its nesting level (e.g. "lvl2_ccf_circuits").
        2. Insert it into the appropriate dict with a unique key.
        3. Add key to the circ fixture params list.
    """
    q_channels = [Channel(type=ChannelType.QUANTUM, label=f"q{i}") for i in range(5)]
    c_channels = [Channel(type=ChannelType.CLASSICAL, label=f"c{i}") for i in range(6)]
    # Base Gates
    one = Circuit("x", duration=1, channels=[q_channels[0]])
    two = Circuit("y", duration=2, channels=[q_channels[1]])
    three = Circuit("z", duration=3, channels=[q_channels[2]])
    five = Circuit("measurement", duration=5, channels=[c_channels[0], q_channels[3]])
    ten = Circuit("cnot", duration=10, channels=[q_channels[1], q_channels[4]])

    # Circuits
    circ1 = Circuit("circ1", [one, two, three])
    circ2 = Circuit("circ2", [five, ten])

    ## If Else Circuits
    if_circ1 = IfElseCircuit(
        if_circuit=one,
        else_circuit=two,
        condition_circuit=Circuit(name=BoolOp.MATCH, channels=[c_channels[0]]),
    )
    if_circ2 = IfElseCircuit(
        if_circuit=circ1,
        else_circuit=circ2,
        condition_circuit=Circuit(name=BoolOp.MATCH, channels=[c_channels[1]]),
    )

    ## Nested If Else Circuits
    nested_if_circ1 = IfElseCircuit(
        if_circuit=circ1,
        else_circuit=if_circ2,
        condition_circuit=Circuit(name=BoolOp.MATCH, channels=[c_channels[2]]),
    )
    nested_if_circ2 = IfElseCircuit(
        if_circuit=if_circ1,
        else_circuit=if_circ2,
        condition_circuit=Circuit(name=BoolOp.MATCH, channels=[c_channels[3]]),
    )

    ## Double Nested If Else Circuits
    double_nested_if_circ1 = IfElseCircuit(
        if_circuit=if_circ1,
        else_circuit=nested_if_circ2,
        condition_circuit=Circuit(name=BoolOp.MATCH, channels=[c_channels[4]]),
    )
    double_nested_if_circ2 = IfElseCircuit(
        if_circuit=nested_if_circ1,
        else_circuit=nested_if_circ2,
        condition_circuit=Circuit(name=BoolOp.MATCH, channels=[c_channels[5]]),
    )

    return {
        "if_circ1": if_circ1,
        "if_circ2": if_circ2,
        "nested_if_circ1": nested_if_circ1,
        "nested_if_circ2": nested_if_circ2,
        "double_nested_if_circ1": double_nested_if_circ1,
        "double_nested_if_circ2": double_nested_if_circ2,
    }


@pytest.fixture(
    params=[
        ("if_circ1"),
        ("if_circ2"),
        ("nested_if_circ1"),
        ("nested_if_circ2"),
        ("double_nested_if_circ1"),
        ("double_nested_if_circ2"),
    ],
    ids=str,
    name="circ",
)
def fixture_circ(request, _example_circuits):
    """Fixture to return parametrized circuit examples. This structure allows us to
    avoid creating a new fixture for each circuit example."""
    name = request.param
    return _example_circuits[name]


@pytest.fixture
def get_circ(_example_circuits):
    """fixture to get one of the example circuits by name."""

    def _get(name: str):
        return _example_circuits[name]

    return _get


@pytest.fixture(name="err_msgs", scope="class")
def fixture_cond_circuit_err_msgs():
    """Fixture to provide error message lambdas for condition circuit validation tests.
    Each lambda takes the relevant parameters to format the expected error message.
    """
    return {
        "wrong_num_2bit": (
            lambda b_op: f"Condition circuit with BoolOp '{b_op}' must have at least "
            "two classical channels."
        ),
        "wrong_num_1bit": (
            lambda b_op: f"Condition circuit with BoolOp '{b_op}' must have only "
            "one classical channel."
        ),
        "invalid_bool_op": (
            lambda b_op: f"Unsupported BoolOp '{b_op}' for condition circuit. "
            "Supported BoolOps are: "
            f"{', '.join(BoolOp.multi_bit_list() + BoolOp.mono_bit_list())}."
        ),
        "non_classical": (
            lambda _: "IfElseCircuit `condition_circuit` must be a circuit with "
            "classical channels only."
        ),
    }


class TestIfElseCircuit:
    """Test suite for IfElseCircuit class."""

    def test_empty_ifelse_circuit(self):
        """Test attributes of an empty IfElseCircuit."""
        c = IfElseCircuit()

        # basic structural expectations
        assert isinstance(c, IfElseCircuit)
        assert isinstance(c.if_circuit, Circuit)
        assert isinstance(c.else_circuit, Circuit)
        assert isinstance(c.condition_circuit, Circuit)
        assert isinstance(c.id, str)

        assert c.name == "if-else_circuit"
        assert c.if_circuit.name == "empty_branch"
        assert c.else_circuit.name == "empty_branch"
        assert c.condition_circuit.name == BoolOp.MATCH
        assert c.circuit == ((c.if_circuit, c.else_circuit),)
        assert set(c.channels) == set(
            c.condition_circuit.channels
            + c.if_circuit.channels
            + c.else_circuit.channels
        )
        assert c.duration == max(c.if_circuit.duration, c.else_circuit.duration)
        assert all(
            channel.type == ChannelType.CLASSICAL
            for channel in c.condition_circuit.channels
        )

    @pytest.mark.parametrize(
        "msg_key, name, channels",
        [
            ("non_classical", BoolOp.MATCH, [Channel(type=ChannelType.QUANTUM)]),
            ("invalid_bool_op", "bazinga", [Channel(type=ChannelType.CLASSICAL)]),
            ("wrong_num_2bit", BoolOp.AND, [Channel(type=ChannelType.CLASSICAL)]),
            ("wrong_num_2bit", BoolOp.OR, [Channel(type=ChannelType.CLASSICAL)]),
            ("wrong_num_2bit", BoolOp.XOR, [Channel(type=ChannelType.CLASSICAL)]),
            (
                "wrong_num_1bit",
                BoolOp.MATCH,
                [
                    Channel(type=ChannelType.CLASSICAL),
                    Channel(type=ChannelType.CLASSICAL),
                ],
            ),
        ],
    )
    def test_condition_circuit_validation(self, msg_key, name, channels, err_msgs):
        """Test condition_circuit validation."""
        with pytest.raises(ValueError) as exc_info:
            wrong_circuit = Circuit(name=name, channels=channels)
            IfElseCircuit(condition_circuit=wrong_circuit)
        assert err_msgs[msg_key](name) in str(exc_info.value)

    @pytest.mark.parametrize(
        "method_name, expected_msg",
        [
            ("as_gate", "IfElseCircuit cannot be represented as a gate."),
            (
                "circuit_seq",
                "IfElseCircuit cannot be converted into a Circuit sequence.",
            ),
            (
                "construct_padded_circuit_time_sequence",
                "IfElseCircuit cannot construct a padded circuit time sequence.",
            ),
        ],
        ids=[
            "as_gate",
            "circuit_seq",
            "construct_padded_circuit_time_sequence",
        ],
    )
    def test_method_overrides(self, method_name, expected_msg):
        """Parametrized test for methods that must raise NotImplementedError.

        This approach avoids duplicating the try/except blocks and makes it
        straightforward to add more methods later.
        """
        c = IfElseCircuit()
        method = getattr(c, method_name)
        with pytest.raises(NotImplementedError) as exc_info:
            method()
        assert expected_msg in str(exc_info.value)

    @pytest.mark.parametrize(
        "keep_channels",
        [
            True,
            False,
        ],
        ids=[
            "keep_channels",
            "drop_channels",
        ],
    )
    def test_clone_method(self, circ, keep_channels):
        """Test the clone method of IfElseCircuit."""
        if keep_channels:
            cloned = circ.clone(circ.channels)
        else:
            new_channels = [
                (
                    Channel(type=ChannelType.CLASSICAL)
                    if channel.is_classical()
                    else Channel(type=ChannelType.QUANTUM)
                )
                for channel in circ.channels
            ]
            cloned = circ.clone(new_channels)

        # Properties that are true for all clones
        assert isinstance(cloned, IfElseCircuit)
        assert cloned.id != circ.id
        assert cloned.duration == circ.duration

        assert cloned == circ
        assert cloned.circuit == circ.circuit
        assert cloned.if_circuit == circ.if_circuit
        assert cloned.else_circuit == circ.else_circuit
        assert cloned.condition_circuit == circ.condition_circuit

        # Properties that depend on keep_channels
        is_same_channels = cloned.channels == circ.channels
        assert is_same_channels == keep_channels
        is_same_channels_in_if = cloned.if_circuit.channels == circ.if_circuit.channels
        assert is_same_channels_in_if == keep_channels
        is_same_channels_in_else = (
            cloned.else_circuit.channels == circ.else_circuit.channels
        )
        assert is_same_channels_in_else == keep_channels
        is_same_channels_in_condition = (
            cloned.condition_circuit.channels == circ.condition_circuit.channels
        )
        assert is_same_channels_in_condition == keep_channels

    def test_nr_of_qubits_in_circuit(self, circ):
        """Test the nr_of_qubits_in_circuit method."""
        num_q = len([True for channel in circ.channels if channel.is_quantum()])
        assert num_q == circ.nr_of_qubits_in_circuit()

    def test_unroll_method(self, circ):
        """Test the unroll class method of IfElseCircuit."""
        unrolled_circ = IfElseCircuit.unroll(circ)[0]

        unrolled_if = unrolled_circ.if_circuit
        unrolled_else = unrolled_circ.else_circuit
        unrolled_condition = unrolled_circ.condition_circuit

        assert isinstance(unrolled_if, Circuit)
        assert isinstance(unrolled_else, Circuit)
        assert isinstance(unrolled_condition, Circuit)

        assert unrolled_if.name == circ.if_circuit.name
        assert unrolled_else.name == circ.else_circuit.name
        assert unrolled_condition.name == circ.condition_circuit.name

    def test_repr_and_str_method(self, circ):
        """
        Test the __repr__ and __str__ methods of IfElseCircuit. Since no __str__ is
        defined in IfElseCircuit, we expect __repr__ and __str__ to return the same
        output.
        """
        circ_repr = repr(circ)
        circ_str = str(circ)

        expected_output = (
            f"{circ.name}\n"
            f"  if: {circ.if_circuit.name}\n"
            f"  else: {circ.else_circuit.name}\n"
            f"  condition: {circ.condition_circuit.name}"
        )
        assert circ_repr == circ_str == expected_output

    def test_detailed_str(self, circ):
        """Test the detailed_str method of IfElseCircuit."""
        detail_str = circ.detailed_str()

        assert "if-else_circuit" in detail_str
        assert f"{circ.name}" in detail_str
        assert f"if: {circ.if_circuit.name}" in detail_str
        assert f"else: {circ.else_circuit.name}" in detail_str
        assert f"condition: {circ.condition_circuit.name}" in detail_str

    @pytest.mark.parametrize(
        "other_circ_builder, expected_equal",
        [
            (
                lambda c: IfElseCircuit(
                    if_circuit=c.if_circuit,
                    else_circuit=c.else_circuit,
                    condition_circuit=c.condition_circuit,
                ),
                True,
            ),
            (
                lambda c: IfElseCircuit(
                    if_circuit=c.if_circuit,
                    else_circuit=c.else_circuit,
                    condition_circuit=Circuit(
                        name=BoolOp.XOR,
                        channels=[
                            Channel(ChannelType.CLASSICAL, "diff_channel"),
                            Channel(ChannelType.CLASSICAL, "diff_channel"),
                        ],
                    ),
                ),
                False,
            ),
            (
                lambda c: IfElseCircuit(
                    if_circuit=c.if_circuit,
                    else_circuit=Circuit("diff_b"),
                    condition_circuit=c.condition_circuit,
                ),
                False,
            ),
            (
                lambda c: IfElseCircuit(
                    if_circuit=Circuit("diff_a"),
                    else_circuit=c.else_circuit,
                    condition_circuit=c.condition_circuit,
                ),
                False,
            ),
        ],
        ids=["same", "diff_condition", "diff_else", "diff_if"],
    )
    def test_equality(self, circ, other_circ_builder, expected_equal):
        """Test the equality operator of IfElseCircuit."""
        assert circ.__class__ != Circuit  # type check

        other = other_circ_builder(circ)
        if expected_equal:
            assert circ == other
        else:
            assert circ != other

    def test_is_condition_single_bit(self, get_circ):
        """Test the is_condition_single_bit cached property."""

        if_circ1 = get_circ("if_circ1")

        assert if_circ1.is_condition_single_bit is True

        # Multi-bit condition circuit
        multi_bit_condition = Circuit(
            name=BoolOp.AND,
            channels=[
                Channel(type=ChannelType.CLASSICAL, label="c1"),
                Channel(type=ChannelType.CLASSICAL, label="c2"),
            ],
        )
        if_else_multi_bit = IfElseCircuit(
            if_circuit=if_circ1.if_circuit,
            else_circuit=Circuit("x", channels=[Channel()]),
            condition_circuit=multi_bit_condition,
        )
        assert if_else_multi_bit.is_condition_single_bit is False

    def test_is_single_gate_conditioned(self, get_circ):
        """Test the is_single_gate_conditioned cached property."""
        if_circ1 = get_circ("if_circ1")

        q_channel = [Channel(type=ChannelType.QUANTUM, label=f"q{i}") for i in range(2)]
        c_channel = [Channel(type=ChannelType.CLASSICAL, label="c1")]

        # Single gate conditioned IfElseCircuit
        single_gate_if = IfElseCircuit(
            if_circuit=Circuit(
                name="single_gate",
                circuit=[[Circuit(name="H", duration=1, channels=q_channel[:1])]],
            ),
            condition_circuit=Circuit(name=BoolOp.MATCH, channels=c_channel),
        )
        assert single_gate_if.is_single_gate_conditioned is True

        # Multi-gate IfElseCircuit
        multi_gate_if = IfElseCircuit(
            if_circuit=Circuit(
                name="multi_gate",
                circuit=[
                    [Circuit(name="H", duration=1, channels=q_channel[:1])],
                    [Circuit(name="CNOT", duration=2, channels=q_channel)],
                ],
            ),
            else_circuit=Circuit(name="CNOT", duration=2, channels=q_channel),
            condition_circuit=Circuit(
                name=BoolOp.MATCH,
                channels=c_channel,
            ),
        )
        assert multi_gate_if.is_single_gate_conditioned is False

        assert if_circ1.is_single_gate_conditioned is False
