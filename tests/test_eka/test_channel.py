"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from __future__ import annotations
import unittest
from pydantic import ValidationError

from loom.eka import Channel, ChannelType
from loom.eka.utilities import uuid_error


class TestChannel(unittest.TestCase):
    """
    Test for the Channel and ChannelType classes.
    """

    def test_channel_default(self):
        """
        Tests that the creation of a default channel is done correctly:
        - type is of the right type,
        - label is of the right string,
        - the id is the correct uuid format.
        """
        ch = Channel()
        uuid_error(ch.id)
        self.assertEqual(ch.type, ChannelType.QUANTUM)
        self.assertEqual(ch.label, "data_qubit")

    def test_channel_custom(self):
        """
        Tests that the creation of a custom channel is done correctly:
        - wrong type of channel raises an exception,
        - wrong format of id raises an exception.
        """
        with self.assertRaises(ValidationError) as context:
            _ = Channel(type="qubit")
        self.assertEqual(
            str(context.exception.errors()[0]["msg"]),
            "Input should be 'quantum' or 'classical'",
        )

        with self.assertRaises(ValidationError) as context:
            _ = Channel(id=1234)
        self.assertEqual(
            str(context.exception.errors()[0]["msg"]), "Input should be a valid string"
        )

        with self.assertRaises(Exception) as context:
            _ = Channel(id="1234")
        self.assertEqual(
            str(context.exception.errors()[0]["msg"]),
            "Value error, Invalid uuid: 1234. UUID must be version 4.",
        )
