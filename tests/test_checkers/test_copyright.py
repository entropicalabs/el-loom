"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import io
import astroid
import pylint.testutils
from checkers.copyright import CopyrightChecker


class TestCopyrightChecker(pylint.testutils.CheckerTestCase):
    """
    Test the CopyrightChecker Class

    We test three scenarios here:

    - Missing copyright notice
    - Present copyright notice with incorrect spelling
    - Present copyright notice with correct spelling
    """

    CHECKER_CLASS = CopyrightChecker

    mock_module = astroid.Module(name="mock_module")
    copyright_full = (
        '"""\nCopyright (c) Entropica Labs Pte Ltd 2025.\n\n'
        "Use, distribution and reproduction of this program in its "
        "source or compiled\nform is prohibited without the "
        "express written consent of Entropica Labs Pte\nLtd."
        '\n\n"""\n'
    )

    def test_missing_copyright(self):
        """Test that checker triggers when copyright message is missing."""
        self.mock_module.stream = lambda: io.BytesIO(b"def foo():\n    pass\n")
        error_msg = pylint.testutils.MessageTest(
            msg_id="file-no-copyright",
            line=0,
        )
        with self.assertAddsMessages(error_msg):
            self.checker.process_module(self.mock_module)

    def test_present_copyright_and_incorrect_copyright_spelling(self):
        """Test that checker triggers when copyright message is spelled incorrectly."""
        misspelt_copyright = self.copyright_full.replace("distribution", "retribution")
        self.mock_module.stream = lambda: io.BytesIO(misspelt_copyright.encode())
        error_msg = pylint.testutils.MessageTest(
            msg_id="file-misspelt-copyright",
            line=3,
        )
        with self.assertAddsMessages(error_msg):
            self.checker.process_module(self.mock_module)

    def test_present_copyright_and_correct_spelling(self):
        """Test that checker does not trigger when copyright message is present
        and spelled correctly."""
        self.mock_module.stream = lambda: io.BytesIO(self.copyright_full.encode())
        with self.assertNoMessages():
            self.checker.process_module(self.mock_module)
