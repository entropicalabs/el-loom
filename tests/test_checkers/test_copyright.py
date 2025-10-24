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
        '"""\nCopyright 2024 Entropica Labs Pte Ltd\n\n'
        'Licensed under the Apache License, Version 2.0 (the "License");\n'
        "you may not use this file except in compliance with the License.\n"
        "You may obtain a copy of the License at\n\n"
        "    http://www.apache.org/licenses/LICENSE-2.0\n\n"
        "Unless required by applicable law or agreed to in writing, software\n"
        'distributed under the License is distributed on an "AS IS" BASIS,\n'
        "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
        "See the License for the specific language governing permissions and\n"
        "limitations under the License.\n\n"
        '"""\n'
    )
    
    
    # copyright_full = (
    #     '"""\nCopyright (c) Entropica Labs Pte Ltd 2025.\n\n'
    #     "Use, distribution and reproduction of this program in its "
    #     "source or compiled\nform is prohibited without the "
    #     "express written consent of Entropica Labs Pte\nLtd."
    #     '\n\n"""\n'
    # )

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
        misspelt_copyright = self.copyright_full.replace("Version", "Verssion")
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
