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

from typing import TYPE_CHECKING
from pylint.checkers import BaseRawFileChecker

from astroid import nodes

if TYPE_CHECKING:
    from pylint.lint import PyLinter


class CopyrightChecker(BaseRawFileChecker):
    """Check all modules for copyright notice"""

    name = "copyright-checker"
    msgs = {
        "C5201": (
            "Include copyright in file",
            "file-no-copyright",
            ("Your file has no copyright"),
        ),
        "C5202": (
            "Copyright misspelt in file",
            "file-misspelt-copyright",
            ("Your file has misspelt copyright"),
        ),
    }
    options = (
        (
            "ignore-copyright",
            {
                "default": False,
                "type": "yn",
                "metavar": "<y or n>",
                "help": "Allow files to not explicitly include a copyright notice",
            },
        ),
    )

    copyright_msg = [
        b'"""\n',
        b"Copyright 2024 Entropica Labs Pte Ltd\n",
        b"\n",
        b'Licensed under the Apache License, Version 2.0 (the "License");\n',
        b"you may not use this file except in compliance with the License.\n",
        b"You may obtain a copy of the License at\n",
        b"\n",
        b"    http://www.apache.org/licenses/LICENSE-2.0\n",
        b"\n",
        b"Unless required by applicable law or agreed to in writing, software\n",
        b'distributed under the License is distributed on an "AS IS" BASIS,\n',
        b"WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n",
        b"See the License for the specific language governing permissions and\n",
        b"limitations under the License.\n",
        b"\n",
        b'"""\n',
    ]

    def process_module(self, node: nodes.Module):
        """Process a module.

        Module's content is accessible via node.stream() function.
        """

        # Check for copyright notice
        with node.stream() as stream:
            lines = []
            for lineno, line in enumerate(stream):
                if lineno < 8:
                    lines.append(line)
                else:
                    break

        if len(lines) < 8 or lines[0] != self.copyright_msg[0]:
            self.add_message("file-no-copyright", line=0)
        else:
            for lineno, line in enumerate(lines):
                if line != self.copyright_msg[lineno]:
                    self.add_message("file-misspelt-copyright", line=lineno)


def register(linter: "PyLinter") -> None:
    """This required method auto registers the checker during initialization.
    :param linter: The linter to register the checker to.
    """
    linter.register_checker(CopyrightChecker(linter))
