"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

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
        b"Copyright (c) Entropica Labs Pte Ltd 2025.\n",
        b"\n",
        b"Use, distribution and reproduction of this program in its source or compiled\n",
        b"form is prohibited without the express written consent of Entropica Labs Pte\n",
        b"Ltd.\n",
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
