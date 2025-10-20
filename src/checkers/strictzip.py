"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from typing import TYPE_CHECKING
from pylint.checkers import BaseChecker

from astroid import nodes

if TYPE_CHECKING:
    from pylint.lint import PyLinter


class StrictZipChecker(BaseChecker):
    """Check all calls to zip for existence of `strict` argument."""

    name = "strict-zip-checker"
    msgs = {
        "C5101": (
            "Zip() does not explicitly use the `strict` argument.",
            "missing-strict-zip",
            "Ensure that all calls to `zip()` explicitly use the `strict` argument.",
        ),
    }
    options = (
        (
            "ignore-strict-zip",
            {
                "default": False,
                "type": "yn",
                "metavar": "<y or n>",
                "help": "Allow zip to not explicitly use the `strict` argument",
            },
        ),
    )

    def visit_call(self, node: nodes.Call):
        """Visit a function call in the AST.

        We only care if the function is zip()"""
        if (
            isinstance(node.func, nodes.Name)
            and node.func.repr_name() == "zip"
            and not any(kw.arg == "strict" for kw in node.keywords if kw.arg)
        ):
            self.add_message("missing-strict-zip", node=node)


def register(linter: "PyLinter") -> None:
    """This required method auto registers the checker during initialization.
    :param linter: The linter to register the checker to.
    """
    linter.register_checker(StrictZipChecker(linter))
