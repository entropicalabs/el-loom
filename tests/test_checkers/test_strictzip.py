"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import astroid
import pylint.testutils
from checkers.strictzip import StrictZipChecker


class TestStrictZipChecker(pylint.testutils.CheckerTestCase):
    """Test the StrictZipChecker Class"""

    CHECKER_CLASS = StrictZipChecker

    def test_present_strict_zip(self):
        """Test that checker does not trigger when `strict` parameter is present."""
        test_cases = [
            (  # calls
                """
                zip([1, 2], [3, 4], strict=False) #@
                zip([1, 2], [3, 4], strict=True) #@
                """,
                lambda node: node,
            ),
            (  # assignments
                """
                def test():
                    a = zip([1, 2], [3, 4], strict=False) #@
                    b = zip([1, 2], [3, 4], strict=True) #@
                    return 
                """,
                lambda node: node.value,
            ),
            (  # returns
                """
                def test():
                    return zip([1, 2], [3, 4], strict=False) #@
                def test():
                    return zip([1, 2], [3, 4], strict=True) #@
                """,
                lambda node: node.value,
            ),
            (  # nested
                """
                enumerate(zip([1, 2], [3, 4], strict=False)) #@
                enumerate(zip([1, 2], [3, 4], strict=True)) #@
                """,
                lambda node: node.args[0],
            ),
        ]

        for src, node_fn in test_cases:
            node_false, node_true = astroid.extract_node(src)
            with self.assertNoMessages():
                # Explicit checks
                self.checker.visit_call(node_fn(node_false))
                self.checker.visit_call(node_fn(node_true))

                # Recursive/Implicit checks
                self.walk(node_false)
                self.walk(node_true)

    # pylint: disable=use-dict-literal
    def test_missing_strict_zip(self):
        """Test that checker triggers when `strict` parameter is missing."""
        test_cases = [
            (  # calls
                "zip([1, 2], [3, 4]) #@",
                lambda node: node,
                dict(line=1, col_offset=0, end_line=1, end_col_offset=19),
            ),
            (  # assignments
                """
                def test():
                    a = zip([1, 2], [3, 4]) #@
                    return 
                """,
                lambda node: node.value,
                dict(line=3, col_offset=8, end_line=3, end_col_offset=27),
            ),
            (  # returns
                """
                def test():
                    return zip([1, 2], [3, 4]) #@
                """,
                lambda node: node.value,
                dict(line=3, col_offset=11, end_line=3, end_col_offset=30),
            ),
            (  # nested
                "enumerate(zip([1, 2], [3, 4])) #@",
                lambda node: node.args[0],
                dict(line=1, col_offset=10, end_line=1, end_col_offset=29),
            ),
        ]

        for src, node_fn, msg_args in test_cases:
            node = astroid.extract_node(src)
            error = pylint.testutils.MessageTest(
                msg_id="missing-strict-zip",
                node=node_fn(node),
                **msg_args,
            )
            # Explicit checks
            with self.assertAddsMessages(error):
                self.checker.visit_call(node_fn(node))

            # Recursive/Implicit checks
            with self.assertAddsMessages(error):
                self.walk(node)
