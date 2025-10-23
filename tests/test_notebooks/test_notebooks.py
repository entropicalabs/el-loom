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

import os
from os.path import dirname
import unittest
import pytest

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


class TestNotebooks(unittest.TestCase):
    """
    Tests to run the notebooks. They work by converting the notebooks
    to a python script via nbconvert and then running the resulting .py file.
    """

    @staticmethod
    def generate_notebook_test_function(test_filename: str):
        """
        Generate a test function for the given notebook file.
        """

        @pytest.mark.notebook
        def test(self):  # pylint: disable=unused-argument
            with open(test_filename, encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)
            ep = ExecutePreprocessor(timeout=300, kernel_name="env")
            ep.preprocess(nb)

        return test

    @classmethod
    def add_notebook_tests(cls):
        """
        Add the test methods to the TestNotebooks class for each .ipynb file in the
        directory. The test methods are named test_<relative_path_to_notebook>
        with underscores replacing path separators.

        This method is called at class definition time to dynamically add tests.
        """
        repo_path = dirname(dirname(dirname(os.path.abspath(__file__))))

        for dirpath, _, filenames in os.walk(repo_path + "/docs/source/notebooks"):
            for f in filenames:
                if f.endswith(".ipynb"):
                    filename = os.path.join(dirpath, f)
                    rel_path = os.path.relpath(filename, repo_path)
                    test_name = "test_" + rel_path.split(".")[0].replace(os.sep, "_")
                    test_func = cls.generate_notebook_test_function(filename)
                    setattr(cls, test_name, test_func)


# Integrate test generation at class definition time
TestNotebooks.add_notebook_tests()

if __name__ == "__main__":
    unittest.main()
