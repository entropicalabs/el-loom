"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from loom_five_qubit_perfect_code.applicator import FiveQubitPerfectCodeApplicator

#: Dictionary to map code names to their applicator classes and block class names.
selector_dict = {
    "loom_five_qubit_perfect_code": {
        "applicator": FiveQubitPerfectCodeApplicator,
        "block_class_name": "FiveQubitPerfectCode",
    },
}
