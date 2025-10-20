"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from loom_steane_code.applicator import SteaneCodeApplicator

#: Dictionary to map code names to their applicator classes and block class names.
selector_dict = {
    "loom_steane_code": {
        "applicator": SteaneCodeApplicator,
        "block_class_name": "SteaneCode",
    },
}
