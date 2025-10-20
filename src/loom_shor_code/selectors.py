"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from loom_shor_code.applicator import ShorCodeApplicator

#: Dictionary to map code names to their applicator classes and block class names.
selector_dict = {
    "loom_shor_code": {
        "applicator": ShorCodeApplicator,
        "block_class_name": "ShorCode",
    },
}
