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

# Populate `__revision__` variable holding the actual git revision. The procedure is as
# follows: (1) Suppose we are running from an editable install. Then the actual revision
# will be known by `git rev-parse HEAD`. (2) If this fails, then we probably run the
# packaged version of Loom. In this case, the `./scripts/update-revision.py` did its job
# and the `_revision.py` module is available.  So we re-export `__revision__` from
# there. (3) If failed, then something unexpected has happened, we set `__revision__` to
# None.

try:
    import subprocess
    from pathlib import Path

    __revision__ = (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=(Path(__file__).parent.parent.parent / ".git").resolve(),
            stderr=subprocess.DEVNULL,
        )
        .decode()
        .strip()
    )
except Exception:  # pylint: disable=broad-exception-caught
    try:
        from ._revision import __revision__
    except ImportError:
        __revision__ = None
