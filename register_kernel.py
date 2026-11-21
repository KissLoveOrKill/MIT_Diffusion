"""Register this environment and select it in the three assignment notebooks."""

import json
import os
from pathlib import Path
import re
import sys

from ipykernel.kernelspec import install


root = Path(__file__).resolve().parent
prefix = Path(sys.prefix)
kernel_name = "mit-diffusion"
display_name = "Python (MIT Diffusion)"
environment = {}
if os.name == "nt":
    # Jupyter may be launched outside an activated Conda terminal.
    paths = [prefix, prefix / "Scripts", prefix / "Library" / "bin"]
    environment["PATH"] = os.pathsep.join(map(str, paths)) + os.pathsep + os.environ["PATH"]

destination = install(
    user=True,
    kernel_name=kernel_name,
    display_name=display_name,
    env=environment,
)
for path in sorted(root.glob("lab_*.ipynb")):
    text = path.read_bytes().decode("utf-8")
    notebook = json.loads(text)
    spec = notebook["metadata"]["kernelspec"]
    spec.update(display_name=display_name, language="python", name=kernel_name)
    # Change only the kernelspec block, retaining source, outputs and formatting.
    replacement = '"kernelspec": ' + json.dumps(spec, indent=1)
    replacement = replacement.replace("\n", "\n  ")
    newline = "\r\n" if "\r\n" in text else "\n"
    replacement = replacement.replace("\n", newline)
    updated, count = re.subn(r'"kernelspec":\s*\{[^{}]*\}', lambda _: replacement, text)
    if count != 1:
        raise RuntimeError(f"Expected one kernelspec in {path.name}, found {count}")
    assert json.loads(updated)["cells"] == notebook["cells"]
    path.write_bytes(updated.encode("utf-8"))
    print(f"Selected {display_name} in {path.name}")
print(f"Installed kernel: {destination}")

