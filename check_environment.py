"""Run a small real Jupyter session without executing assignment exercises."""

import ast
import json
from pathlib import Path
import subprocess
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager


root = Path(__file__).resolve().parent
kernel_name = "mit-diffusion"
kernel = KernelSpecManager().get_kernel_spec(kernel_name)
assert Path(kernel.argv[0]).resolve() == Path(sys.executable).resolve(), kernel.argv

cells = []
notebooks = sorted(root.glob("lab_*.ipynb"))
for path in notebooks:
    notebook = nbformat.read(path, as_version=4)
    assert notebook.metadata.kernelspec.name == kernel_name, path.name
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        if any(isinstance(node, ast.Import) and any(alias.name == "torch" for alias in node.names)
               for node in tree.body):
            cells.append(nbformat.v4.new_code_cell(cell.source))
            break
    else:
        raise RuntimeError(f"No import cell found in {path.name}")

cells.append(nbformat.v4.new_code_cell(r'''
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from celluloid import Camera
from IPython.display import HTML
from sklearn.datasets import make_moons, make_circles
from torchvision.ops import nms
from torchvision import datasets, transforms
from einops import rearrange
from torch.func import jacrev, vmap

assert torch.cuda.is_available(), "CUDA is unavailable"
device = torch.device("cuda")
torch.manual_seed(42)

# Exercise matrix multiplication, autograd and the CNN path needed by Lab 3.
x = torch.randn(32, 32, device=device, requires_grad=True)
loss = (x @ x.T).square().mean()
loss.backward()
assert torch.isfinite(x.grad).all()
model = torch.nn.Sequential(torch.nn.Conv2d(1, 8, 3, padding=1), torch.nn.SiLU(),
                            torch.nn.Flatten(), torch.nn.Linear(8 * 32 * 32, 10)).to(device)
output = model(torch.randn(2, 1, 32, 32, device=device))
output.square().mean().backward()
assert output.shape == (2, 10) and torch.isfinite(output).all()
assert hasattr(torch.nn, "Buffer"), "Lab 3 requires torch.nn.Buffer"
samples = torch.randn(4, 2, device=device)
actual = vmap(jacrev(lambda z: z.square()))(samples)
assert torch.allclose(actual, torch.diag_embed(2 * samples))
boxes = torch.tensor([[0., 0., 2., 2.], [0., 0., 1., 1.]], device=device)
assert nms(boxes, torch.tensor([0.9, 0.8], device=device), 0.5).numel() == 2
torch.cuda.synchronize()

# Encode the same H.264 HTML5 video type used by Lab 1.
ffmpeg = shutil.which("ffmpeg")
assert ffmpeg, "FFmpeg is missing from the kernel PATH"
ffmpeg_version = subprocess.check_output([ffmpeg, "-version"], text=True).splitlines()[0]
with tempfile.TemporaryDirectory(prefix="mit-diffusion-check-") as temporary:
    temporary = Path(temporary)
    fig, ax = plt.subplots(figsize=(3.2, 2.4), dpi=50)
    camera = Camera(fig)
    for end in (0.25, 0.5, 1.0):
        ax.plot([0, 1], [0, end], color="blue")
        camera.snap()
    fig.savefig(temporary / "plot.png")
    assert (temporary / "plot.png").stat().st_size > 0
    animation = camera.animate(interval=100)
    video_html = animation.to_html5_video()
    assert "video/mp4" in video_html
    plt.close(fig)

names = ("torch", "torchvision", "numpy", "matplotlib", "seaborn", "scikit-learn",
         "tqdm", "einops", "celluloid", "jupyterlab", "ipykernel", "ipywidgets", "nbformat")
report = {
    "python": sys.version.split()[0],
    "executable": sys.executable,
    "packages": {name: importlib.metadata.version(name) for name in names},
    "cuda_available": torch.cuda.is_available(),
    "cuda_runtime": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "compute_capability": list(torch.cuda.get_device_capability(0)),
    "ffmpeg": ffmpeg_version,
    "checks": ["all_notebook_import_cells", "cuda_matmul_backward", "cuda_cnn_backward",
               "torch_func_vmap_jacrev", "torchvision_cuda_nms", "matplotlib_png",
               "celluloid_h264_html5_video", "jupyter_kernel"],
}
Path("environment-check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
'''))

smoke = nbformat.v4.new_notebook(cells=cells)
client = NotebookClient(smoke, timeout=180, kernel_name=kernel_name,
                        resources={"metadata": {"path": str(root)}})
client.execute()
for output in smoke.cells[-1].outputs:
    if output.output_type == "stream":
        print(output.text, end="")
subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
print(f"PASS: {len(notebooks)} notebook imports, GPU, plotting, video, Jupyter and pip dependencies.")
