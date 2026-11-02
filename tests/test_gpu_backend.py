"""Host-side callback needs CPU even when the model computes on CUDA."""
import ast
from pathlib import Path


def test_gpu_training_keeps_cpu_callback_backend_available():
    path=Path(__file__).resolve().parents[1]/'fold_pi05/train.py'
    tree=ast.parse(path.read_text())
    assignments=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Subscript) and isinstance(target.slice,ast.Constant) and target.slice.value=='JAX_PLATFORMS':
                    assignments.append(node.value.value)
    assert assignments==['cuda,cpu']
