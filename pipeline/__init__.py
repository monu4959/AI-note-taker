"""Meeting intelligence pipeline."""

import os
import sys


def _register_cuda_dlls() -> None:
    """Let CTranslate2 (faster-whisper) find cuBLAS/cuDNN on Windows.

    PyTorch's CUDA wheel already ships cublas64_12.dll and cudnn64_9.dll in torch/lib.
    CTranslate2 looks for them on the DLL search path, so we add that folder instead of
    installing a second CUDA toolkit. No-op on other platforms.
    """
    if sys.platform != "win32":
        return
    import importlib.util

    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.submodule_search_locations:
        return
    torch_lib = os.path.join(spec.submodule_search_locations[0], "lib")
    if os.path.isdir(torch_lib):
        os.add_dll_directory(torch_lib)
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")


_register_cuda_dlls()


def _load_env() -> None:
    """Load project-local secrets (.env at the project root, gitignored), e.g. HF_TOKEN."""
    from pathlib import Path

    from dotenv import load_dotenv

    # override=True: a project-local value wins over a stale global one (e.g. ANTHROPIC_API_KEY).
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)


_load_env()
