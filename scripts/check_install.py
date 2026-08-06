import importlib
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UPSTREAM_ROOT = os.path.join(REPO_ROOT, "external", "4DGaussians")
if not os.path.isdir(UPSTREAM_ROOT):
    raise SystemExit("Missing external/4DGaussians. Run: bash scripts/bootstrap_external.sh")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, UPSTREAM_ROOT)


def import_version(name):
    module = importlib.import_module(name)
    version = getattr(module, "__version__", "unknown")
    return module, version


def main():
    results = {"repo_root": REPO_ROOT, "upstream_root": UPSTREAM_ROOT}
    failures = []

    for module_name in [
        "torch",
        "torchvision",
        "torchaudio",
        "plyfile",
        "open3d",
        "mmcv",
        "kornia",
        "imageio",
        "lpips",
        "pytorch_msssim",
        "cv2",
        "arguments",
        "gaussian_renderer",
        "scene",
        "diff_gaussian_rasterization",
        "simple_knn._C",
    ]:
        try:
            _, version = import_version(module_name)
            results[module_name] = version
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{module_name}: {exc}")

    if "torch" in results:
        import torch

        results["cuda_available"] = torch.cuda.is_available()
        results["cuda_device_count"] = torch.cuda.device_count()
        results["cuda_runtime"] = torch.version.cuda
        results["cuda_home"] = os.environ.get("CUDA_HOME")

        results["upstream_4dgs_revision"] = "843d5ac636c37e4b611242287754f3d4ed150144"

    print(json.dumps(results, indent=2))
    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
