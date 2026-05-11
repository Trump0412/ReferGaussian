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
        "refergaussian.temporal",
        "utils.config_utils",
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

        from argparse import Namespace
        from refergaussian.temporal import build_temporal_warp
        from utils.config_utils import load_config_dict

        dummy_args = Namespace(
            warp_enabled=True,
            temporal_warp_type="mlp",
            warp_hidden_dim=16,
            warp_num_layers=2,
            warp_num_bins=128,
        )
        warp = build_temporal_warp(dummy_args, device="cuda" if torch.cuda.is_available() else "cpu")
        probe = torch.linspace(0.0, 1.0, 4, device=warp.device).unsqueeze(-1)
        results["temporal_probe"] = warp(probe).detach().cpu().tolist()
        config = load_config_dict(os.path.join(UPSTREAM_ROOT, "arguments", "dnerf", "bouncingballs.py"))
        results["config_loader"] = config["ModelHiddenParams"]["kplanes_config"]["resolution"]

    try:
        _, version = import_version("open3d")
        results["open3d"] = version
    except Exception as exc:  # noqa: BLE001
        results["open3d"] = f"optional: {exc}"

    try:
        _, version = import_version("mmcv")
        results["mmcv"] = version
    except Exception as exc:  # noqa: BLE001
        results["mmcv"] = f"optional: {exc}"

    print(json.dumps(results, indent=2))
    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
