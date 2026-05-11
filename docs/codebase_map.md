# 4DGaussians Codebase Map

This map documents the official upstream snapshot vendored at `external/4DGaussians/`.

## Data And Time Entry

1. `scene/dataset_readers.py`
   - Blender / D-NeRF scenes are parsed by `readNerfSyntheticInfo()`.
   - It normalizes frame timestamps through `read_timeline()` and stores the value in `CameraInfo.time`.
   - HyperNeRF / NeRFies scenes are parsed by `readHyperDataInfos()` and likewise provide normalized times through the dataset object.
2. `scene/dataset.py`
   - `FourDGSdataset.__getitem__()` forwards `caminfo.time` into the runtime `Camera`.
3. `scene/cameras.py`
   - `Camera.time` is stored as a scalar per view.

## Render And Deformation Chain

1. `gaussian_renderer/__init__.py::render()`
   - Reads `viewpoint_camera.time`.
   - Expands it to a per-Gaussian tensor with shape `[num_gaussians, 1]`.
   - Passes the time tensor to `pc._deformation(...)`.
2. `scene/deformation.py`
   - `deform_network.forward_dynamic()` receives `times_sel`.
   - The spatial-temporal HexPlane field uses the time tensor together with point coordinates.
3. `scene/gaussian_model.py`
   - Holds canonical Gaussian state, deformation network, and optimizer state.
   - Updates Gaussian parameters through `training_setup()`, optimizer steps, densification, and pruning.

## Training Loop

1. `train.py::training()`
   - Builds `GaussianModel`, `Scene`, and enters coarse then fine reconstruction.
2. `train.py::scene_reconstruction()`
   - Samples cameras, renders predictions, computes reconstruction loss, applies regularizers, and steps optimizers.
3. `Scene.save()`
   - Saves Gaussian point clouds and deformation weights under `point_cloud/iteration_*`.

## Safe Phase B Insertion Point

The safest Phase B hook is before deformation, after the per-view timestamp has been expanded:

```text
Camera.time
  -> gaussian_renderer.render()
  -> temporal warp phi(t) or density integral tau(t)
  -> scene.deformation.deform_network(..., times_sel=tau)
```

This preserves:
- dataset protocol
- Gaussian parameterization
- rasterizer behavior
- official render output layout

## Areas Intentionally Left Untouched

- `submodules/depth-diff-gaussian-rasterization/`
- low-level CUDA rasterizer bindings
- dataset storage formats
- coarse/fine stage structure

## Phase B Integration Summary

Phase B introduces `refergaussian.temporal` and only requires glue in:
- `external/4DGaussians/arguments/__init__.py`
- `external/4DGaussians/train.py`
- `external/4DGaussians/render.py`
- `external/4DGaussians/gaussian_renderer/__init__.py`

