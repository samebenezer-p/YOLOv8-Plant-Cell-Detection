# Release Checklist: Pre-Publication Hygiene & Roadmap

This document enumerates the remaining technical, legal, and operational actions required before publishing the PlantCell Tool and YOLOv8 plant cell detection repository to a public platform (e.g. GitHub, Zenodo, Hugging Face).

---

## 1. Pending Pre-Release Action Items

| Item # | Domain | Requirement | Current Status | Action Required Prior to Release |
|---|---|---|---|---|
| **1** | **Legal / Licensing** | Choose and attach an open-source LICENSE | Missing (No LICENSE file exists) | Author / Guide must select a license (e.g., MIT, Apache 2.0, or CC-BY-NC 4.0 for non-commercial academic research). **Do not commit code without an explicit license.** |
| **2** | **Data Redistribution Rights** | TopoInherit dataset redistribution terms | Included locally in `data/` and `real_dataset/` (now git-ignored) | Verify copyright and redistribution rights for TopoInherit raw micrographs. Do NOT commit raw micrographs or full annotations to Git; publish an automated fetch script or Zenodo accession link instead. |
| **3** | **Checkpoint Hosting** | External hosting for `.pt` model weights | Checkpoints located in `runs/` (ignored by Git) | Upload the 7 model checkpoints (`E1`, `E2`, `E3`, `E4`, `E5a`, `E5b`, `E8`, total ~42 MB) to a GitHub Releases tag or Zenodo repository. Provide a download script verifying SHA256 checksums. |
| **4** | **Hardcoded Path Sanitization** | Remove machine-specific `E:/` paths in YAML/scripts | `data/real/data.yaml` has `path: E:/Yolov8_Plant_Cell_Detection/data/real` | Replace absolute paths with relative paths (`path: data/real`) or environment variable anchors (`$REPO_ROOT`) so the repository is portable across Linux, macOS, and other Windows drives. |
| **5** | **Repository Redundancy Cleanup** | Duplicate dataset directories | `real_dataset/` duplicates data in `data/real/` | Safely remove the redundant `real_dataset/` directory to save ~1.2 GB of disk space. |
| **6** | **Root Documentation Alignment** | Root `README.md` describes an obsolete 3-class prototype | `README.md` references a 3-class prototype from earlier iterations | Rewrite root `README.md` to reference the final frozen 1-class study (`plantcell_tool/`, ablation suite E1..E8, and conference paper results). |
| **7** | **Git Remote Configuration** | Remote origin URL | `git remote -v` outputs nothing | Configure a remote GitHub or GitLab repository (`git remote add origin <url>`) and establish default branch protection. |

---

## 2. Git Staging & Footprint Audit (Post `.gitignore` Update)

Following the append-only `.gitignore` update, large artifacts (`.venv_gpu/`, `data/`, `real_dataset/`, `images/`, `labels/`, `outputs_tool/`, `*.onnx`, `*.engine`) are cleanly excluded.

### Untracked Files Preview (`git ls-files --others --exclude-standard`)
```text
app/streamlit_app.py
benchmark_gpu_rigorous.py
benchmark_gpu_throughput.py
configs/yolov8n_nc1.yaml
configs/yolov8n_p2_nc1.yaml
demo_inference.py
download_and_install_gpu.py
install_gpu_env.py
plantcell_tool/__init__.py
plantcell_tool/__main__.py
plantcell_tool/cli.py
plantcell_tool/export.py
plantcell_tool/imageio.py
plantcell_tool/infer.py
plantcell_tool/models.py
plantcell_tool/validate_full_images.py
prepare_real_data.py
release/MODEL_CARD.md
release/README_TOOL.md
release/RELEASE_CHECKLIST.md
requirements-app.txt
requirements.txt
tests/test_cli.py
tests/test_coordinates.py
tests/test_imageio.py
tests/test_loader.py
tests/test_tiling.py
```

### Staging Size Footprint
- **Total Untracked Files**: 101 files
- **Total Staged Size**: **7.98 MB**
- **Binary Footprint**: 0 binary checkpoints (`*.pt` are fully ignored). Zero raw micrographs staged.
