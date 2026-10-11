# Notice: licence, third-party software and data attributions

## This project

The code, configurations, run records and documentation in this repository, and the trained weights published with it, are licensed under the
**GNU Affero General Public License, version 3 (AGPL-3.0)**; the full text is in `LICENSE`. Copyright (C) 2026 the authors of this repository (see the git history).

Why the AGPL: the detectors are fine-tuned from Ultralytics YOLOv8 code and COCO-pretrained weights, which are themselves AGPL-3.0, so the trained weights and the code that
uses them are shared under the same licence. If you modify this software and let others interact with it over a network (for example the Gradio apps in `demo/` and `edge/`),
section 13 of the AGPL requires you to offer those users the source of your modified version; the apps print a source link taken from `$FISH_SOURCE_URL`, so set it to the address of your published source.

## Third-party software

| Software | Licence | How it is used |
|---|---|---|
| Ultralytics YOLOv8 / YOLO11 | AGPL-3.0 | training and evaluation of the YOLO detectors; the COCO-pretrained starting weights come from Ultralytics |
| PyTorch, torchvision | BSD-style / BSD-3-Clause | training and inference; the second detector family (Faster R-CNN) |
| NumPy, OpenCV, PyYAML, imageio | BSD-3-Clause, Apache-2.0, MIT, BSD-2-Clause | data handling, image reading, configuration |
| Gradio | Apache-2.0 | the demo and board web applications |
| NVIDIA TensorRT, CUDA, JetPack | NVIDIA proprietary licences | used on the Jetson Orin Nano; **not included** in this repository |
| YOLO-Fish (Muksit et al., 2022) | GPL-3.0 | its published detectors are only **evaluated** here (comparison); the code and weights are not included or redistributed |

Other Python dependencies are installed from PyPI (see the `requirements-*.txt` files); none is copied into this repository, and each keeps its own licence (see its package metadata).

## Datasets and attributions

No dataset is included in this repository; the scripts download or convert them from the original sources, and the conversions are changes to the originals as stated below.
Trained weights were produced from the training datasets, which are openly licensed.

| Dataset | Licence | Credit | What was changed here |
|---|---|---|---|
| DeepFish | MIT (dataset repository); the paper is CC BY 4.0 | Saleh, A., Laradji, I. H., Konovalov, D. A., Bradley, M., Wang, D. and Sheaves, M. (2020). A realistic fish-habitat dataset to evaluate algorithms for underwater visual analysis. Scientific Reports 10, 14671. https://doi.org/10.1038/s41598-020-71639-x | bounding boxes come from the darknet export distributed by the YOLO-Fish authors (Muksit et al., 2022); converted to one class, split by capture site |
| OzFish | CC BY 3.0 AU | Australian Institute of Marine Science, University of Western Australia and Curtin University (2019). OzFish Dataset. https://doi.org/10.25845/5e28f062c5097 | boxes converted to one-class YOLO labels, frames resized to 960 or 1280 px wide, a subset of 779 frames split by deployment: the converted labels are a derivative work of OzFish |
| Kakadu | CC BY 4.0 | Jansen, A., Walden, D., Walker, S. and Buccella, C. (2022). A deep learning dataset for underwater object detection of tropical freshwater fish species in northern Australia. Zenodo. https://doi.org/10.5281/zenodo.7250921 | 23 taxa merged into one class, a subset of 7,318 of 44,112 frames, resized to 960 or 1280 px wide |
| DEEPECOMAR (Mediterranean fish) | CC BY 4.0 | Catalan, I. and Alvarez-Ellacuria, A. (2023). Labelled Mediterranean fish dataset. Zenodo. https://doi.org/10.5281/zenodo.7534425 | used only for evaluation; species merged into one class, lens-stain boxes removed, duplicate frames removed |
| COCO | annotations CC BY 4.0; images under their Flickr terms | Lin, T.-Y. et al. (2014). Microsoft COCO: Common Objects in Context. ECCV | only through the pretrained starting weights; no COCO data is included |

The Caltech Fish Counting dataset (sonar) was not used.

## What this notice is not

It is a good-faith record of the licences known to the authors, not legal advice. If you redistribute this work, keep `LICENSE` and this notice, keep the attributions above, and
check the licences of any component you add.
