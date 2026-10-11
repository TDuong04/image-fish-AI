# Licence and availability: options for the owner to decide

Prepared 2026-10-11. This is a summary of the licence facts recorded in the project (`CLAUDE.md`, `docs/data_audit.md`, `docs/PROGRESS.md`), not legal advice;
check the final wording with your supervisor or the institution's research office before publishing. A journal will ask for a code and data availability
statement, so this decision gates the publication.

## Facts

| Item | Licence | What follows |
|---|---|---|
| Ultralytics YOLOv8 code and the COCO-pretrained `yolov8*.pt` weights | AGPL-3.0 | Weights fine-tuned from them are encumbered too. The trigger is **conveying** a copy (including to a partner), not selling; network use can also trigger it. |
| torchvision Faster R-CNN code | BSD-3-Clause | Permissive. The licence of the COCO-pretrained torchvision weights should be checked before shipping them (the code licence is not automatically the weights licence). |
| YOLOX, NanoDet-Plus (alternatives named in `CLAUDE.md`) | Apache-2.0 | Permissive; not used here. |
| YOLO-Fish code and weights | GPL-3.0 | Used only to evaluate (comparison); not redistributed by this project. |
| DeepFish | MIT (dataset repository); article CC BY 4.0 | Attribution. |
| Kakadu (Jansen et al., 2022) | CC BY 4.0 | Attribution; our 7,318-frame subset and converted labels are an adaptation: say so. |
| OzFish (AIMS, UWA, Curtin, 2019) | CC BY 3.0 AU | Attribution; **our converted labels are a derivative work** and need the attribution and a statement that they were changed. A third-party mirror would break the chain, so cite the AIMS record. |
| DEEPECOMAR (Catalan and Alvarez-Ellacuria, 2023) | CC BY 4.0 | Attribution; used for evaluation only, with labels converted to one class. |

## Options

1. **Publish code and weights under AGPL-3.0 (simplest, compliant).** Release the repository and the trained weights openly with the AGPL-3.0 licence and the data attributions above.
   Anyone who receives them gets the AGPL rights. This fits a paper and a thesis. It rules out handing a closed-source device to a partner, because they would have to receive the source.
2. **Publish the code under AGPL-3.0 but not the weights.** Weaker for reproducibility; reviewers may ask for weights. The conveying question still applies to anything shipped on a device.
3. **Use the permissive path for the deployed model.** Train and ship a permissively licensed detector (the torchvision Faster R-CNN screen in `configs/plan_tv.yaml` is the first step; YOLOX or NanoDet-Plus are the other candidates in `CLAUDE.md`) and keep the YOLOv8 results as the research finding. Costs training and Orin-timing work, and the
   permissive model may be less accurate; the second-family result (F-number to follow) shows by how much on our data.
4. **Keep everything internal (no copy leaves the institution).** The AGPL does not bite, but a journal still wants availability, so this usually means option 1 or 2 for the paper anyway.

## Suggested statement (option 1), to adapt

> Code, configurations, run records and the trained weights are available at [repository URL] under the AGPL-3.0 licence (the Ultralytics YOLOv8 code and COCO-pretrained weights are AGPL-3.0, and the fine-tuned weights are distributed under the same licence).
> The datasets are third-party and openly licensed: DeepFish (Saleh et al., 2020), Kakadu (Jansen et al., 2022; CC BY 4.0), OzFish (Australian Institute of Marine Science et al., 2019; CC BY 3.0 AU; our converted labels are a derivative work)
> and DEEPECOMAR (Catalan and Alvarez-Ellacuria, 2023; CC BY 4.0). The scripts that rebuild the splits and converted labels are included; the datasets themselves are obtained from their original sources.

## Decision (2026-10-11)

The owner chose **option 1: share everything openly under the AGPL-3.0**, and asked for the licensing to be handled properly. Done in the working tree: `LICENSE`, `NOTICE.md`, `docs/MODEL_CARD.md`, the AGPL section 13 source offer in the apps and a README section; what still needs the owner before publication is in `docs/PUBLISH_CHECKLIST.md`.

## What I needed from you (answered above)

- Which option, or "ask my supervisor".
- Where the repository would be published (the project has a Gitea and a GitHub remote; nothing has been pushed yet).
- Whether the weights may be published (option 1) or not (option 2).
