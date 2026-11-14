# TopoInherit Dataset

This dataset contains continuous high-resolution microscopic image sequences of plant cells, specifically curated for tracing cell lineage, tracking cell topologies, and analyzing cell division events over time.

## Dataset Overview
The dataset consists of 32 high-resolution microscopic images collected across two time points (0h & 24h), accumulating a total of **3,359 fully annotated cell instances**. Across these cells, the dataset provides ground truth for **1,255 cell tracking pairs** and **283 cell division events**.

The dataset is divided into two major Arabidopsis plant lines across 16 experimental subsets (`plant_1` to `plant_16`):

### 1. Sansha-5 Line (`plant_1` to `plant_8`)
* **Scope:** Contains 8 pairs of 0h/24h microscopic cell images.
* **Total Annotated Cell Instances:** 1,613
* **Tracking Relationship Pairs:** 643
* **Cell Division Relationship Pairs:** 109

### 2. WO115-2 Line (`plant_9` to `plant_16`)
* **Scope:** Contains 8 pairs of 0h/24h microscopic cell images.
* **Total Annotated Cell Instances:** 1,746
* **Tracking Relationship Pairs:** 612
* **Cell Division Relationship Pairs:** 174

## Data Structure
Each `plant_x` folder is structured as follows:

* `img/`: Contains the raw, high-resolution original microscopic images (typically in `.tif` or `.tiff` formats) for the respective time points.
* `json/`: Contains the manually annotated ground truth files generated via LabelMe. These JSON files store the precise geometric contours of each annotated cell, along with the tracking and lineage relationship mappings across the two time points.

## Usage Notes
This dataset is published concurrently with the TopoInherit framework. The detailed geometry bounds and temporal mappings within the `json/` folder can be directly parsed for executing cell lineage tracking validations or training new geometric cell models.
