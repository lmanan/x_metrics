# x_metrics

Adapters for popular evaluation metrics in deep learning experiments, with a focus on segmentation and tracking tasks.

## Installation

```bash
pip install -e .
```

## Usage

### Segmentation Quality Metrics

Compute IoU-based matching metrics (precision, recall, F1, accuracy) between predicted and ground truth segmentation masks.

```python
from x_metrics import SegmentationAdapter

adapter = SegmentationAdapter(
    zarr_path="path/to/data.zarr",
    group="experiment_01",
    pred_dataset="predictions",
    target_dataset="ground_truth",
    thresh=0.5,  # IoU threshold for considering a match
)

results = adapter.compute()

print(f"Precision: {results['precision']:.3f}")
print(f"Recall:    {results['recall']:.3f}")
print(f"F1 Score:  {results['f1']:.3f}")
print(f"Accuracy:  {results['accuracy']:.3f}")
print(f"Mean IoU:  {results['mean_true_score']:.3f}")
```

#### Multiple Thresholds

Evaluate at multiple IoU thresholds simultaneously:

```python
adapter = SegmentationAdapter(
    zarr_path="data.zarr",
    group="experiment_01",
    pred_dataset="predictions",
    target_dataset="ground_truth",
    thresh=(0.5, 0.75, 0.9),
)

results = adapter.compute()  # Returns list of dicts, one per threshold

for r in results:
    print(f"Threshold {r['thresh']}: F1={r['f1']:.3f}")
```

#### Additional Options

```python
adapter = SegmentationAdapter(
    zarr_path="data.zarr",
    group="experiment_01",
    pred_dataset="predictions",
    target_dataset="ground_truth",
    thresh=0.5,
    by_image=True,      # Average metrics per frame (default: global)
    show_progress=True, # Show progress bar
    parallel=True,      # Use parallel processing
)
```

### HOTA Tracking Metrics

Compute Higher Order Tracking Accuracy (HOTA) metrics for evaluating object tracking performance. HOTA decomposes into detection accuracy (DetA) and association accuracy (AssA).

#### Data Format

- **Zarr datasets**: Instance segmentation masks with shape `(C, T, Y, X)` where `C=1`
- **CSV files**: Track information with columns:
  - `unique_id`: Unique identifier for each object
  - `time`: Frame/time index
  - `y`, `x`: Object coordinates
  - `parent_id`: Parent object ID (0 if track starts, otherwise links to parent)

```python
from x_metrics import HOTAAdapter

adapter = HOTAAdapter(
    zarr_path="path/to/data.zarr",
    group="experiment_01",
    pred_dataset="predictions",
    target_dataset="ground_truth",
    pred_csv_path="path/to/pred_tracks.csv",
    target_csv_path="path/to/gt_tracks.csv",
    iou_threshold=0.5,
)

results = adapter.compute()

print(f"HOTA: {results['HOTA']:.3f}")
print(f"DetA: {results['DetA']:.3f}")  # Detection accuracy
print(f"AssA: {results['AssA']:.3f}")  # Association accuracy
print(f"LocA: {results['LocA']:.3f}")  # Localization accuracy
```

#### Detailed Per-Alpha Results

HOTA is computed across multiple alpha (IoU) thresholds. Access per-threshold results:

```python
results = adapter.compute()

for alpha_idx, hota_val in enumerate(results['per_alpha']['HOTA']):
    print(f"Alpha {alpha_idx}: HOTA={hota_val:.3f}")
```

## Creating Custom Adapters

Extend `BaseMetricAdapter` to create your own metric adapters:

```python
from x_metrics import BaseMetricAdapter

class MyCustomAdapter(BaseMetricAdapter):
    def __init__(self, pred_path, target_path):
        self.pred_path = pred_path
        self.target_path = target_path

    def compute(self) -> dict:
        # Load data and compute metrics
        # ...
        return {"my_metric": value}
```
