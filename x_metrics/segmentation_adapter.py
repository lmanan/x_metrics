"""Segmentation quality metric adapter for zarr datasets."""

from pathlib import Path
from typing import Any

import numpy as np
import zarr

from x_metrics.base_adapter import BaseMetricAdapter
from x_metrics.segmentation_metrics import matching_dataset


class SegmentationAdapter(BaseMetricAdapter):
    """Adapter for computing segmentation quality metrics on zarr datasets.

    Computes IoU-based matching metrics including precision, recall, accuracy,
    and F1 score between predicted and ground truth segmentation masks.

    Parameters
    ----------
    zarr_path : str or Path
        Path to zarr container.
    group : str
        Name of the group containing prediction and target datasets.
    pred_dataset : str
        Name of the dataset containing predicted segmentation (shape: C T Y X, C=1).
    target_dataset : str
        Name of the dataset containing ground truth segmentation (shape: C T Y X, C=1).
    thresh : float or tuple of float, optional
        IoU threshold(s) for considering a match. Default is 0.5.
    criterion : str, optional
        Matching criterion. Default is "iou".
    by_image : bool, optional
        If True, metrics are averaged per image. If False, metrics are
        computed globally across all images. Default is False.
    show_progress : bool, optional
        Whether to show progress bar. Default is True.
    parallel : bool, optional
        Whether to use parallel processing. Default is False.
    """

    def __init__(
        self,
        zarr_path: str | Path,
        group: str,
        pred_dataset: str,
        target_dataset: str,
        thresh: float | tuple[float, ...] = 0.5,
        criterion: str = "iou",
        by_image: bool = False,
        show_progress: bool = True,
        parallel: bool = False,
    ):
        self.zarr_path = Path(zarr_path)
        self.group = group
        self.pred_dataset = pred_dataset
        self.target_dataset = target_dataset
        self.thresh = thresh
        self.criterion = criterion
        self.by_image = by_image
        self.show_progress = show_progress
        self.parallel = parallel

    def _load_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Load zarr arrays as numpy arrays."""
        root = zarr.open(self.zarr_path, mode="r")
        group = root[self.group]

        # Load datasets and take first channel (C=1)
        pred_arr = np.asarray(group[self.pred_dataset][0], dtype=np.uint32)
        target_arr = np.asarray(group[self.target_dataset][0], dtype=np.uint32)

        return pred_arr, target_arr

    def compute(self) -> dict[str, Any]:
        """Compute segmentation quality metrics.

        Returns
        -------
        dict
            Dictionary containing segmentation metrics:
            - precision: TP / (TP + FP)
            - recall: TP / (TP + FN)
            - accuracy: TP / (TP + FP + FN)
            - f1: 2*TP / (2*TP + FP + FN)
            - tp: True positives
            - fp: False positives
            - fn: False negatives
            - n_true: Number of ground truth objects
            - n_pred: Number of predicted objects
            - mean_true_score: Mean IoU of matched objects
            - thresh: IoU threshold used
            - criterion: Matching criterion used
        """
        pred_arr, target_arr = self._load_data()

        # Convert to list of 2D frames for matching_dataset
        # Shape is T Y X after removing C dimension
        y_pred = [pred_arr[t] for t in range(pred_arr.shape[0])]
        y_true = [target_arr[t] for t in range(target_arr.shape[0])]

        result = matching_dataset(
            y_true=y_true,
            y_pred=y_pred,
            thresh=self.thresh,
            criterion=self.criterion,
            by_image=self.by_image,
            show_progress=self.show_progress,
            parallel=self.parallel,
        )

        # Convert namedtuple to dict
        if hasattr(result, "_asdict"):
            return result._asdict()
        else:
            # Multiple thresholds returns tuple of namedtuples
            return [r._asdict() for r in result]
