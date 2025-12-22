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
    pred_zarr_path : str or Path
        Path to predicted segmentation zarr (shape: C T Y X, C=1).
    target_zarr_path : str or Path
        Path to ground truth segmentation zarr (shape: C T Y X, C=1).
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
        pred_zarr_path: str | Path,
        target_zarr_path: str | Path,
        thresh: float | tuple[float, ...] = 0.5,
        criterion: str = "iou",
        by_image: bool = False,
        show_progress: bool = True,
        parallel: bool = False,
    ):
        self.pred_zarr_path = Path(pred_zarr_path)
        self.target_zarr_path = Path(target_zarr_path)
        self.thresh = thresh
        self.criterion = criterion
        self.by_image = by_image
        self.show_progress = show_progress
        self.parallel = parallel

    def _load_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Load zarr arrays as numpy arrays."""
        pred_zarr = zarr.open(self.pred_zarr_path, mode="r")
        target_zarr = zarr.open(self.target_zarr_path, mode="r")

        # Handle zarr as array or group, take first channel (C=1)
        if isinstance(pred_zarr, zarr.Array):
            pred_arr = np.asarray(pred_zarr[0], dtype=np.uint32)
        else:
            pred_arr = np.asarray(pred_zarr["data"][0], dtype=np.uint32)

        if isinstance(target_zarr, zarr.Array):
            target_arr = np.asarray(target_zarr[0], dtype=np.uint32)
        else:
            target_arr = np.asarray(target_zarr["data"][0], dtype=np.uint32)

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
