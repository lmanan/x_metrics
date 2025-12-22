"""HOTA (Higher Order Tracking Accuracy) metric adapter using TrackEval."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr
from trackeval.metrics import HOTA

from x_metrics.base_adapter import BaseMetricAdapter


class HOTAAdapter(BaseMetricAdapter):
    """Adapter for computing HOTA metric on zarr segmentation data with CSV tracks.

    HOTA decomposes tracking performance into:
    - DetA (Detection Accuracy): How well objects are detected
    - AssA (Association Accuracy): How well objects are associated across time
    - HOTA = sqrt(DetA * AssA)

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
    pred_csv_path : str or Path
        Path to predicted tracks CSV with columns: unique_id, time, y, x, parent_id.
    target_csv_path : str or Path
        Path to ground truth tracks CSV with columns: unique_id, time, y, x, parent_id.
    iou_threshold : float, optional
        IoU threshold for considering a detection as matched. Default is 0.5.
    """

    def __init__(
        self,
        zarr_path: str | Path,
        group: str,
        pred_dataset: str,
        target_dataset: str,
        pred_csv_path: str | Path,
        target_csv_path: str | Path,
        iou_threshold: float = 0.5,
    ):
        self.zarr_path = Path(zarr_path)
        self.group = group
        self.pred_dataset = pred_dataset
        self.target_dataset = target_dataset
        self.pred_csv_path = Path(pred_csv_path)
        self.target_csv_path = Path(target_csv_path)
        self.iou_threshold = iou_threshold

        self._hota_metric = HOTA()

    def _load_data(self) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
        """Load zarr arrays and CSV tracking data."""
        root = zarr.open(self.zarr_path, mode="r")
        group = root[self.group]

        # Load datasets and take first channel (C=1)
        pred_arr = np.asarray(group[self.pred_dataset][0])
        target_arr = np.asarray(group[self.target_dataset][0])

        pred_csv = pd.read_csv(self.pred_csv_path)
        target_csv = pd.read_csv(self.target_csv_path)

        return pred_arr, target_arr, pred_csv, target_csv

    def _build_tracks(self, csv_df: pd.DataFrame) -> dict[int, list[int]]:
        """Build track ID mapping from CSV.

        Each detection gets assigned to a track. Detections with parent_id=0
        start new tracks. Detections with parent_id!=0 continue the parent's track.

        Returns
        -------
        dict
            Mapping from unique_id to track_id.
        """
        # Sort by time to process in temporal order
        csv_df = csv_df.sort_values("time").reset_index(drop=True)

        unique_id_to_track = {}
        next_track_id = 1

        for _, row in csv_df.iterrows():
            unique_id = int(row["unique_id"])
            parent_id = int(row["parent_id"])

            if parent_id == 0:
                # New track starts
                unique_id_to_track[unique_id] = next_track_id
                next_track_id += 1
            else:
                # Continue parent's track (or start new if parent not found)
                if parent_id in unique_id_to_track:
                    unique_id_to_track[unique_id] = unique_id_to_track[parent_id]
                else:
                    unique_id_to_track[unique_id] = next_track_id
                    next_track_id += 1

        return unique_id_to_track

    def _compute_iou_matrix(
        self,
        pred_mask: np.ndarray,
        target_mask: np.ndarray,
        pred_ids: list[int],
        target_ids: list[int],
    ) -> np.ndarray:
        """Compute IoU matrix between predicted and target detections in a frame."""
        n_pred = len(pred_ids)
        n_target = len(target_ids)

        if n_pred == 0 or n_target == 0:
            return np.zeros((n_target, n_pred))

        iou_matrix = np.zeros((n_target, n_pred))

        for i, tid in enumerate(target_ids):
            target_region = target_mask == tid
            for j, pid in enumerate(pred_ids):
                pred_region = pred_mask == pid
                intersection = np.logical_and(target_region, pred_region).sum()
                union = np.logical_or(target_region, pred_region).sum()
                if union > 0:
                    iou_matrix[i, j] = intersection / union

        return iou_matrix

    def _prepare_trackeval_data(
        self,
        pred_arr: np.ndarray,
        target_arr: np.ndarray,
        pred_csv: pd.DataFrame,
        target_csv: pd.DataFrame,
    ) -> dict[str, Any]:
        """Prepare data in TrackEval HOTA format."""
        # Build track mappings
        pred_unique_to_track = self._build_tracks(pred_csv)
        target_unique_to_track = self._build_tracks(target_csv)

        n_frames = pred_arr.shape[0]

        # Collect unique track IDs
        gt_track_ids = set(target_unique_to_track.values())
        pred_track_ids = set(pred_unique_to_track.values())

        # Initialize accumulators
        data = {
            "num_timesteps": n_frames,
            "num_gt_ids": len(gt_track_ids),
            "num_tracker_ids": len(pred_track_ids),
            "num_gt_dets": 0,
            "num_tracker_dets": 0,
            "gt_ids": [],
            "tracker_ids": [],
            "similarity_scores": [],
        }

        for t in range(n_frames):
            pred_frame = pred_arr[t]
            target_frame = target_arr[t]

            # Get detection IDs present in this frame from masks
            pred_ids_in_frame = [int(x) for x in np.unique(pred_frame) if x != 0]
            target_ids_in_frame = [int(x) for x in np.unique(target_frame) if x != 0]

            # Map to track IDs
            gt_track_ids_frame = np.array(
                [target_unique_to_track.get(uid, -1) for uid in target_ids_in_frame]
            )
            pred_track_ids_frame = np.array(
                [pred_unique_to_track.get(uid, -1) for uid in pred_ids_in_frame]
            )

            # Filter out unmapped IDs
            valid_gt = gt_track_ids_frame != -1
            valid_pred = pred_track_ids_frame != -1

            gt_track_ids_frame = gt_track_ids_frame[valid_gt]
            pred_track_ids_frame = pred_track_ids_frame[valid_pred]
            target_ids_in_frame = [
                uid for uid, v in zip(target_ids_in_frame, valid_gt) if v
            ]
            pred_ids_in_frame = [
                uid for uid, v in zip(pred_ids_in_frame, valid_pred) if v
            ]

            # Compute IoU similarity matrix
            similarity = self._compute_iou_matrix(
                pred_frame, target_frame, pred_ids_in_frame, target_ids_in_frame
            )

            data["gt_ids"].append(gt_track_ids_frame)
            data["tracker_ids"].append(pred_track_ids_frame)
            data["similarity_scores"].append(similarity)
            data["num_gt_dets"] += len(gt_track_ids_frame)
            data["num_tracker_dets"] += len(pred_track_ids_frame)

        return data

    def compute(self) -> dict[str, Any]:
        """Compute HOTA metric.

        Returns
        -------
        dict
            Dictionary containing HOTA metrics:
            - HOTA: Main HOTA score (geometric mean of DetA and AssA)
            - DetA: Detection accuracy
            - AssA: Association accuracy
            - DetRe: Detection recall
            - DetPr: Detection precision
            - AssRe: Association recall
            - AssPr: Association precision
            - LocA: Localization accuracy
        """
        pred_arr, target_arr, pred_csv, target_csv = self._load_data()
        data = self._prepare_trackeval_data(pred_arr, target_arr, pred_csv, target_csv)

        # Run HOTA computation
        results = self._hota_metric.eval_sequence(data)

        # Extract key metrics (averaged over alpha thresholds)
        output = {}
        for key in ["HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA"]:
            if key in results:
                # HOTA returns arrays for different alpha thresholds, take mean
                values = results[key]
                if isinstance(values, np.ndarray):
                    output[key] = float(np.mean(values))
                else:
                    output[key] = float(values)

        # Also include per-alpha results for detailed analysis
        output["per_alpha"] = {
            key: results[key].tolist()
            if isinstance(results.get(key), np.ndarray)
            else results.get(key)
            for key in ["HOTA", "DetA", "AssA"]
            if key in results
        }

        return output
