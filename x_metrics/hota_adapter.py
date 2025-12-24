"""HOTA and Identity metrics adapter using TrackEval."""

from pathlib import Path
from typing import Any

import numpy as np
import zarr
from data_utils import load_csv_data
from tqdm import tqdm
from trackeval.metrics import HOTA, Identity

from x_metrics.base_adapter import BaseMetricAdapter


class HOTAAdapter(BaseMetricAdapter):
    """Adapter for computing HOTA and Identity metrics on zarr segmentation data with CSV tracks.

    HOTA decomposes tracking performance into:
    - DetA (Detection Accuracy): How well objects are detected
    - AssA (Association Accuracy): How well objects are associated across time
    - HOTA = sqrt(DetA * AssA)

    Identity metrics measure ID consistency:
    - IDF1: Identity F1 score (harmonic mean of IDP and IDR)
    - IDP: Identity Precision
    - IDR: Identity Recall

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
        Path to predicted tracks CSV with columns: sequence, id, t, y, x, parent_id
        (may have additional columns which will be dropped).
    target_csv_path : str or Path
        Path to ground truth tracks CSV with columns: sequence, id, t, y, x, parent_id
        (may have additional columns which will be dropped).
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
        self._identity_metric = Identity()

    def _load_csv(self, csv_path: Path) -> np.ndarray:
        """Load and filter CSV tracking data.

        Filters rows where sequence == group.

        Returns
        -------
        np.ndarray
            Numerical data array with columns [id, t, y, x, parent_id].
        """
        numerical_data, _, _, _ = load_csv_data(
            str(csv_path), sequences=[self.group]
        )
        return numerical_data

    def _load_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load zarr arrays and CSV tracking data."""
        root = zarr.open(self.zarr_path, mode="r")
        group = root[self.group]

        # Load datasets and take first channel (C=1)
        pred_arr = np.asarray(group[self.pred_dataset][0])
        target_arr = np.asarray(group[self.target_dataset][0])

        pred_csv = self._load_csv(self.pred_csv_path)
        target_csv = self._load_csv(self.target_csv_path)

        return pred_arr, target_arr, pred_csv, target_csv

    def _build_tracks(self, csv_data: np.ndarray) -> dict[int, int]:
        """Build track ID mapping from CSV data.

        Each detection gets assigned to a track. Detections with parent_id=0
        start new tracks. Detections with parent_id!=0 continue the parent's track.

        Parameters
        ----------
        csv_data : np.ndarray
            Numerical data array with columns [id, t, y, x, parent_id].

        Returns
        -------
        dict
            Mapping from id to track_id.
        """
        # Sort by t (column 1) to process in temporal order
        sorted_indices = np.argsort(csv_data[:, 1])
        sorted_data = csv_data[sorted_indices]

        id_to_track = {}
        next_track_id = 0

        for row in sorted_data:
            id_ = int(row[0])  # id column
            parent_id = int(row[4])  # parent_id column

            if parent_id == 0:
                # New track starts
                id_to_track[id_] = next_track_id
                next_track_id += 1
            else:
                # Continue parent's track (or start new if parent not found)
                if parent_id in id_to_track:
                    id_to_track[id_] = id_to_track[parent_id]
                else:
                    id_to_track[id_] = next_track_id
                    next_track_id += 1

        return id_to_track

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
        pred_csv: np.ndarray,
        target_csv: np.ndarray,
    ) -> dict[str, Any]:
        """Prepare data in TrackEval HOTA format."""
        # Build track mappings
        pred_id_to_track = self._build_tracks(pred_csv)
        target_id_to_track = self._build_tracks(target_csv)

        n_frames = pred_arr.shape[0]

        # Collect unique track IDs
        gt_track_ids = set(target_id_to_track.values())
        pred_track_ids = set(pred_id_to_track.values())

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

        for t in tqdm(range(n_frames), desc="Processing frames"):
            pred_frame = pred_arr[t]
            target_frame = target_arr[t]

            # Get detection IDs present in this frame from masks
            pred_ids_in_frame = [int(x) for x in np.unique(pred_frame) if x != 0]
            target_ids_in_frame = [int(x) for x in np.unique(target_frame) if x != 0]

            # Map to track IDs
            gt_track_ids_frame = np.array(
                [target_id_to_track.get(id_, -1) for id_ in target_ids_in_frame]
            )
            pred_track_ids_frame = np.array(
                [pred_id_to_track.get(id_, -1) for id_ in pred_ids_in_frame]
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

    def find_association_errors(
        self, iou_threshold: float | None = None
    ) -> list[dict[str, Any]]:
        """Find locations where predicted links differ from ground truth.

        This identifies detections where the predicted track assignment doesn't
        match the ground truth track assignment, helping debug tracking errors.

        Parameters
        ----------
        iou_threshold : float, optional
            IoU threshold for matching detections. Defaults to self.iou_threshold.

        Returns
        -------
        list[dict]
            List of association errors, each containing:
            - t: Frame index
            - y: Y coordinate (centroid)
            - x: X coordinate (centroid)
            - detection_id: The detection ID in the segmentation
            - gt_track_id: Ground truth track ID
            - pred_track_id: Predicted track ID
            - expected_pred_track: The pred track that should match this GT track
            - error_type: 'id_switch' if detection assigned to wrong track,
                         'fragmentation' if GT track split across pred tracks
        """
        if iou_threshold is None:
            iou_threshold = self.iou_threshold

        pred_arr, target_arr, pred_csv, target_csv = self._load_data()

        # Build track mappings
        pred_id_to_track = self._build_tracks(pred_csv)
        target_id_to_track = self._build_tracks(target_csv)

        # Build detection ID to (y, x) coordinate mapping from CSV
        # CSV columns: [id, t, y, x, parent_id]
        target_id_to_coords = {
            int(row[0]): (float(row[2]), float(row[3])) for row in target_csv
        }

        n_frames = pred_arr.shape[0]

        # First pass: count overlap between GT tracks and pred tracks
        # track_overlap[gt_track][pred_track] = count of matched detections
        track_overlap: dict[int, dict[int, int]] = {}

        matched_detections = []  # Store for second pass

        for t in tqdm(range(n_frames), desc="Analyzing associations"):
            pred_frame = pred_arr[t]
            target_frame = target_arr[t]

            # Get detection IDs
            pred_ids_in_frame = [int(x) for x in np.unique(pred_frame) if x != 0]
            target_ids_in_frame = [int(x) for x in np.unique(target_frame) if x != 0]

            # Filter to valid IDs (present in CSV)
            target_ids_valid = [
                uid for uid in target_ids_in_frame if uid in target_id_to_track
            ]
            pred_ids_valid = [
                uid for uid in pred_ids_in_frame if uid in pred_id_to_track
            ]

            if not target_ids_valid or not pred_ids_valid:
                continue

            # Compute IoU matrix
            iou_matrix = self._compute_iou_matrix(
                pred_frame, target_frame, pred_ids_valid, target_ids_valid
            )

            # Match detections based on IoU threshold
            for i, gt_det_id in enumerate(target_ids_valid):
                # Find best matching pred detection
                best_j = np.argmax(iou_matrix[i])
                if iou_matrix[i, best_j] >= iou_threshold:
                    pred_det_id = pred_ids_valid[best_j]
                    gt_track = target_id_to_track[gt_det_id]
                    pred_track = pred_id_to_track[pred_det_id]

                    # Count overlap
                    if gt_track not in track_overlap:
                        track_overlap[gt_track] = {}
                    track_overlap[gt_track][pred_track] = (
                        track_overlap[gt_track].get(pred_track, 0) + 1
                    )

                    # Store for second pass
                    coords = target_id_to_coords.get(gt_det_id, (0.0, 0.0))
                    matched_detections.append(
                        {
                            "t": t,
                            "y": coords[0],
                            "x": coords[1],
                            "detection_id": gt_det_id,
                            "gt_track_id": gt_track,
                            "pred_track_id": pred_track,
                        }
                    )

        # Build optimal GT track → pred track mapping (most common assignment)
        gt_to_pred_track: dict[int, int] = {}
        for gt_track, pred_counts in track_overlap.items():
            if pred_counts:
                gt_to_pred_track[gt_track] = max(pred_counts, key=pred_counts.get)

        # Second pass: find errors
        errors = []
        for det in matched_detections:
            gt_track = det["gt_track_id"]
            pred_track = det["pred_track_id"]
            expected_pred = gt_to_pred_track.get(gt_track)

            if expected_pred is not None and pred_track != expected_pred:
                # This is an association error
                error_type = "id_switch"
                # Check if this GT track is fragmented (appears in multiple pred tracks)
                if len(track_overlap.get(gt_track, {})) > 1:
                    error_type = "fragmentation"

                errors.append(
                    {
                        "t": det["t"],
                        "y": det["y"],
                        "x": det["x"],
                        "detection_id": det["detection_id"],
                        "gt_track_id": gt_track,
                        "pred_track_id": pred_track,
                        "expected_pred_track": expected_pred,
                        "error_type": error_type,
                    }
                )

        return errors

    def compute(self) -> dict[str, Any]:
        """Compute HOTA and Identity metrics.

        Returns
        -------
        dict
            Dictionary containing tracking metrics:
            - HOTA: Main HOTA score (geometric mean of DetA and AssA)
            - DetA: Detection accuracy
            - AssA: Association accuracy
            - DetRe: Detection recall
            - DetPr: Detection precision
            - AssRe: Association recall
            - AssPr: Association precision
            - LocA: Localization accuracy
            - IDF1: Identity F1 score
            - IDP: Identity precision
            - IDR: Identity recall
        """
        pred_arr, target_arr, pred_csv, target_csv = self._load_data()
        data = self._prepare_trackeval_data(pred_arr, target_arr, pred_csv, target_csv)

        # Run HOTA computation
        hota_results = self._hota_metric.eval_sequence(data)

        # Run Identity computation
        identity_results = self._identity_metric.eval_sequence(data)

        # Extract key HOTA metrics (averaged over alpha thresholds)
        output = {}
        for key in ["HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA"]:
            if key in hota_results:
                # HOTA returns arrays for different alpha thresholds, take mean
                values = hota_results[key]
                if isinstance(values, np.ndarray):
                    output[key] = float(np.mean(values))
                else:
                    output[key] = float(values)

        # Extract Identity metrics (IDF1, IDP, IDR)
        for key in ["IDF1", "IDP", "IDR"]:
            if key in identity_results:
                values = identity_results[key]
                if isinstance(values, np.ndarray):
                    output[key] = float(np.mean(values))
                else:
                    output[key] = float(values)

        # Also include per-alpha results for detailed analysis
        output["per_alpha"] = {
            key: hota_results[key].tolist()
            if isinstance(hota_results.get(key), np.ndarray)
            else hota_results.get(key)
            for key in ["HOTA", "DetA", "AssA"]
            if key in hota_results
        }

        return output
