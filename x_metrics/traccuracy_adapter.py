"""Traccuracy cell tracking metrics adapter."""

from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import zarr
from data_utils import load_csv_data
from traccuracy import TrackingGraph
from traccuracy.matchers._base import Matcher
from traccuracy.metrics import (
    AOGMMetrics,
    CTCMetrics,
    DivisionMetrics,
)

from x_metrics.base_adapter import BaseMetricAdapter


class TraccuracyAdapter(BaseMetricAdapter):
    """Adapter for computing cell tracking metrics using traccuracy.

    Traccuracy provides comprehensive cell tracking evaluation metrics including:
    - CTC Metrics: TRA, DET, LNK scores from the Cell Tracking Challenge
    - Division Metrics: Precision, Recall, F1 for cell divisions
    - AOGM: Acyclic Oriented Graph Metric

    Parameters
    ----------
    zarr_path : str or Path
        Path to zarr container.
    group : str
        Name of the group containing prediction and target datasets.
    pred_dataset : str
        Name of the dataset containing predicted segmentation.
        Shape: C T Y X (C=1) for 2D, or C T Z Y X (C=1) for 3D.
    target_dataset : str
        Name of the dataset containing ground truth segmentation.
        Shape: C T Y X (C=1) for 2D, or C T Z Y X (C=1) for 3D.
    pred_csv_path : str or Path
        Path to predicted tracks CSV with columns: group, id, t, y, x, parent_id
        for 2D, or group, id, t, z, y, x, parent_id for 3D
        (may have additional columns which will be dropped).
    target_csv_path : str or Path
        Path to ground truth tracks CSV with columns: group, id, t, y, x, parent_id
        for 2D, or group, id, t, z, y, x, parent_id for 3D
        (may have additional columns which will be dropped).
    matcher : traccuracy.matchers.Matcher
        A traccuracy Matcher object for matching detections between ground truth
        and predictions. Options include:
        - IOUMatcher(iou_threshold=0.5, one_to_one=True): Match by segmentation IoU
        - PointMatcher(threshold=10.0): Match by point distance
        - CTCMatcher(): Cell Tracking Challenge matching
    voxel_size : dict[str, float], optional
        Scaling factors for spatial coordinates. Keys should be 'x', 'y', and
        optionally 'z' for 3D data. Including 'z' enables 3D mode.
        Default is {"x": 1.0, "y": 1.0} (2D).
    """

    def __init__(
        self,
        zarr_path: str | Path,
        group: str,
        pred_dataset: str,
        target_dataset: str,
        pred_csv_path: str | Path,
        target_csv_path: str | Path,
        matcher: Matcher,
        voxel_size: dict[str, float] | None = None,
    ):
        if voxel_size is None:
            voxel_size = {"x": 1.0, "y": 1.0}
        self.zarr_path = Path(zarr_path)
        self.group = group
        self.pred_dataset = pred_dataset
        self.target_dataset = target_dataset
        self.pred_csv_path = Path(pred_csv_path)
        self.target_csv_path = Path(target_csv_path)
        self.matcher = matcher
        self.voxel_size = voxel_size
        self.is_3d = "z" in voxel_size

    def _load_csv(self, csv_path: Path) -> np.ndarray:
        """Load and filter CSV tracking data.

        Filters rows where group == group.

        Returns
        -------
        np.ndarray
            Numerical data array with columns [id, t, y, x, parent_id] for 2D,
            or [id, t, z, y, x, parent_id] for 3D.
        """
        numerical_data, *_ = load_csv_data(
            str(csv_path), voxel_size=self.voxel_size, groups=[self.group]
        )
        return numerical_data

    def _load_data(
        self,
    ) -> tuple[zarr.Array, zarr.Array, np.ndarray, np.ndarray]:
        """Load zarr arrays (lazy) and CSV tracking data.

        Zarr arrays are returned without loading into memory - frames are
        loaded on-demand when accessed during iteration.
        """
        root = zarr.open(self.zarr_path, mode="r")
        group = root[self.group]

        # Keep as zarr arrays for lazy loading (first channel, C=1)
        pred_arr = group[self.pred_dataset][0]
        target_arr = group[self.target_dataset][0]

        pred_csv = self._load_csv(self.pred_csv_path)
        target_csv = self._load_csv(self.target_csv_path)

        return pred_arr, target_arr, pred_csv, target_csv

    def _build_tracking_graph(
        self,
        csv_data: np.ndarray,
        seg_arr: zarr.Array,
    ) -> TrackingGraph:
        """Build a traccuracy TrackingGraph from CSV and segmentation data.

        Parameters
        ----------
        csv_data : np.ndarray
            Numerical data array with columns [id, t, y, x, parent_id] for 2D,
            or [id, t, z, y, x, parent_id] for 3D.
        seg_arr : zarr.Array
            Segmentation array with shape (T, Y, X) for 2D or (T, Z, Y, X) for 3D.

        Returns
        -------
        TrackingGraph
            Traccuracy tracking graph with nodes and edges.
        """
        graph = nx.DiGraph()

        # Column indices differ for 2D vs 3D
        # 2D: [id, t, y, x, parent_id]
        # 3D: [id, t, z, y, x, parent_id]
        if self.is_3d:
            col_z, col_y, col_x, col_parent = 2, 3, 4, 5
        else:
            col_y, col_x, col_parent = 2, 3, 4

        # Build node info mapping
        node_info = {}
        for row in csv_data:
            node_id = int(row[0])
            info = {
                "t": int(row[1]),
                "y": float(row[col_y]),
                "x": float(row[col_x]),
                "parent_id": int(row[col_parent]),
            }
            if self.is_3d:
                info["z"] = float(row[col_z])
            node_info[node_id] = info

        # Add nodes with attributes
        for node_id, info in node_info.items():
            attrs = {
                "t": info["t"],
                "y": info["y"],
                "x": info["x"],
                "segmentation_id": node_id,
            }
            if self.is_3d:
                attrs["z"] = info["z"]
            graph.add_node(node_id, **attrs)

        # Add edges based on parent_id relationships
        for node_id, info in node_info.items():
            parent_id = info["parent_id"]
            if parent_id != 0 and parent_id in node_info:
                # Edge goes from parent to child (forward in time)
                graph.add_edge(parent_id, node_id)

        # Load segmentation into memory for TrackingGraph
        # TrackingGraph expects segmentation as numpy array
        print("Loading segmentation data...")
        seg_np = np.array(seg_arr)

        location_keys = ("z", "y", "x") if self.is_3d else ("y", "x")

        return TrackingGraph(
            graph=graph,
            segmentation=seg_np,
            frame_key="t",
            label_key="segmentation_id",
            location_keys=location_keys,
        )

    def compute(
        self,
        metrics: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compute traccuracy cell tracking metrics.

        Parameters
        ----------
        metrics : list[str], optional
            List of metric types to compute. Options are:
            - "ctc": CTC metrics (TRA, DET, LNK)
            - "division": Division metrics (precision, recall, F1)
            - "aogm": AOGM metric
            If None, computes all metrics.

        Returns
        -------
        dict
            Dictionary containing tracking metrics:
            - CTC_TRA: Cell Tracking Challenge TRA score
            - CTC_DET: Cell Tracking Challenge DET score
            - CTC_LNK: Cell Tracking Challenge LNK score
            - Division_Precision: Division detection precision
            - Division_Recall: Division detection recall
            - Division_F1: Division detection F1 score
            - AOGM: Acyclic Oriented Graph Metric
        """
        if metrics is None:
            metrics = ["ctc", "division", "aogm"]

        pred_arr, target_arr, pred_csv, target_csv = self._load_data()

        # Build tracking graphs
        print("Building ground truth tracking graph...")
        gt_graph = self._build_tracking_graph(target_csv, target_arr)

        print("Building prediction tracking graph...")
        pred_graph = self._build_tracking_graph(pred_csv, pred_arr)

        # Compute matching using provided matcher
        print("Computing matching...")
        matched = self.matcher.compute_mapping(gt_graph, pred_graph)

        output = {}

        # Compute CTC metrics
        if "ctc" in metrics:
            print("Computing CTC metrics...")
            ctc_metric = CTCMetrics()
            ctc_results = ctc_metric.compute(matched)

            # Extract results from the Results object
            for key, value in ctc_results.results.items():
                output[f"CTC_{key}"] = value

        # Compute Division metrics
        if "division" in metrics:
            print("Computing Division metrics...")
            div_metric = DivisionMetrics(max_frame_buffer=1)
            div_results = div_metric.compute(matched)

            for key, value in div_results.results.items():
                output[f"Division_{key}"] = value

        # Compute AOGM metric
        if "aogm" in metrics:
            print("Computing AOGM metric...")
            aogm_metric = AOGMMetrics()
            aogm_results = aogm_metric.compute(matched)

            for key, value in aogm_results.results.items():
                output[f"AOGM_{key}"] = value

        return output
