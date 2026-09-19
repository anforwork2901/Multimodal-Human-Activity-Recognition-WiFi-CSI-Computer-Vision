"""CSI signal processing filters and preprocessing pipeline."""

import ast
from typing import Any, List, Union
import numpy as np
import pandas as pd
from src.domain.interfaces import ICSIPreprocessor


def moving_average_filter(data: List[float], window_size: int = 5) -> List[float]:
    """Moving average filter for CSI denoising (GaitFi methodology)."""
    if len(data) < window_size:
        return data

    filtered_data = []
    n = len(data)
    half_window = window_size // 2
    for i in range(n):
        start_idx = max(0, i - half_window)
        end_idx = min(n, i + half_window + 1)
        window = data[start_idx:end_idx]
        filtered_data.append(float(np.mean(window)))

    return filtered_data


def enhanced_hampel_filter(
    data: Union[List[float], np.ndarray], K: int = 3, n_sigmas: float = 3.0
) -> List[float]:
    """
    Enhanced Hampel filter with Median Absolute Deviation (MAD) for outlier removal.
    """
    arr = np.array(data, dtype=float)
    filtered = arr.copy()
    n = len(arr)

    for i in range(n):
        start_idx = max(0, i - K)
        end_idx = min(n, i + K + 1)
        window = arr[start_idx:end_idx]

        median = np.median(window)
        mad = np.median(np.abs(window - median))

        if mad > 0:
            threshold = n_sigmas * 1.4826 * mad
            if np.abs(arr[i] - median) > threshold:
                filtered[i] = median
        else:
            std_dev = np.std(window)
            if std_dev > 0:
                threshold = n_sigmas * std_dev
                if np.abs(arr[i] - median) > threshold:
                    filtered[i] = median

    return filtered.tolist()


def normalize_csi(csi_data: Union[List[float], np.ndarray]) -> List[float]:
    """Normalize CSI data array to the [-1.0, 1.0] range."""
    arr = np.array(csi_data, dtype=float)
    if len(arr) == 0:
        return []

    min_val = np.min(arr)
    max_val = np.max(arr)

    if max_val == min_val:
        return np.zeros_like(arr).tolist()

    normalized = 2.0 * (arr - min_val) / (max_val - min_val) - 1.0
    return normalized.tolist()


class CSIPreprocessor(ICSIPreprocessor):
    """
    Standard CSI Preprocessing Pipeline adhering to ICSIPreprocessor.
    Handles string parsing, filtering, normalization, and dimension alignment.
    """

    def __init__(self, target_length: int = 102):
        self.target_length = target_length

    def preprocess(self, raw_csi: Any) -> List[float]:
        try:
            if pd.isna(raw_csi) or raw_csi is None or raw_csi == "":
                return [0.0] * self.target_length

            # 1. Parse string or array to list of floats
            csi_data: List[float] = []
            if isinstance(raw_csi, str):
                try:
                    parsed = ast.literal_eval(raw_csi)
                    if isinstance(parsed, (list, tuple, np.ndarray)):
                        csi_data = [float(x) for x in parsed if not pd.isna(x)]
                except (ValueError, SyntaxError):
                    clean = raw_csi.strip("[]").replace(" ", "")
                    if clean:
                        csi_data = [float(x) for x in clean.split(",") if x.strip()]
            elif isinstance(raw_csi, (list, tuple, np.ndarray)):
                csi_data = [float(x) for x in raw_csi if not pd.isna(x)]

            if not csi_data:
                return [0.0] * self.target_length

            # 2. Moving average smoothing
            if len(csi_data) >= 3:
                csi_smoothed = moving_average_filter(csi_data, window_size=3)
            else:
                csi_smoothed = csi_data

            # 3. Hampel outlier filtering
            if len(csi_smoothed) >= 5:
                csi_filtered = enhanced_hampel_filter(csi_smoothed, K=2, n_sigmas=3.0)
            else:
                csi_filtered = csi_smoothed

            # 4. Normalize to [-1, 1]
            csi_norm = normalize_csi(csi_filtered)

            # 5. Dimension alignment (Pad with mean or truncate)
            if len(csi_norm) < self.target_length:
                mean_val = float(np.mean(csi_norm)) if csi_norm else 0.0
                csi_norm.extend([mean_val] * (self.target_length - len(csi_norm)))
            elif len(csi_norm) > self.target_length:
                csi_norm = csi_norm[: self.target_length]

            return csi_norm

        except Exception:
            return [0.0] * self.target_length
