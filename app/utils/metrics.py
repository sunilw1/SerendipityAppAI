"""
Metrics and Statistics Utilities
==================================

Statistical functions for baseline learning and data quality assessment.

Key Functions:
- compute_percentiles: Calculate percentile values
- compute_baseline_stats: Generate statistical baselines
- rolling_statistics: Compute rolling window statistics
- outlier_detection: Identify statistical outliers
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import math

import numpy as np
from scipy import stats as scipy_stats

from app.models.schemas import BaselineMetrics


def compute_percentiles(
    values: Union[List[float], np.ndarray],
    percentiles: List[float] = [5, 25, 50, 75, 95]
) -> Dict[str, float]:
    """
    Compute percentile values for a dataset.
    
    Args:
        values: List or array of numeric values
        percentiles: List of percentiles to compute (0-100)
        
    Returns:
        Dict mapping percentile labels to values
    """
    if len(values) == 0:
        return {f"p{p}": float('nan') for p in percentiles}
    
    arr = np.asarray(values)
    arr = arr[~np.isnan(arr)]  # Remove NaN values
    
    if len(arr) == 0:
        return {f"p{p}": float('nan') for p in percentiles}
    
    results = {}
    for p in percentiles:
        results[f"p{p}"] = float(np.percentile(arr, p))
    
    return results


def compute_baseline_stats(
    values: Union[List[float], np.ndarray],
    metric_name: str
) -> BaselineMetrics:
    """
    Compute comprehensive baseline statistics for a metric.
    
    Args:
        values: List or array of numeric values
        metric_name: Name of the metric
        
    Returns:
        BaselineMetrics object with full statistics
    """
    arr = np.asarray(values)
    arr = arr[~np.isnan(arr)]  # Remove NaN values
    
    if len(arr) == 0:
        # Return default values for empty data
        return BaselineMetrics(
            metric_name=metric_name,
            mean=0.0,
            median=0.0,
            std=0.0,
            min_value=0.0,
            max_value=0.0,
            p5=0.0,
            p25=0.0,
            p75=0.0,
            p95=0.0,
            sample_size=0
        )
    
    return BaselineMetrics(
        metric_name=metric_name,
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        std=float(np.std(arr)),
        min_value=float(np.min(arr)),
        max_value=float(np.max(arr)),
        p5=float(np.percentile(arr, 5)),
        p25=float(np.percentile(arr, 25)),
        p75=float(np.percentile(arr, 75)),
        p95=float(np.percentile(arr, 95)),
        sample_size=len(arr)
    )


def rolling_mean(
    values: np.ndarray,
    window_size: int
) -> np.ndarray:
    """
    Compute rolling mean with the specified window size.
    
    Args:
        values: Array of values
        window_size: Size of rolling window
        
    Returns:
        Array of rolling means (first window_size-1 values are NaN)
    """
    if len(values) < window_size:
        return np.full_like(values, np.nan, dtype=float)
    
    cumsum = np.cumsum(np.insert(values, 0, 0))
    rolling = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    
    # Pad with NaN for initial values
    result = np.empty(len(values))
    result[:window_size-1] = np.nan
    result[window_size-1:] = rolling
    
    return result


def rolling_std(
    values: np.ndarray,
    window_size: int
) -> np.ndarray:
    """
    Compute rolling standard deviation.
    
    Args:
        values: Array of values
        window_size: Size of rolling window
        
    Returns:
        Array of rolling standard deviations
    """
    if len(values) < window_size:
        return np.full_like(values, np.nan, dtype=float)
    
    result = np.empty(len(values))
    result[:window_size-1] = np.nan
    
    for i in range(window_size - 1, len(values)):
        window = values[i - window_size + 1:i + 1]
        result[i] = np.std(window)
    
    return result


def detect_outliers_iqr(
    values: Union[List[float], np.ndarray],
    multiplier: float = 1.5
) -> Tuple[np.ndarray, float, float]:
    """
    Detect outliers using the IQR (Interquartile Range) method.
    
    Args:
        values: Array of values
        multiplier: IQR multiplier for bounds (1.5 is standard)
        
    Returns:
        Tuple of (boolean mask of outliers, lower bound, upper bound)
    """
    arr = np.asarray(values)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    outliers = (arr < lower_bound) | (arr > upper_bound)
    
    return outliers, lower_bound, upper_bound


def detect_outliers_zscore(
    values: Union[List[float], np.ndarray],
    threshold: float = 3.0
) -> Tuple[np.ndarray, float, float]:
    """
    Detect outliers using Z-score method.
    
    Args:
        values: Array of values
        threshold: Z-score threshold (3.0 is standard)
        
    Returns:
        Tuple of (boolean mask of outliers, mean, std)
    """
    arr = np.asarray(values)
    mean = np.mean(arr)
    std = np.std(arr)
    
    if std == 0:
        return np.zeros(len(arr), dtype=bool), mean, std
    
    z_scores = np.abs((arr - mean) / std)
    outliers = z_scores > threshold
    
    return outliers, mean, std


def detect_outliers_mad(
    values: Union[List[float], np.ndarray],
    threshold: float = 3.5
) -> Tuple[np.ndarray, float, float]:
    """
    Detect outliers using Median Absolute Deviation (robust to outliers).
    
    This method is more robust than Z-score for datasets with many outliers.
    
    Args:
        values: Array of values
        threshold: MAD threshold
        
    Returns:
        Tuple of (boolean mask of outliers, median, MAD)
    """
    arr = np.asarray(values)
    median = np.median(arr)
    mad = np.median(np.abs(arr - median))
    
    if mad == 0:
        return np.zeros(len(arr), dtype=bool), median, mad
    
    # Modified Z-score
    modified_z = 0.6745 * (arr - median) / mad
    outliers = np.abs(modified_z) > threshold
    
    return outliers, median, mad


def compute_histogram(
    values: Union[List[float], np.ndarray],
    bins: int = 20,
    range_: Optional[Tuple[float, float]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute histogram for a dataset.
    
    Args:
        values: Array of values
        bins: Number of bins
        range_: Optional (min, max) range
        
    Returns:
        Tuple of (counts, bin_edges)
    """
    arr = np.asarray(values)
    arr = arr[~np.isnan(arr)]
    
    return np.histogram(arr, bins=bins, range=range_)


def compute_distribution_stats(
    values: Union[List[float], np.ndarray]
) -> Dict[str, float]:
    """
    Compute distribution shape statistics.
    
    Args:
        values: Array of values
        
    Returns:
        Dict with skewness, kurtosis, and normality test p-value
    """
    arr = np.asarray(values)
    arr = arr[~np.isnan(arr)]
    
    if len(arr) < 8:  # Need minimum samples for these tests
        return {
            "skewness": float('nan'),
            "kurtosis": float('nan'),
            "normality_pvalue": float('nan')
        }
    
    result = {
        "skewness": float(scipy_stats.skew(arr)),
        "kurtosis": float(scipy_stats.kurtosis(arr)),
    }
    
    # Shapiro-Wilk test for normality (limited to 5000 samples)
    if len(arr) <= 5000:
        _, p_value = scipy_stats.shapiro(arr)
        result["normality_pvalue"] = float(p_value)
    else:
        # Use D'Agostino-Pearson for larger samples
        _, p_value = scipy_stats.normaltest(arr)
        result["normality_pvalue"] = float(p_value)
    
    return result


def compute_correlation(
    x: Union[List[float], np.ndarray],
    y: Union[List[float], np.ndarray]
) -> Tuple[float, float]:
    """
    Compute Pearson correlation coefficient with p-value.
    
    Args:
        x: First variable
        y: Second variable
        
    Returns:
        Tuple of (correlation coefficient, p-value)
    """
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    
    # Remove pairs with NaN
    mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
    x_clean = x_arr[mask]
    y_clean = y_arr[mask]
    
    if len(x_clean) < 3:
        return float('nan'), float('nan')
    
    corr, p_value = scipy_stats.pearsonr(x_clean, y_clean)
    return float(corr), float(p_value)


def exponential_moving_average(
    values: np.ndarray,
    alpha: float = 0.3
) -> np.ndarray:
    """
    Compute exponential moving average.
    
    Args:
        values: Array of values
        alpha: Smoothing factor (0-1, higher = less smoothing)
        
    Returns:
        Array of EMA values
    """
    ema = np.empty(len(values))
    ema[0] = values[0]
    
    for i in range(1, len(values)):
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i-1]
    
    return ema


def compute_activity_distribution(
    activities: List[str]
) -> Dict[str, float]:
    """
    Compute distribution of activity types.
    
    Args:
        activities: List of activity type strings
        
    Returns:
        Dict mapping activity type to percentage (0-1)
    """
    if not activities:
        return {}
    
    counts: Dict[str, int] = {}
    for activity in activities:
        counts[activity] = counts.get(activity, 0) + 1
    
    total = len(activities)
    return {k: v / total for k, v in counts.items()}


def compute_confidence_distribution(
    confidence_scores: List[float],
    buckets: List[float] = [0.25, 0.5, 0.75, 0.9]
) -> Dict[str, int]:
    """
    Compute distribution of confidence scores across buckets.
    
    Args:
        confidence_scores: List of confidence scores (0-1)
        buckets: Bucket boundaries
        
    Returns:
        Dict with count in each bucket
    """
    extended_buckets = [0.0] + buckets + [1.01]
    labels = []
    for i in range(len(extended_buckets) - 1):
        labels.append(f"{extended_buckets[i]:.2f}-{extended_buckets[i+1]:.2f}")
    
    counts = {label: 0 for label in labels}
    
    for score in confidence_scores:
        for i in range(len(extended_buckets) - 1):
            if extended_buckets[i] <= score < extended_buckets[i+1]:
                counts[labels[i]] += 1
                break
    
    return counts
