"""
Principal Component Analysis (NumPy-only).

This module provides a simple, dependency-free implementation of PCA that
mimics the core behavior of ``sklearn.decomposition.PCA`` while staying
lightweight and easy to read. It supports:

- Automatic centering of data (stores ``mean_``)
- Singular Value Decomposition (SVD)-based PCA
- Flexible component selection:
  * ``n_components=None``  -> keep all components
  * ``n_components=int``   -> keep the first k components
  * ``n_components=float`` -> keep enough components to explain
                              the given fraction of variance in (0, 1]
- Optional whitening via ``whiten=True``
- Standard attributes:
  ``components_``, ``explained_variance_``, ``explained_variance_ratio_``,
  ``singular_values_``, ``mean_``, ``n_components_``, ``n_features_in_``,
  ``noise_variance_``
- Methods:
  ``fit``, ``transform``, ``fit_transform``, ``inverse_transform``

The implementation uses only NumPy and is designed for teaching and small-scale
experiments, not for large production workloads.

Examples
--------
Basic usage with an exact 1D subspace:

>>> import numpy as np
>>> from rice_ml.decomposition.pca import PCA
>>> X = np.array([[0., 0.],
...               [1., 1.],
...               [2., 2.]], dtype=float)
>>> pca = PCA(n_components=1).fit(X)
>>> X_t = pca.transform(X)
>>> X_t.shape
(3, 1)
>>> X_inv = pca.inverse_transform(X_t)
>>> bool(np.allclose(X, X_inv))  # rank-1 data, so 1 component is enough
True

Keeping all components:

>>> X = np.array([[1., 2., 3.],
...               [4., 5., 6.]], dtype=float)
>>> pca = PCA().fit(X)
>>> pca.components_.shape
(2, 3)
>>> X_t = pca.transform(X)
>>> X_inv = pca.inverse_transform(X_t)
>>> bool(np.allclose(X, X_inv))
True

Explained variance ratio:

>>> X = np.array([[0., 0.],
...               [1., 0.],
...               [0., 1.],
...               [1., 1.]], dtype=float)
>>> pca = PCA(n_components=2).fit(X)
>>> round(float(pca.explained_variance_ratio_.sum()), 6)
1.0
"""

from __future__ import annotations

from typing import Optional, Union, Sequence

import numpy as np

__all__ = [
    "PCA",
]

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


# ----------------------------- Helper Functions -----------------------------


def _ensure_2d_float(X: ArrayLike, name: str = "X") -> np.ndarray:
    """Ensure X is a 2D numeric ndarray of dtype float.

    Parameters
    ----------
    X : array_like
        Input data.
    name : str, default="X"
        Name used in error messages.

    Returns
    -------
    arr : ndarray, shape (n_samples, n_features)
        2D float64 array.

    Raises
    ------
    ValueError
        If X is not 2D or empty.
    TypeError
        If X cannot be converted to a numeric array.
    """
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array; got {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.issubdtype(arr.dtype, np.number):
        try:
            arr = arr.astype(float, copy=False)
        except (TypeError, ValueError) as e:
            raise TypeError(f"All elements of {name} must be numeric.") from e
    else:
        arr = arr.astype(float, copy=False)
    return arr


def _validate_n_components(n_components, n_max: int) -> None:
    """Validate n_components against maximum allowed value.

    Parameters
    ----------
    n_components : None, int, or float
        Desired components.
    n_max : int
        Maximum allowable number of components.

    Raises
    ------
    ValueError
        If ``n_components`` is out of range.
    TypeError
        If ``n_components`` is of unsupported type.
    """
    if n_components is None:
        return
    if isinstance(n_components, (int, np.integer)):
        if n_components < 1 or n_components > n_max:
            raise ValueError(
                f"n_components={n_components} must be in [1, {n_max}] "
                f"for the given data."
            )
    elif isinstance(n_components, float):
        if not (0.0 < n_components <= 1.0):
            raise ValueError(
                "When n_components is a float, it must be in the interval (0, 1]."
            )
    else:
        raise TypeError(
            "n_components must be None, an integer, or a float in (0, 1]."
        )


# ----------------------------------- PCA -----------------------------------


class PCA:
    """
    Principal Component Analysis (PCA).

    This class implements PCA using Singular Value Decomposition (SVD) on
    mean-centered data. It is inspired by ``sklearn.decomposition.PCA`` but
    uses only NumPy and a simplified interface.

    Parameters
    ----------
    n_components : None, int, or float, default=None
        Number of components to keep.

        - If ``None``: keep all components (i.e., ``min(n_samples, n_features)``).
        - If ``int``: keep exactly ``n_components``.
        - If ``float`` in (0, 1]: keep the minimum number of components such that
          the cumulative explained variance ratio is at least that fraction.
    whiten : bool, default=False
        If True, the projected data is divided by ``sqrt(explained_variance_)``
        so that each component has unit variance. Whitening can sometimes
        improve performance of downstream models but may also amplify noise.

    Attributes
    ----------
    components_ : ndarray, shape (n_components_, n_features)
        Principal axes in feature space. Each row is a principal component.
    explained_variance_ : ndarray, shape (n_components_,)
        Variance explained by each of the selected components.
    explained_variance_ratio_ : ndarray, shape (n_components_,)
        Fraction of total variance explained by each selected component.
    singular_values_ : ndarray, shape (n_components_,)
        Singular values corresponding to each selected component.
    mean_ : ndarray, shape (n_features,)
        Per-feature empirical mean, subtracted from data during fitting.
    n_components_ : int
        Number of components actually kept after fitting.
    n_features_in_ : int
        Number of features in the input data used to fit the model.
    n_samples_ : int
        Number of samples in the input data used to fit the model.
    noise_variance_ : float
        Estimated noise variance (mean of discarded eigenvalues). Zero when
        all components are kept.

    Notes
    -----
    - This implementation always uses a full SVD via ``np.linalg.svd``.
    - Data is always centered before SVD.
    - There is no support for sparse input or incremental fitting.

    Examples
    --------
    Basic usage:

    >>> import numpy as np
    >>> from rice_ml.decomposition.pca import PCA
    >>> X = np.array([[0., 0.],
    ...               [1., 0.],
    ...               [0., 1.],
    ...               [1., 1.]], dtype=float)
    >>> pca = PCA(n_components=2).fit(X)
    >>> X_t = pca.transform(X)
    >>> X_t.shape
    (4, 2)
    >>> X_inv = pca.inverse_transform(X_t)
    >>> bool(np.allclose(X, X_inv))
    True

    Using a variance fraction:

    >>> X = np.array([[2., 0.],
    ...               [0., 2.],
    ...               [1., 1.],
    ...               [3., 3.]], dtype=float)
    >>> pca = PCA(n_components=0.9).fit(X)
    >>> pca.n_components_  # doctest: +SKIP
    1
    """

    def __init__(self, n_components: Optional[Union[int, float]] = None, *, whiten: bool = False) -> None:
        self.n_components = n_components
        self.whiten = bool(whiten)

        # Attributes set during fit
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.singular_values_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.n_components_: Optional[int] = None
        self.n_features_in_: Optional[int] = None
        self.n_samples_: Optional[int] = None
        self.noise_variance_: Optional[float] = None

    # ------------------------------ Internals ------------------------------

    def _check_is_fitted(self) -> None:
        """Check that the PCA instance is fitted."""
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("PCA instance is not fitted yet. Call 'fit(X)' first.")

    # ------------------------------ API Methods ------------------------------

    def fit(self, X: ArrayLike, y: None = None) -> "PCA":
        """
        Fit the PCA model on X.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training data.
        y : None, ignored
            Present for API compatibility; not used.

        Returns
        -------
        self : PCA
            Fitted estimator.
        """
        X_arr = _ensure_2d_float(X, "X")
        n_samples, n_features = X_arr.shape
        self.n_samples_ = n_samples
        self.n_features_in_ = n_features

        # Center data
        self.mean_ = np.mean(X_arr, axis=0)
        X_centered = X_arr - self.mean_

        # SVD
        # X_centered = U * S * Vt
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Eigenvalues of covariance matrix (S^2 / (n_samples - 1))
        if n_samples > 1:
            eigenvalues = (S ** 2) / (n_samples - 1)
        else:
            # Degenerate case: single sample
            eigenvalues = np.zeros_like(S, dtype=float)

        total_variance = float(eigenvalues.sum())
        n_max = min(n_samples, n_features)
        _validate_n_components(self.n_components, n_max)

        # Determine how many components to keep
        if self.n_components is None:
            n_keep = n_max
        elif isinstance(self.n_components, (int, np.integer)):
            n_keep = int(self.n_components)
        else:  # float in (0, 1]
            target = float(self.n_components)
            if total_variance == 0.0:
                # All-zero variance: keep 1 component by convention.
                n_keep = 1
            else:
                cumsum_ratio = np.cumsum(eigenvalues) / total_variance
                # Smallest index where cumulative ratio >= target
                n_keep = int(np.searchsorted(cumsum_ratio, target) + 1)

        # Slice SVD outputs
        self.n_components_ = n_keep
        self.components_ = Vt[:n_keep, :]
        self.singular_values_ = S[:n_keep]
        self.explained_variance_ = eigenvalues[:n_keep]

        if total_variance > 0.0:
            self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        else:
            # Zero variance everywhere; define ratios as zeros
            self.explained_variance_ratio_ = np.zeros_like(self.explained_variance_)

        # Noise variance: mean of discarded eigenvalues
        if n_keep < len(eigenvalues):
            self.noise_variance_ = float(np.mean(eigenvalues[n_keep:]))
        else:
            self.noise_variance_ = 0.0

        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Apply dimensionality reduction to X.

        The data is projected onto the principal components learned during
        ``fit``, optionally whitened.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Data to project.

        Returns
        -------
        X_transformed : ndarray, shape (n_samples, n_components_)
            Projected data in principal component space.

        Raises
        ------
        RuntimeError
            If called before ``fit``.
        ValueError
            If the number of features in X does not match the fitted model.
        """
        self._check_is_fitted()
        X_arr = _ensure_2d_float(X, "X")

        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, but PCA was fitted on "
                f"{self.n_features_in_} features."
            )

        X_centered = X_arr - self.mean_
        # project
        X_proj = X_centered @ self.components_.T  # (n_samples, n_components_)

        if self.whiten:
            # avoid division by zero in degenerate directions
            var = np.maximum(self.explained_variance_, 1e-15)
            X_proj = X_proj / np.sqrt(var)

        return X_proj

    def fit_transform(self, X: ArrayLike, y: None = None) -> np.ndarray:
        """
        Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Data to fit and transform.
        y : None, ignored
            Present for API compatibility; not used.

        Returns
        -------
        X_transformed : ndarray, shape (n_samples, n_components_)
            Projected data.
        """
        return self.fit(X, y=y).transform(X)

    def inverse_transform(self, X: ArrayLike) -> np.ndarray:
        """
        Transform data back to the original feature space.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_components_)
            Data in principal component space.

        Returns
        -------
        X_reconstructed : ndarray, shape (n_samples, n_features)
            Approximate reconstruction of the original data.

        Raises
        ------
        RuntimeError
            If called before ``fit``.
        ValueError
            If the number of components in X does not match the fitted model.
        """
        self._check_is_fitted()
        Z = np.asarray(X, dtype=float)
        if Z.ndim != 2:
            raise ValueError(f"Input must be 2D for inverse_transform; got {Z.ndim}D.")

        if Z.shape[1] != self.n_components_:
            raise ValueError(
                f"X has {Z.shape[1]} components, but PCA was fitted with "
                f"{self.n_components_} components."
            )

        # Undo whitening if necessary
        if self.whiten:
            var = np.maximum(self.explained_variance_, 1e-15)
            Z = Z * np.sqrt(var)

        X_reconstructed = Z @ self.components_ + self.mean_
        return X_reconstructed
