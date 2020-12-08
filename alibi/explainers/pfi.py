# flake8: noqa E131
import copy
import math
import numpy as np
from sklearn.inspection import permutation_importance
from itertools import count
from functools import partial
from typing import Callable, List, Optional, Tuple, Union, TYPE_CHECKING, no_type_check
from typing_extensions import Literal
from sklearn.base import BaseEstimator, TransformerMixin

from alibi.api.interfaces import Explainer, Explanation
from alibi.api.defaults import DEFAULT_META_PFI, DEFAULT_DATA_PFI

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


# This is merely a wrapper around the sklearn functionality based heavily on the ALE implementation.
class PFI(Explainer):

    def __init__(self,
                 estimator: BaseEstimator,
                 feature_names: Optional[List[str]] = None) -> None:
        """
        Permutation Feature Importance for tabular datasets.

        Parameters
        ----------
        estimator
            A sklearn Estimator instance that can be fitted and used for predictions
        feature_names
            A list of feature names used for displaying results.
        """
        super().__init__(meta=copy.deepcopy(DEFAULT_META_PFI))

        self.estimator = estimator
        self.feature_names = feature_names

    def explain(self, X: np.ndarray, y: np.ndarray = None, scoring: Optional[Union[str, callable]] = None,
                n_repeats: int = 5, random_state: Optional[Union[int, np.random.RandomState]] = None) -> Explanation:
        """
        Calculate the PFI for each feature with respect to the dataset `X`.

        Parameters
        ----------
        X
            An NxF tabular dataset used to calculate the ALE curves. This is typically the training dataset
            or a representative sample.
        y
            A array of length N containing the target values
        scoring
            A scoring method
        n_repeats
            Number of times to permute a feature
        random_state
            Pseudo-random number generator to control the permutations of each feature. Pass an int to get reproducible
            results across function calls.

        Returns
        -------
            An `Explanation` object containing the data and the metadata of the calculated PFI.

        """
        if y is None:
            raise ValueError("The target values (y) for dataset X must be given to calculate the PFI.")

        if X.ndim != 2:
            raise ValueError('The array X must be 2-dimensional')
        n_features = X.shape[1]

        # Generate feature names if none are given, copied from the ALE functionality
        if self.feature_names is None:
            self.feature_names = [f'f_{i}' for i in range(n_features)]
        self.feature_names = np.array(self.feature_names)

        r = permutation_importance(self.estimator, X, y, scoring=scoring, n_repeats=n_repeats,
                                   random_state=random_state)

        feature_importance = [self.feature_names[i] for i in r.importances_mean.argsort()[::-1]]
        feature_importance_means = [r.importances_mean[i] for i in r.importances_mean.argsort()[::-1]]
        feature_importance_stds = [r.importances_std[i] for i in r.importances_mean.argsort()[::-1]]

        return self.build_explanation(
            feature_importance=feature_importance,
            feature_importance_means=feature_importance_means,
            feature_importance_stds=feature_importance_stds,
        )

    def build_explanation(self,
                          feature_importance: List[str],
                          feature_importance_means: List[float],
                          feature_importance_stds: List[float]) -> Explanation:
        """
        Helper method to build the Explanation object.
        """
        data = copy.deepcopy(DEFAULT_DATA_PFI)
        data.update(
            feature_importance=feature_importance,
            feature_importance_means=feature_importance_means,
            feature_importance_stds=feature_importance_stds,
            feature_names=self.feature_names,
        )

        return Explanation(meta=copy.deepcopy(self.meta), data=data)


# no_type_check is needed because exp is a generic explanation and so mypy doesn't know that the
# attributes actually exist... As a side effect the type information does not show up in the static
# docs. Will need to re-think this.
@no_type_check
def plot_pfi(exp: Explanation,
             ax: Optional['plt.Axes'] = None,
             bar_kw: Optional[dict] = None,
             error_kw: Optional[dict] = None,
             fig_kw: Optional[dict] = None) -> 'np.ndarray':
    """
    Plot ALE curves on matplotlib axes.

    Parameters
    ----------
    exp
        An `Explanation` object produced by a call to the `ALE.explain` method.
    ax
        A `matplotlib` axes object to plot on.
    bar_kw
        Keyword arguments passed to the `plt.barh` function.
    error_kw
        Keyword arguments passed to the `plt.errorbar` function.
    fig_kw
        Keyword arguments passed to the `fig.set` function.

    Returns
    -------
    A matplotlib axes with the resulting PFI values plotted.

    """
    import matplotlib.pyplot as plt

    def _default_kw(default_kw, kw):
        if kw is None:
            kw = {}
        return {**default_kw, **kw}

    if ax is None:
        ax = plt.gca()

    fig = ax.figure

    bar_kw = _default_kw({"align": 'center'}, bar_kw)
    error_kw = _default_kw({}, error_kw)
    fig_kw = _default_kw({'tight_layout': 'tight'}, fig_kw)

    ax.barh(exp.feature_importance[::-1], exp.feature_importance_means[::-1], xerr=exp.feature_importance_stds[::-1],
            error_kw=error_kw, **bar_kw)
    fig.set(**fig_kw)

    return ax
