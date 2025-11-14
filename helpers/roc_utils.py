import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score


def plot_roc(model, X, y, title="ROC Curve"):
    """
    Plot ROC curve for any binary classifier.

    Parameters
    ----------
    model : fitted estimator
        Must implement predict_proba() or decision_function().
    X : array-like
        Test features.
    y : array-like
        True test labels.
    title : str
        Title of the plot.
    """

    # 1. Get scores (use predict_proba if possible)
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X)
    else:
        raise ValueError(
            "Model must support predict_proba() or decision_function(). "
            "You passed a classifier that only outputs hard labels."
        )

    # 2. ROC + AUC
    fpr, tpr, _ = roc_curve(y, y_scores)
    auc = roc_auc_score(y, y_scores)

    # 3. Prepare DataFrame
    roc_df = pd.DataFrame({"False Positive Rate": fpr, "True Positive Rate": tpr})

    # 4. Plot
    plt.figure(figsize=(6, 5))
    plt.title(title)
    sns.lineplot(
        data=roc_df,
        x="False Positive Rate",
        y="True Positive Rate",
        linewidth=2,
        label=f"AUC = {auc:.2f}",
    )
    return auc
