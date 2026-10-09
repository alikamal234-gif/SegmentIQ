from sklearn.model_selection import cross_val_score


def cross_validate_model(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
):
    """
    Effectue une cross-validation du modèle.

    Parameters
    ----------
    model : modèle sklearn
        Modèle à évaluer.

    X : DataFrame
        Features.

    y : Series
        Target.

    cv : int
        Nombre de folds.

    scoring : str
        Métrique utilisée.

    Returns
    -------
    scores : array
        Score obtenu pour chaque fold.
    """

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    return scores


def cross_validation_summary(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
):
    """
    Effectue la cross-validation et retourne
    les scores ainsi que leur moyenne et écart-type.
    """

    scores = cross_validate_model(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring
    )

    result = {
        "scores": scores,
        "mean": scores.mean(),
        "std": scores.std()
    }

    return result