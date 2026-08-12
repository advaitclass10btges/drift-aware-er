import numpy as np


def apply_missingness_shift(
    df,
    columns,
    probability=0.5,
    random_state=42
):
    """
    Simulates source attribute loss.

    Randomly removes values from selected columns.
    """

    shifted = df.copy()

    rng = np.random.default_rng(
        random_state
    )

    for col in columns:

        if col not in shifted.columns:
            continue

        mask = rng.random(
            len(shifted)
        ) < probability


        shifted.loc[
            mask,
            col
        ] = None


    return shifted
