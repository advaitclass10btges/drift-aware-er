import re


def remove_descriptive_tokens(
    value,
    keep_tokens=2
):
    """
    Simulates catalog title shortening.

    Keeps first N informative tokens.

    Example:

    Samsung Galaxy S21 Ultra 5G 128GB

    becomes:

    Samsung Galaxy
    """

    if value is None:
        return value

    tokens = re.findall(
        r"\w+",
        str(value)
    )

    if len(tokens) <= keep_tokens:
        return value

    return " ".join(
        tokens[:keep_tokens]
    )



def apply_title_degradation(
    df,
    keep_tokens=2
):

    shifted = df.copy()

    if "title" in shifted.columns:

        shifted["title"] = (
            shifted["title"]
            .apply(
                lambda x:
                remove_descriptive_tokens(
                    x,
                    keep_tokens
                )
            )
        )

    return shifted
