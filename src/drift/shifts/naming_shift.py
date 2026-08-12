import re


def uppercase_hyphen_shift(value):
    """
    Simulates vendor naming convention change.

    Example:

    Before:
        Apple iPhone 13 Pro Max

    After:
        APPLE-IPHONE-13-PRO-MAX
    """

    if value is None:
        return value

    value = str(value)

    value = value.upper()

    value = re.sub(
        r"\s+",
        "-",
        value.strip()
    )

    return value



def apply_naming_shift(df, columns):
    """
    Apply naming convention shift
    to selected text columns.
    """

    shifted = df.copy()

    for col in columns:
        if col in shifted.columns:
            shifted[col] = shifted[col].apply(
                uppercase_hyphen_shift
            )

    return shifted
