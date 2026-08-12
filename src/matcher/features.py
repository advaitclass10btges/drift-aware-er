import numpy as np
from rapidfuzz.fuzz import ratio
import jellyfish


def safe_string(x):
    """
    Convert missing values safely into empty strings.
    """
    if x is None:
        return ""

    if isinstance(x, float) and np.isnan(x):
        return ""

    return str(x).lower().strip()


def string_features(a, b):
    """
    Extract string similarity features.
    """

    a = safe_string(a)
    b = safe_string(b)

    if not a and not b:
        return {
            "jaro_winkler": 0.0,
            "token_ratio": 0.0,
            "length_ratio": 0.0
        }

    return {
        "jaro_winkler": jellyfish.jaro_winkler_similarity(a, b),

        "token_ratio": ratio(a, b) / 100.0,

        "length_ratio":
            min(len(a), len(b)) / max(len(a), len(b))
            if max(len(a), len(b)) > 0
            else 0.0
    }


def price_features(a, b):
    """
    Numerical price similarity features.
    """

    if np.isnan(a) or np.isnan(b):
        return {
            "price_difference": 0.0,
            "price_ratio": 0.0,
            "price_log_difference": 0.0
        }

    diff = abs(a - b)

    price_ratio = (
        min(a, b) / max(a, b)
        if max(a, b) > 0
        else 0.0
    )

    log_difference = (
        np.log1p(abs(a) + 1)
        -
        np.log1p(abs(b) + 1)
    )

    return {
        "price_difference": diff,
        "price_ratio": price_ratio,
        "price_log_difference": log_difference
    }


def build_features(left, right):
    """
    Build one feature vector for an entity pair.

    Input:
        left  -> table A record
        right -> table B record

    Output:
        dictionary of numerical features
    """

    features = {}

    title = string_features(
        left["title"],
        right["title"]
    )

    manufacturer = string_features(
        left["manufacturer"],
        right["manufacturer"]
    )

    price = price_features(
        left["price"],
        right["price"]
    )


    features.update({

        # Title features
        "title_jaro":
            title["jaro_winkler"],

        "title_token":
            title["token_ratio"],

        "title_length":
            title["length_ratio"],


        # Manufacturer features
        "manufacturer_jaro":
            manufacturer["jaro_winkler"],

        "manufacturer_token":
            manufacturer["token_ratio"],

        "manufacturer_length":
            manufacturer["length_ratio"],


        # Price features
        "price_difference":
            price["price_difference"],

        "price_ratio":
            price["price_ratio"],

        "price_log_difference":
            price["price_log_difference"]
    })


    return features
