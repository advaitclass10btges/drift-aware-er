import joblib


def save_matcher(model, path):
    """
    Save trained matcher.
    """
    joblib.dump(model, path)


def load_matcher(path):
    """
    Load frozen matcher.
    """
    return joblib.load(path)
