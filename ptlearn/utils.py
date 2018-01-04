def str2val(maybe_str, mapping):
    """ Map key to value if argument is a string."""
    if isinstance(maybe_str, str):
        return mapping[maybe_str]

    return maybe_str
