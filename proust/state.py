aliases = None
nlp = None


def get_loaded_aliases():
    return aliases


def set_aliases(value):
    global aliases
    aliases = value


def reset_aliases():
    set_aliases(None)


def get_loaded_nlp():
    return nlp


def set_nlp(value):
    global nlp
    nlp = value


def reset_nlp():
    set_nlp(None)
