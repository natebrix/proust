import warnings

warnings.warn(
    "proust_names is a compatibility module; prefer importing from 'proust' and using create_session() or ProustSession.",
    DeprecationWarning,
    stacklevel=2,
)

from proust import *  # noqa: F401,F403
