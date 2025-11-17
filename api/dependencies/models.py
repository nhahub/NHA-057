from functools import lru_cache

from CV.inference.sign_recognizer import SignRecognizer


@lru_cache(maxsize=1)
def get_sign_recognizer() -> SignRecognizer:
    """Return a singleton SignRecognizer instance.

    The first call will construct the recognizer and load the underlying
    I3D model and label mappings. Subsequent calls reuse the same
    instance within this process, which avoids reloading weights for
    every request.
    """

    return SignRecognizer()
