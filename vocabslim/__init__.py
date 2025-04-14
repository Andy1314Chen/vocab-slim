__version__ = "0.1.1"

def __getattr__(name):
    if name == "VocabSlim":
        from .core import VocabSlim
        return VocabSlim
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ["__version__", "VocabSlim"]
