import os
import importlib
from .base import DexHand


class DexHandFactory:
    _registry = {}

    @classmethod
    def register(cls, dexhand_type: str, hand_class):
        """Register a new hand type."""
        cls._registry[dexhand_type] = hand_class

    @classmethod
    def create_hand(cls, dexhand_type: str, side: str, *args, **kwargs) -> DexHand:
        assert side in ["left", "right"], f"Invalid side '{side}', must be 'left' or 'right'."
        """Create a hand instance by type."""
        dexhand_type += "_rh" if side == "right" else "_lh"
        if dexhand_type not in cls._registry:
            raise ValueError(f"Hand type '{dexhand_type}' not registered.")
        return cls._registry[dexhand_type](*args, **kwargs)

    @classmethod
    def auto_register_hands(cls, directory: str, base_package: str):
        """Automatically import all hand modules in the directory."""
        for filename in os.listdir(directory):
            if filename.endswith(".py") and filename != "__init__.py":
                module_name = f"{base_package}.{filename[:-3]}"
                importlib.import_module(module_name)


def register_dexhand(dexhand_type):
    def decorator(cls):
        DexHandFactory.register(dexhand_type, cls)
        return cls

    return decorator