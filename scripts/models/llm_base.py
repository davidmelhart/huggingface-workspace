import torch
import gc
from abc import ABC, abstractmethod

class VisionLanguageModelPrototype(ABC):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()

    @abstractmethod
    def load(self):
        """Load the model and processor (to be implemented by subclass)."""
        pass

    def unload(self):
        """Unload the model and processor to free up memory."""
        mem_before = torch.cuda.memory_allocated()
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        gc.collect()
        torch.cuda.empty_cache()
        mem_after = torch.cuda.memory_allocated()
        print(f"\n>>> Model {self.model_name} unloaded successfully!\n>>> Memory allocated (before/after): {mem_before}/{mem_after}\n")

    @abstractmethod
    def infer(self, *args, **kwargs):
        """Run inference (to be implemented by subclass)."""
        pass
