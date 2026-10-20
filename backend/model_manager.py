"""
Models module for AutoExtract backend application.

This module handles the initialization and management of machine learning
models used for graph analysis.

The YOLO model (and PyTorch) is only needed by the interactive Bar Analyzer,
so it is loaded on first use rather than at import: the app and Batch
Extract start in seconds, and work offline, without touching torch.hub.
"""

import os
import threading


class ModelManager:
    """
    Manages the BarGraphAnalyzer model instance.

    This class provides a centralized way to manage the graph analysis
    model, including lazy initialization and access to the analyzer instance.
    """

    def __init__(self):
        """Create the manager; the analyzer itself is loaded on first use."""
        self.analyzer = None
        self._init_error: str | None = None
        self._tried = False
        self._lock = threading.Lock()

    @property
    def init_error(self) -> str | None:
        """Human-readable reason initialization failed, or None if ready."""
        return self._init_error

    def _initialize_analyzer(self) -> None:
        """
        Initialize the BarGraphAnalyzer with configured parameters.

        Creates a new BarGraphAnalyzer instance using the model path,
        class names, and tesseract command from configuration.
        """
        from config import MODEL_PATH, CLASS_NAMES, TESSERACT_CMD

        self._tried = True
        if not os.path.isfile(MODEL_PATH):
            self._init_error = (
                f"Model weights missing at {MODEL_PATH}. "
                "Restore backend/models/best.pt or rebuild the Docker image. "
                "To use custom weights: -v /path/to/best.pt:/app/backend/models/best.pt:ro"
            )
            print(f"Failed to initialize BarGraphAnalyzer: {self._init_error}")
            return
        try:
            from bar_graph_analyzer import BarGraphAnalyzer
            self.analyzer = BarGraphAnalyzer(
                model_path=MODEL_PATH,
                class_names=CLASS_NAMES,
                pytesseract_cmd=TESSERACT_CMD
            )
        except Exception as e:
            self._init_error = str(e)
            print(f"Failed to initialize BarGraphAnalyzer: {e}")
            self.analyzer = None

    def get_analyzer(self):
        """
        Get the BarGraphAnalyzer instance, loading it on first call.

        Returns:
            BarGraphAnalyzer: The initialized analyzer instance

        Raises:
            RuntimeError: If the analyzer failed to initialize
        """
        with self._lock:
            if self.analyzer is None and not self._tried:
                self._initialize_analyzer()
        if self.analyzer is None:
            detail = f": {self._init_error}" if self._init_error else ""
            raise RuntimeError(f"BarGraphAnalyzer not initialized{detail}")
        return self.analyzer

    def is_ready(self) -> bool:
        """
        Check if the model is loaded.

        Returns:
            bool: True if the analyzer is initialized, False otherwise
        """
        return self.analyzer is not None


# Global model manager instance
model_manager = ModelManager()
