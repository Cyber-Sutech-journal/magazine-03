"""Factory for creating :class:`~mot_counting.interfaces.detector.IDetector` instances (§4.1).

Design contract
---------------
``DetectorFactory`` is intentionally trivial.  The composition root is
responsible for loading the YOLO26 model weights (the expensive step) and for
validating the configured class list against ``model.names`` (§4.1, §4.6).
The Factory's only job is to wrap the *already-loaded* model object into the
correct concrete ``IDetector`` implementation and return it.

No model-loading code may ever live inside this Factory.
"""

from __future__ import annotations

from mot_counting.interfaces.detector import IDetector

# Supported model-variant prefixes and the concrete class they map to.
# Populated in T19 once the concrete YOLO26Detector implementation exists.
# Until then, every known variant correctly reaches the TODO(T19) stub so
# this module compiles and can be tested in isolation with a mock model object.
_SUPPORTED_VARIANTS: frozenset[str] = frozenset(
    {"yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"}
)


class DetectorFactory:
    """Wraps an already-loaded model object into a concrete ``IDetector``.

    Usage (composition root)::

        raw_model = YOLO("yolo26m.pt")        # loaded once, outside this Factory
        factory = DetectorFactory(
            confidence_threshold=config.detection.confidence_threshold,
            classes=config.detection.classes,
            imgsz=config.detection.imgsz,
        )
        detector: IDetector = factory.create("yolo26m", raw_model)
    """

    def __init__(self, confidence_threshold: float, classes: list[str], imgsz: int) -> None:
        """Initialise the factory with detection filter parameters.

        Args:
            confidence_threshold: Global confidence threshold forwarded to the
                concrete detector at construction time.
            classes: List of class-name strings the detector should keep.
            imgsz: Inference image size forwarded from ``detection.imgsz``.
        """
        self._confidence_threshold = confidence_threshold
        self._classes = classes
        self._imgsz = imgsz

    def create(self, model_variant: str, loaded_model: object) -> IDetector:
        """Wrap *loaded_model* into the ``IDetector`` matching *model_variant*.

        Args:
            model_variant: YOLO26 variant string from the configuration (e.g.
                ``"yolo26m"``).  Must be one of the supported variants;
                anything else raises ``ValueError`` for fail-fast startup
                (§12.1).
            loaded_model: An already-initialised YOLO model object provided
                by the composition root.  This Factory must never load model
                weights itself (§4.1).

        Returns:
            A concrete ``IDetector`` implementation wrapping *loaded_model*.

        Raises:
            ValueError: If *model_variant* is not a recognised YOLO26 variant.
            NotImplementedError: Temporary — raised until the concrete
                ``Yolo26Detector`` class is wired in T19.
        """
        if model_variant not in _SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unknown model_variant {model_variant!r}.  "
                f"Supported variants: {sorted(_SUPPORTED_VARIANTS)}."
            )

        if model_variant.startswith("yolo26"):
            from mot_counting.detectors.yolo26_detector import Yolo26Detector

            return Yolo26Detector(
                model=loaded_model,
                imgsz=self._imgsz,
                confidence_threshold=self._confidence_threshold,
                allowed_classes=self._classes,
            )

        # Unreachable given _SUPPORTED_VARIANTS above; guards future additions.
        raise ValueError(f"Unhandled model_variant {model_variant!r}.")  # pragma: no cover
