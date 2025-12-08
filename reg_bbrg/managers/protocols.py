import torch
from reg_bbrg.nn import RegBBrgOutput, RegBBrgModule
from typing import Generic, Protocol, TypeVar


class Predictable(Protocol):
    """The predictable interface for the diffusion model that can direct return prediction instead of objective."""
    return_prediction: bool


P = TypeVar("P", bound=Predictable)


class PredictionContext(Generic[P]):
    """
    The context for prediction.

    * generic type `P` should be a subclass of `Predictable`.

    - Properties:
        - previous_return_prediction: The previous return prediction `bool` flag.
        - predictable_target: The `Predictable` target.
    """
    predictable_target: P
    previous_return_prediction: bool

    def __init__(self, predictable_target: P, target_return_prediction: bool = True) -> None:
        """
        Construct a PredictionContext.

        - Parameters:
            - predictable_target: The `Predictable` target.
        """
        self.predictable_target = predictable_target
        self.previous_return_prediction = target_return_prediction

    def __enter__(self) -> P:
        target_return_prediction = self.previous_return_prediction
        self.previous_return_prediction = self.predictable_target.return_prediction
        self.predictable_target.return_prediction = target_return_prediction
        return self.predictable_target

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        target_return_prediction = self.previous_return_prediction
        self.predictable_target.return_prediction = self.previous_return_prediction
        self.previous_return_prediction = target_return_prediction
