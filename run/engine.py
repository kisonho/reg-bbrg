from diffusion import metrics
from torch.utils.data import DataLoader
from torchmanager import callbacks
from torchmanager_core import devices, torch, view
from torchmanager_core.protocols import MonitorType
from torchmanager_core.typing import Callable, TypeVar, cast

from reg_bbrg import compile
from reg_bbrg.data import MedicalTranslationDataset as Dataset
from reg_bbrg.configs import EvalConfigs, TrainingConfigs
from reg_bbrg.nn import RegularizedBrownianBridgeModule as RegBBrgModule
from reg_bbrg.managers import AdversarialDiffusionManager as Manager

G1 = TypeVar("G1", bound=torch.nn.Module)
G2 = TypeVar("G2", bound=torch.nn.Module)
D = TypeVar("D", bound=torch.nn.Module)


def eval(cfgs: EvalConfigs, testing_dataset: Dataset | DataLoader) -> dict[str, float]:
    """
    Test with `diffusion.configs.TestingConfigs`

    - Parameters:
        - model: An optional pre-trained `torch.nn.Module`
        - configs: A `diffusion.configs.TestingConfigs` for testing
    - Returns: A `dict` of results with name as `str` and value as `float`
    """
    # initialize metrics
    denormalize_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: (x * 0.5 + 0.5).clamp(0, 1)
    metric_fns = {
        "lpips": metrics.LPIPS(),
        "mae": metrics.MAE(),
        "psnr": metrics.PSNR(denormalize_fn=denormalize_fn),
        "ssim": metrics.SSIM(1, denormalize_fn=denormalize_fn, pixel_range=1),
    }

    # load checkpoint
    if cfgs.model.endswith(".model"):
        # load checkpoint
        manager = cast(Manager[RegBBrgModule], Manager.from_checkpoint(cfgs.model, map_location=devices.CPU))
        manager.reset()

        # set metrics
        manager.metric_fns = metric_fns
    else:
        model = cast(RegBBrgModule, torch.load(cfgs.model, map_location=devices.CPU))
        manager = Manager(model, metrics=metric_fns)

    # enables return predictions
    manager.raw_model.return_prediction = not cfgs.disable_fast_sampling
    manager.loss_fn = None

    # set time steps
    if cfgs.time_steps is not None:
        manager.raw_model.time_steps = cfgs.time_steps

    # evaluation
    result = manager.test(testing_dataset, sampling_images=cfgs.disable_fast_sampling, device=cfgs.devices, use_multi_gpus=cfgs.use_multi_gpus, show_verbose=cfgs.show_verbose)
    return result


def train(cfgs: TrainingConfigs, training_dataset: Dataset | DataLoader, /, model: RegBBrgModule[G1, G2, D] | None = None, *, validation_dataset: Dataset | DataLoader | None = None) -> torch.nn.Module:
    """
    Train with C2B2

    - Parameters:
        - cfgs: The training configurations in `c2b2.configs.TrainingConfigs`.
        - training_dataset: The training dataset in `torchmanager.data.Dataset`.
        - model: The model to train in `torch.nn.Module`.
        - validation_dataset: The validation dataset in `torchmanager.data.Dataset`.
    - Returns: The trained model in `torch.nn.Module`.
    """
    # load model
    if cfgs.ckpt_path is None:
        # assert model and discriminator are not None
        assert model is not None, "Model is required for training."
        manager = compile(model, lambda_rec=cfgs.lambda_rec, lambda_adv=cfgs.lambda_adv, use_ema=not cfgs.disable_ema)
    else:
        # load manager from checkpoint
        manager = cast(Manager[RegBBrgModule[G1, G2, D]], Manager.from_checkpoint(cfgs.ckpt_path))
        manager.reset()

        # check if model has given
        if model is not None:
            manager.model.load_state_dict(model.state_dict())

    # initialize callbacks
    if callbacks.Experiment is NotImplemented:
        view.warnings.warn("TensorBoard is not installed, no TensorBoard callback will be used.", RuntimeWarning)
        callbacks_list: list[callbacks.Callback] = [callbacks.LastCheckpoint(manager, cfgs.output_model)]
    else:
        experiment_callback = callbacks.Experiment(cfgs.experiment, manager, monitors={"loss": MonitorType.MIN, "diffusion_loss": MonitorType.MIN})
        callbacks_list: list[callbacks.Callback] = [experiment_callback]

    # train model
    model = manager.fit(training_dataset, epochs=cfgs.epochs, val_dataset=validation_dataset, callbacks_list=callbacks_list, device=cfgs.devices, use_multi_gpus=cfgs.use_multi_gpus, show_verbose=cfgs.show_verbose)

    # save final model
    torch.save(model, cfgs.output_model)
    return model
