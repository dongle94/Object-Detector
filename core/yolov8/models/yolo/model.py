# Ultralytics YOLO 🚀, AGPL-3.0 license

import inspect
from pathlib import Path
from typing import Union

# from core.yolov8.engine.model import Model
from core.yolov8.models import yolo
from core.yolov8.nn import attempt_load_one_weight
from core.yolov8.nn.tasks import DetectionModel   # , OBBModel

from core.yolov8.utils import callbacks, checks


class YOLOV8(object):
    """
        A base class to unify APIs for all models.

        Args:
            model (str, Path): Path to the model file to load or create.
            task (Any, optional): Task type for the YOLO model. Defaults to None.

        Attributes:
            predictor (Any): The predictor object.
            model (Any): The model object.
            trainer (Any): The trainer object.
            task (str): The type of model task.
            ckpt (Any): The checkpoint object if the model loaded from *.pt file.
            cfg (str): The model configuration if loaded from *.yaml file.
            ckpt_path (str): The checkpoint file path.
            overrides (dict): Overrides for the trainer object.
            metrics (Any): The data for metrics.

        Methods:
            __call__(source=None, stream=False, **kwargs):
                Alias for the predict method.
            _new(cfg:str, verbose:bool=True) -> None:
                Initializes a new model and infers the task type from the model definitions.
            _load(weights:str, task:str='') -> None:
                Initializes a new model and infers the task type from the model head.
            _check_is_pytorch_model() -> None:
                Raises TypeError if the model is not a PyTorch model.
            reset() -> None:
                Resets the model modules.
            info(verbose:bool=False) -> None:
                Logs the model info.
            fuse() -> None:
                Fuses the model for faster inference.
            predict(source=None, stream=False, **kwargs) -> List[ultralytics.engine.results.Results]:
                Performs prediction using the YOLO model.

        Returns:
            list(ultralytics.engine.results.Results): The prediction results.
        """
    def __init__(self, model: Union[str, Path] = "yolov8n.pt", task=None) -> None:
        """
        Initializes the YOLO model.

        Args:
            model (Union[str, Path], optional): Path or name of the model to load or create. Defaults to 'yolov8n.pt'.
            task (Any, optional): Task type for the YOLO model. Defaults to None.
        """
        super().__init__()
        self.callbacks = callbacks.get_default_callbacks()
        self.predictor = None  # reuse predictor
        self.model = None  # model object
        self.trainer = None  # trainer object
        self.ckpt = None  # if loaded from *.pt
        self.cfg = None  # if loaded from *.yaml
        self.ckpt_path = None
        self.overrides = {}  # overrides for trainer object
        self.metrics = None  # validation/training metrics
        self.session = None  # HUB session
        self.task = task  # task type
        self.model_name = model = str(model).strip()  # strip spaces

        # Load or create new YOLO model
        model = checks.check_model_file_from_stem(model)  # add suffix, i.e. yolov8n -> yolov8n.pt
        if Path(model).suffix in (".yaml", ".yml"):
            self._new(model, task)
        else:
            self._load(model, task)

        self.model_name = model

    def __call__(self, source=None, stream=False, **kwargs):
        """Calls the predict() method with given arguments to perform object detection."""
        return self.predict(source, stream, **kwargs)

    def _new(self, cfg: str, task=None, model=None, verbose=True):
        """
        Initializes a new model and infers the task type from the model definitions.

        Args:
            cfg (str): model configuration file
            task (str | None): model task
            model (BaseModel): Customized model.
            verbose (bool): display model info on load
        """
        pass

    def _load(self, weights: str, task=None):
        """
        Initializes a new model and infers the task type from the model head.

        Args:
            weights (str): model checkpoint to be loaded
            task (str | None): model task
        """
        suffix = Path(weights).suffix
        if suffix == ".pt":
            self.model, self.ckpt = attempt_load_one_weight(weights)
            self.task = self.model.args["task"]
            self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
            self.ckpt_path = self.model.pt_path

    def predict(self, source=None, stream=False, predictor=None, **kwargs):
        """
        Perform prediction using the YOLO model.

        Args:
            source (str | int | PIL | np.ndarray): The source of the image to make predictions on.
                Accepts all source types accepted by the YOLO model.
            stream (bool): Whether to stream the predictions or not. Defaults to False.
            predictor (BasePredictor): Customized predictor.
            **kwargs : Additional keyword arguments passed to the predictor.
                Check the 'configuration' section in the documentation for all available options.

        Returns:
            (List[ultralytics.engine.results.Results]): The prediction results.
        """
        is_cli = False

        custom = {"conf": 0.25, "save": is_cli}  # method defaults
        args = {**self.overrides, **custom, **kwargs, "mode": "predict"}  # highest priority args on the right
        # prompts = args.pop("prompts", None)  # for SAM-type models

        if not self.predictor:
            self.predictor = predictor or self._smart_load("predictor")(overrides=args, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=is_cli)
        # else:  # only update args if predictor is already setup
        #     self.predictor.args = get_cfg(self.predictor.args, args)
        #     if "project" in args or "name" in args:
        #         self.predictor.save_dir = get_save_dir(self.predictor.args)

        # if prompts and hasattr(self.predictor, "set_prompts"):  # for SAM-type models
        #     self.predictor.set_prompts(prompts)

        return self.predictor(source=source, stream=stream)

    @staticmethod
    def _reset_ckpt_args(args):
        """Reset arguments when loading a PyTorch model."""
        include = {"imgsz", "data", "task", "single_cls"}  # only remember these arguments when loading a PyTorch model
        return {k: v for k, v in args.items() if k in include}

    def _smart_load(self, key):
        """Load model/trainer/validator/predictor."""
        try:
            return self.task_map[self.task][key]
        except Exception as e:
            name = self.__class__.__name__
            mode = inspect.stack()[1][3]  # get the function name.
            raise NotImplementedError(
                print(f"WARNING ⚠️ '{name}' model does not support '{mode}' mode for '{self.task}' task yet.")
            ) from e

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": DetectionModel,
                # "trainer": yolo.detect.DetectionTrainer,
                # "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
            # "obb": {
            #     "model": OBBModel,
            #     "trainer": yolo.obb.OBBTrainer,
            #     "validator": yolo.obb.OBBValidator,
            #     "predictor": yolo.obb.OBBPredictor,
            # },
        }
