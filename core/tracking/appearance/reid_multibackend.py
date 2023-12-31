import torch.nn as nn
import torch
import numpy as np
import torchvision.transforms as T
# import gdown
from pathlib import Path
from collections import OrderedDict, namedtuple
from os.path import exists as file_exists

from utils.logger import get_logger
from ..tracking_utils.torch_utils import select_device

from .reid_model_factory import (
    show_downloadable_models,
    get_model_url,
    get_model_name,
    load_pretrained_weights
)
from .backbones import build_model


# from ..utils.checks import TestRequirements
# __tr = TestRequirements()


def check_suffix(file='osnet_x0_25_msmt17.pt', suffix=('.pt',), msg=''):
    # Check file(s) for acceptable suffix
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # file suffix
            if len(s):
                try:
                    assert s in suffix
                except AssertionError as err:
                    get_logger().error(f"{err}{f} acceptable suffix is {suffix}")


class ReIDDetectMultiBackend(nn.Module):
    # ReID models MultiBackend class for python inference on various backends
    def __init__(self, weights='osnet_x0_25_msmt17.pt', device=torch.device('cpu'), fp16=False):
        super().__init__()

        w = weights[0] if isinstance(weights, list) else weights
        self.pt, self.onnx, self.engine = self.model_type(w)  # get backend
        self.fp16 = fp16
        self.fp16 &= self.pt or self.engine  # FP16

        # Build transform functions
        self.device = select_device(device)
        self.image_size = (256, 128)
        self.pixel_mean = [0.485, 0.456, 0.406]
        self.pixel_std = [0.229, 0.224, 0.225]
        self.transforms = []
        self.transforms += [T.Resize(self.image_size)]
        self.transforms += [T.ToTensor()]
        self.transforms += [T.Normalize(mean=self.pixel_mean, std=self.pixel_std)]
        self.preprocess = T.Compose(self.transforms)
        self.to_pil = T.ToPILImage()

        model_name = get_model_name(w)

        # if w.suffix == '.pt':
        #     model_url = get_model_url(w)
        #     print(w, model_url)
        #     if not file_exists(w) and model_url is not None:
        #         get_logger().error("tracking model is not exist")
        #         raise "tracking weight is not exist"
        #         # gdown.download(model_url, str(w), quiet=False)
        #     elif file_exists(w):
        #         pass
        #     else:
        #         get_logger().error(f'No URL associated to the chosen StrongSORT weights ({w}). Choose between:')
        #         show_downloadable_models()
        #         exit()

        # Build model
        self.model = build_model(
            model_name,
            num_classes=1,
            pretrained=not (w and w.is_file()),
            use_gpu=device
        )

        if self.pt:  # PyTorch
            # populate model arch with weights
            if w and w.is_file() and w.suffix == '.pt':
                load_pretrained_weights(self.model, w)
            self.model.to(device).eval()
            self.model.half() if self.fp16 else self.model.float()
        elif self.onnx:  # ONNX Runtime
            get_logger().info(f'Loading {w} for ONNX Runtime inference...')
            cuda = torch.cuda.is_available() and device.type != 'cpu'
            # __tr.check_packages(['onnx', 'onnxruntime-gpu' if cuda else 'onnxruntime'])
            import onnxruntime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            self.session = onnxruntime.InferenceSession(str(w), providers=providers)
        elif self.engine:  # TensorRT
            get_logger().info(f'Loading {w} for TensorRT inference...')
            # __tr.check_packages(('nvidia-tensorrt',))
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
            if device.type == 'cpu':
                device = torch.device('cuda:0')
            Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
                self.model_ = runtime.deserialize_cuda_engine(f.read())
            self.context = self.model_.create_execution_context()
            self.bindings = OrderedDict()
            self.fp16 = False  # default updated below
            dynamic = False
            for index in range(self.model_.num_bindings):
                name = self.model_.get_binding_name(index)
                dtype = trt.nptype(self.model_.get_binding_dtype(index))
                if self.model_.binding_is_input(index):
                    if -1 in tuple(self.model_.get_binding_shape(index)):  # dynamic
                        dynamic = True
                        self.context.set_binding_shape(index, tuple(self.model_.get_profile_shape(0, index)[2]))
                    if dtype == np.float16:
                        self.fp16 = True
                shape = tuple(self.context.get_binding_shape(index))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
            batch_size = self.bindings['images'].shape[0]  # if dynamic, this is instead max batch size
        else:
            get_logger().error('This model framework is not supported yet!')
            exit()

    @staticmethod
    def model_type(p='path/to/model.pt'):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        from . import export_formats
        sf = list(export_formats().Suffix)  # export suffixes
        check_suffix(p, sf)  # checks
        types = [s in Path(p).name for s in sf]
        return types

    def _preprocess(self, im_batch):

        images = []
        for element in im_batch:
            image = self.to_pil(element)
            image = self.preprocess(image)
            images.append(image)

        images = torch.stack(images, dim=0)
        images = images.to(self.device)

        return images

    def forward(self, im_batch):

        # preprocess batch
        im_batch = self._preprocess(im_batch)

        # batch to half
        if self.fp16 and im_batch.dtype != torch.float16:
            im_batch = im_batch.half()

        # batch processing
        features = []
        if self.pt:
            features = self.model(im_batch)
        elif self.onnx:  # ONNX Runtime
            im_batch = im_batch.cpu().numpy()  # torch to numpy
            features = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: im_batch})[0]
        elif self.engine:  # TensorRT
            if True and im_batch.shape != self.bindings['images'].shape:
                i_in, i_out = (self.model_.get_binding_index(x) for x in ('images', 'output'))
                self.context.set_binding_shape(i_in, im_batch.shape)  # reshape if dynamic
                self.bindings['images'] = self.bindings['images']._replace(shape=im_batch.shape)
                self.bindings['output'].data.resize_(tuple(self.context.get_binding_shape(i_out)))
            s = self.bindings['images'].shape
            assert im_batch.shape == s, f"input size {im_batch.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs['images'] = int(im_batch.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            features = self.bindings['output'].data
        else:
            get_logger().error('Framework not supported at the moment, leave an enhancement suggestion')
            exit()

        if isinstance(features, (list, tuple)):
            return self.from_numpy(features[0]) if len(features) == 1 else [self.from_numpy(x) for x in features]
        else:
            return self.from_numpy(features)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=[(256, 128, 3)]):
        # Warmup model by running inference once
        warmup_types = self.pt, self.onnx, self.engine
        if any(warmup_types) and self.device.type != 'cpu':
            im = [np.empty(*imgsz).astype(np.uint8)]  # input
            for _ in range(2 if self.jit else 1):  #
                self.forward(im)  # warmup
