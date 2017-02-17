import numpy as np
import chainer
from chainer import cuda

class wrapper:
    def __init__(self, gpu_id):
        self.gpu_id = gpu_id
        self.use_gpu = True if gpu_id >= 0 else False
        self.xp = cuda.cupy if gpu_id >= 0 else np

    def init(self):
        if self.use_gpu:
            cuda.check_cuda_available()
            cuda.get_device(self.gpu_id).use()

    def make_var(self, array, dtype=np.float32):
        var = np.array(array, dtype=dtype)
        if self.use_gpu: var = chainer.cuda.to_gpu(var)
        return chainer.Variable(var)

    def get_data(self, variable):
        return cuda.to_cpu(variable.data)

    def zeros(self, shape, dtype=np.float32):
        return chainer.Variable(self.xp.zeros(shape, dtype=dtype))

    def ones(self, shape, dtype=np.float32):
        return chainer.Variable(self.xp.ones(shape, dtype=dtype))

    def make_model(self, **kwargs):
        model = chainer.FunctionSet(**kwargs)
        if self.use_gpu: model.to_gpu()
        return model

    def begin_model_access(self, model):
        if self.use_gpu: model.to_cpu()

    def end_model_access(self, model):
        if self.use_gpu: model.to_gpu()
