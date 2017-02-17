import numpy
import chainer
from chainer import cuda

class wrapper:
    @staticmethod
    def init(gpu_id):
        cuda.check_cuda_available()
        cuda.get_device(gpu_id).use()


    @staticmethod
    def make_var(array, dtype=cuda.cupy.float32):
        return chainer.Variable(chainer.cuda.to_gpu(numpy.array(array, dtype=dtype)))
        #return chainer.Variable(cuda.cupy.array(array, dtype=dtype))

    @staticmethod
    def get_data(variable):
        return cuda.to_cpu(variable.data)

    @staticmethod
    def zeros(shape, dtype=cuda.cupy.float32):
        return chainer.Variable(cuda.cupy.zeros(shape, dtype=dtype))

    @staticmethod
    def ones(shape, dtype=cuda.cupy.float32):
        return chainer.Variable(cuda.cupy.ones(shape, dtype=dtype))

    @staticmethod
    def make_model(**kwargs):
        return chainer.FunctionSet(**kwargs).to_gpu()

    @staticmethod
    def begin_model_access(model):
        model.to_cpu()

    @staticmethod
    def end_model_access(model):
        model.to_gpu()
