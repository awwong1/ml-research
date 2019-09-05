import os
from torch import cuda


def set_cuda_devices(cpu_only, gpu_ids=None, logger=None):
    if gpu_ids is None:
        gpu_ids = list(range(cuda.device_count()))
    use_cuda = cuda.is_available() and not cpu_only and gpu_ids
    if use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = " ".join(map(str, gpu_ids))
        if logger:
            for gpu_id in gpu_ids:
                device_name = cuda.get_device_name(gpu_id)
                logger.info("CUDA %s : %s", gpu_id, device_name)
    elif logger:
        gpu_ids = []
        logger.info("CUDA is disabled, CPU only")
    return use_cuda, gpu_ids
