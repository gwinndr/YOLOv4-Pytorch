# For all things related to devices
#### ONLY USE PROVIDED FUNCTIONS, DO NOT USE GLOBAL CONSTANTS ####

import torch

TORCH_CPU_DEVICE = torch.device("cpu")

if(torch.cuda.device_count() > 0):
    TORCH_CUDA_DEVICE = torch.device("cuda")
else:
    print("----- WARNING: No CUDA device detected, expect model to run slower! -----")
    print("")
    TORCH_CUDA_DEVICE = None

USE_CUDA = True

# use_cuda
def use_cuda(cuda_bool):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    Sets whether to use CUDA (if available), or use the CPU (not recommended)
    If this function is not called, the default is to use CUDA if available
    ----------
    """

    global USE_CUDA
    USE_CUDA = cuda_bool

# get_device
def get_device():
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    Grabs the default device.
    Default device is CUDA if available and USE_CUDA is True, CPU otherwise.
    ----------
    """

    if((not USE_CUDA) or (TORCH_CUDA_DEVICE is None)):
        return TORCH_CPU_DEVICE
    else:
        return TORCH_CUDA_DEVICE

# synchronize_device
def synchronize_device():
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Synchronizes queued tensor operations
    - Used for fps benchmarking purposes (should not be used otherwise)
    - Only cuda operations are asynchronous, so does nothing if no cuda
    ----------
    """

    if(USE_CUDA and (TORCH_CUDA_DEVICE is not None)):
        torch.cuda.synchronize()


# gpu_device_name
def gpu_device_name():
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Gets the name of the gpu
    - Gives the string "None" if no gpu or it is not being used
    ----------
    """

    if((not USE_CUDA) or (TORCH_CUDA_DEVICE is None)):
        return "None"
    else:
        return torch.cuda.get_device_name()


# cuda_device
def cuda_device():
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    Grabs the cuda device (may be None if CUDA is not available)
    ----------
    """

    return TORCH_CUDA_DEVICE

# cpu_device
def cpu_device():
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    Grabs the cpu device
    ----------
    """

    return TORCH_CPU_DEVICE
