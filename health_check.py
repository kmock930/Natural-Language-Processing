import tensorflow as tf
# Check GPU device
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    print(f"Number of GPUs available: {len(gpu_devices)}")
    for gpu in gpu_devices:
        print(f"GPU Device: {gpu}")
        assert tf.test.is_gpu_available()
else:
    print("No GPU device found.")

# Check CUDA version
cuda_version = tf.sysconfig.get_build_info()['cuda_version']
print(f"CUDA Version: {cuda_version}")