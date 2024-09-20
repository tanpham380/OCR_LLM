import pycuda.driver as cuda
cuda.init()
gpu = cuda.Device(0)
print(gpu.name())
print("Compute Capability: ", gpu.compute_capability())
