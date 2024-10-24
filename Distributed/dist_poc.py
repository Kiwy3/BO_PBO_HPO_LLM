import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

# Function to initialize the process group and specify the GPU for each process
def init_process(rank, size, fn, backend='nccl'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Set the GPU to use for each process
    torch.cuda.set_device(rank)

    # Initialize the process group for GPU communication using NCCL backend
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)
    dist.destroy_process_group()

# Function to perform all_reduce operation on GPU
def run(rank, size):
    # Create a tensor on the assigned GPU for this process
    tensor = torch.ones(1).cuda(rank) * rank
    print(f"Process {rank} has data {tensor[0].item()} on GPU {rank} before all_reduce")
    
    # Perform all-reduce operation across GPUs
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    print(f"Process {rank} has data {tensor[0].item()} on GPU {rank} after all_reduce")

# Spawn multiple processes for distributed GPU communication
def spawn_processes():
    """
    Spawn multiple processes for distributed GPU communication.

    This function spawns multiple processes, each assigned to a different GPU,
    and performs all_reduce operation across GPUs using NCCL backend.

    :return: None
    """
    size = torch.cuda.device_count()  # Number of GPUs available
    processes = []

    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    spawn_processes()
    print("end of program")