# benchmark_pytorch.py
# The script is copied from https://leimao.github.io/blog/PyTorch-Benchmark/
from timeit import default_timer as timer
import torch
import torch.nn as nn
import torchvision
import timm
import torch.utils.benchmark as benchmark

from config import get_config,_update_config_from_file
from models import build_model


@torch.no_grad()
def run_inference(model: nn.Module,
                  input_tensor: torch.Tensor) -> torch.Tensor:

    return model.forward(input_tensor)

@torch.no_grad()
def measure_time_device(
    model: nn.Module,
    input_tensor: torch.Tensor,
    num_repeats: int = 100,
    num_warmups: int = 10,
    synchronize: bool = True,
    continuous_measure: bool = True,
) -> float:

    '''
        synchronize and continuous_measure should always be True
    '''
    assert synchronize == True and continuous_measure == True

    for _ in range(num_warmups):
        _ = model.forward(input_tensor)
    torch.cuda.synchronize()

    elapsed_time_ms = 0

    if continuous_measure:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(num_repeats):
            # what you do

            _ = model.forward(input_tensor)


        end_event.record()
        if synchronize:
            # This has to be synchronized to compute the elapsed time.
            # Otherwise, there will be runtime error.
            torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)

    else:
        for _ in range(num_repeats):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

            _ = model.forward(input_tensor)


            end_event.record()
            if synchronize:
                # This has to be synchronized to compute the elapsed time.
                # Otherwise, there will be runtime error.
                torch.cuda.synchronize()
            elapsed_time_ms += start_event.elapsed_time(end_event)

    return elapsed_time_ms / num_repeats

def main() -> None:

    num_warmups = 100
    num_repeats = 1000
    input_shape = (1, 3, 224, 224)

    device = torch.device("cuda:0")

    config=get_config(None)
    _update_config_from_file(config, './configs/pict.yaml')
    model = build_model(config)
    cpt = torch.load('/data/tangwenhao/output/pict/model/pict_swin_small_patch4_window7_224clu_3_new_best_model.pth', map_location='cpu')
    model.load_state_dict(cpt['state_dict'], strict=True)
    #model = torchvision.models.s(pretrained=False)
    #model = timm.create_model(model_name='tf_efficientnet_b3')## tf_efficientnet_b3  swin_small_patch4_window7_224
    # model = nn.Conv2d(in_channels=input_shape[1],
    #                   out_channels=256,
    #                   kernel_size=(5, 5))



    model.to(device)
    model.eval()

    # Input tensor
    input_tensor = torch.rand(input_shape, device=device)

    torch.cuda.synchronize()

    latency_ms = measure_time_device(
                    model=model,
                    input_tensor=input_tensor,
                    num_repeats=num_repeats,
                    num_warmups=num_warmups,
                )

    print( f"Latency: {latency_ms:.5f} ms| ")
    print("Latency Measurement Using PyTorch Benchmark...")
    num_threads = 1
    timer = benchmark.Timer(stmt="run_inference(model, input_tensor)",
                            setup="from __main__ import run_inference",
                            globals={
                                "model": model,
                                "input_tensor": input_tensor
                            },
                            num_threads=num_threads,
                            label="Latency Measurement",
                            sub_label="torch.utils.benchmark.")

    profile_result = timer.timeit(num_repeats)
    # https://pytorch.org/docs/stable/_modules/torch/utils/benchmark/utils/common.html#Measurement
    print(f"Latency: {profile_result.mean * 1000:.5f} ms")


if __name__ == "__main__":

    main()