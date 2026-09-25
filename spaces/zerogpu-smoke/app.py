"""Minimal ZeroGPU smoke test.

One button: requests a ZeroGPU slot, reports what hardware was granted,
and times a matmul to prove the GPU actually executes work. If this Space
works, the full Granite Switch demo's @spaces.GPU path will too.
"""

import time

import gradio as gr
import spaces
import torch


@spaces.GPU(duration=60)
def gpu_check():
    lines = [f"torch {torch.__version__}"]
    if not torch.cuda.is_available():
        lines.append("FAIL: torch.cuda.is_available() is False inside @spaces.GPU")
        return "\n".join(lines)

    props = torch.cuda.get_device_properties(0)
    lines.append(f"device: {props.name}")
    lines.append(f"vram:   {props.total_memory / 2**30:.1f} GiB")
    lines.append(f"cuda:   {torch.version.cuda}")

    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        a = a @ a.T
        a = a / a.norm()
    torch.cuda.synchronize()
    lines.append(f"10x 4096^2 bf16 matmul: {(time.time() - t0) * 1000:.1f} ms")
    lines.append("PASS: ZeroGPU allocated and computing")
    return "\n".join(lines)


with gr.Blocks(title="ZeroGPU Smoke Test") as demo:
    gr.Markdown(
        "# ZeroGPU Smoke Test\n"
        "Click the button to request a ZeroGPU slot and run a quick compute "
        "check. Precursor to the Granite Switch 4.1 30B demo — for the 30B "
        "model in BF16 (~60 GB) the reported VRAM should be ≥ 70 GiB "
        "(`xlarge`); 48 GiB (`large`) needs quantization."
    )
    btn = gr.Button("Run GPU check", variant="primary")
    out = gr.Textbox(label="Result", lines=8)
    btn.click(fn=gpu_check, outputs=out)


if __name__ == "__main__":
    demo.launch()
