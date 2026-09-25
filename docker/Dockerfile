# Custom vLLM image for IBM Granite Switch.
#
# Stock vLLM/Transformers do not recognize the `granite_switch` architecture — it
# ships only in the separate `granite-switch` package. This image adds it so vLLM
# registers the architecture at startup and can serve the preview checkpoint.
#
# Build via GitHub Actions (see .github/workflows/build.yml), NOT locally on a Mac:
# Inference Endpoints require linux/amd64, and a Mac produces arm64.
#
# Version note: the `[vllm]` extra pins vLLM 0.19.1 (CUDA 12.x), which matches HF's
# L4 hosts. `[vllm20]` needs CUDA 13+, which those hosts do not have — do not use it.

FROM vllm/vllm-openai:v0.19.1

# git is needed to fetch the package
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install Granite Switch (vLLM extra). This registers the granite_switch architecture
# with vLLM via the package's plugin, so the stock OpenAI server entrypoint can serve it.
#
# Pinned to a known-good commit. granite-switch 6013c7f (2026-09-10) dropped the
# SingleSwitch format that the ibm-granite/granite-switch-4.1-3b-preview checkpoint
# uses, without bumping its version (both report 0.1.0), so an unpinned clone breaks
# the endpoint at startup. Bump this only together with a MultiSwitch checkpoint.
ARG GRANITE_SWITCH_REF=756f946640d571a5beef2a49d4cab6614c61f18a
RUN git clone https://github.com/generative-computing/granite-switch.git /opt/granite-switch \
 && git -C /opt/granite-switch checkout "$GRANITE_SWITCH_REF" \
 && pip install "/opt/granite-switch[vllm]"

# The base image's vLLM OpenAI server entrypoint is inherited.
# Inference Endpoints passes runtime flags via the "Container Arguments" field, e.g.:
#   --model /repository --served-model-name ibm-granite/granite-switch-4.1-3b-preview \
#   --host 0.0.0.0 --enable-prompt-tokens-details
# (HF mounts the selected model at /repository, so --model points there, not at the Hub.)
