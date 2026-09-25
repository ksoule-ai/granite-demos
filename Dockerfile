FROM vllm/vllm-openai:v0.19.1

# git needed to fetch the package
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install Granite Switch (vLLM extra) so vLLM registers the granite_switch architecture at startup.
# Pinned: granite-switch 6013c7f (2026-09-10) dropped the SingleSwitch format that the
# ibm-granite/granite-switch-4.1-3b-preview checkpoint uses, without bumping its version.
ARG GRANITE_SWITCH_REF=756f946640d571a5beef2a49d4cab6614c61f18a
RUN git clone https://github.com/generative-computing/granite-switch.git /opt/granite-switch \
 && git -C /opt/granite-switch checkout "$GRANITE_SWITCH_REF" \
 && pip install "/opt/granite-switch[vllm]"

# Base image's vLLM OpenAI entrypoint is inherited.
# We'll pass --model / --port / --host via the endpoint's Container Arguments.