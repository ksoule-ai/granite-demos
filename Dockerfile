FROM vllm/vllm-openai:v0.19.1

# git needed to fetch the package
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install Granite Switch (vLLM extra) so vLLM registers the granite_switch architecture at startup
RUN git clone https://github.com/generative-computing/granite-switch.git /opt/granite-switch \
 && pip install "/opt/granite-switch[vllm]"

# Base image's vLLM OpenAI entrypoint is inherited.
# We'll pass --model / --port / --host via the endpoint's Container Arguments.