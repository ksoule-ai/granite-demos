"""Mellea LocalHFBackend glue for Granite Switch *embedded* adapters.

Mellea 0.6.0 supports Granite Switch's embedded adapters only on its
OpenAI/vLLM backend (``load_embedded_adapters=True``). Its HuggingFace
backend (``LocalHFBackend``) assumes every adapter is a separately
published PEFT weight directory: ``add_adapter`` downloads weights,
``load_adapter`` PEFT-loads them, and generation activates them with
``model.set_adapter``. None of that applies to a Granite Switch
checkpoint, where the LoRA weights are baked into the model and an
adapter is activated by a control token that the chat template splices
in when ``apply_chat_template(..., adapter_name=...)`` is passed.

``SwitchBackend`` closes that gap with three small overrides:

* registration accepts ``EmbeddedIntrinsicAdapter`` (no weights to
  download — only the io.yaml config matters);
* the PEFT load/activate/assert steps are skipped for embedded adapters
  (generation still serializes on the backend's lock);
* while an embedded intrinsic is being generated, ``apply_chat_template``
  is called with ``adapter_name=<intrinsic>``, which makes the model's
  chat template insert the adapter's activation token. GraniteSwitch's
  switch layer detects that token at inference time and applies the
  adapter's LoRA deltas — no PEFT involved.

Pinned against mellea==0.6.0 internals (``_generate_from_intrinsic``,
``_generate_with_adapter_lock``, ``_added_adapters``); revisit on upgrade —
upstream intends to support embedded adapters on the HF backend natively.
"""

import contextvars

import torch
from mellea.backends.adapters import AdapterType
from mellea.backends.adapters.adapter import (
    Adapter,
    EmbeddedIntrinsicAdapter,
    IntrinsicAdapter,
    get_adapter_for_intrinsic,
)
from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.model_options import ModelOption

# The intrinsic name whose adapter must be activated by the chat template
# for the generation currently being prepared. Context-local so concurrent
# non-intrinsic generations never see it.
_ACTIVE_EMBEDDED_ADAPTER: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "active_embedded_adapter", default=None
)


class SwitchEmbeddedAdapter(EmbeddedIntrinsicAdapter, IntrinsicAdapter):
    """An embedded adapter that passes LocalHFBackend's IntrinsicAdapter checks.

    ``LocalHFBackend._generate_from_intrinsic`` rejects anything that is not
    an ``IntrinsicAdapter``; inheriting from both keeps embedded (no-weights)
    semantics while satisfying that check. Neither parent ``__init__`` fits
    (IntrinsicAdapter's validates against the PEFT catalog), so this
    initializes ``Adapter`` directly, mirroring EmbeddedIntrinsicAdapter.
    """

    def __init__(self, intrinsic_name: str, config: dict, technology: str = "alora"):
        if technology not in ("lora", "alora"):
            raise ValueError(f"technology must be 'lora' or 'alora', got '{technology}'")
        adapter_type = AdapterType.ALORA if technology == "alora" else AdapterType.LORA
        Adapter.__init__(self, intrinsic_name, adapter_type)
        self.intrinsic_name = intrinsic_name
        self.config = config
        self.technology = technology

    @classmethod
    def from_embedded(cls, adapter: EmbeddedIntrinsicAdapter) -> "SwitchEmbeddedAdapter":
        return cls(adapter.intrinsic_name, adapter.config, adapter.technology)

    def get_local_hf_path(self, base_model_name: str) -> str:
        # Weights live inside the switch checkpoint; there is nothing to
        # download. add_adapter stores this as the (unused) adapter path.
        return ""


def _patch_apply_chat_template(tokenizer):
    """Make the tokenizer add ``adapter_name`` to apply_chat_template calls.

    mellea renders intrinsic prompts via
    ``granite_formatters.base.util.chat_completion_request_to_transformers_inputs``,
    which calls ``tokenizer.apply_chat_template`` without any way to pass
    template kwargs. The Granite Switch template only inserts an adapter's
    activation token when ``adapter_name`` is given, so this patches the
    tokenizer *instance* to inject it whenever
    :data:`_ACTIVE_EMBEDDED_ADAPTER` is set. Patching the instance (not
    wrapping it in a proxy) keeps ``type(tokenizer)`` intact for libraries
    that type-check it (xgrammar, llguidance).

    The patch also returns the bare ``input_ids`` tensor for
    ``return_tensors="pt"`` calls: transformers >= 5 (which granite-switch
    requires) returns a BatchEncoding there, but mellea 0.6.0's standard
    generation path feeds the result straight into ``model.generate`` and
    slices it with ``.shape`` — both of which need the tensor.
    """
    if getattr(tokenizer, "_switch_adapter_patch", False):
        return
    original = tokenizer.apply_chat_template

    def apply_chat_template(*args, **kwargs):
        adapter_name = _ACTIVE_EMBEDDED_ADAPTER.get()
        if adapter_name is not None:
            kwargs.setdefault("adapter_name", adapter_name)
        result = original(*args, **kwargs)
        if kwargs.get("return_tensors") == "pt" and not torch.is_tensor(result):
            result = result["input_ids"]
        return result

    tokenizer.apply_chat_template = apply_chat_template
    tokenizer._switch_adapter_patch = True


class SwitchBackend(LocalHFBackend):
    """LocalHFBackend for Granite Switch checkpoints with embedded adapters."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _patch_apply_chat_template(self._tokenizer)
        # mellea's stdlib intrinsic helpers (core.requirement_check, ...)
        # auto-register missing adapters from this repo when the backend
        # declares it uses embedded adapters.
        self._uses_embedded_adapters = True
        self._adapter_source = self._hf_model_id

    def register_embedded_adapters(self, intrinsic_name: str | None = None) -> list[str]:
        """Register all (or one of) the checkpoint's embedded adapters.

        Reads ``adapter_index.json`` + ``io_configs/*/io.yaml`` from the model
        repo. Returns the registered intrinsic names.
        """
        adapters = EmbeddedIntrinsicAdapter.from_source(
            self._hf_model_id, intrinsic_name=intrinsic_name
        )
        for adapter in adapters:
            self.add_adapter(adapter)
        return [a.intrinsic_name for a in adapters]

    def has_embedded_adapter(self, intrinsic_name: str) -> bool:
        return any(
            isinstance(a, SwitchEmbeddedAdapter) and a.intrinsic_name == intrinsic_name
            for a in self._added_adapters.values()
        )

    # ------------------------------------------------------------ registration
    def add_adapter(self, adapter):
        if isinstance(adapter, EmbeddedIntrinsicAdapter):
            if not isinstance(adapter, SwitchEmbeddedAdapter):
                adapter = SwitchEmbeddedAdapter.from_embedded(adapter)
            if adapter.qualified_name in self._added_adapters:
                return  # idempotent for embedded adapters (nothing is loaded)
        return super().add_adapter(adapter)

    def load_adapter(self, adapter_qualified_name: str):
        adapter = self._added_adapters.get(adapter_qualified_name)
        if isinstance(adapter, SwitchEmbeddedAdapter):
            # Weights are baked into the checkpoint — nothing to load. Track
            # it so list_adapters() reflects availability.
            self._loaded_adapters[adapter_qualified_name] = adapter
            return
        return super().load_adapter(adapter_qualified_name)

    def unload_adapter(self, adapter_qualified_name: str):
        adapter = self._added_adapters.get(adapter_qualified_name)
        if isinstance(adapter, SwitchEmbeddedAdapter):
            self._loaded_adapters.pop(adapter_qualified_name, None)
            return
        return super().unload_adapter(adapter_qualified_name)

    # ------------------------------------------------------------- generation
    def _generate_with_adapter_lock(self, adapter_name, generate_func, *args, **kwargs):
        if isinstance(self._added_adapters.get(adapter_name), SwitchEmbeddedAdapter):
            # Activation happens via the control token already present in the
            # rendered prompt; PEFT set_adapter/asserts don't apply. Keep the
            # lock: the parent serializes all generation through it.
            with self._generation_lock:
                return generate_func(*args, **kwargs)
        return super()._generate_with_adapter_lock(
            adapter_name, generate_func, *args, **kwargs
        )

    async def _generate_from_intrinsic(self, action, ctx, *, model_options, tool_calls=False):
        adapter = get_adapter_for_intrinsic(
            action.intrinsic_name, action.adapter_types, self._added_adapters
        )
        if not isinstance(adapter, SwitchEmbeddedAdapter):
            return await super()._generate_from_intrinsic(
                action, ctx, model_options=model_options, tool_calls=tool_calls
            )
        # Adapters are judges, not generators: their io.yaml fixes greedy
        # decoding and a small token budget. Sampling strategies forward the
        # caller's model_options to validation calls too, so strip generation
        # tuning here or a user temperature would override the judge's
        # temperature=0.0 from io.yaml.
        model_options = {
            k: v
            for k, v in model_options.items()
            if k
            not in ("temperature", "do_sample", "top_p", "top_k", "max_new_tokens")
            and k != ModelOption.MAX_NEW_TOKENS
        }
        # The prompt for this intrinsic is rendered synchronously inside the
        # parent coroutine (before any thread handoff), so the context var is
        # visible to the tokenizer proxy exactly for this generation.
        token = _ACTIVE_EMBEDDED_ADAPTER.set(action.intrinsic_name)
        try:
            return await super()._generate_from_intrinsic(
                action, ctx, model_options=model_options, tool_calls=tool_calls
            )
        finally:
            _ACTIVE_EMBEDDED_ADAPTER.reset(token)
