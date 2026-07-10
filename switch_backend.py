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
import logging

import torch
from transformers import DynamicCache

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

logger = logging.getLogger(__name__)


def _common_prefix_len(a, b):
    n = min(a.shape[-1], b.shape[-1])
    if n == 0:
        return 0
    neq = (a[:n] != b[:n]).nonzero()
    return int(neq[0]) if len(neq) else n


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
        # Per-interaction KV prefix reuse (inert until begin_interaction()):
        # the interaction's DynamicCache, the token ids its prefix was
        # computed from, and the (hits, total) stats per generation.
        self._kv_cache = None
        self._kv_identity = None
        self._kv_stats = []

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

    # ----------------------------------------------------- KV prefix reuse
    # One Space interaction (one @spaces.GPU slot) shares a DynamicCache
    # across its generations: judge/adapter turns reuse the KV of their
    # common token prefix with the draft conversation, and retry attempts
    # reuse the instruction prefill. Sound for Granite Switch's aLoRA
    # adapters: KV before an activation token is base-weight KV, and two
    # prompts using different adapters differ at their control tokens, so a
    # common prefix can never extend past the first control token —
    # adapter-weight KV is only ever reused by the same adapter judging an
    # identical prefix.
    def begin_interaction(self):
        """Enable KV prefix reuse until end_interaction(); resets any state."""
        self._kv_cache = DynamicCache()
        self._kv_identity = None
        self._kv_stats = []

    def end_interaction(self):
        """Drop the interaction cache (frees its GPU tensors)."""
        self._kv_cache = None
        self._kv_identity = None
        self._kv_stats = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def pop_kv_stats(self):
        """Drain and return the (hits, total_prompt_tokens) recorded since
        the last pop, in generation order (generation is serialized)."""
        stats, self._kv_stats = self._kv_stats, []
        return stats

    def _generate_with_kv_reuse(self, generate_func, args, kwargs):
        """Call generate_func with the interaction cache's shared prefix.

        Two call shapes reach here (both funneled through the adapter lock):
        mellea's standard path (generate_func is model.generate, args[0] is
        the full-prompt ids tensor) and the intrinsic path (generate_func is
        granite's generate_with_transformers, args[2]["input_tokens"] is the
        prompt; extra keys flow to model.generate). Injecting past_key_values
        plus a FULL-length attention_mask makes generate trim the cached
        prefix internally (transformers only trims when the mask length
        matches input_ids).

        hits are measured against _kv_identity — the token ids the cache's
        contents were actually computed from — never against anything else:
        after a judge runs, the cache holds the judge's tokens, and comparing
        a new prompt to an older draft sequence could overstate the overlap
        and reuse invalid KV. Judge prompts are re-tokenized renders while
        draft identities hold generated ids, and BPE re-encode is not the
        identity, so measured overlap may stop partway into the draft answer;
        the stats report actual reused tokens.

        Any failure in the cache math falls back to a cache-less call with
        the pristine args. A failed generate itself is never retried: on the
        streaming path the streamer has already emitted tokens to the UI.
        """
        call_args, call_kwargs = args, kwargs
        prompt_ids = None
        mode = None
        injected = False
        try:
            if args and torch.is_tensor(args[0]):
                mode = "standard"
                prompt_ids = args[0]
                usable = (
                    prompt_ids.dim() == 2
                    and prompt_ids.shape[0] == 1
                    and kwargs.get("use_cache", True) is not False
                    and kwargs.get("num_return_sequences") in (None, 1)
                )
            elif (
                len(args) >= 3
                and isinstance(args[2], dict)
                and "input_tokens" in args[2]
            ):
                mode = "intrinsic"
                prompt_ids = args[2]["input_tokens"]
                usable = (
                    torch.is_tensor(prompt_ids)
                    and prompt_ids.dim() == 2
                    and prompt_ids.shape[0] == 1
                    and args[2].get("num_return_sequences") in (None, 1)
                )
            else:
                usable = False

            if usable:
                total = int(prompt_ids.shape[1])
                hits = 0
                if self._kv_identity is not None:
                    hits = max(0, min(
                        _common_prefix_len(
                            self._kv_identity.to(prompt_ids.device), prompt_ids[0]
                        ),
                        self._kv_cache.get_seq_length(),
                        total - 1,  # generate needs at least one new token
                    ))
                if hits == 0:
                    self._kv_cache = DynamicCache()
                else:
                    self._kv_cache.crop(hits)
                mask = torch.ones_like(prompt_ids)
                if mode == "standard":
                    call_kwargs = {
                        **kwargs,
                        "past_key_values": self._kv_cache,
                        "attention_mask": mask,
                    }
                else:
                    generate_input = dict(args[2])
                    generate_input["past_key_values"] = self._kv_cache
                    generate_input["attention_mask"] = mask
                    call_args = (*args[:2], generate_input, *args[3:])
                injected = True
                self._kv_stats.append((hits, total))
            elif torch.is_tensor(prompt_ids) and prompt_ids.dim() == 2:
                # Unusual call shape: run cache-less but still report it.
                self._kv_stats.append((0, int(prompt_ids.shape[1])))
        except Exception:
            logger.warning(
                "KV cache reuse disabled for this generation", exc_info=True
            )
            self._kv_cache = DynamicCache()
            self._kv_identity = None
            call_args, call_kwargs = args, kwargs
            injected = False
            if torch.is_tensor(prompt_ids) and prompt_ids.dim() == 2:
                self._kv_stats.append((0, int(prompt_ids.shape[1])))

        result = generate_func(*call_args, **call_kwargs)

        if injected:
            try:
                if mode == "standard" and hasattr(result, "sequences"):
                    # Full prompt + generated ids: exactly what the cache holds
                    # (minus the last token generate never feeds forward).
                    self._kv_identity = result.sequences[0].detach()
                else:
                    # generate_with_transformers returns a ChatCompletionResponse
                    # (no sequences); the prompt is the reusable-identity part.
                    self._kv_identity = prompt_ids[0].detach()
            except Exception:
                logger.warning("KV identity update failed; resetting cache", exc_info=True)
                self._kv_cache = DynamicCache()
                self._kv_identity = None
        return result

    # ------------------------------------------------------------- generation
    def _has_peft_adapters(self):
        return any(
            not isinstance(a, SwitchEmbeddedAdapter)
            for a in self._loaded_adapters.values()
        )

    def _generate_with_adapter_lock(self, adapter_name, generate_func, *args, **kwargs):
        embedded = isinstance(
            self._added_adapters.get(adapter_name), SwitchEmbeddedAdapter
        )
        if embedded or (adapter_name == "" and not self._has_peft_adapters()):
            # Embedded adapters activate via the control token already present
            # in the rendered prompt, and with no PEFT adapter ever loaded the
            # plain path has nothing to deactivate — in both cases the parent's
            # set_adapter([]) dance would require the peft package (it raises
            # "PEFT is not installed" rather than the "No adapter loaded"
            # ValueError the parent swallows). Keep the lock: the parent
            # serializes all generation through it.
            with self._generation_lock:
                if self._kv_cache is None:  # no interaction in progress
                    return generate_func(*args, **kwargs)
                return self._generate_with_kv_reuse(generate_func, args, kwargs)
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
