# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Let an InternVL (`-HF`) EC consumer accept a metadata-only image, so the
grid arms (`cpu-grid`, `example-grid`) run for InternVL3.5 and Intern-S1.

The EPD proxy's grid rewrite replaces each image with the metadata the
encoder published for it. What gets published is whatever the model's data
parser declares in `embedding_fields`; InternVL's (`interns1.py`) declares
nothing, so the encoder publishes an empty dict and the proxy has nothing to
rewrite. Even with metadata, InternVL sizes an embeddings input's
placeholder from the embedding tensor itself, which a metadata-only request
does not carry.

The tile count is all that sizes the placeholder: `image_num_patches` tiles of
`image_seq_length` (256) tokens each. This patch, in the pattern Qwen2-VL
uses (`Qwen2VLMultiModalDataParser`), makes `image_num_patches` the image
metadata and `image_embeds` its values:

  * `InternS1ProcessingInfo.get_data_parser` returns a parser declaring those
    fields that turns a dict image input into `DictEmbeddingItems`. On the
    producer this decides what the connector publishes; on the consumer, what
    a rewritten request must carry.
  * `InternS1MultiModalProcessor._get_prompt_updates` sizes a dict input's
    placeholder as `image_num_patches * image_seq_length`.
  * `_get_mm_fields_config` sizes `image_embeds` per image from the same
    count, so a request carrying embeddings for several images splits them.

Video stays as it is: the benchmark disables video input.

Applies to vLLM v0.32.0's `vllm/model_executor/models/interns1.py` (the
`InternVLForConditionalGeneration` architecture of the `-HF` releases).
Loaded from `sitecustomize.py`, so it reaches every process a server starts.
Importing the model module here is deferred until something else imports
it, so a server for another model never pays for it.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys

_TARGET = "vllm.model_executor.models.interns1"
_MARKER = "_ec_bench_metadata_patch"


def _patch(mod) -> None:
    import torch

    from vllm.multimodal.inputs import MultiModalFieldConfig
    from vllm.multimodal.parse import DictEmbeddingItems, MultiModalDataParser

    info_cls = mod.InternS1ProcessingInfo
    proc_cls = mod.InternS1MultiModalProcessor
    if getattr(proc_cls, _MARKER, False):
        return

    def _image_embeds_config(hf_inputs):
        # One tile count per image. Metadata arriving over the API is a
        # 1-element vector per image, so the stacked batch is (N, 1).
        num_patches = torch.as_tensor(
            hf_inputs.get("image_num_patches", torch.empty(0))
        ).reshape(-1)
        return dict(
            image_embeds=MultiModalFieldConfig.flat_from_sizes(
                "image", num_patches * _IMAGE_SEQ_LENGTH[0]
            ),
            image_num_patches=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
        )

    class InternS1MultiModalDataParser(MultiModalDataParser):
        # The tile count is what sizes the placeholder range.
        embedding_fields = {
            "image": {"image_embeds": "values", "image_num_patches": "metadata"},
        }

        def _parse_image_data(self, data):
            if isinstance(data, dict):
                required, optional = self.embedding_field_sets("image")
                return DictEmbeddingItems(
                    data,
                    modality="image",
                    required_fields=required,
                    optional_fields=optional,
                    fields_factory=_image_embeds_config,
                )
            return super()._parse_image_data(data)

    # image_seq_length comes from the HF processor, which the field factory
    # cannot reach; the first get_data_parser call records it.
    _IMAGE_SEQ_LENGTH = [256]

    def get_data_parser(self):
        _IMAGE_SEQ_LENGTH[0] = self.get_hf_processor().image_seq_length
        return InternS1MultiModalDataParser(
            expected_hidden_size=self._get_expected_hidden_size(),
            allow_missing_mm_embeddings=self.allow_missing_mm_embeddings,
        )

    info_cls.get_data_parser = get_data_parser

    orig_fields = proc_cls._get_mm_fields_config

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        if "image_num_patches" in hf_inputs and "pixel_values" not in hf_inputs:
            # The stock config sizes `pixel_values` from the same tile counts,
            # so it needs them one per image too.
            hf_inputs = type(hf_inputs)(
                {
                    **hf_inputs,
                    "image_num_patches": torch.as_tensor(
                        hf_inputs["image_num_patches"]
                    ).reshape(-1),
                }
            )
            config = dict(orig_fields(self, hf_inputs, hf_processor_mm_kwargs))
            config.update(_image_embeds_config(hf_inputs))
            return config
        return orig_fields(self, hf_inputs, hf_processor_mm_kwargs)

    proc_cls._get_mm_fields_config = _get_mm_fields_config

    orig_updates = proc_cls._get_prompt_updates

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        updates = list(
            orig_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs)
        )
        images = mm_items.get("image")
        if not isinstance(images, DictEmbeddingItems):
            return updates
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        repl_tpl = (
            hf_processor.start_image_token,
            hf_processor.image_token,
            hf_processor.end_image_token,
        )

        def replacement(item_idx: int):
            num_patches = int(images.get(item_idx)["image_num_patches"])
            start, ctx, end = repl_tpl
            ids = mod.cached_encode(
                tokenizer,
                start + ctx * (num_patches * hf_processor.image_seq_length) + end,
                add_special_tokens=False,
            )
            return mod.PromptUpdateDetails.select_token_id(
                ids, hf_processor.image_token_id
            )

        return [
            mod.PromptReplacement(
                modality="image", target=u.target, replacement=replacement
            )
            if u.modality == "image"
            else u
            for u in updates
        ]

    proc_cls._get_prompt_updates = _get_prompt_updates
    setattr(proc_cls, _MARKER, True)


class _PatchOnImport(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Runs `_patch` on the target module right after it executes."""

    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        self._inner = spec.loader
        spec.loader = self
        return spec

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        _patch(module)


def install() -> None:
    if _TARGET in sys.modules:
        _patch(sys.modules[_TARGET])
    elif not any(isinstance(f, _PatchOnImport) for f in sys.meta_path):
        sys.meta_path.insert(0, _PatchOnImport())
