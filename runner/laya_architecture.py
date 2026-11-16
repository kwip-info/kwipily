"""Reviewed Laya sequence/head implementation, Apache-2.0.
Derived without algorithm changes from NandhaKishorM/laya common.py at
6f0496f797dc7127265d62e1543a62de507d943c. See accompanying laya-LICENSE.txt (canonical third_party/laya/LICENSE).
Only inert tokenization/architecture helpers retained. No upstream downloader,
router, hooks, compilation backend, SDK loader or remote-code path is bundled.
"""
import json,threading
from contextlib import nullcontext
from contextvars import ContextVar
from typing import Any,Dict,List,Optional,Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
QTYPES={'choice':0,'score':1,'noul':2}
_DEFAULT_NOUL_LABELS={'false':'false','true':'true'}
_TOKENIZE_LOCK=threading.RLock()
_QUESTION_TOKEN_CACHE=ContextVar('laya_question_tokens',default=None)
def _autocast_enabled(device_type: str) -> bool:
    """Whether an autocast scope covers `device_type`.

    `torch.is_autocast_enabled` only accepted a `device_type` from torch 2.4; before that it
    reported CUDA alone and CPU and MPS had no autocast to report. This package declares
    `torch>=2.0.0`, so call the device-aware form where it exists and fall back otherwise,
    using torch's separate CPU predicate.
    """
    try:
        return torch.is_autocast_enabled(device_type)
    except TypeError:
        # torch < 2.4 took no `device_type`: the no-argument call reports CUDA alone, CPU has
        # its own predicate, and MPS had no autocast to report.
        if device_type == "cpu":
            return torch.is_autocast_cpu_enabled()
        return torch.is_autocast_enabled()

def encode_text(tok, text, **kwargs):
    """Tokenize `text` while holding the lock a shared fast tokenizer needs."""
    with _TOKENIZE_LOCK:
        return tok(text, **kwargs)

def _encode_question_text(tok, text, **kwargs):
    scope = _QUESTION_TOKEN_CACHE.get()
    cache = scope["tokens"] if scope is not None and scope["thread"] == threading.get_ident() else None
    if cache is None:
        return encode_text(tok, text, **kwargs)["input_ids"]
    # Key the rendered, mask-sanitized text and encoding settings, not a JSON question:
    # option order, structured criteria and custom noul labels must keep their meaning.
    key = (id(tok), text, tuple(kwargs.items()))
    if key not in cache:
        # Keep the tokenizer alive so its identity cannot be reused within this scope.
        cache[key] = (tok, tuple(encode_text(tok, text, **kwargs)["input_ids"]))
    # Sequence assembly must not mutate token lists retained for later states.
    return list(cache[key][1])

def serialize_state(state: Union[str, dict, list]) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)

def render_criterion(value) -> str:
    """Render one criterion value as text.

    Strings pass through; anything structured (dict, list, number) becomes a single-line JSON
    document with the default separators -- ``", "`` between members, ``": "`` before a value --
    so a rubric reads as JSON rather than a Python repr. Without this a dict-valued criterion
    crashed `noul` outright and leaked `{'desc': ...}` into `choice` and `score` prompts.
    """
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)

def _resolve_noul_labels(labels=None):
    if labels is None:
        labels = _DEFAULT_NOUL_LABELS
    if not isinstance(labels, dict) or set(labels) != {"false", "true"}:
        raise ValueError("noul labels must map exactly 'false' and 'true' to distinct non-empty strings")
    false_label, true_label = labels["false"], labels["true"]
    if not isinstance(false_label, str) or not isinstance(true_label, str):
        raise ValueError("noul labels must map exactly 'false' and 'true' to distinct non-empty strings")
    false_label, true_label = false_label.strip(), true_label.strip()
    if not false_label or not true_label or false_label == true_label:
        raise ValueError("noul labels must map exactly 'false' and 'true' to distinct non-empty strings")
    return false_label, true_label

def render_options(q: Dict) -> List[str]:
    """Render option texts in label-index order. Noul semantic order is always [false, true]."""
    t, crit = q["t"], q.get("crit")
    if t != "noul" and "labels" in q:
        raise ValueError("labels is only supported for noul questions")
    if t == "choice":
        # only None/"" mean "no description"; 0 and False are legitimate criterion values.
        # `str(k)` unconditionally: a label with no description is rendered as itself, so an int
        # label used to come back as an int from a function annotated `-> List[str]` and then
        # reached `build_sequence`, which calls `.replace` on it and raised an AttributeError
        # naming neither the question nor the label. With a description the same label already
        # went through `"%s: %s" %` and was a str, which is why only the undescribed form broke.
        # `structured._enum_field` stringifies labels the same way; the returned answer still
        # carries the caller's original label, which is unchanged.
        return [str(k) if v is None or v == "" else "%s: %s" % (k, render_criterion(v))
                for k, v in crit.items()]
    if t == "score":
        return ["level %d: %s" % (i, render_criterion(c)) for i, c in enumerate(crit)]
    crit = crit or {}
    false_label, true_label = _resolve_noul_labels(q.get("labels"))
    false_crit, true_crit = crit.get("false"), crit.get("true")
    return [
        false_label + ": "
        + (render_criterion(false_crit) if false_crit not in (None, "") else "no, the statement does not hold"),
        true_label + ": "
        + (render_criterion(true_crit) if true_crit not in (None, "") else "yes, the statement holds"),
    ]

def build_sequence(
    tok,
    state: Union[str, dict, list],
    q: Dict,
    max_len: int = 512,
    head_max_len: int = 192,
    option_order: Optional[List[int]] = None,
    truncate_left: bool = False,
    state_ids: Optional[List[int]] = None,
    return_stats: bool = False,
    return_truncation_stats: bool = False,
    return_layout: bool = False,
):
    """Format: [CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP].

    `state_ids` lets a caller tokenize the shared state once and reuse it across every question,
    instead of re-serializing and re-tokenizing the same document per question.

    `return_stats` adds a third return value describing what the head budget did to the options:
    `options` (how many the question defines), `options_distinct` (how many still have a token
    span of their own) and `tokens_per_option` (the cap applied to each, or None when none was).

    The state is clamped to whatever room is left after the head, so a long state loses tokens
    here silently. `return_truncation_stats=True` adds one more return value, after the option
    stats when both are asked for, reporting that clamp:

        {"state_tokens": int, "state_tokens_used": int, "state_tokens_dropped": int,
         "truncated": bool}

    Callers cannot reconstruct this from the outside. The budget is in tokens, not characters,
    and the room left for the state depends on `max_len`, `head_max_len`, the instruction and
    the rendered options - so it moves per checkpoint and per question. A caller guessing with a
    fixed character threshold is wrong in both directions: it reports truncation that did not
    happen, and stays silent while evidence is being dropped (issue #174).

    The question half is `build_head`; `state_room` reports how much of `max_len` is left for the
    state after it, which is what a caller must size a window against.

    `return_layout=True` adds one last return value, the parallel option layout from
    `parallel_layout`: `{"position_ids": [...], "option_ids": [...]}`, one entry per token.
    """
    ids, markers, stats = build_head(tok, q, head_max_len, option_order=option_order)
    head_len = len(ids)
    room = max(0, max_len - len(ids) - 1)
    if state_ids is None:
        state_ids = encode_text(tok, serialize_state(state).replace(tok.mask_token, " "),
                                add_special_tokens=False)["input_ids"]
    # not state_ids[-room:]: with no room left, state_ids[-0:] is the whole state rather than none of it
    st = state_ids[max(0, len(state_ids) - room):] if truncate_left else state_ids[:room]
    ids = ids + st + [tok.sep_token_id]
    # Laid out before the clamp: a dropped trailing option must not widen the last surviving span
    layout = parallel_layout(markers, head_len, len(ids)) if return_layout else None
    ids, markers = ids[:max_len], [m for m in markers if m < max_len]
    extra = ()
    if return_truncation_stats:
        # `room` leaves space for the closing [SEP], so every token in `st` survives the [:max_len] clamp
        extra = ({
            "state_tokens": len(state_ids),
            "state_tokens_used": len(st),
            "state_tokens_dropped": len(state_ids) - len(st),
            "truncated": len(st) < len(state_ids),
        },)
    if return_layout:
        extra += ({k: v[:max_len] for k, v in layout.items()},)
    if not return_stats:
        return (ids, markers) + extra
    return (ids, markers, stats) + extra

def build_head(tok, q: Dict, head_max_len: int = 192, option_order: Optional[List[int]] = None):
    """The question half of a sequence: `[CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP]`.

    `build_sequence` appends the state to this and `state_room` measures what is left over for it,
    so neither can disagree with the other about what the head costs.

    Returns `(ids, markers, stats)`: the head token ids, the `[MASK]` position of each option in
    `option_order`, and the stats dict `build_sequence` returns for `return_stats`.
    """
    mask_tok = tok.mask_token
    opts = render_options(q)
    order = option_order if option_order is not None else list(range(len(opts)))
    ins = str(q["ins"]).replace(mask_tok, " ")
    head_ids = _encode_question_text(tok, "%s question: %s" % (q["t"], ins), add_special_tokens=False)
    opt_ids = []
    for i in order:
        # Cap at the tokenizer, not after the fact: `[:48]` still makes the tokenizer process the
        # whole (possibly long) description. truncation=True, max_length=48 keeps the first 48
        # tokens, which is exactly what the previous slice produced.
        opt_tokens = _encode_question_text(
            tok,
            " " + opts[i].replace(mask_tok, " "),
            add_special_tokens=False,
            truncation=True,
            max_length=48,
        )
        opt_ids.append([tok.mask_token_id] + opt_tokens)
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    per_option = None
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        per_option = per
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[: max(8, opt_budget)]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)
    # Two options that share a prefix can come out of the cut as the same token span: the marker
    # count still matches the option count, so the guard in `Agent._encode_state` passes and
    # nothing downstream can tell that the question lost the ability to name them apart. Counted
    # on the capped option ids, before assembly: re-slicing the finished sequence cannot close
    # the last option's span -- it runs on into the serialized state, which differs per request,
    # so the last option always looks distinguishable however it collided (#538).
    return ids, markers, {
        "options": len(opt_ids),
        "options_distinct": len({tuple(o) for o in opt_ids}),
        "tokens_per_option": per_option,
    }

def parallel_layout(markers: List[int], head_len: int, length: int) -> Dict[str, List[int]]:
    """Position ids and option ids that make the encoder blind to option order.

    In the sequential layout option `s` sits at positions after option `s-1`, and every option
    attends to every other, so where an option is listed changes its embedding: the slot logits
    of five identical options spread by 3.59 (`tests/test_option_order.py`). Here every option
    starts at the same position, the first one after the instruction's [SEP], and the head's
    closing [SEP] and the state continue after the longest option. Together with
    `parallel_option_masks`, which stops an option from attending to another, reordering the
    options only reorders the marker embeddings.

    `option_ids` is 0 for shared tokens (instruction, closing [SEP], state, padding) and `s + 1`
    for the tokens of slot `s`, which runs from its [MASK] up to the next marker or, for the last
    slot, up to the [SEP] at `head_len - 1`.
    """
    if not markers:
        return {"position_ids": list(range(length)), "option_ids": [0] * length}
    start = markers[0]
    spans = list(zip(markers, markers[1:] + [head_len - 1]))
    position_ids, option_ids = list(range(start)), [0] * start
    for s, (a, b) in enumerate(spans):
        position_ids += range(start, start + b - a)
        option_ids += [s + 1] * (b - a)
    after = start + max(b - a for a, b in spans)
    position_ids += range(after, after + length - len(position_ids))
    option_ids += [0] * (length - len(option_ids))
    return {"position_ids": position_ids, "option_ids": option_ids}

def parallel_option_masks(attention_mask: torch.Tensor, position_ids: torch.Tensor, option_ids: torch.Tensor,
                          sliding_window: Optional[int]) -> Dict[str, torch.Tensor]:
    """Per-layer-type boolean attention masks for the parallel layout (`parallel_layout`).

    A token attends to every real token except those of a *different* option. ModernBERT's local
    layers keep their window, measured in position ids rather than sequence index: options share
    positions, so a sequence-index window would let slot 0 see more of the state than slot 3.
    Returned in the `{"full_attention": ..., "sliding_attention": ...}` form ModernBERT takes
    in place of a padding mask; each mask is `[batch, 1, query, key]`, True where attention is allowed.
    """
    keys = attention_mask.bool()[:, None, :]
    other_option = (option_ids[:, :, None] > 0) & (option_ids[:, None, :] > 0) \
        & (option_ids[:, :, None] != option_ids[:, None, :])
    full = keys & ~other_option
    masks = {"full_attention": full[:, None]}
    if sliding_window is not None:
        near = (position_ids[:, :, None] - position_ids[:, None, :]).abs() <= sliding_window
        masks["sliding_attention"] = (full & near)[:, None]
    return masks

class _DynamicMultiheadAttention(nn.MultiheadAttention):
    """`nn.MultiheadAttention` that keeps the shapes it traces.

    The stock module reshapes the packed projection with sizes captured while tracing, so under
    the legacy ONNX exporter the traced sequence length becomes a constant and a model exported
    from a short dummy input only runs at that length. The parameters and the maths are the same
    here; the reshape uses only constant shape arguments (`chunk` / `unflatten` / `flatten`) and
    `scaled_dot_product_attention`, the kernel the stock path already uses when weights are not
    requested.
    """

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        if (need_weights or self.in_proj_weight is None or self.bias_k is not None
                or self.bias_v is not None
                or (attn_mask is not None and attn_mask.dtype != torch.bool)):
            # Weight averaging, additive masks and the optional k/v bias are not on the traced
            # path; the stock implementation keeps them correct.
            return super().forward(query, key, value, key_padding_mask=key_padding_mask,
                                   need_weights=need_weights, attn_mask=attn_mask,
                                   average_attn_weights=average_attn_weights, is_causal=is_causal)
        if self.batch_first:
            query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
        # (T, B, E) from here, matching the stock module's internals; attention runs on (B, H, T, D).
        if query is key is value:
            q, k, v = (part.unflatten(-1, (self.num_heads, self.head_dim)).permute(1, 2, 0, 3)
                       for part in F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1))
        else:
            embed_dim = query.shape[-1]
            wq, wk, wv = self.in_proj_weight.split(embed_dim, dim=0)
            bq, bk, bv = ((None, None, None) if self.in_proj_bias is None
                          else self.in_proj_bias.split(embed_dim, dim=0))
            q, k, v = (
                F.linear(t, w, b).unflatten(-1, (self.num_heads, self.head_dim)).permute(1, 2, 0, 3)
                for t, w, b in ((query, wq, bq), (key, wk, bk), (value, wv, bv))
            )
        mask = None
        if attn_mask is not None:
            mask = ~attn_mask
        if key_padding_mask is not None:
            # `== 0` keeps this correct for a bool mask and for the 0 / -inf float mask the encoder
            # layer hands over (`F._canonical_mask`), where `~` would not be defined.
            keep = key_padding_mask[:, None, None, :] == 0
            mask = keep if mask is None else mask & keep
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal and mask is None,
                                              dropout_p=self.dropout if self.training else 0.0)
        attn = attn.permute(2, 0, 1, 3).flatten(-2)
        if self.batch_first:
            attn = attn.transpose(0, 1)
        return self.out_proj(attn), None

class DecisionModel(nn.Module):
    """Bidirectional transformer encoder backbone + typed decision head."""

    def __init__(self, encoder: nn.Module, head_layers: int = 2, n_act: int = 2, dropout: float = 0.1,
                 no_init: bool = False):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        # `no_init` means the caller is about to load every parameter from a checkpoint, so the
        # head's initial values are pure overhead -- and not just time. transformers'
        # `no_init_weights()` patches the torch.nn.init functions, but something inside
        # nn.TransformerEncoderLayer draws from the RNG outside them, so building it still
        # advanced the global generator and made `load()` a visible side effect. On the meta
        # device no initialisation kernel runs at all; the layers are materialised empty and the
        # load fills them.
        with torch.device("meta") if no_init else nullcontext():
            nhead = max(1, d // 64)
            layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True, norm_first=True)
            # The stock attention bakes the traced length into an exported graph; see
            # _DynamicMultiheadAttention. Same parameters, same maths, traceable shapes.
            layer.self_attn = _DynamicMultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
            self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False) if head_layers > 0 else None
            self.type_emb = nn.Embedding(3, d)
            self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
            self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
            self.register_buffer("temperature", torch.ones(3))
        if no_init:
            # Only the modules created above are on the meta device; the encoder is already real
            # and may hold non-persistent buffers (RoPE frequencies) that to_empty would wipe.
            for module in (self.head, self.type_emb, self.scorer, self.act_head):
                if module is not None:
                    module.to_empty(device="cpu")
            self.temperature = torch.empty_like(self.temperature, device="cpu")
        self.head_checkpointing = False

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder: bool = False,
                position_ids=None, option_ids=None):
        if option_ids is None:
            h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        else:
            # Parallel option layout (`parallel_layout`). The head below has no positional
            # encoding, so with order-blind marker embeddings the logits permute with the options.
            masks = parallel_option_masks(attention_mask, position_ids, option_ids,
                                          getattr(self.encoder.config, "sliding_window", None))
            h = self.encoder(input_ids=input_ids, attention_mask=masks, position_ids=position_ids).last_hidden_state
        if detach_encoder:
            h = h.detach()
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                if self.head_checkpointing and self.training and torch.is_grad_enabled():
                    # Non-reentrant checkpointing also trains the head when its input
                    # is frozen. Default RNG preservation keeps dropout consistent.
                    h = checkpoint(layer, h, src_key_padding_mask=pad, use_reentrant=False)
                else:
                    h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)

        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        if p.size(-1) >= 2:
            top2 = p.topk(2, -1).values
        else:
            # A single-option question has exactly one marker, so p.topk(2, ...)
            # has nothing to select for the second slot and raises. The answer
            # is still well-defined: softmax over one logit is 1.0 regardless of
            # its value, so pad the missing second entry with 0.0 - that gives
            # the act head top1 - top2 == 1.0, the same "fully decided" signal
            # it would see for any other unambiguous top-1-vs-rest gap.
            top1 = p.topk(1, -1).values
            top2 = torch.cat([top1, torch.zeros_like(top1)], dim=-1)
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        pooled = h[:, 0].float()
        act_input = torch.cat([pooled, feats], -1)
        # Explicit-dtype exports run without autocast; keep the confidence maths in fp32,
        # then match the head's weights. Autocast already chooses the linear's input dtype.
        if not _autocast_enabled(h.device.type):
            act_input = act_input.to(self.act_head[0].weight.dtype)
        act_logits = self.act_head(act_input)
        return logits, act_logits

def collate_items(batch, pad_id: int):
    items = [it for group in batch for it in group]
    if not items:
        return None
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = np.full((n, L), pad_id, dtype=np.int64)
    att = np.zeros((n, L), dtype=np.int64)
    mpos = np.zeros((n, kmax), dtype=np.int64)
    mmask = np.zeros((n, kmax), dtype=bool)
    has_target = any("target" in it for it in items)
    target = np.zeros((n, kmax), dtype=np.float32) if has_target else None

    for i, it in enumerate(items):
        it_ids = it["ids"]
        l_ids = len(it_ids)
        ids[i, :l_ids] = it_ids
        att[i, :l_ids] = 1
        markers = it["markers"]
        k = len(markers)
        mpos[i, :k] = markers
        mmask[i, :k] = True
        if has_target and "target" in it:
            t = it["target"]
            if len(t) > k:
                # Otherwise this lands as "The expanded size of the tensor (k) must match the
                # existing size (kmax)" from inside the assignment, which says nothing about the
                # actual mistake: a target with more entries than the item has options. The limit
                # is this item's own marker count, not the batch-wide kmax: in a mixed-width
                # batch a longer sibling row must not legitimise extra entries (#311).
                raise ValueError(
                    "collate_items: item %d has %d target entries but only %d marker positions; "
                    "a target needs one entry per option" % (i, len(t), k))
            target[i, : len(t)] = t

    res = {
        "input_ids": torch.from_numpy(ids),
        "attention_mask": torch.from_numpy(att),
        "marker_pos": torch.from_numpy(mpos),
        "marker_mask": torch.from_numpy(mmask),
        "qtype": torch.from_numpy(np.array([it["qtype"] for it in items], dtype=np.int64)),
        "label": torch.from_numpy(np.array([it.get("label", -1) for it in items], dtype=np.int64)),
        "meta": [{k: it[k] for k in it if k not in ("ids", "markers", "target", "layout")} for it in items],
    }
    if target is not None:
        res["target"] = torch.from_numpy(target)
    if any("layout" in it for it in items):
        # All or none: a batch that mixes layouts would run half its rows through the wrong masks
        if not all("layout" in it for it in items):
            raise ValueError("collate_items: some items carry a parallel layout and some do not")
        pos = np.zeros((n, L), dtype=np.int64)
        opt = np.zeros((n, L), dtype=np.int64)
        for i, it in enumerate(items):
            l_ids = len(it["ids"])
            pos[i, :l_ids] = it["layout"]["position_ids"]
            opt[i, :l_ids] = it["layout"]["option_ids"]
        res["position_ids"], res["option_ids"] = torch.from_numpy(pos), torch.from_numpy(opt)
    return res
