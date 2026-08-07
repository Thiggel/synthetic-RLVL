import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "data" / "pack_document_preserving_nanoset.py"
SPEC = importlib.util.spec_from_file_location("pack_document_preserving_nanoset", SCRIPT)
assert SPEC and SPEC.loader
PACKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKER  # dataclasses require the module to be registered
SPEC.loader.exec_module(PACKER)

EOS = 7
PAD = 99
WINDOW = 17  # seq_len 16 + 1


def make_doc(doc_id: int, length: int):
    assert length >= 1
    return doc_id, [doc_id % 5 + 1] * (length - 1) + [EOS]


def test_documents_are_never_split_and_padding_is_tail_only(tmp_path):
    lengths = [5, 9, 17, 3, 12, 8, 2, 16, 6, 4, 11, 10]
    docs = [make_doc(i, n) for i, n in enumerate(lengths)]
    result = PACKER.pack_documents(docs, window_len=WINDOW, pad_id=PAD, pool_size=4)
    assert result.overlength == []
    assert result.n_docs_packed == len(docs)
    # every document appears contiguously inside exactly one window
    seen = []
    for w in result.windows:
        assert len(w.tokens) <= WINDOW
        assert w.doc_ends[-1] == len(w.tokens)
        start = 0
        for doc_id, end in zip(w.doc_ids, w.doc_ends):
            assert w.tokens[end - 1] == EOS
            assert w.tokens[start:end] == dict(docs)[doc_id]
            start = end
            seen.append(doc_id)
    assert sorted(seen) == list(range(len(docs)))

    info = PACKER.write_nanoset_folder(result, tmp_path, tokenizer_name="fake", token_size=4)
    raw = np.fromfile(tmp_path / "00000_docpack.ds", dtype=np.uint32)
    assert raw.size == info["total_tokens"] == WINDOW * len(result.windows)
    for k in range(len(result.windows)):
        window = raw[k * WINDOW : (k + 1) * WINDOW].tolist()
        assert window[0] not in (PAD, EOS)
        if PAD in window:
            first_pad = window.index(PAD)
            assert all(t == PAD for t in window[first_pad:])
            assert window[first_pad - 1] == EOS
        else:
            assert window[-1] == EOS
    ends = np.fromfile(tmp_path / "00000_docpack.ds.index", dtype=np.uint64)
    assert len(ends) == len(docs)
    assert all(raw[int(e) - 1] == EOS for e in ends)
    meta = (tmp_path / "00000_docpack.ds.metadata").read_text().splitlines()
    assert meta[0] == "fake|4"
    assert int(meta[1]) == info["total_tokens"]


def test_overlength_documents_are_excluded_and_counted():
    docs = [make_doc(0, 5), make_doc(1, WINDOW + 1), make_doc(2, 30), make_doc(3, WINDOW)]
    result = PACKER.pack_documents(docs, window_len=WINDOW, pad_id=PAD, pool_size=2)
    assert [o["doc_id"] for o in result.overlength] == [1, 2]
    assert result.n_docs_packed == 2
    packed_ids = [d for w in result.windows for d in w.doc_ids]
    assert sorted(packed_ids) == [0, 3]


def test_first_fit_backfills_open_windows():
    # 12 leaves room for 5 in the first window even after a 9 forces a second window
    docs = [make_doc(0, 12), make_doc(1, 9), make_doc(2, 5)]
    result = PACKER.pack_documents(docs, window_len=WINDOW, pad_id=PAD, pool_size=4)
    by_window = [w.doc_ids for w in result.windows]
    assert [0, 2] in by_window and [1] in by_window


def test_pad_token_in_document_rejected():
    with pytest.raises(ValueError):
        PACKER.pack_documents([(0, [1, PAD, EOS])], window_len=WINDOW, pad_id=PAD)


def test_shuffle_is_deterministic():
    docs = [make_doc(i, 3 + (i % 7)) for i in range(40)]
    a = PACKER.shuffle_windows(PACKER.pack_documents(docs, WINDOW, PAD, 4), seed=42)
    b = PACKER.shuffle_windows(PACKER.pack_documents(docs, WINDOW, PAD, 4), seed=42)
    assert [w.doc_ids for w in a.windows] == [w.doc_ids for w in b.windows]


def test_summary_loss_token_accounting():
    docs = [make_doc(0, 10), make_doc(1, 6), make_doc(2, 4)]
    result = PACKER.pack_documents(docs, window_len=WINDOW, pad_id=PAD, pool_size=1)
    summary = PACKER.summarize(result, [10, 6, 4])
    # labels per window are window_len-1; padding labels masked
    assert summary["loss_tokens"] == sum((WINDOW - 1) - p for p in result.pad_counts)
    assert summary["overlength_count"] == 0
    assert summary["real_tokens"] == 20
    assert summary["pad_tokens"] == WINDOW * len(result.windows) - 20


def test_collator_masks_exactly_padding_labels():
    clm_collator = pytest.importorskip("nanotron.data.clm_collator")
    clm_collator.dist.get_rank = lambda pg=None: 0

    class PC:
        pp_pg = None
        cp_pg = None
        context_parallel_size = 1

    seq = 8
    collator = clm_collator.DataCollatorForCLMWithPositionIds(
        sequence_length=seq, input_pp_rank=0, output_pp_rank=0, parallel_context=PC(),
        use_doc_masking=False, padding_label_id=PAD,
    )
    ids = np.array([1, 2, EOS, 3, EOS, PAD, PAD, PAD, PAD], dtype=np.int64)
    out = collator([{"input_ids": ids, "positions": np.arange(seq + 1)}])
    assert out["label_mask"][0].tolist() == [True, True, True, True, False, False, False, False]
    # without padding_label_id the mask is all ones (original Dolmino behavior)
    plain = clm_collator.DataCollatorForCLMWithPositionIds(
        sequence_length=seq, input_pp_rank=0, output_pp_rank=0, parallel_context=PC(), use_doc_masking=False,
    )
    assert plain([{"input_ids": ids, "positions": np.arange(seq + 1)}])["label_mask"][0].all()
