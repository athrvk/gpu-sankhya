"""Data plumbing for the two model-side experiments in data_wild/REPORT.md:

- `train.negatives_to_examples` / `--extra-negatives` (real all-O sentences
  mixed into each epoch at a ratio),
- `sankhya.pretrain`'s masking (15% of real characters, loss only there),
- `train.py --init-from` (trunk copied, heads fresh).

Everything here is deliberately tiny (a few hundred examples, 1-2 epochs) so
the file runs in seconds on CPU.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sankhya import classes as C
from sankhya import train as T
from sankhya import pretrain as P
from sankhya.model import SankhyaCNN, ARCHS

PY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestNegativesToExamples(unittest.TestCase):
    def test_all_o_labels(self):
        rows = [{"text": "अरब सागर के किनारे", "lang": "hi_deva"},
                {"text": "anna hazare ka andolan", "lang": "hi_latn"}]
        ex = T.negatives_to_examples(rows)
        self.assertEqual(len(ex), 2)
        for e, r in zip(ex, rows):
            self.assertEqual(e["text"], r["text"])
            self.assertEqual(e["spans"], [])
            self.assertTrue(e["bio"])
            self.assertEqual(set(e["bio"]), {0})
            self.assertEqual(set(e["cls"]), {C.CLASSES.index("O")})
            self.assertEqual(len(e["bio"]), len(e["cls"]))

    def test_labels_truncated_to_max_len(self):
        long_text = "क " * 200
        ex = T.negatives_to_examples([{"text": long_text, "lang": "hi_deva"}])[0]
        self.assertEqual(len(ex["bio"]), T.MAX_LEN)

    def test_tensorizes_as_all_o(self):
        vocab = ["<pad>", "<unk>", "a", "b", " "]
        c2i = {c: i for i, c in enumerate(vocab)}
        ex = T.negatives_to_examples([{"text": "ab ba", "lang": "hi_latn"}])
        chars, bio, cls, mask = T.tensorize(ex, c2i)
        self.assertEqual(int(mask.sum().item()), 5)
        self.assertEqual(int(bio.sum().item()), 0)
        self.assertEqual(int(cls.sum().item()), 0)


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _synthetic_examples(n, lang="hi_latn"):
    """Minimal train/val rows in the generator's schema."""
    out = []
    for i in range(n):
        text = f"das hazaar {i}"
        out.append({"text": text, "lang": lang,
                     "bio": [0] * len(text), "cls": [0] * len(text), "spans": []})
    return out


class TestExtraNegativeMixing(unittest.TestCase):
    """The ratio arithmetic is what a training run actually depends on, so it
    is asserted end to end through `train.main` on a tiny model."""

    def test_ratio_reported_and_epoch_size(self):
        with tempfile.TemporaryDirectory() as d:
            tr = os.path.join(d, "train.jsonl")
            va = os.path.join(d, "val.jsonl")
            neg = os.path.join(d, "neg.jsonl")
            _write_jsonl(tr, _synthetic_examples(200))
            _write_jsonl(va, _synthetic_examples(20))
            _write_jsonl(neg, [{"text": f"koi amount nahi hai {i}", "lang": "hi_latn"}
                                for i in range(50)])
            out = os.path.join(d, "models")
            import io
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                T.main(["--train", tr, "--val", va, "--out", out,
                        "--lang", "hi_latn", "--epochs", "1", "--batch", "64",
                        "--channels", "8", "--embed-dim", "4",
                        "--extra-negatives", neg, "--extra-negative-ratio", "0.15"])
            log = buf.getvalue()
            self.assertIn("loaded 50 real-negative lines", log)
            # 200 main examples, ratio 0.15 -> 35 negatives (35/235 = 0.149)
            self.assertIn("mixing in 35 real negatives/epoch", log)
            self.assertTrue(os.path.exists(os.path.join(out, "sankhya.pt")))

    def test_ratio_zero_disables(self):
        with tempfile.TemporaryDirectory() as d:
            tr = os.path.join(d, "train.jsonl")
            va = os.path.join(d, "val.jsonl")
            neg = os.path.join(d, "neg.jsonl")
            _write_jsonl(tr, _synthetic_examples(100))
            _write_jsonl(va, _synthetic_examples(10))
            _write_jsonl(neg, [{"text": "kuch bhi nahi", "lang": "hi_latn"}])
            import io
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                T.main(["--train", tr, "--val", va, "--out", os.path.join(d, "m"),
                        "--lang", "hi_latn", "--epochs", "1", "--batch", "64",
                        "--channels", "8", "--embed-dim", "4",
                        "--extra-negatives", neg, "--extra-negative-ratio", "0"])
            self.assertNotIn("mixing in", buf.getvalue())


class TestPretrainMasking(unittest.TestCase):
    def test_mask_rate_and_loss_positions(self):
        torch.manual_seed(0)
        B, L, V = 64, 100, 30
        chars = torch.randint(2, V, (B, L))
        mask = torch.ones(B, L)
        gen = torch.Generator()
        gen.manual_seed(7)
        inp, tgt = P.apply_mask(chars, mask, mask_id=1, rate=0.15, rng=gen)
        sel = tgt != P.IGNORE_INDEX
        rate = sel.float().mean().item()
        self.assertAlmostEqual(rate, 0.15, delta=0.02)
        # masked positions carry the mask id in the input and the ORIGINAL id
        # as the target; every other position is untouched and ignored.
        self.assertTrue((inp[sel] == 1).all())
        self.assertTrue((tgt[sel] == chars[sel]).all())
        self.assertTrue((inp[~sel] == chars[~sel]).all())
        self.assertTrue((tgt[~sel] == P.IGNORE_INDEX).all())

    def test_padding_never_masked(self):
        chars = torch.randint(2, 20, (16, 50))
        mask = torch.zeros(16, 50)
        mask[:, :10] = 1.0
        gen = torch.Generator()
        gen.manual_seed(1)
        inp, tgt = P.apply_mask(chars, mask, mask_id=1, rate=0.5, rng=gen)
        self.assertTrue((tgt[:, 10:] == P.IGNORE_INDEX).all())
        self.assertTrue((inp[:, 10:] == chars[:, 10:]).all())

    def test_loss_ignores_unmasked_positions(self):
        """Changing the logits at an UNMASKED position must not move the loss."""
        torch.manual_seed(0)
        V = 12
        ce = torch.nn.CrossEntropyLoss(ignore_index=P.IGNORE_INDEX)
        logits = torch.randn(1, 8, V)
        tgt = torch.full((1, 8), P.IGNORE_INDEX)
        tgt[0, 2] = 5
        base = ce(logits.reshape(-1, V), tgt.reshape(-1)).item()
        bumped = logits.clone()
        bumped[0, 6] += 100.0  # an unmasked position
        self.assertAlmostEqual(ce(bumped.reshape(-1, V), tgt.reshape(-1)).item(), base, places=6)
        bumped2 = logits.clone()
        bumped2[0, 2, 5] += 100.0  # the masked position
        self.assertLess(ce(bumped2.reshape(-1, V), tgt.reshape(-1)).item(), base)

    def test_trunk_state_dict_keys_match_tagger(self):
        m = P.PretrainModel(vocab_size=40, arch=ARCHS["v2"], channels=8, embed_dim=4)
        tagger = SankhyaCNN(vocab_size=40, n_cls=len(C.CLASSES), arch=ARCHS["v2"],
                            channels=8, embed_dim=4)
        trunk = m.trunk_state_dict()
        tsd = tagger.state_dict()
        for k, v in trunk.items():
            self.assertIn(k, tsd)
            self.assertEqual(tuple(v.shape), tuple(tsd[k].shape))
        # every trunk tensor of the tagger is covered
        for k in tsd:
            if k.startswith("embed.") or k.startswith("conv"):
                self.assertIn(k, trunk)
        # and the heads are NOT part of it
        self.assertFalse(any(k.startswith(("bio_head", "cls_head")) for k in trunk))


class TestInitFrom(unittest.TestCase):
    def test_loads_trunk_leaves_heads_fresh(self):
        with tempfile.TemporaryDirectory() as d:
            tr = os.path.join(d, "train.jsonl")
            va = os.path.join(d, "val.jsonl")
            _write_jsonl(tr, _synthetic_examples(64))
            _write_jsonl(va, _synthetic_examples(8))

            # build a pretrain-shaped checkpoint on the real charset
            from sankhya.charset import build_charset
            from sankhya.langs import base as langbase
            vocab = build_charset(langbase.get_pack("hi_latn"))
            pm = P.PretrainModel(vocab_size=len(vocab), arch=None, channels=8, embed_dim=4)
            for p in pm.trunk.parameters():
                torch.nn.init.constant_(p, 0.25)
            ck = os.path.join(d, "pretrain.pt")
            torch.save({"state_dict": pm.trunk_state_dict(), "vocab": vocab}, ck)

            out = os.path.join(d, "m")
            import io
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                T.main(["--train", tr, "--val", va, "--out", out,
                        "--lang", "hi_latn", "--epochs", "1", "--batch", "64",
                        "--channels", "8", "--embed-dim", "4", "--lr", "1e-9",
                        "--init-from", ck])
            self.assertIn("initialised trunk", buf.getvalue())
            self.assertIn("heads are fresh", buf.getvalue())

            got = torch.load(os.path.join(out, "sankhya.pt"), map_location="cpu")["state_dict"]
            # with lr ~ 0 the trunk should still be the constant we saved
            self.assertTrue(torch.allclose(got["conv1.weight"],
                                            torch.full_like(got["conv1.weight"], 0.25),
                                            atol=1e-4))
            # heads were never in the pretrain checkpoint, so they are the
            # fresh torch init -- not the constant.
            self.assertFalse(torch.allclose(got["bio_head.weight"],
                                             torch.full_like(got["bio_head.weight"], 0.25),
                                             atol=1e-3))

    def test_shape_mismatch_is_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            tr = os.path.join(d, "train.jsonl")
            va = os.path.join(d, "val.jsonl")
            _write_jsonl(tr, _synthetic_examples(16))
            _write_jsonl(va, _synthetic_examples(4))
            from sankhya.charset import build_charset
            from sankhya.langs import base as langbase
            vocab = build_charset(langbase.get_pack("hi_latn"))
            pm = P.PretrainModel(vocab_size=len(vocab), arch=None, channels=16, embed_dim=4)
            ck = os.path.join(d, "pretrain.pt")
            torch.save({"state_dict": pm.trunk_state_dict(), "vocab": vocab}, ck)
            with self.assertRaises(SystemExit):
                T.main(["--train", tr, "--val", va, "--out", os.path.join(d, "m"),
                        "--lang", "hi_latn", "--epochs", "1", "--batch", "16",
                        "--channels", "8", "--embed-dim", "4", "--init-from", ck])


class TestWildNegativesTool(unittest.TestCase):
    def test_gold_wild_keys_nonempty(self):
        from sankhya import wild
        keys = wild.gold_wild_keys()
        self.assertGreaterEqual(len(keys), 350)

    def test_committed_negatives_are_disjoint_from_gold(self):
        """If the (gitignored) pool has been built locally, it must not share
        a line with the evaluation set."""
        from sankhya import wild
        path = wild.DEFAULT_NEGATIVES
        if not os.path.exists(path):
            self.skipTest("data_wild/negatives.jsonl not built here")
        gold = wild.gold_wild_keys()
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.assertNotIn(wild._dedup_key(json.loads(line)["text"]), gold)


if __name__ == "__main__":
    unittest.main()
