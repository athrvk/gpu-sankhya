# Running gpu-sankhya training on Colab

From a fresh Colab notebook (CPU runtime is fine), in `python/` of this repo:

```bash
!pip install -q torch numpy onnx onnxruntime
```

```bash
!python -m sankhya.generator --n 80000 --seed 1 --out data/train.jsonl --lang hi_latn
!python -m sankhya.generator --n 5000 --seed 2 --out data/val.jsonl --lang hi_latn
```

```bash
!python -m sankhya.train --train data/train.jsonl --val data/val.jsonl --epochs 8 --batch 128 --lr 3e-3 --out models/ --dilation 2
```

```bash
!python -m sankhya.export --ckpt models/sankhya.pt --val data/val.jsonl --out models/
```

```bash
!python -m sankhya.eval_gold --gold tests/gold.jsonl --ckpt models/sankhya.pt
```

Outputs land in `models/`: `sankhya.pt` (torch checkpoint), `sankhya.onnx`,
`sankhya.weights.json` (float32, human-readable), `sankhya.weights.int8.json`
(quantized), plus `charset.json` / `classes.json`.
