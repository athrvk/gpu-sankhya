# python/sankhya

Language-independent semantic classes (`sankhya/classes.py`), a deterministic
evaluation core (`sankhya/core.py`), a language-pack system
(`sankhya/langs/`), and a synthetic labelled-data generator
(`sankhya/generator.py`). See `docs/DATA_GRAMMAR.md` for the spec.

## Usage

```bash
cd python
python -m sankhya.generator --n 80000 --seed 1 --out ../data/train.jsonl --lang hi_latn
python -m sankhya.generator --n 5000 --seed 2 --out ../data/val.jsonl --lang hi_latn
python -m sankhya.charset --lang hi_latn --out ../data/charset.json
```

## Tests

```bash
python -m pytest tests/
# or, dependency-free:
python tests/test_core.py
python tests/test_generator.py
```

## Adding a new language pack

Subclass/construct `sankhya.langs.base.LanguagePack` with a lexicon, symbol
units, templates and a `noise_fn`, then `base.register(pack)`. Nothing in
`core.py` or `generator.py` should ever need to change — they operate purely
on class tokens.
