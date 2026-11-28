"""Profile text prompts at upstream 2800baa29705. Usage: python gsp.py /path/to/sglang"""
import ast
import contextlib
import csv
import gc
import importlib.util
import io
import os
import pickle
import random
import sys
import tempfile
import tracemalloc
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

bench = Path(sys.argv[1]).resolve() / "python/sglang/benchmark"
for name in ("sglang", "sglang.benchmark", "sglang.benchmark.datasets", "sglang.benchmark.utils"):
    sys.modules[name] = types.ModuleType(name)
utils = ast.parse((bench / "utils.py").read_text(encoding="utf8"))
exec(compile(ast.Module([n for n in utils.body if getattr(n, "name", "") == "remove_suffix"], []),
             "utils.py", "exec"), sys.modules["sglang.benchmark.utils"].__dict__)
for short in ("common", "generated_shared_prefix"):
    source = bench / "datasets" / (short + ".py")
    name = "sglang.benchmark.datasets." + short
    spec = importlib.util.spec_from_file_location(name, source)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)

line, = [i for i, s in enumerate(source.read_text(encoding="utf8").splitlines(), 1)
         if 'turn_prompts = [f"{system_prompt}' in s]
vocab = {"<unk>": 0, **{str(i): i for i in range(1, 1000)}}
backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
backend.pre_tokenizer = WhitespaceSplit()
tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")

def measure(length, count, question=0, groups=1, fast=False):
    random.seed(42)
    np.random.seed(42)
    encoded_first_turns = 0
    encode = tokenizer.encode

    def counted(text, *args, **kwargs):
        nonlocal encoded_first_turns
        if "\n\n" in text:
            encoded_first_turns += 1
        return encode(text, *args, **kwargs)

    with tempfile.TemporaryDirectory() as tmp, patch.object(Path, "home", return_value=Path(tmp)):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), patch.object(tokenizer, "encode", counted):
            tracemalloc.start()
            rows = mod.sample_generated_shared_prefix_requests(groups, count // groups, length,
                question, 1, 1., tokenizer, 42, ordered=True, fast_prepare=fast)
            gc.collect()
            snap = tracemalloc.take_snapshot()
            tracemalloc.stop()
        retained = sum(stat.size for stat in snap.statistics("lineno")
                       if stat.traceback[0].filename == str(source) and stat.traceback[0].lineno == line)
        path, = Path(tmp).rglob("*.pkl")
        on_disk = path.stat().st_size
        reloaded = pickle.loads(path.read_bytes())
        assert rows == reloaded
        distinct = len({id(row.prompt) for row in rows})
        assert distinct == len({id(row.prompt) for row in reloaded})
        assert all(row.prompt_len == (1 if fast else length + question) for row in rows)
        if question == 0:
            assert all(row.prompt.endswith("\n\n") for row in rows)
            assert len({row.prompt for row in rows}) == groups
        canonical = {}
        for row in reloaded:
            row.prompt = canonical.setdefault(row.prompt, row.prompt)
        assert reloaded == rows
        return dict(length=length, rows=count, groups=groups, question=question, fast=fast,
                    copies=distinct, distinct_texts=len(canonical), encode_calls=encoded_first_turns,
                    live_bytes=retained, cache_bytes=on_disk, shared_pickle_bytes=len(pickle.dumps(reloaded)))

if __name__ == "__main__":
    with open("gsp.csv", "w", newline="") as dest:
        writer = None
        for args in ([(length, n) for n in (1, 8, 32, 64) for length in (4096, 16384, 65536)]
                     + [(16384, 32, 0, 4), (16384, 32, 1), (16384, 32, 0, 1, True)]):
            row = measure(*args)
            if writer is None:
                writer = csv.DictWriter(dest, fieldnames=row)
                writer.writeheader()
            writer.writerow(row)
            dest.flush()
            print(row, flush=True)

