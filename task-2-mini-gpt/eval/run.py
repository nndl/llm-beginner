"""任务二自检：tokenizer / KV cache / 困惑度。"""
import json
import sys
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def test_tokenizer_roundtrip():
    from src.tokenizer import BPETokenizer
    tok_path = ROOT / "ckpt" / "tokenizer.json"
    if not tok_path.exists():
        return {"test": "tokenizer_roundtrip", "pass": None,
                "skip": "ckpt/tokenizer.json 不存在"}
    tok = BPETokenizer.from_pretrained(str(tok_path))
    samples = ["床前明月光", "Hello, world!", "深度学习需要数学基础"]
    fails = []
    for s in samples:
        ids = tok.encode(s)
        s2 = tok.decode(ids)
        if s != s2:
            fails.append({"in": s, "out": s2})
    return {"test": "tokenizer_roundtrip", "pass": not fails, "failures": fails}


def test_kv_cache_equivalence():
    from src.model import load_for_eval
    ckpt = ROOT / "ckpt" / "best.pt"
    if not ckpt.exists():
        return {"test": "kv_cache_equivalence", "pass": None, "skip": "ckpt/best.pt 不存在"}
    model, tok = load_for_eval(str(ckpt))
    model.eval()
    ids = torch.tensor([tok.encode("从前有座山")], dtype=torch.long)
    with torch.no_grad():
        logits_full = model(ids)
        cache = None
        logits_inc = []
        for i in range(ids.size(1)):
            out, cache = model(ids[:, i:i + 1], kv_cache=cache, return_cache=True)
            logits_inc.append(out)
        logits_inc = torch.cat(logits_inc, dim=1)
    diff = (logits_full - logits_inc).abs().max().item()
    return {"test": "kv_cache_equivalence", "pass": diff < 1e-4,
            "max_abs_diff": diff}


def test_perplexity():
    from src.model import load_for_eval
    ckpt = ROOT / "ckpt" / "best.pt"
    dev_path = ROOT / "data" / "dev.txt"
    if not ckpt.exists():
        return {"test": "perplexity_on_dev", "pass": None, "skip": "ckpt/best.pt 不存在"}
    if not dev_path.exists():
        return {"test": "perplexity_on_dev", "pass": None,
                "skip": "data/dev.txt 不存在；训练时请留出 dev split"}
    model, tok = load_for_eval(str(ckpt))
    model.eval()
    text = dev_path.read_text(encoding="utf-8")[:5000]
    ids = torch.tensor([tok.encode(text)], dtype=torch.long)
    with torch.no_grad():
        logits = model(ids)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1))
    ppl = torch.exp(loss).item()
    return {"test": "perplexity_on_dev", "pass": ppl < 200,
            "perplexity": round(ppl, 2)}


def main():
    results = []
    for fn in [test_tokenizer_roundtrip, test_kv_cache_equivalence, test_perplexity]:
        try:
            r = fn()
        except Exception as e:
            r = {"test": fn.__name__.replace("test_", ""),
                 "pass": False, "error": str(e),
                 "trace": traceback.format_exc().splitlines()[-3:]}
        results.append(r)
        tag = "[通过]" if r.get("pass") is True else \
              ("[跳过]" if r.get("pass") is None else "[失败]")
        print(f"{tag} {r['test']}: {json.dumps(r, ensure_ascii=False)}")
    out = ROOT / "eval" / "result.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果写入 {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
