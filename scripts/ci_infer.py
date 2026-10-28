#!/usr/bin/env python3
"""按 download.json 拉取测试模型，用多条提示词跑 CPU 推理，并把每次输出写进仓库根目录的 infer_log.log。"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
FILES = ("merges", "model", "model_info", "vocab")
PROMPTS = (
    ("你好，你是谁？", 24),
    ("1+1=？只回答数字。", 8),
    ("用一句话形容秋天。", 24),
    ("用 Python 写一个冒泡排序函数。", 48),
)
ISAS = ("auto", "classic", "avx", "fma")
BAD = ("装载失败", "找不到模型", "推理失败", "错误:")
SEP = "=" * 78
SUB = "-" * 78


def setup_stdout():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def download(url, dest):
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as o:
        while True:
            buf = r.read(1 << 20)
            if not buf:
                break
            o.write(buf)
    tmp.replace(dest)


def ensure_model(cfg, name, base):
    if name not in cfg:
        raise SystemExit("download.json 里没有模型 %s，可选：%s" % (name, ", ".join(cfg)))
    entry = cfg[name]
    dest = base / name
    dest.mkdir(parents=True, exist_ok=True)
    for key in FILES:
        fname = entry[key]
        f = dest / fname
        if f.is_file() and f.stat().st_size > 0:
            print("  %-14s 已存在 %d 字节" % (fname, f.stat().st_size))
            continue
        url = entry["begin"] + fname
        print("  %-14s 下载 %s" % (fname, url))
        download(url, f)
    for key in FILES:
        f = dest / entry[key]
        if not f.is_file() or f.stat().st_size == 0:
            raise SystemExit("模型文件不全：%s" % f)
    return dest


def run_once(exe, model, prompt, max_new, isa):
    env = dict(os.environ)
    env["JIEYU_ISA"] = isa
    cmd = [str(exe), "-once", str(model), prompt, str(max_new), "0", "-noinfo"]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(ROOT), env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    cost = time.time() - t0
    text = p.stdout.decode("utf-8", "replace")
    lines = [l for l in text.splitlines() if l.strip() and not l.startswith("[")]
    return p.returncode, "\n".join(lines), cost


def split_isa(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default="resut/jieyu")
    ap.add_argument("--model", default=os.environ.get("JIEYU_TEST_MODEL", ""))
    ap.add_argument("--dir", default="_file/dl")
    ap.add_argument("--isa", default=",".join(ISAS))
    ap.add_argument("--extra-isa", default="auto")
    ap.add_argument("--json", default=str(ROOT / "download.json"))
    ap.add_argument("--log", default="infer_log.log")
    a = ap.parse_args()

    exe = (ROOT / a.exe).resolve()
    if not exe.is_file():
        raise SystemExit("找不到可执行文件 %s" % exe)
    cfg = load_config(a.json)
    names = split_isa(a.model) or list(cfg)
    base = (ROOT / a.dir).resolve()
    isas = split_isa(a.isa)
    extra = split_isa(a.extra_isa)

    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    L = [SEP, "推理测试日志", SEP,
         "时间      %s UTC" % stamp,
         "可执行    %s" % a.exe,
         "模型      %s" % "、".join(names),
         "ISA       %s（其余模型：%s）" % (",".join(isas), ",".join(extra) or "无"),
         SEP, ""]

    fails = 0
    checks = 0
    for idx, name in enumerate(names):
        use = isas if idx == 0 else (extra or isas)
        print("==== %s （ISA %s）====" % (name, ",".join(use)))
        model = ensure_model(cfg, name, base)
        L += ["[%s]" % name, "模型目录  %s" % model, ""]
        for i, (prompt, max_new) in enumerate(PROMPTS, 1):
            got = {}
            cost = {}
            for isa in use:
                rc, out, sec = run_once(exe, model, prompt, max_new, isa)
                got[isa] = out.strip()
                cost[isa] = sec
                checks += 1
                ok = rc == 0 and got[isa] != "" and not any(b in got[isa] for b in BAD)
                if not ok:
                    fails += 1
                print("  %s %-8s %-18s %6.1fs %4d 字  %s" %
                      ("OK  " if ok else "FAIL", isa, prompt, sec, len(got[isa]),
                       got[isa].replace("\n", " ")[:50]))
                if not ok:
                    print("       rc=%d 输出=%r" % (rc, got[isa][:400]))
            uniq = {v for v in got.values() if v}
            same = len(uniq) <= 1
            checks += 1
            if not same:
                fails += 1
            speed = " / ".join("%s %.1fs" % (k, v) for k, v in cost.items())
            L += [SUB, "#%d  %s" % (i, prompt),
                  "上限 %d token；用时 %s；跨 ISA 一致：%s" %
                  (max_new, speed, "是" if same else "否"),
                  "", got[use[0]] or "（空输出）", ""]
            if not same:
                for isa, v in got.items():
                    L.append("  %-8s %s" % (isa, v[:120].replace("\n", " ")))
                L.append("")

    L += [SEP, "汇总  检查项 %d / 通过 %d / 失败 %d" % (checks, checks - fails, fails), SEP, ""]

    log_path = ROOT / a.log
    with open(log_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L))
    print("日志写到 %s" % log_path)
    print("==== 推理测试 %d/%d ====" % (checks - fails, checks))
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
