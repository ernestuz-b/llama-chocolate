#!/usr/bin/env python3
"""Compare the committed synthetic MoE/LoRA fixture in Chocolate ON/OFF builds."""

import argparse
import hashlib
import os
from pathlib import Path
import re
import subprocess
import tempfile


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def compare(args, destination):
    for backend in args.backends:
        logs = {}
        outputs = {}
        for name, binary in (("on", args.on), ("off", args.off)):
            output = destination / f"{name}-{backend}.bin"
            logfile = destination / f"{name}-{backend}.log"
            dump = destination / f"{name}-{backend}.dump"
            dump.unlink(missing_ok=True)
            env = dict(os.environ, LLAMA_CHOCOLATE_DUMP=str(dump))
            with logfile.open("w") as stream:
                subprocess.run([str(binary), "--chocolate-probe", str(output), backend],
                               env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
            logs[name] = logfile.read_text()
            outputs[name] = output.read_bytes()
        a, b = logs["on"], logs["off"]
        on, off = outputs["on"], outputs["off"]
        graph = lambda text: [line for line in text.splitlines() if line.startswith(("## SPLIT", "node #", "graph_reserve:", "graph_realloc:"))]
        counters = lambda text: [line for line in text.splitlines() if line.startswith("llama_moe_cache:")]
        check(len(on) == 190464 and on == off, f"{backend}: logits differ")
        check(graph(a) and graph(a) == graph(b), f"{backend}: graph assignment differs")
        check(counters(a) == counters(b), f"{backend}: native cache counters differ")
        check(on[:47616] != on[47616:95232] and on[95232:142848] != on[142848:], f"{backend}: LoRA must change logits")
        observations = re.findall(r"chocolate metadata: requests = (\d+), selections = (\d+), tracked = (\d+), actual resident = (\d+), rejected = (\d+)", a)
        reliability = re.findall(r"chocolate reliability: samples = (\d+), complete = (\d+), aborted = (\d+), incomplete = (\d+), missing routers = (\d+), rejected staging = (\d+), desynchronized groups = (\d+), lost events = (\d+), history valid = (\d+), management failed = (\d+)", a)
        if backend == "CPU":
            check(not observations and not reliability, "CPU-only context unexpectedly has a native cache observer")
        else:
            check([int(row[0]) for row in observations] == [96, 96, 14, 14], f"{backend}: incomplete router coverage")
            for row in observations:
                check(tuple(map(int, row[1:])) == (372, 16, 10 if backend == "both" else 5, 0), f"{backend}: residency or selection coverage differs: {row}")
            check(len(reliability) == 4, f"{backend}: missing reliability report")
            for row, expected in zip(reliability, (48, 48, 7, 7)):
                check(tuple(map(int, row)) == (expected, expected, 0, 0, 0, 0, 0, 0, 1, 0), f"{backend}: invalid sample or mirror: {row}")
        if backend == "both":
            check("layer   0 assigned to device CUDA0" in a and "layer   1 assigned to device CUDA1" in a, "two-GPU fixture did not split layers")
        print(f"{backend}: 47616 identical logits; graph_lines={len(graph(a))}; native counters equal; nonzero LoRA verified; sha256={hashlib.sha256(on).hexdigest()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--on", type=Path, required=True)
    parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--backends", nargs="+", default=["CPU"])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    args.on = args.on.resolve()
    args.off = args.off.resolve()
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        compare(args, args.output_dir.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="chocolate-integration-") as directory:
            compare(args, Path(directory))


if __name__ == "__main__":
    main()
