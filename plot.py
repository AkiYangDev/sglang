from pathlib import Path
import csv
from statistics import median
import matplotlib.pyplot as plt
import numpy as np

def read(name):
    with open(name, newline="") as src:
        return list(csv.DictReader(src))

def heatmap(data, cols, rows, xlabel, ylabel, title, subtitle, footer, unit, fmt, name):
    fig = plt.figure(figsize=(8.1, 5.05), facecolor="white")
    ax = fig.add_axes([0.19, 0.23, 0.64, 0.55])
    cax = fig.add_axes([0.86, 0.23, 0.018, 0.55])
    fig.text(0.055, 0.92, title, fontsize=16, weight="bold", color="#102b39")
    fig.text(0.055, 0.85, subtitle, fontsize=10, color="#405763")
    img = ax.imshow(data, aspect="auto", cmap="YlGnBu", vmin=0)
    fig.colorbar(img, cax=cax)
    cax.set_title(unit, fontsize=9)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xlabel(xlabel, labelpad=7)
    ax.set_ylabel(ylabel, labelpad=7)
    for r in range(len(rows)):
        for c in range(len(cols)):
            val = data[r,c]
            ax.text(c,r,format(val,fmt),ha="center",va="center",fontsize=11,
                    color="white" if img.norm(val) > 0.55 else "#102b39")
    fig.text(0.055,0.052,footer,fontsize=9,color="#405763")
    fig.savefig(name, dpi=160)
    plt.close(fig)

g = [r for r in read("gsp.csv") if r["groups"]=="1" and r["question"]=="0" and r["fast"]=="False"]
lengths = [4096,16384,65536]
counts = [1,8,32,64]
live = {(int(r["rows"]),int(r["length"])):int(r["live_bytes"])/2**20 for r in g}
heatmap(np.array([[live[n,l] for l in lengths] for n in counts]),["4,096","16,384","65,536"],counts,
        "Tokens in the shared prefix","Requests in one group","Memory held by identical text prompts",
        "Live Python allocations at the prompt-formatting line after sampling and GC (MiB).",
        "Empty question  ·  cold dataset cache  ·  CPython 3.12.14  ·  SGLang 2800baa29705",
        "MiB",".2f","gsp.png")

o = read("offload.csv")
lengths = [4096,16384,65536,131072]
submitted = [0,4096,16384]
peak = [[median(int(r["peak_bytes"]) for r in o if r["variant"]=="upstream" and
        int(r["prompt_tokens"])==l and int(r["submitted_tokens"])==s)/1024 for l in lengths] for s in submitted]
heatmap(np.array(peak),["4,096","16,384","65,536","131,072"],["0","4,096","16,384"],
        "Tokens in the original prompt","Tokens already offloaded¹","One decode offload check with no new block",
        "Peak traced Python allocation per call (KiB). Every call returned False.",
        "¹ Decode-side incremental tokens. One more is ready, 64 are required. Median of 3 CPU calls.",
        "KiB",".1f","offload.png")
