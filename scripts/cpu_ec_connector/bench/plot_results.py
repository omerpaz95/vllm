#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Charts and an offline report from `run_bench.py`'s `bench.json`.

Reads one or more `bench.json` files -- several files are replications of the
same sweep, aggregated as the mean over reps with min/max error bars -- and
writes a PNG and an SVG per chart plus a self-contained `report.html` that
embeds the SVGs inline. The report opens with no network access.

The charts answer, in order: how much throughput each arm delivers, how that
compares with the baseline, what it costs in time-to-first-token, where the
EPD time goes per stage, what the connector actually moved (the mechanism,
not the timing), how fast it moved it, and whether the load points measured
work rather than queue depth.

Typical use:

    python plot_results.py --out-dir report results/*/bench.json

    # No hardware needed: synthesise a plausible sweep and render it.
    python plot_results.py --demo --out-dir /tmp/report_demo
"""

from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import partial
from html import escape
from pathlib import Path
from string import Template
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

matplotlib.use("Agg")

# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------

SURFACE = "#fcfcfb"
PLANE = "#f9f9f7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"

# Baseline neutral grey; the CPU connector in blues and the Example connector
# in oranges, each family shaded so a "grid" arm is darker than its "data"
# counterpart. Validated with the dataviz palette checker (light surface,
# adjacent pairs): lightness band, CVD separation and normal-vision floor all
# pass. The baseline is deliberately below the chroma floor -- it is the
# reference series and reads as grey on purpose.
GREY = "#8b8985"
CPU_LIGHT, CPU_MID, CPU_DARK = "#7cb1f0", "#3282e4", "#1c5096"
EX_MID, EX_DARK = "#ef8f3c", "#a5490f"
SPARE = ("#4a3aa7", "#1baf7a", "#e87ba4", "#008300")

ARM_ORDER = (
    "baseline",
    "offload",
    "cpu-data",
    "cpu-grid",
    "example-data",
    "example-grid",
)
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")


@dataclass(frozen=True)
class ArmStyle:
    """How one arm is drawn everywhere it appears."""

    color: str
    marker: str
    dashes: tuple[float, ...] | None


def _family(arm: str) -> tuple[str, int]:
    """(hue family, shade rank) inferred from the arm name.

    Unknown arms -- a future harness may add `multinode-cpu-grid` -- are keyed
    off the connector and rewrite words in their name so they land in the same
    hue family as the arm they extend.
    """
    name = arm.lower()
    if "baseline" in name:
        return "base", 0
    shade = 2 if "grid" in name else 1 if "data" in name else 0
    if "example" in name:
        return "example", shade
    if "cpu" in name or "offload" in name or "nixl" in name:
        return "cpu", shade
    return "other", shade


def arm_styles(arms: Sequence[str]) -> dict[str, ArmStyle]:
    """A stable colour, marker and dash pattern per arm."""
    ramps = {
        "base": {0: GREY, 1: GREY, 2: GREY},
        "cpu": {0: CPU_LIGHT, 1: CPU_MID, 2: CPU_DARK},
        "example": {0: EX_MID, 1: EX_MID, 2: EX_DARK},
        "other": dict(enumerate(SPARE)),
    }
    used: set[str] = set()
    spare = iter(SPARE)
    styles: dict[str, ArmStyle] = {}
    for index, arm in enumerate(arms):
        family, shade = _family(arm)
        ramp = ramps[family]
        color = ramp.get(shade, next(iter(ramp.values())))
        while color in used:
            color = next(spare, GREY)
        used.add(color)
        # Redundant encoding: the family also differs in marker, and a
        # single-instance arm is dashed where an EPD arm is solid.
        dashes = (5.0, 2.5) if family == "cpu" and shade == 0 else None
        styles[arm] = ArmStyle(color, MARKERS[index % len(MARKERS)], dashes)
    return styles


def sort_arms(arms: Iterable[str]) -> list[str]:
    """Canonical arms first, in the order `run_bench.py` declares them."""
    known = [a for a in ARM_ORDER if a in set(arms)]
    return known + sorted(set(arms) - set(known))


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

Point = tuple[str, int]  # (request_rate, concurrency)
Getter = Callable[[dict[str, Any]], float | None]


@dataclass(frozen=True)
class Stat:
    """One aggregated number: the mean over reps, and their spread."""

    mean: float
    lo: float
    hi: float
    n: int

    @property
    def err(self) -> tuple[float, float]:
        return (self.mean - self.lo, self.hi - self.mean)


def _stat(values: Sequence[float]) -> Stat | None:
    if not values:
        return None
    return Stat(float(np.mean(values)), min(values), max(values), len(values))


@dataclass
class Bench:
    """Every rep's entries, indexed by (arm, request_rate, concurrency)."""

    sources: list[Path]
    docs: list[dict[str, Any]]
    entries: dict[tuple[str, Point], list[dict[str, Any]]]

    @property
    def reps(self) -> int:
        return len(self.docs)

    @property
    def arms(self) -> list[str]:
        return sort_arms({arm for arm, _ in self.entries})

    @property
    def points(self) -> list[Point]:
        return sorted({p for _, p in self.entries}, key=_point_key)

    def points_for(self, arm: str) -> list[Point]:
        return sorted(
            {p for a, p in self.entries if a == arm and self.entries[a, p]},
            key=_point_key,
        )

    def rows(self, arm: str, point: Point) -> list[dict[str, Any]]:
        return self.entries.get((arm, point), [])

    def stat(self, arm: str, point: Point, get: Getter) -> Stat | None:
        values = [v for r in self.rows(arm, point) if (v := get(r)) is not None]
        return _stat(values)

    def series(
        self, arm: str, get: Getter, points: Sequence[Point] | None = None
    ) -> list[tuple[Point, Stat]]:
        out = []
        for point in points if points is not None else self.points_for(arm):
            stat = self.stat(arm, point, get)
            if stat is not None:
                out.append((point, stat))
        return out


def _point_key(point: Point) -> tuple[float, float]:
    rate, conc = point
    try:
        rate_value = float(rate)
    except (TypeError, ValueError):
        rate_value = math.inf
    # 0 means unbounded concurrency, which is the heaviest point, not the
    # lightest.
    return (math.inf if conc == 0 else conc, rate_value)


def load_bench(paths: Sequence[Path]) -> Bench:
    """Read every file and index its entries; files may only partly overlap."""
    docs: list[dict[str, Any]] = []
    entries: dict[tuple[str, Point], list[dict[str, Any]]] = {}
    for path in paths:
        doc = json.loads(path.read_text())
        results = doc.get("results")
        if not isinstance(results, list):
            raise SystemExit(f"[plot] {path}: no 'results' list; not a bench.json")
        docs.append(doc)
        for entry in results:
            point = (str(entry.get("request_rate")), int(entry.get("concurrency", 0)))
            entries.setdefault((entry["arm"], point), []).append(entry)
    if not entries:
        raise SystemExit("[plot] the inputs hold no results")
    return Bench(list(paths), docs, entries)


def baseline_arm(bench: Bench) -> str | None:
    """The arm the ratios are taken against.

    `baseline` by name when it is present; otherwise the arm that loaded
    nothing anywhere, which is what "no connector" looks like in the data.
    """
    if any(arm == "baseline" for arm, _ in bench.entries):
        return "baseline"
    for arm in bench.arms:
        rows = [r for (a, _), rs in bench.entries.items() if a == arm for r in rs]
        if rows and not any(r.get("server", {}).get("ec_load_entries") for r in rows):
            return arm
    return None


# --------------------------------------------------------------------------
# Metric accessors
# --------------------------------------------------------------------------


def client(key: str) -> Getter:
    def get(entry: dict[str, Any]) -> float | None:
        value = entry.get("client", {}).get(key)
        return float(value) if isinstance(value, (int, float)) else None

    return get


def server(key: str) -> Getter:
    def get(entry: dict[str, Any]) -> float | None:
        value = entry.get("server", {}).get(key)
        return float(value) if isinstance(value, (int, float)) else None

    return get


def stage(key: str) -> Getter:
    def get(entry: dict[str, Any]) -> float | None:
        value = entry.get("stages", {}).get(f"{key}_ms_median")
        return float(value) if isinstance(value, (int, float)) else None

    return get


def entry_field(key: str) -> Getter:
    def get(entry: dict[str, Any]) -> float | None:
        value = entry.get(key)
        return float(value) if isinstance(value, (int, float)) else None

    return get


def producer_saves(entry: dict[str, Any]) -> float | None:
    """Saves wherever they happened: the consumer's, or the encoder's on EPD."""
    consumer = entry.get("server", {}).get("ec_save_entries") or 0
    encoder = entry.get("encoder", {}).get("ec_save_entries") or 0
    return float(consumer or encoder)


def producer_save_bytes(entry: dict[str, Any]) -> float | None:
    consumer = entry.get("server", {}).get("ec_save_bytes") or 0
    encoder = entry.get("encoder", {}).get("ec_save_bytes") or 0
    return float(consumer or encoder)


def producer_gbps(entry: dict[str, Any]) -> float | None:
    """Save bandwidth wherever the saving happened."""
    consumer = entry.get("server", {}).get("ec_save_gbps") or 0
    encoder = entry.get("encoder", {}).get("ec_save_gbps") or 0
    return float(consumer or encoder)


def queue_instances(bench: Bench) -> list[str]:
    """Instance names that reported a waiting queue, decode-side first."""
    names: set[str] = set()
    for rows in bench.entries.values():
        for row in rows:
            for key in row.get("queue", {}):
                if key.endswith("_waiting_max"):
                    names.add(key[: -len("_waiting_max")])
    order = {"single": 0, "decode": 1}
    return sorted(names, key=lambda n: (order.get(n, 2), n))


def queue_metric(instance: str, suffix: str) -> Getter:
    def get(entry: dict[str, Any]) -> float | None:
        value = entry.get("queue", {}).get(f"{instance}_{suffix}")
        return float(value) if isinstance(value, (int, float)) else None

    return get


def max_waiting(entry: dict[str, Any]) -> float | None:
    depths = [
        v
        for k, v in entry.get("queue", {}).items()
        if k.endswith("_waiting_max") and isinstance(v, (int, float))
    ]
    return float(max(depths)) if depths else None


def ratio_stat(bench: Bench, arm: str, base: str, point: Point, get: Getter) -> Stat:
    """Per-rep ratio against the baseline, so rep noise cancels where it can."""
    arm_rows = bench.rows(arm, point)
    base_rows = bench.rows(base, point)
    base_values = [v for r in base_rows if (v := get(r)) is not None and v]
    if not base_values:
        return Stat(math.nan, math.nan, math.nan, 0)
    paired = len(arm_rows) == len(base_values)
    ratios = []
    for index, row in enumerate(arm_rows):
        value = get(row)
        if value is None:
            continue
        ref = base_values[index] if paired else float(np.mean(base_values))
        ratios.append(value / ref)
    return _stat(ratios) or Stat(math.nan, math.nan, math.nan, 0)


# --------------------------------------------------------------------------
# Load-point axis
# --------------------------------------------------------------------------


@dataclass
class LoadAxis:
    """How load points map onto the x axis of every load-sweep chart."""

    label: str
    positions: dict[Point, float]
    ticks: list[float]
    ticklabels: list[str]
    log: bool

    def x(self, point: Point) -> float:
        return self.positions[point]

    def apply(self, ax: plt.Axes) -> None:
        if self.log:
            ax.set_xscale("log", base=2)
        ax.set_xticks(self.ticks)
        ax.set_xticklabels(self.ticklabels)
        ax.minorticks_off()
        ax.set_xlabel(self.label)


def load_axis(points: Sequence[Point]) -> LoadAxis:
    concs = {c for _, c in points}
    rates = {r for r, _ in points}
    if len(concs) > 1 and 0 not in concs:
        values = sorted(concs)
        span = max(values) / min(values)
        return LoadAxis(
            "Offered concurrency (max in-flight requests)",
            {p: float(p[1]) for p in points},
            [float(v) for v in values],
            [str(v) for v in values],
            log=span > 8,
        )
    if len(concs) == 1 and len(rates) > 1 and all(_is_number(r) for r in rates):
        values = sorted(float(r) for r in rates)
        span = max(values) / min(values) if min(values) else 1.0
        return LoadAxis(
            "Request rate (requests/s)",
            {p: float(p[0]) for p in points},
            values,
            [_trim(v) for v in values],
            log=span > 8,
        )
    ordered = sorted(set(points), key=_point_key)
    return LoadAxis(
        "Load point (rate / concurrency)",
        {p: float(i) for i, p in enumerate(ordered)},
        [float(i) for i in range(len(ordered))],
        [_point_label(p) for p in ordered],
        log=False,
    )


def _is_number(text: str) -> bool:
    try:
        float(text)
    except (TypeError, ValueError):
        return False
    return True


def _trim(value: float) -> str:
    return f"{value:g}"


def _point_label(point: Point) -> str:
    rate, conc = point
    return f"{rate} / {'unbounded' if conc == 0 else f'c={conc}'}"


# --------------------------------------------------------------------------
# Chart chrome
# --------------------------------------------------------------------------

RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
    "font.size": 10.5,
    "axes.titlesize": 13,
    "axes.labelsize": 10.5,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.9,
    "grid.linestyle": "-",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.frameon": False,
    "legend.fontsize": 9.5,
    "legend.labelcolor": INK_2,
    "lines.linewidth": 2.0,
    "lines.markersize": 6.0,
    "lines.markeredgewidth": 1.5,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "svg.fonttype": "path",
}


def style_axes(ax: plt.Axes, *, y_only_grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0)
    ax.grid(axis="y", visible=True, color=GRID, linewidth=0.9)
    if y_only_grid:
        ax.grid(axis="x", visible=False)
    else:
        ax.grid(axis="x", visible=True, color=GRID, linewidth=0.9)


def titles(fig: plt.Figure, title: str, subtitle: str) -> None:
    """Title and subtitle at a fixed distance from the top, in inches."""
    height = fig.get_figheight()
    fig.suptitle(
        title,
        x=0.012,
        y=1 - 0.16 / height,
        ha="left",
        va="top",
        color=INK,
        fontweight="bold",
    )
    fig.text(
        0.012,
        1 - 0.44 / height,
        subtitle,
        ha="left",
        va="top",
        color=INK_2,
        fontsize=10,
    )


def source_note(fig: plt.Figure, note: str) -> None:
    """A wrapped provenance line under the plot, inside the figure width."""
    wrapped = "\n".join(textwrap.wrap(note, width=int(fig.get_figwidth() * 15)))
    fig.text(
        0.012,
        0.14 / fig.get_figheight(),
        wrapped,
        ha="left",
        va="bottom",
        color=MUTED,
        fontsize=8.5,
        linespacing=1.4,
    )


def layout(fig: plt.Figure, *, top_in: float, bottom_in: float, **kwargs) -> None:
    height = fig.get_figheight()
    fig.subplots_adjust(top=1 - top_in / height, bottom=bottom_in / height, **kwargs)


def legend(ax: plt.Axes, *, ncol: int = 3, loc: str = "upper left") -> None:
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) >= 2:
        ax.legend(handles, labels, ncol=ncol, loc=loc, handlelength=1.8)


def figure_legend(fig: plt.Figure, ax: plt.Axes, *, ncol: int) -> None:
    """One legend for a multi-panel figure, on its own row under the subtitle."""
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) < 2:
        return
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.012, 1 - 0.62 / fig.get_figheight()),
        ncol=ncol,
        handlelength=1.8,
        columnspacing=1.6,
        borderaxespad=0.0,
    )


def log_span(values: Sequence[float], factor: float = 10.0) -> bool:
    """True when the positive values span enough to warrant a log axis."""
    positive = [v for v in values if v > 0]
    return bool(positive) and max(positive) / min(positive) > factor


def log_y(ax: plt.Axes) -> None:
    """A log y axis with 1/2/5 ticks, so it stays as readable as a linear one."""
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.6g}"))


def direct_labels(
    ax: plt.Axes, items: Sequence[tuple[float, float, str]], *, dx: float = 6.0
) -> None:
    """Label the named points, pushed apart so they never overlap."""
    if not items:
        return
    ordered = sorted(items, key=lambda it: it[1])
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.055
    placed: list[float] = []
    for _, y, _ in ordered:
        want = y if not placed else max(y, placed[-1] + gap)
        placed.append(want)
    for (x, _, text), y in zip(ordered, placed):
        ax.annotate(
            text,
            (x, y),
            textcoords="offset points",
            xytext=(dx, 0),
            va="center",
            ha="left",
            fontsize=9,
            color=INK_2,
        )


def bar_group(
    ax: plt.Axes,
    centers: np.ndarray,
    values: Sequence[float],
    errs: Sequence[tuple[float, float]] | None,
    width: float,
    color: str,
    label: str | None = None,
    *,
    hatch: str | None = None,
) -> None:
    """Bars with a 2px surface gap and min/max whiskers."""
    ax.bar(
        centers,
        values,
        width=width,
        color=color,
        label=label,
        edgecolor=SURFACE,
        linewidth=1.5,
        hatch=hatch,
        zorder=2,
    )
    if errs and any(e[0] or e[1] for e in errs):
        ax.errorbar(
            centers,
            values,
            yerr=np.array(errs).T,
            fmt="none",
            ecolor=INK_2,
            elinewidth=1.0,
            capsize=3,
            capthick=1.0,
            zorder=3,
        )


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------


@dataclass
class Chart:
    """A rendered chart: files on disk plus the SVG the report inlines."""

    name: str
    title: str
    caption: str
    svg: str
    png: Path


def _save(fig: plt.Figure, out_dir: Path, name: str) -> tuple[str, Path]:
    png = out_dir / f"{name}.png"
    svg = out_dir / f"{name}.svg"
    fig.savefig(png, dpi=200)
    fig.savefig(svg, format="svg")
    plt.close(fig)
    return _inline_svg(svg.read_text()), png


def _inline_svg(text: str) -> str:
    body = text[text.index("<svg") :]
    body = re.sub(r'\swidth="[\d.]+pt"\s+height="[\d.]+pt"', " ", body, count=1)
    return body.replace("<svg ", '<svg class="chart" ', 1)


def rep_note(bench: Bench, extra: str = "") -> str:
    files = ", ".join(p.name for p in bench.sources[:3])
    if len(bench.sources) > 3:
        files += f", +{len(bench.sources) - 3} more"
    spread = "mean of reps, bars/bands span min-max" if bench.reps > 1 else "single rep"
    note = f"Source: run_bench.py bench.json (n={bench.reps} reps: {files}) - {spread}"
    return f"{note}. {extra}" if extra else note


def chart_throughput(bench: Bench, axis: LoadAxis, styles, out_dir: Path) -> Chart:
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    layout(fig, top_in=0.92, bottom_in=1.0, left=0.085, right=0.965)
    ends = []
    for arm in bench.arms:
        pairs = bench.series(arm, client("output_throughput"))
        if not pairs:
            continue
        style = styles[arm]
        xs = [axis.x(p) for p, _ in pairs]
        ys = [s.mean for _, s in pairs]
        ax.plot(
            xs,
            ys,
            color=style.color,
            marker=style.marker,
            label=arm,
            dashes=style.dashes or (),
            markeredgecolor=SURFACE,
            zorder=3,
        )
        ax.fill_between(
            xs,
            [s.lo for _, s in pairs],
            [s.hi for _, s in pairs],
            color=style.color,
            alpha=0.14,
            linewidth=0,
            zorder=1,
        )
        ends.append((xs[-1], ys[-1], arm))
    style_axes(ax)
    axis.apply(ax)
    ax.set_ylabel("Output throughput (generated tokens/s)")
    ax.set_ylim(bottom=0)
    legend(ax, ncol=2)
    titles(
        fig,
        "Output throughput scales with concurrency on every arm",
        "Generated tokens per second at each offered load point; higher is better.",
    )
    source_note(fig, rep_note(bench))
    svg, png = _save(fig, out_dir, "01_throughput")
    return Chart(
        "01_throughput",
        "Output throughput vs offered load",
        "Client-observed generation rate. The band is the min-max over reps; "
        "where a line stops, that arm has no data for the remaining points.",
        svg,
        png,
    )


def chart_speedup(
    bench: Bench, axis: LoadAxis, styles, base: str, out_dir: Path
) -> Chart:
    fig, ax = plt.subplots(figsize=(9.9, 5.6))
    layout(fig, top_in=0.92, bottom_in=1.0, left=0.082, right=0.865)
    firsts = []
    for arm in bench.arms:
        if arm == base:
            continue
        points = [p for p in bench.points_for(arm) if bench.rows(base, p)]
        stats = [
            (p, ratio_stat(bench, arm, base, p, client("output_throughput")))
            for p in points
        ]
        stats = [(p, s) for p, s in stats if s.n and not math.isnan(s.mean)]
        if not stats:
            continue
        style = styles[arm]
        xs = [axis.x(p) for p, _ in stats]
        ys = [s.mean for _, s in stats]
        ax.errorbar(
            xs,
            ys,
            yerr=np.array([s.err for _, s in stats]).T,
            color=style.color,
            marker=style.marker,
            label=arm,
            dashes=style.dashes or (),
            markeredgecolor=SURFACE,
            elinewidth=1.0,
            capsize=3,
            zorder=3,
        )
        firsts.append((xs[-1], ys[-1], arm))
    ax.axhline(1.0, color=AXIS, linewidth=1.4, zorder=2)
    ax.annotate(
        f"1.0 = {base}",
        (0.004, 1.0),
        xycoords=("axes fraction", "data"),
        textcoords="offset points",
        xytext=(0, 6),
        fontsize=9,
        color=MUTED,
    )
    style_axes(ax)
    axis.apply(ax)
    ax.set_ylabel(f"Output throughput / {base} (x)")
    ax.margins(y=0.12)
    direct_labels(ax, firsts)
    legend(ax, ncol=2, loc="upper right")
    titles(
        fig,
        "Speedup over baseline is largest where the encoder is on the critical path",
        "Ratio of output throughput to the baseline arm at the same load point.",
    )
    source_note(fig, rep_note(bench, "Ratios formed per rep, then averaged."))
    svg, png = _save(fig, out_dir, "02_speedup")
    return Chart(
        "02_speedup",
        "Speedup vs baseline",
        "Above 1.0 the arm delivers more tokens per second than the baseline at "
        "the same offered load. Whiskers span the min-max over reps.",
        svg,
        png,
    )


def chart_ttft(bench: Bench, axis: LoadAxis, styles, out_dir: Path) -> Chart:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.4), sharex=True)
    layout(fig, top_in=1.28, bottom_in=0.9, left=0.075, right=0.985, wspace=0.2)
    panels = (("median_ttft_ms", "Median TTFT"), ("p99_ttft_ms", "p99 TTFT"))
    for ax, (key, name) in zip(axes, panels):
        seen: list[float] = []
        for arm in bench.arms:
            pairs = bench.series(arm, client(key))
            if not pairs:
                continue
            style = styles[arm]
            xs = [axis.x(p) for p, _ in pairs]
            seen += [s.mean for _, s in pairs]
            ax.plot(
                xs,
                [s.mean for _, s in pairs],
                color=style.color,
                marker=style.marker,
                label=arm,
                dashes=style.dashes or (),
                markeredgecolor=SURFACE,
                zorder=3,
            )
            ax.fill_between(
                xs,
                [s.lo for _, s in pairs],
                [s.hi for _, s in pairs],
                color=style.color,
                alpha=0.14,
                linewidth=0,
                zorder=1,
            )
        style_axes(ax)
        axis.apply(ax)
        ax.set_title(name, color=INK, fontsize=11.5, loc="left", pad=8)
        if log_span(seen):
            log_y(ax)
        else:
            ax.set_ylim(bottom=0)
    logged = axes[0].get_yscale() == "log"
    axes[0].set_ylabel(
        "Time to first token (ms, log scale)" if logged else "Time to first token (ms)"
    )
    figure_legend(fig, axes[0], ncol=6)
    titles(
        fig,
        "Time to first token: median and tail",
        "Lower is better; the tail panel is where a queued encoder shows up first."
        + (" Log y scale." if logged else ""),
    )
    source_note(fig, rep_note(bench))
    svg, png = _save(fig, out_dir, "03_ttft")
    return Chart(
        "03_ttft",
        "Time to first token (median and p99)",
        "TTFT carries the encoder cost, so it is the latency the connector is "
        "expected to move. Bands span the min-max over reps.",
        svg,
        png,
    )


def _stage_keys(bench: Bench) -> list[str]:
    preferred = ["encode", "rewrite", "decode_ttfb"]
    present = {
        key[: -len("_ms_median")]
        for rows in bench.entries.values()
        for row in rows
        for key in row.get("stages", {})
        if key.endswith("_ms_median")
    }
    keys = [k for k in preferred if k in present]
    return keys or sorted(present - {"decode_total"})


def epd_arms(bench: Bench) -> list[str]:
    return [
        arm
        for arm in bench.arms
        if any(
            r.get("stages") for p in bench.points_for(arm) for r in bench.rows(arm, p)
        )
    ]


def chart_stages(bench: Bench, styles, point: Point, out_dir: Path) -> Chart | None:
    arms = [a for a in epd_arms(bench) if bench.rows(a, point)]
    keys = _stage_keys(bench)
    if not arms or not keys:
        return None
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(11.0, 5.8), gridspec_kw={"width_ratios": [1.15, 1.0]}
    )
    layout(fig, top_in=1.3, bottom_in=1.25, left=0.075, right=0.985, wspace=0.2)

    shades = _stage_shades(keys)
    centers = np.arange(len(arms), dtype=float)
    bottoms = np.zeros(len(arms))
    for key in keys:
        values = np.array(
            [(bench.stat(a, point, stage(key)) or Stat(0, 0, 0, 0)).mean for a in arms]
        )
        ax.bar(
            centers,
            values,
            bottom=bottoms,
            width=0.62,
            color=shades[key],
            edgecolor=SURFACE,
            linewidth=1.5,
            label=key.replace("_", " "),
            zorder=2,
        )
        bottoms += values
    for x, total in zip(centers, bottoms):
        ax.annotate(
            f"{total:.0f} ms",
            (x, total),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=9,
            color=INK_2,
        )
    style_axes(ax)
    ax.set_xticks(centers)
    ax.set_xticklabels([_short_arm(a) for a in arms], rotation=18, ha="right")
    ax.set_ylabel("Proxy stage median (ms)")
    ax.set_ylim(0, bottoms.max() * 1.18)
    ax.set_title(
        f"Stacked stages at {_point_label(point)}",
        color=INK,
        fontsize=11.5,
        loc="left",
        pad=8,
    )
    figure_legend(fig, ax, ncol=len(keys))

    width = 0.8 / max(len(arms), 1)
    positions = np.arange(len(keys), dtype=float)
    for index, arm in enumerate(arms):
        offset = (index - (len(arms) - 1) / 2) * width
        stats = [bench.stat(arm, point, stage(k)) for k in keys]
        bar_group(
            ax2,
            positions + offset,
            [s.mean if s else 0.0 for s in stats],
            [s.err if s else (0.0, 0.0) for s in stats],
            width,
            styles[arm].color,
            arm,
        )
    style_axes(ax2)
    ax2.set_xticks(positions)
    ax2.set_xticklabels([k.replace("_", " ") for k in keys])
    ax2.set_ylabel("Stage median (ms)")
    ax2.margins(y=0.28)
    ax2.set_ylim(bottom=0)
    ax2.set_title(
        "Same stages, side by side", color=INK, fontsize=11.5, loc="left", pad=8
    )
    legend(ax2, ncol=2, loc="upper left")
    titles(
        fig,
        "Where the EPD request spends its time",
        "Proxy-reported medians per stage; decode_ttfb is what the client waits for.",
    )
    source_note(fig, rep_note(bench))
    svg, png = _save(fig, out_dir, "04_stages")
    return Chart(
        "04_stages",
        "EPD stage breakdown",
        f"Stage medians the EPD proxy logs, at {_point_label(point)}. The grid "
        "rewrite trades a little encoder time for a shorter decode time-to-first-byte.",
        svg,
        png,
    )


def _stage_shades(keys: Sequence[str]) -> dict[str, str]:
    ramp = ["#9ec5f4", "#3987e5", "#1c5cab", "#104281"]
    return {k: ramp[min(i, len(ramp) - 1)] for i, k in enumerate(keys)}


def chart_stages_by_load(bench: Bench, styles, out_dir: Path) -> Chart | None:
    arms = epd_arms(bench)
    keys = _stage_keys(bench)
    points = [p for p in bench.points if any(bench.rows(a, p) for a in arms)]
    if not arms or not keys or len(points) < 2:
        return None
    cols = min(len(points), 6)
    rows = math.ceil(len(points) / cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(2.05 * cols + 0.9, 3.4 * rows + 1.9), sharey=True
    )
    axes = np.atleast_1d(axes).ravel()
    layout(fig, top_in=1.15, bottom_in=1.45, left=0.075, right=0.99, wspace=0.12)
    shades = _stage_shades(keys)
    for ax, point in zip(axes, points):
        present = [a for a in arms if bench.rows(a, point)]
        centers = np.arange(len(present), dtype=float)
        bottoms = np.zeros(len(present))
        for key in keys:
            values = np.array(
                [
                    (bench.stat(a, point, stage(key)) or Stat(0, 0, 0, 0)).mean
                    for a in present
                ]
            )
            ax.bar(
                centers,
                values,
                bottom=bottoms,
                width=0.66,
                color=shades[key],
                edgecolor=SURFACE,
                linewidth=1.2,
                label=key.replace("_", " "),
                zorder=2,
            )
            bottoms += values
        style_axes(ax)
        ax.set_xticks(centers)
        ax.set_xticklabels([_short_arm(a) for a in present], rotation=45, ha="right")
        ax.set_title(_point_label(point), color=INK_2, fontsize=10, loc="left")
    for ax in axes[len(points) :]:
        ax.set_visible(False)
    axes[0].set_ylabel("Stage median (ms)")
    figure_legend(fig, axes[0], ncol=len(keys))
    titles(
        fig,
        "Stage breakdown across the load sweep",
        "One panel per load point; the same stacked stages, on a shared scale.",
    )
    source_note(fig, rep_note(bench))
    svg, png = _save(fig, out_dir, "05_stages_by_load")
    return Chart(
        "05_stages_by_load",
        "EPD stage breakdown per load point",
        "Small multiples of the stage stack, so a stage that only grows under "
        "load is visible as such.",
        svg,
        png,
    )


def _short_arm(arm: str) -> str:
    return arm.replace("example", "ex").replace("multinode", "mn")


def chart_mechanism(bench: Bench, styles, point: Point, out_dir: Path) -> Chart:
    arms = [a for a in bench.arms if bench.rows(a, point)]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 5.8))
    layout(fig, top_in=1.3, bottom_in=1.25, left=0.075, right=0.985, wspace=0.24)

    metrics = (
        ("encoder inputs (consumer)", server("encoder_inputs_computed")),
        ("EC entries loaded", server("ec_load_entries")),
        ("EC entries saved", producer_saves),
    )
    width = 0.8 / len(metrics)
    centers = np.arange(len(arms), dtype=float)
    ramp = ["#8b8985", "#2a78d6", "#86b6ef"]
    for index, (label, get) in enumerate(metrics):
        offset = (index - (len(metrics) - 1) / 2) * width
        stats = [bench.stat(a, point, get) for a in arms]
        bar_group(
            ax,
            centers + offset,
            [s.mean if s else 0.0 for s in stats],
            [s.err if s else (0.0, 0.0) for s in stats],
            width,
            ramp[index],
            label,
        )
    style_axes(ax)
    ax.set_xticks(centers)
    ax.set_xticklabels([_short_arm(a) for a in arms], rotation=18, ha="right")
    ax.set_ylabel("Entries per run")
    ax.margins(y=0.3)
    ax.set_ylim(bottom=0)
    ax.set_title(
        f"What each arm did at {_point_label(point)}",
        color=INK,
        fontsize=11.5,
        loc="left",
        pad=8,
    )
    figure_legend(fig, ax, ncol=3)

    base = baseline_arm(bench)
    ref = bench.stat(base, point, server("encoder_inputs_computed")) if base else None
    moved, avoided = [], []
    for arm in arms:
        load_bytes = bench.stat(arm, point, server("ec_load_bytes"))
        save_bytes = bench.stat(arm, point, producer_save_bytes)
        moved.append(
            (
                (load_bytes.mean if load_bytes else 0.0)
                + (save_bytes.mean if save_bytes else 0.0)
            )
            / 1e9
        )
        avoided.append(_bytes_not_recomputed(bench, arm, point, ref) / 1e9)
    width2 = 0.38
    bar_group(
        ax2,
        centers - width2 / 2,
        moved,
        None,
        width2,
        "#2a78d6",
        "bytes moved over the connector",
    )
    bar_group(
        ax2,
        centers + width2 / 2,
        avoided,
        None,
        width2,
        "#1baf7a",
        "embedding bytes not recomputed",
    )
    style_axes(ax2)
    ax2.set_xticks(centers)
    ax2.set_xticklabels([_short_arm(a) for a in arms], rotation=18, ha="right")
    ax2.set_ylabel("GB per run")
    ax2.margins(y=0.3)
    ax2.set_ylim(bottom=0)
    ax2.set_title(
        "Bytes moved vs bytes avoided", color=INK, fontsize=11.5, loc="left", pad=8
    )
    legend(ax2, ncol=1, loc="upper left")
    titles(
        fig,
        "Mechanism evidence: the consumer stopped running the vision tower",
        "Counters from the servers' own logs, not inferred from the timings.",
    )
    source_note(
        fig,
        rep_note(
            bench,
            "'Not recomputed' prices the skipped encoder inputs at the arm's "
            "own bytes/entry.",
        ),
    )
    svg, png = _save(fig, out_dir, "06_mechanism")
    return Chart(
        "06_mechanism",
        "Mechanism evidence",
        "A connector arm must load entries and compute fewer encoder inputs than "
        "the baseline; otherwise the arms differ by configuration, not mechanism.",
        svg,
        png,
    )


def _bytes_not_recomputed(
    bench: Bench, arm: str, point: Point, ref: Stat | None
) -> float:
    """Encoder inputs skipped vs baseline, priced at the arm's bytes/entry."""
    if ref is None:
        return 0.0
    computed = bench.stat(arm, point, server("encoder_inputs_computed"))
    entries = bench.stat(arm, point, server("ec_load_entries"))
    nbytes = bench.stat(arm, point, server("ec_load_bytes"))
    if computed is None or not entries or not entries.mean or nbytes is None:
        return 0.0
    per_entry = nbytes.mean / entries.mean
    return max(ref.mean - computed.mean, 0.0) * per_entry


def chart_bandwidth(
    bench: Bench, axis: LoadAxis, styles, out_dir: Path
) -> Chart | None:
    directions = (
        (server("ec_load_gbps"), "load (consumer)"),
        (producer_gbps, "save (producer)"),
    )
    arms = [
        a
        for a in bench.arms
        if any(
            (bench.stat(a, p, get) or Stat(0, 0, 0, 0)).mean
            for p in bench.points_for(a)
            for get, _ in directions
        )
    ]
    if not arms:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.4), sharey=True)
    layout(fig, top_in=1.28, bottom_in=0.9, left=0.075, right=0.985, wspace=0.1)
    seen: list[float] = []
    for ax, (key, name) in zip(axes, directions):
        for arm in arms:
            pairs = [(p, s) for p, s in bench.series(arm, key) if s.mean > 0]
            if not pairs:
                continue
            style = styles[arm]
            seen += [s.mean for _, s in pairs]
            ax.errorbar(
                [axis.x(p) for p, _ in pairs],
                [s.mean for _, s in pairs],
                yerr=np.array([s.err for _, s in pairs]).T,
                color=style.color,
                marker=style.marker,
                label=arm,
                dashes=style.dashes or (),
                markeredgecolor=SURFACE,
                elinewidth=1.0,
                capsize=3,
                zorder=3,
            )
        style_axes(ax)
        axis.apply(ax)
        ax.set_title(f"{name} bandwidth", color=INK, fontsize=11.5, loc="left", pad=8)
    logged = log_span(seen)
    for ax in axes:
        if logged:
            log_y(ax)
        else:
            ax.set_ylim(bottom=0)
    axes[0].set_ylabel(
        "Transfer bandwidth (GB/s, log scale)"
        if logged
        else "Transfer bandwidth (GB/s)"
    )
    figure_legend(fig, axes[0], ncol=6)
    titles(
        fig,
        "Connector transfer bandwidth",
        "Bytes over the connector's own transfer time; arms that moved nothing "
        "are omitted.",
    )
    source_note(fig, rep_note(bench))
    svg, png = _save(fig, out_dir, "07_bandwidth")
    return Chart(
        "07_bandwidth",
        "Connector transfer bandwidth",
        "This is transfer time only, not end-to-end: it says how fast the "
        "transport ran, not how much of the request it covered.",
        svg,
        png,
    )


def chart_queues(bench: Bench, axis: LoadAxis, styles, out_dir: Path) -> Chart | None:
    instances = queue_instances(bench)
    if not instances:
        return None
    cols = min(len(instances), 3)
    rows = math.ceil(len(instances) / cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(3.7 * cols + 0.8, 4.4 * rows + 2.1), sharey=True
    )
    axes = np.atleast_1d(axes).ravel()
    layout(fig, top_in=1.35, bottom_in=0.95, left=0.08, right=0.985, wspace=0.12)
    concs = [p for p in bench.points if p[1] > 0]
    handles: dict[str, Any] = {}
    for ax, instance in zip(axes, instances):
        for arm in bench.arms:
            pairs = [
                (p, s)
                for p, s in bench.series(arm, queue_metric(instance, "waiting_max"))
            ]
            if not pairs:
                continue
            style = styles[arm]
            (line,) = ax.plot(
                [axis.x(p) for p, _ in pairs],
                [s.mean for _, s in pairs],
                color=style.color,
                marker=style.marker,
                label=arm,
                dashes=style.dashes or (),
                markeredgecolor=SURFACE,
                zorder=3,
            )
            handles.setdefault(arm, line)
        if concs:
            xs = [axis.x(p) for p in concs]
            (line,) = ax.plot(
                xs,
                [p[1] for p in concs],
                color=MUTED,
                linewidth=1.2,
                dashes=(4, 3),
                label="offered concurrency",
                zorder=2,
            )
            handles.setdefault("offered concurrency", line)
        style_axes(ax)
        axis.apply(ax)
        ax.set_title(instance, color=INK, fontsize=11.5, loc="left", pad=8)
        ax.set_ylim(bottom=0)
    for ax in axes[len(instances) :]:
        ax.set_visible(False)
    axes[0].set_ylabel("Max requests waiting (peak queue depth)")
    fig.legend(
        list(handles.values()),
        list(handles),
        loc="upper left",
        bbox_to_anchor=(0.012, 1 - 0.62 / fig.get_figheight()),
        ncol=min(len(handles), 7),
        handlelength=1.8,
        columnspacing=1.6,
        borderaxespad=0.0,
    )
    titles(
        fig,
        "Validity: is the point measuring work, or the queue?",
        "A waiting queue that tracks the offered concurrency means the instance "
        "is saturated and the latencies describe queueing.",
    )
    source_note(fig, rep_note(bench, "Sampled once per second during each point."))
    svg, png = _save(fig, out_dir, "08_queues")
    return Chart(
        "08_queues",
        "Queue depth per instance",
        "Where a line meets the dashed offered-concurrency reference, the arm's "
        "numbers at that point are queue-bound and comparisons get weaker.",
        svg,
        png,
    )


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


@dataclass
class Summary:
    """The handful of numbers the executive block states."""

    best_arm: str | None
    best_speedup: float
    best_point: Point | None
    ttft_drop: float
    ttft_arm: str | None
    ttft_point: Point | None
    bytes_moved: float
    bytes_avoided: float
    loads: float
    saves: float
    inputs_avoided_pct: float


def summarise(bench: Bench, base: str | None) -> Summary:
    best = (None, 1.0, None)
    ttft_best = (None, 0.0, None)
    if base:
        for arm in bench.arms:
            if arm == base:
                continue
            for point in bench.points_for(arm):
                if not bench.rows(base, point):
                    continue
                ratio = ratio_stat(bench, arm, base, point, client("output_throughput"))
                if ratio.n and ratio.mean > best[1]:
                    best = (arm, ratio.mean, point)
                ttft = ratio_stat(bench, arm, base, point, client("median_ttft_ms"))
                if ttft.n and not math.isnan(ttft.mean):
                    drop = 1.0 - ttft.mean
                    if drop > ttft_best[1]:
                        ttft_best = (arm, drop, point)
    moved = avoided = loads = saves = 0.0
    inputs_ref = inputs_arm = 0.0
    for (arm, point), rows in bench.entries.items():
        if arm == base:
            continue
        for row in rows:
            srv = row.get("server", {})
            moved += (srv.get("ec_load_bytes") or 0) + (producer_save_bytes(row) or 0)
            loads += srv.get("ec_load_entries") or 0
            saves += producer_saves(row) or 0
    for arm in bench.arms:
        if arm == base or not base:
            continue
        for point in bench.points_for(arm):
            ref = bench.stat(base, point, server("encoder_inputs_computed"))
            got = bench.stat(arm, point, server("encoder_inputs_computed"))
            if ref is None or got is None:
                continue
            inputs_ref += ref.mean
            inputs_arm += got.mean
            avoided += _bytes_not_recomputed(bench, arm, point, ref)
    pct = (1 - inputs_arm / inputs_ref) * 100 if inputs_ref else 0.0
    reps = max(bench.reps, 1)
    return Summary(
        best[0],
        best[1],
        best[2],
        ttft_best[1],
        ttft_best[0],
        ttft_best[2],
        moved / reps,
        avoided,
        loads / reps,
        saves / reps,
        pct,
    )


def gate_rows(bench: Bench, base: str | None) -> list[dict[str, Any]]:
    """`run_bench.py --check-gates` restated from the persisted numbers."""
    if not base:
        return []
    rows = []
    for arm in bench.arms:
        if arm == base:
            continue
        for point in bench.points_for(arm):
            ref = bench.stat(base, point, server("encoder_inputs_computed"))
            got = bench.stat(arm, point, server("encoder_inputs_computed"))
            loads = bench.stat(arm, point, server("ec_load_entries"))
            if ref is None or got is None or loads is None:
                continue
            passed = loads.mean > 0 and got.mean < ref.mean
            rows.append(
                {
                    "arm": arm,
                    "point": point,
                    "loads": loads.mean,
                    "computed": got.mean,
                    "baseline": ref.mean,
                    "avoided": (1 - got.mean / ref.mean) * 100 if ref.mean else 0.0,
                    "pass": passed,
                }
            )
    return rows


def _fmt(value: float | None, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.{digits}f}"


def _gb(value: float) -> str:
    return f"{value / 1e9:,.1f} GB" if value else "0 GB"


def table_rows(bench: Bench, base: str | None) -> list[dict[str, Any]]:
    rows = []
    for arm in bench.arms:
        for point in bench.points_for(arm):
            ratio = (
                ratio_stat(bench, arm, base, point, client("output_throughput"))
                if base and arm != base and bench.rows(base, point)
                else None
            )
            get = partial(bench.stat, arm, point)
            rows.append(
                {
                    "arm": arm,
                    "rate": point[0],
                    "conc": point[1],
                    "reps": len(bench.rows(arm, point)),
                    "ttft_p50": get(client("median_ttft_ms")),
                    "ttft_p99": get(client("p99_ttft_ms")),
                    "tpot": get(client("mean_tpot_ms")),
                    "out_tps": get(client("output_throughput")),
                    "ratio": ratio,
                    "loads": get(server("ec_load_entries")),
                    "saves": get(producer_saves),
                    "inputs": get(server("encoder_inputs_computed")),
                    "encode": get(stage("encode")),
                    "dec_ttfb": get(stage("decode_ttfb")),
                    "qmax": get(max_waiting),
                    "completed": get(client("completed")),
                    "coverage": get(entry_field("rewrite_coverage")),
                }
            )
    return rows


CSS = Template(
    """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; background: ${plane}; color: ${ink};
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  font-size: 15px; line-height: 1.55; }
.wrap { max-width: 1180px; margin: 0 auto; padding: 48px 28px 72px; }
header { border-bottom: 1px solid ${grid}; padding-bottom: 22px; margin-bottom: 30px; }
h1 { font-size: 30px; line-height: 1.2; margin: 0 0 8px; letter-spacing: -0.4px; }
h2 { font-size: 19px; margin: 44px 0 6px; letter-spacing: -0.2px; }
h3 { font-size: 15px; margin: 26px 0 6px; }
p, li { color: ${ink2}; margin: 0 0 12px; }
.sub { color: ${muted}; font-size: 13.5px; margin: 0; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 14px; margin: 26px 0 8px; }
.tile { background: ${surface}; border: 1px solid ${grid}; border-radius: 10px;
  padding: 16px 18px; }
.tile .k { font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em;
  color: ${muted}; margin: 0 0 6px; }
.tile .v { font-size: 30px; line-height: 1.1; color: ${ink}; margin: 0; }
.tile .n { font-size: 12.5px; color: ${muted}; margin: 6px 0 0; }
figure { background: ${surface}; border: 1px solid ${grid}; border-radius: 10px;
  margin: 20px 0 0; padding: 14px 16px 6px; }
figure svg.chart { width: 100%; height: auto; display: block; }
figcaption { color: ${muted}; font-size: 13px; padding: 6px 2px 12px; }
table { border-collapse: collapse; width: 100%; font-size: 13px;
  font-variant-numeric: tabular-nums; background: ${surface}; }
.scroll { overflow-x: auto; border: 1px solid ${grid}; border-radius: 10px;
  margin-top: 14px; }
th, td { text-align: right; padding: 7px 11px; border-bottom: 1px solid ${grid};
  white-space: nowrap; }
th { color: ${muted}; font-weight: 600; font-size: 11.5px; text-transform: uppercase;
  letter-spacing: 0.05em; position: sticky; top: 0; background: ${surface}; }
td:first-child, th:first-child { text-align: left; }
tbody tr:last-child td { border-bottom: none; }
.swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px;
  margin-right: 8px; vertical-align: baseline; }
.pass { color: ${good}; font-weight: 600; }
.fail { color: ${bad}; font-weight: 600; }
.warn { color: #a5490f; font-weight: 600; }
.note { background: ${surface}; border: 1px solid ${grid}; border-left: 3px solid
  ${axis}; border-radius: 8px; padding: 14px 18px; margin-top: 16px; }
footer { margin-top: 52px; border-top: 1px solid ${grid}; padding-top: 18px;
  color: ${muted}; font-size: 12.5px; }
"""
).substitute(
    plane=PLANE,
    surface=SURFACE,
    ink=INK,
    ink2=INK_2,
    muted=MUTED,
    grid=GRID,
    axis=AXIS,
    good=GOOD,
    bad=CRITICAL,
)


def _tile(key: str, value: str, note: str) -> str:
    return (
        f'<div class="tile"><p class="k">{escape(key)}</p>'
        f'<p class="v">{escape(value)}</p><p class="n">{escape(note)}</p></div>'
    )


def _table(headers: Sequence[str], body: Sequence[Sequence[str]]) -> str:
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    rows = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in body
    )
    return (
        f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{rows}</tbody></table></div>"
    )


def build_report(
    bench: Bench,
    charts: Sequence[Chart],
    styles: dict[str, ArmStyle],
    base: str | None,
    title: str,
) -> str:
    summary = summarise(bench, base)
    gates = gate_rows(bench, base)
    rows = table_rows(bench, base)
    args = bench.docs[0].get("args", {})
    expected = bench.docs[0].get("manifest_expected", {})

    tiles = [
        _tile(
            "Best throughput gain",
            f"{summary.best_speedup:.2f}x" if summary.best_arm else "-",
            f"{summary.best_arm} at {_point_label(summary.best_point)}"
            if summary.best_arm
            else "no baseline arm in the data",
        ),
        _tile(
            "Best TTFT reduction",
            f"-{summary.ttft_drop * 100:.0f}%" if summary.ttft_arm else "-",
            f"median TTFT, {summary.ttft_arm} at {_point_label(summary.ttft_point)}"
            if summary.ttft_arm
            else "no baseline arm in the data",
        ),
        _tile(
            "Encoder work avoided",
            f"{summary.inputs_avoided_pct:.0f}%",
            "encoder inputs the consumer did not compute, vs baseline",
        ),
        _tile(
            "Embeddings moved / not recomputed",
            f"{summary.bytes_moved / 1e9:,.0f} / {summary.bytes_avoided / 1e9:,.0f} GB",
            f"{summary.loads:,.0f} loads, {summary.saves:,.0f} saves - summed "
            "over connector arms and load points, per rep",
        ),
    ]

    figures = "".join(
        f'<figure id="{escape(c.name)}"><h3>{escape(c.title)}</h3>{c.svg}'
        f"<figcaption>{escape(c.caption)} n={bench.reps} reps."
        f"</figcaption></figure>"
        for c in charts
    )

    headers = [
        "arm",
        "rate",
        "conc",
        "reps",
        "ttft p50 (ms)",
        "ttft p99 (ms)",
        "tpot (ms)",
        "out tok/s",
        "x base",
        "loads",
        "saves",
        "enc inputs",
        "encode (ms)",
        "dec ttfb (ms)",
        "queue max",
    ]
    body = []
    for row in rows:
        swatch = (
            '<span class="swatch" style="background:'
            f'{styles[row["arm"]].color}"></span>'
        )
        ratio = row["ratio"]
        body.append(
            [
                f"{swatch}{escape(row['arm'])}",
                escape(str(row["rate"])),
                "unbounded" if row["conc"] == 0 else str(row["conc"]),
                str(row["reps"]),
                _fmt(row["ttft_p50"].mean if row["ttft_p50"] else None),
                _fmt(row["ttft_p99"].mean if row["ttft_p99"] else None),
                _fmt(row["tpot"].mean if row["tpot"] else None, 2),
                _fmt(row["out_tps"].mean if row["out_tps"] else None),
                f"{ratio.mean:.2f}" if ratio and ratio.n else "-",
                _fmt(row["loads"].mean if row["loads"] else None, 0),
                _fmt(row["saves"].mean if row["saves"] else None, 0),
                _fmt(row["inputs"].mean if row["inputs"] else None, 0),
                _fmt(row["encode"].mean if row["encode"] else None),
                _fmt(row["dec_ttfb"].mean if row["dec_ttfb"] else None),
                _fmt(row["qmax"].mean if row["qmax"] else None, 0),
            ]
        )

    gate_body = [
        [
            f'<span class="swatch" style="background:{styles[g["arm"]].color}"></span>'
            f"{escape(g['arm'])}",
            _point_label(g["point"]),
            _fmt(g["loads"], 0),
            f"{_fmt(g['computed'], 0)} vs {_fmt(g['baseline'], 0)}",
            f"{g['avoided']:.1f}%",
            '<span class="pass">PASS</span>'
            if g["pass"]
            else '<span class="fail">FAIL</span>',
        ]
        for g in gates
    ]

    validity_body = []
    completions = [r["completed"].mean for r in rows if r["completed"]]
    target = max(completions) if completions else 0.0
    for row in rows:
        qmax = row["qmax"].mean if row["qmax"] else None
        conc = row["conc"]
        queue_bound = qmax is not None and conc and qmax >= 0.5 * conc
        done = row["completed"].mean if row["completed"] else None
        coverage = row["coverage"].mean if row["coverage"] else None
        flags = []
        if done is not None and target and done < target:
            flags.append('<span class="fail">incomplete</span>')
        if queue_bound:
            flags.append('<span class="warn">queue-bound</span>')
        if "grid" in row["arm"] and (coverage or 0) < 0.9:
            flags.append('<span class="warn">low rewrite coverage</span>')
        validity_body.append(
            [
                f'<span class="swatch" '
                f'style="background:{styles[row["arm"]].color}"></span>'
                f"{escape(row['arm'])}",
                _point_label((row["rate"], conc)),
                _fmt(done, 0),
                _fmt(qmax, 0),
                f"{coverage * 100:.0f}%" if coverage else "-",
                " ".join(flags) or '<span class="pass">ok</span>',
            ]
        )

    gates_ok = all(g["pass"] for g in gates) if gates else None
    verdict = (
        "every connector arm loaded entries and computed fewer encoder inputs "
        "than the baseline"
        if gates_ok
        else "at least one arm did not clear the mechanism gate -- read the table "
        "below before quoting a speedup"
        if gates_ok is False
        else "no baseline arm is present, so the mechanism gate could not be checked"
    )

    workload = ""
    if expected:
        workload = (
            f"Workload: {expected.get('first_encodes', '-')} distinct images, "
            f"{expected.get('reuses', '-')} reuses, max hit rate "
            f"{float(expected.get('max_hit_rate', 0)) * 100:.0f}%, working set "
            f"{float(expected.get('working_set_bytes', 0)) / 1024**3:.2f} GiB."
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title><style>{CSS}</style></head>
<body><div class="wrap">
<header>
<h1>{escape(title)}</h1>
<p class="sub">{escape(str(args.get("model", "")))} &middot;
{bench.reps} replication(s) &middot; {len(bench.arms)} arms &middot;
{len(bench.points)} load points &middot; baseline: {escape(base or "none")}</p>
</header>

<h2>Executive summary</h2>
<p>{escape(verdict.capitalize())}. {escape(workload)}</p>
<div class="tiles">{"".join(tiles)}</div>
<div class="note"><p>Read the charts in order: throughput and speedup say
<em>how much</em>, TTFT says <em>what the user feels</em>, the stage breakdown
says <em>where the time went</em>, and the mechanism and validity sections say
<em>whether the comparison is real</em>. Every number here is the mean over
{bench.reps} replication(s); error bars and bands span the min and max.</p></div>

<h2>Charts</h2>
{figures}

<h2>The numbers</h2>
<p>Mean over reps at each load point. <em>x base</em> is output throughput
relative to the baseline arm at the same point; <em>loads</em>/<em>saves</em>
are connector entries, <em>enc inputs</em> is what the consumer still computed
itself.</p>
{_table(headers, body)}

<h2>Validity</h2>
<p>{escape(verdict.capitalize())}.</p>
<h3>Mechanism gate</h3>
{
        _table(
            [
                "arm",
                "load point",
                "loads",
                "consumer inputs vs baseline",
                "avoided",
                "gate",
            ],
            gate_body,
        )
        if gate_body
        else "<p>No baseline arm, so no gate to check.</p>"
    }
<h3>Completion, queueing and rewrite coverage</h3>
<p>A point whose peak waiting queue reaches half the offered concurrency is
measuring the queue as much as the work; a run that did not complete every
request describes only the requests that survived.</p>
{
        _table(
            [
                "arm",
                "load point",
                "completed",
                "peak queue",
                "rewrite coverage",
                "flags",
            ],
            validity_body,
        )
    }

<footer>Generated by
<code>scripts/cpu_ec_connector/bench/plot_results.py</code> from
{escape(", ".join(p.name for p in bench.sources))}. Charts are inline SVG; this
page needs no network access.</footer>
</div></body></html>
"""


# --------------------------------------------------------------------------
# Demo fixture
# --------------------------------------------------------------------------

# arm -> (topology, throughput ratio at the lightest and heaviest point,
#         TTFT ratio at the lightest and heaviest point, rewrites)
DEMO_ARMS = {
    "baseline": ("single", 1.00, 1.00, 1.00, 1.00, 0),
    "offload": ("single", 1.09, 1.01, 0.90, 0.98, 0),
    "cpu-data": ("epd", 1.15, 1.02, 0.80, 0.94, 0),
    "cpu-grid": ("epd", 1.27, 1.04, 0.68, 0.89, 1),
    "example-data": ("epd", 1.12, 1.01, 0.83, 0.95, 0),
    "example-grid": ("epd", 1.23, 1.03, 0.72, 0.91, 1),
}
DEMO_CONCS = (1, 2, 4, 8, 16, 32)
DEMO_PROMPTS = 600
DEMO_REFS = 744
DEMO_DISTINCT = 186
DEMO_ENTRY_BYTES = 6_291_456


def _decay(value: float, floor: float, conc: int) -> float:
    """Interpolate `value` down to `floor` across the concurrency sweep."""
    frac = math.log2(max(conc, 1)) / math.log2(max(DEMO_CONCS))
    return value + (floor - value) * frac


def demo_entry(rng, arm: str, conc: int) -> dict[str, Any]:
    topology, gain0, gain1, ttft0, ttft1, rewrite = DEMO_ARMS[arm]

    def jitter(scale: float = 0.012) -> float:
        return float(1 + rng.normal(0, scale))

    base_tps = 1400 * conc / (conc + 6.0)
    out_tps = base_tps * _decay(gain0, gain1, conc) * jitter()
    base_ttft = 165 * (1 + conc**1.05 / 2.4)
    ttft = base_ttft * _decay(ttft0, ttft1, conc) * jitter(0.02)
    duration = DEMO_PROMPTS * 128 / out_tps

    client_block = {
        "completed": DEMO_PROMPTS,
        "duration": round(duration, 3),
        "request_throughput": round(DEMO_PROMPTS / duration, 3),
        "output_throughput": round(out_tps, 2),
        "total_token_throughput": round(out_tps * 4.1, 2),
    }
    shapes = {
        "ttft_ms": (ttft, 1.10, 1.55, 1.9, 0.42),
        "tpot_ms": (11.5 + conc * 0.42, 1.04, 1.25, 1.4, 0.1),
        "itl_ms": (11.0 + conc * 0.40, 1.05, 1.30, 1.5, 0.12),
        "e2el_ms": (ttft + 128 * (11.5 + conc * 0.42), 1.06, 1.35, 1.6, 0.15),
    }
    for key, (mean, p50, p95, p99, std) in shapes.items():
        client_block[f"mean_{key}"] = round(mean, 2)
        client_block[f"median_{key}"] = round(mean / p50, 2)
        client_block[f"p95_{key}"] = round(mean * p95, 2)
        client_block[f"p99_{key}"] = round(mean * p99, 2)
        client_block[f"std_{key}"] = round(mean * std, 2)

    if arm == "baseline":
        consumer_inputs, loads = 690, 0
    elif arm == "offload":
        consumer_inputs, loads = 205, 485
    elif rewrite:
        consumer_inputs, loads = 0, DEMO_REFS
    else:
        consumer_inputs, loads = 12, DEMO_REFS - 12
    cpu = "cpu" in arm or arm == "offload"
    gbps = (21.4 if cpu else 1.7) * jitter(0.04)
    saves = 0 if topology == "epd" else (DEMO_DISTINCT if loads else 0)

    def side(entries: int, direction: str) -> dict[str, Any]:
        nbytes = entries * DEMO_ENTRY_BYTES
        ms = nbytes / (gbps * 1e9) * 1000 if entries else 0.0
        return {
            f"ec_{direction}_entries": entries,
            f"ec_{direction}_bytes": nbytes,
            f"ec_{direction}_ms": round(ms, 3),
            f"ec_{direction}_gbps": round(gbps, 1) if entries else 0.0,
            f"ec_{direction}_transfers": entries if not cpu else max(entries // 2, 1),
        }

    server_block = {
        **side(saves, "save"),
        **side(loads, "load"),
        "encoder_inputs_computed": consumer_inputs,
        "encoder_embeds_computed": consumer_inputs * 1280,
    }
    entry: dict[str, Any] = {
        "arm": arm,
        "topology": topology,
        "request_rate": "inf",
        "concurrency": conc,
        "client": client_block,
        "server": server_block,
        "encoders": 1 if topology == "epd" else 0,
        "per_encoder_inputs": {"encoder0": DEMO_DISTINCT} if topology == "epd" else {},
    }
    names = ["single"] if topology == "single" else ["decode", "encoder0"]
    queue: dict[str, Any] = {"samples": int(duration * len(names))}
    for index, name in enumerate(names):
        waiting = max(0.0, (conc - 4) * (0.62 if index == 0 else 0.40)) * jitter(0.05)
        running = min(conc, 12 + conc * 0.2)
        queue[f"{name}_waiting_max"] = round(waiting, 1)
        queue[f"{name}_waiting_mean"] = round(waiting * 0.55, 2)
        queue[f"{name}_running_max"] = round(running, 1)
        queue[f"{name}_running_mean"] = round(running * 0.8, 2)
    entry["queue"] = queue

    if topology == "epd":
        encoder_block = {
            **side(DEMO_DISTINCT, "save"),
            **side(0, "load"),
            "encoder_inputs_computed": DEMO_DISTINCT,
            "encoder_embeds_computed": DEMO_DISTINCT * 1280,
        }
        encode = (81.5 if cpu else 76.5) * (1 + conc * 0.055) * jitter(0.03)
        ttfb_base = (241 if cpu else 268) if rewrite else (296 if cpu else 312)
        decode_ttfb = ttfb_base * (1 + conc * 0.34) * jitter(0.03)
        stages = {
            "requests": DEMO_PROMPTS,
            "modes": ["rewrite"] if rewrite else ["forward"],
            "encode_ms_median": round(encode, 2),
            "decode_ttfb_ms_median": round(decode_ttfb, 2),
            "decode_total_ms_median": round(decode_ttfb + 640 * (1 + conc * 0.3), 2),
        }
        if rewrite:
            stages["rewrite_ms_median"] = round(0.8 * jitter(0.1), 2)
        entry["encoder"] = encoder_block
        entry["stages"] = stages
        entry["rewritten"] = int(DEMO_REFS * 0.985) if rewrite else 0
        entry["rewrite_coverage"] = 0.985 if rewrite else 0.0
    return entry


def write_demo(out_dir: Path, reps: int = 2) -> list[Path]:
    """A plausible two-rep sweep, so the tool can be exercised without hardware."""
    demo_dir = out_dir / "demo_data"
    demo_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for rep in range(reps):
        rng = np.random.default_rng(20260908 + rep)
        results = [
            demo_entry(rng, arm, conc) for arm in DEMO_ARMS for conc in DEMO_CONCS
        ]
        doc = {
            "args": {
                "model": "Qwen/Qwen2.5-VL-7B-Instruct",
                "arms": ",".join(DEMO_ARMS),
                "num_prompts": str(DEMO_PROMPTS),
                "max_concurrency": ",".join(str(c) for c in DEMO_CONCS),
                "request_rate": "inf",
                "workload_dir": "/data/wl",
                "min_rewrite_coverage": "0.9",
                "synthetic": "True",
            },
            "manifest_expected": {
                "first_encodes": DEMO_DISTINCT,
                "reuses": DEMO_REFS - DEMO_DISTINCT,
                "max_hit_rate": round((DEMO_REFS - DEMO_DISTINCT) / DEMO_REFS, 4),
                "working_set_bytes": DEMO_DISTINCT * DEMO_ENTRY_BYTES,
                "max_embeds_per_request": 3 * 1280,
                "suggested_ec_cpu_bytes": int(DEMO_DISTINCT * DEMO_ENTRY_BYTES * 1.25),
                "fragmentation_arm_ec_cpu_bytes": int(
                    DEMO_DISTINCT * DEMO_ENTRY_BYTES * 0.5
                ),
            },
            "results": results,
        }
        path = demo_dir / f"bench_rep{rep + 1}.json"
        path.write_text(json.dumps(doc, indent=2))
        paths.append(path)
    return paths


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def pick_stage_point(bench: Bench, base: str | None) -> Point:
    """The load point the stage and mechanism charts default to.

    The point where the best arm leads by the most, because that is the one a
    reader will want explained; ties fall back to the heaviest point.
    """
    best, best_ratio = None, -math.inf
    if base:
        for arm in bench.arms:
            if arm == base:
                continue
            for point in bench.points_for(arm):
                if not bench.rows(base, point):
                    continue
                ratio = ratio_stat(bench, arm, base, point, client("output_throughput"))
                if ratio.n and ratio.mean > best_ratio:
                    best, best_ratio = point, ratio.mean
    return best or bench.points[-1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render charts and an offline report from run_bench.py output."
    )
    p.add_argument("files", nargs="*", type=Path, help="one or more bench.json files")
    p.add_argument("--out-dir", type=Path, default=Path("report"))
    p.add_argument("--title", default="vLLM encoder-cache connector benchmark")
    p.add_argument(
        "--demo",
        action="store_true",
        help="synthesise a two-rep bench.json set into --out-dir and render it",
    )
    p.add_argument(
        "--stage-concurrency",
        type=int,
        default=None,
        help="concurrency for the stage and mechanism bar charts (default: the "
        "point with the largest speedup)",
    )
    args = p.parse_args()
    if not args.files and not args.demo:
        p.error("give at least one bench.json, or --demo")
    return args


def main() -> int:
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    files = list(args.files)
    if args.demo:
        files = write_demo(out_dir) + files
        print(f"[plot] wrote demo fixtures to {out_dir / 'demo_data'}")

    bench = load_bench(files)
    base = baseline_arm(bench)
    styles = arm_styles(bench.arms)
    axis = load_axis(bench.points)
    point = next(
        (p for p in bench.points if p[1] == args.stage_concurrency),
        pick_stage_point(bench, base),
    )

    charts: list[Chart | None] = []
    with plt.rc_context(RC):
        charts.append(chart_throughput(bench, axis, styles, out_dir))
        if base:
            charts.append(chart_speedup(bench, axis, styles, base, out_dir))
        charts.append(chart_ttft(bench, axis, styles, out_dir))
        charts.append(chart_stages(bench, styles, point, out_dir))
        charts.append(chart_stages_by_load(bench, styles, out_dir))
        charts.append(chart_mechanism(bench, styles, point, out_dir))
        charts.append(chart_bandwidth(bench, axis, styles, out_dir))
        charts.append(chart_queues(bench, axis, styles, out_dir))
    rendered = [c for c in charts if c is not None]

    report = out_dir / "report.html"
    report.write_text(build_report(bench, rendered, styles, base, args.title))
    print(
        f"[plot] {len(rendered)} charts (PNG + SVG) and {report} "
        f"from {len(files)} file(s), {bench.reps} rep(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
