"""Electric-field scan: run <-> cathode HV.

Single source for the run map, shared by the batch scripts in deconvolve/
and fit/ and by the field plots.

The scan was taken twice: rising voltage (17 points, ~10 kV steps) and
falling voltage (9 points, ~20 kV steps). Run 039510 (0 kV) is physically the
same run in both branches, so it is not an independent measurement.
"""

# HV label [kV] -> run number, rising branch
EF_SCAN_UP = {
    0: "039510", 10: "039511", 20: "039512", 30: "039514", 40: "039515",
    50: "039516", 60: "039517", 70: "039518", 80: "039519", 90: "039521",
    100: "039522", 110: "039523", 120: "039525", 130: "039526",
    140: "039527", 150: "039528", 155.7: "039529",
}

# HV label [kV] -> run number, falling branch
EF_SCAN_DOWN = {
    155.7: "039500", 140: "039501", 120: "039502", 100: "039503",
    80: "039504", 60: "039506", 40: "039507", 20: "039508", 0: "039510",
}

# run -> (HV [kV], branch)
FIELD_OF_RUN = {}
for _kv, _run in EF_SCAN_UP.items():
    FIELD_OF_RUN[_run] = (_kv, "up")
for _kv, _run in EF_SCAN_DOWN.items():
    FIELD_OF_RUN.setdefault(_run, (_kv, "down"))


def runs_of(branch="up"):
    """Ordered [(HV, run)] of a branch ("up", "down" or "all").

    "all" keeps both branches and deduplicates by run, so the 0 kV run that
    appears in both is listed once and the total is 25, not 26.
    """
    pairs = []
    if branch in ("up", "all"):
        pairs += list(EF_SCAN_UP.items())
    if branch in ("down", "all"):
        pairs += list(EF_SCAN_DOWN.items())

    seen, out = set(), []
    for kv, run in pairs:
        if run in seen:
            continue
        seen.add(run)
        out.append((kv, run))
    return sorted(out)


def field_of(run):
    """HV [kV] of a run, or None if the run is not in the scan."""
    entry = FIELD_OF_RUN.get(str(run))
    return entry[0] if entry else None
