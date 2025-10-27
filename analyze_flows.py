#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_flows.py

Reads .p (pickle) flow files created by extract_flows_from_pcap.py (new rich format),
optionally supports the old Counter format, analyzes overlaps, and exports clean CSVs
for machine-learning classification.

It runs BOTH passes automatically:
- VPN     -> uses .p files whose stem contains "vpn"
- Non-VPN -> uses .p files whose stem does NOT contain "vpn"

Outputs one CSV per application per pass, e.g.:
  datasets/csvs/facebook_vpn.csv
  datasets/csvs/facebook_nonvpn.csv
"""

import sys
import re
import pickle
from pathlib import Path
from typing import Dict, Tuple, Any

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# === Paths ===
FLOWS_DIR = Path("./datasets/flows")
OUT_DIR   = Path("./datasets/csvs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# === App names (order matters: "tor" occurs inside "torrent") ===
APP_NAMES = [
    "torrent", "tor", "aim", "email", "facebook", "ftps", "gmail", "hangout",
    "icq", "netflix", "scp", "sftp", "skype", "spotify", "voipbuster", "vimeo", "youtube",
]

NAME_MAPPING = {
    "AIM chat": "aim", "Email": "email", "Facebook": "facebook", "FTPS": "ftps",
    "Gmail": "gmail", "Hangouts": "hangout", "ICQ": "icq", "Netflix": "netflix",
    "SCP": "scp", "SFTP": "sftp", "Skype": "skype", "Spotify": "spotify",
    "Torrent": "torrent", "Tor": "tor", "VoipBuster": "voipbuster",
    "Vimeo": "vimeo", "Youtube": "youtube",
}

VPN_RE     = re.compile(r"\bvpn\b", re.I)
NONVPN_RE  = re.compile(r"\bnon[-_ ]?vpn\b", re.I)

def infer_app_from_stem(stem: str) -> str | None:
    s = stem.lower()
    for app in APP_NAMES:
        if app in s:
            return app
    return None

def is_vpn_file(path: Path) -> bool | None:
    """Return True if filename indicates VPN, False if indicates non-VPN, else None."""
    s = path.stem.lower()
    if NONVPN_RE.search(s):
        return False
    if VPN_RE.search(s):
        return True
    # Some datasets mark VPN only by including "vpn" in the name; non-VPN has none.
    return None

def load_flow_pickle(p: Path) -> Dict[Tuple[Any, ...], Any]:
    """
    Supports:
      NEW format: { (proto,(A_ip,A_port),(B_ip,B_port)) -> {stats_dict} }
      OLD format: Counter{ (src_ip, dst_ip, sport, dport, proto) -> n_packets }
    Returns a dict in NEW-like shape:
      key: (proto, (A_ip,A_port), (B_ip,B_port))
      value: stats dict with full fields (missing ones filled with best-effort defaults)
    """
    with open(p, "rb") as f:
        obj = pickle.load(f)

    # NEW format detection: dict of dicts with known fields
    if isinstance(obj, dict) and obj:
        any_val = next(iter(obj.values()))
        if isinstance(any_val, dict) and ("total_packets" in any_val or "duration" in any_val):
             # Looks like new rich format already
             return obj

    # OLD format (Counter-like): convert to minimal stats
    from collections import Counter
    if isinstance(obj, Counter) or (isinstance(obj, dict) and obj and isinstance(next(iter(obj.values())), int)):
        converted = {}
        for (src_ip, dst_ip, sport, dport, proto), n_packets in obj.items():
            # canonicalize endpoints in the same way extractor does (A <= B)
            A = (str(src_ip), int(sport))
            B = (str(dst_ip), int(dport))
            if A <= B:
                key = (str(proto), A, B)
                a_to_b = n_packets
                b_to_a = 0
            else:
                key = (str(proto), B, A)
                a_to_b = 0
                b_to_a = n_packets

            if key not in converted:
                converted[key] = {
                    "duration": 0.0,
                    "total_packets": 0,
                    "total_bytes": 0,
                    "packets_a_to_b": 0,
                    "packets_b_to_a": 0,
                    "bytes_a_to_b": 0,
                    "bytes_b_to_a": 0,
                    "avg_pkt_size": 0.0,
                    "std_pkt_size": 0.0,
                    "mean_iat": 0.0,
                    "std_iat": 0.0,
                    "syn": 0, "ack": 0, "fin": 0, "rst": 0, "psh": 0, "urg": 0,
                }
            converted[key]["total_packets"]  += n_packets
            converted[key]["packets_a_to_b"] += a_to_b
            converted[key]["packets_b_to_a"] += b_to_a
        return converted

    # Empty or unknown => return empty dict
    return {}

def export_app_csv(app: str, rows: list[dict], vpn_flag: bool):
    if not rows:
        return False
    df = pd.DataFrame(rows)
    # Ensure column ordering is stable
    cols = [
        "protocol", "a_port", "b_port",
        "duration", "total_packets", "total_bytes",
        "packets_a_to_b", "packets_b_to_a", "bytes_a_to_b", "bytes_b_to_a",
        "avg_pkt_size", "std_pkt_size", "mean_iat", "std_iat",
        "syn", "ack", "fin", "rst", "psh", "urg",
        "application", "is_vpn",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    df = df[cols]
    out_path = OUT_DIR / f"{app}_{'vpn' if vpn_flag else 'nonvpn'}.csv"
    df.to_csv(out_path, index=False)
    return True

def analyze_and_export(vpn_mode: bool) -> dict:
    """
    vpn_mode=True  -> only files with 'vpn' in stem (or explicitly tagged VPN)
    vpn_mode=False -> only files without 'vpn' in stem (or explicitly tagged NON-VPN)
    Returns summary stats for the summary table.
    """
    files = sorted(FLOWS_DIR.glob("*.p"))
    if not files:
        print(f"[ERROR] No .p files found in {FLOWS_DIR.resolve()}. Run extract_flows_from_pcap.py first.")
        sys.exit(1)

    # Build app -> { reduced_flow_key -> weight } for overlap analysis
    flows_by_app: dict[str, dict[tuple, int]] = {app: {} for app in APP_NAMES}

    csv_exported = 0
    total_flows_cnt = 0
    total_packets_cnt = 0

    for flow_file in files:
        tag = is_vpn_file(flow_file)
        # Decide inclusion
        if vpn_mode is True:
            if tag is False:   # explicitly non-vpn
                continue
            if tag is None and "vpn" not in flow_file.stem.lower():
                continue
        else:
            if tag is True:    # explicitly vpn
                continue
            if tag is None and "vpn" in flow_file.stem.lower():
                continue

        app = infer_app_from_stem(flow_file.stem)
        if not app:
            continue

        rich = load_flow_pickle(flow_file)
        if not rich:
            continue

        # For CSV rows and overlap key, reduce to (proto, a_port, b_port)
        rows = []
        for (proto, (a_ip, a_port), (b_ip, b_port)), st in rich.items():
            # Build overlap key
            red_key = (proto, int(a_port), int(b_port))
            flows_by_app[app][red_key] = flows_by_app[app].get(red_key, 0) + int(st.get("total_packets", 0))

            # Accumulate overall counts
            total_flows_cnt += 1
            total_packets_cnt += int(st.get("total_packets", 0))

            # CSV row (full stats; drop IPs)
            rows.append({
                "protocol": proto,
                "a_port": int(a_port),
                "b_port": int(b_port),
                "duration": float(st.get("duration", 0.0)),
                "total_packets": int(st.get("total_packets", 0)),
                "total_bytes": int(st.get("total_bytes", 0)),
                "packets_a_to_b": int(st.get("packets_a_to_b", 0)),
                "packets_b_to_a": int(st.get("packets_b_to_a", 0)),
                "bytes_a_to_b": int(st.get("bytes_a_to_b", 0)),
                "bytes_b_to_a": int(st.get("bytes_b_to_a", 0)),
                "avg_pkt_size": float(st.get("avg_pkt_size", 0.0)),
                "std_pkt_size": float(st.get("std_pkt_size", 0.0)),
                "mean_iat": float(st.get("mean_iat", 0.0)),
                "std_iat": float(st.get("std_iat", 0.0)),
                "syn": int(st.get("syn", 0)),
                "ack": int(st.get("ack", 0)),
                "fin": int(st.get("fin", 0)),
                "rst": int(st.get("rst", 0)),
                "psh": int(st.get("psh", 0)),
                "urg": int(st.get("urg", 0)),
                "application": app,
                "is_vpn": int(vpn_mode),
            })

        if export_app_csv(app, rows, vpn_mode):
            csv_exported += 1

    # Overlap / uniqueness stats
    n = len(APP_NAMES)
    overlapping_by_app = {a: {} for a in APP_NAMES}
    unique_flows_by_app = {a: 0.0 for a in APP_NAMES}
    unambiguous_packets_by_app = {a: 0.0 for a in APP_NAMES}

    n_flows_total = 0
    n_packets_total = 0
    n_unique_flows_total = 0
    n_unambig_packets_total = 0

    for i, app1 in enumerate(APP_NAMES):
        flows1 = flows_by_app[app1]
        if not flows1:
            continue
        n_flows_this_app = len(flows1)
        n_packets_this_app = sum(flows1.values())

        overlaps = set()
        ambig_packets_sum = 0
        for j, app2 in enumerate(APP_NAMES):
            if i == j:
                continue
            flows2 = flows_by_app[app2]
            if not flows2:
                continue
            common = set(flows1.keys()) & set(flows2.keys())
            overlaps |= common
            # count ambiguous packets by flows shared
            ambig_packets_sum += sum(flows1[k] for k in common)

        n_non_unique = len(overlaps)
        n_unique = n_flows_this_app - n_non_unique
        unique_ratio = (n_unique / n_flows_this_app) if n_flows_this_app else 0.0

        unambig_packets = n_packets_this_app - ambig_packets_sum
        unambig_ratio = (unambig_packets / n_packets_this_app) if n_packets_this_app else 0.0

        unique_flows_by_app[app1] = unique_ratio
        unambiguous_packets_by_app[app1] = unambig_ratio

        n_flows_total += n_flows_this_app
        n_packets_total += n_packets_this_app
        n_unique_flows_total += n_unique
        n_unambig_packets_total += unambig_packets

    summary = {
        "csv_exported": csv_exported,
        "n_flows_total": n_flows_total,
        "n_packets_total": n_packets_total,
        "unique_ratio_all": (n_unique_flows_total / n_flows_total) if n_flows_total else 0.0,
        "unambig_ratio_all": (n_unambig_packets_total / n_packets_total) if n_packets_total else 0.0,
        "unique_flows_by_app": unique_flows_by_app,
        "unambiguous_packets_by_app": unambiguous_packets_by_app,
    }

    # Summary print
    print(f"[INFO] {'VPN' if vpn_mode else 'Non-VPN'}: "
          f"{summary['unique_ratio_all']:.2%} flows unique, "
          f"{summary['unambig_ratio_all']:.2%} packets unambiguous. "
          f"Exported {csv_exported} CSVs.")

    return summary

def render_summary_table(summ_nonvpn: dict, summ_vpn: dict):
    # Build a simple combined table (averaging per-app VPN/Non-VPN)
    col_labels = ("Application", "Unique flows (avg)", "Unambiguous packets (avg)")
    rows = []
    for human, app in NAME_MAPPING.items():
        u1 = summ_nonvpn["unique_flows_by_app"].get(app, 0.0)
        u2 = summ_vpn["unique_flows_by_app"].get(app, 0.0)
        p1 = summ_nonvpn["unambiguous_packets_by_app"].get(app, 0.0)
        p2 = summ_vpn["unambiguous_packets_by_app"].get(app, 0.0)
        rows.append([human, round((u1 + u2) / 2, 2), round((p1 + p2) / 2, 2)])

    table_content = np.array(rows, dtype=object)

    fig = plt.figure(figsize=(9, 10))
    ax = plt.gca()
    the_table = ax.table(cellText=table_content, colLabels=col_labels, loc="center")
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(9)
    ax.axis("off")
    plt.tight_layout()
    out_path = OUT_DIR / "flow_summary_table.png"
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[OK] Saved flow summary table to {out_path}")

def main():
    # Clean run messages
    all_p = sorted(FLOWS_DIR.glob("*.p"))
    if not all_p:
        print(f"[ERROR] No .p files found in {FLOWS_DIR.resolve()}. Run extract_flows_from_pcap.py first.")
        sys.exit(1)

    print(f"[INFO] Found {len(all_p)} .p files in {FLOWS_DIR.resolve()}")

    # Process both passes automatically
    summ_vpn    = analyze_and_export(vpn_mode=True)
    summ_nonvpn = analyze_and_export(vpn_mode=False)

    # Summary table
    render_summary_table(summ_nonvpn, summ_vpn)

if __name__ == "__main__":
    main()
