#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_to_csv.py

Direct PCAP -> CSV (per app & class), skipping pickle stage.
Outputs into ./datasets/csvs/<app>_(vpn|nonvpn).csv  (appended safely with header)

Features per flow:
  protocol, a_port, b_port,
  duration, total_packets, total_bytes,
  packets_a_to_b, packets_b_to_a, bytes_a_to_b, bytes_b_to_a,
  avg_pkt_size, std_pkt_size, mean_iat, std_iat,
  syn, ack, fin, rst, psh, urg,
  application, is_vpn
"""

import argparse
from pathlib import Path
from collections import defaultdict
import csv
import time

from scapy.layers.dns import DNS
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6
from scapy.utils import PcapReader

CSV_DIR = Path("./datasets/csvs")
PCAP_DIR = Path("./datasets/pcaps")
CSV_DIR.mkdir(parents=True, exist_ok=True)

APP_NAMES = [
    "torrent", "tor", "aim", "email", "facebook", "ftps", "gmail", "hangout",
    "icq", "netflix", "scp", "sftp", "skype", "spotify", "voipbuster", "vimeo", "youtube",
]

CSV_COLS = [
    "protocol","a_port","b_port",
    "duration","total_packets","total_bytes",
    "packets_a_to_b","packets_b_to_a","bytes_a_to_b","bytes_b_to_a",
    "avg_pkt_size","std_pkt_size","mean_iat","std_iat",
    "syn","ack","fin","rst","psh","urg",
    "application","is_vpn",
]

def infer_app_from_name(stem: str) -> str | None:
    s = stem.lower()
    for app in APP_NAMES:
        if app in s:
            return app
    return None

def infer_vpn_from_name(stem: str) -> int:
    """
    Robust substring logic (no regex word boundaries).
    - If name contains nonvpn/no-vpn/non_vpn/benign/normal => non-VPN (0)
    - If name contains 'vpn' anywhere => VPN (1)
    - Else (no tag) => assume non-VPN (0)
    """
    s = stem.lower()
    if ("nonvpn" in s) or ("non-vpn" in s) or ("non_vpn" in s) or ("no-vpn" in s) or ("no_vpn" in s) \
       or ("benign" in s) or ("normal" in s):
        return 0
    if "vpn" in s:
        return 1
    return 0

def canonicalize(a_ip: str, a_port: int, b_ip: str, b_port: int):
    """Return canonical (A,B,is_forward) where A/B are ((ip,port), ...) and
    is_forward=True if original packet direction is A->B."""
    A = (a_ip, int(a_port))
    B = (b_ip, int(b_port))
    if A <= B:  # lexicographic
        return A, B, True
    else:
        return B, A, False

def finalize_mean_std(n, mean, M2):
    if n <= 1:
        return float(mean), 0.0
    # population std (pstdev)
    return float(mean), float((M2 / n) ** 0.5)

def process_pcap_to_rows(pcap: Path, app: str, vpn_flag: int) -> list[dict]:
    """Aggregate flows, return list of row dicts for CSV."""
    # per-flow accumulators (bi-directional, canonicalized)
    flow = lambda: {
        "first_ts": None, "last_ts": None,
        "total_packets": 0, "total_bytes": 0,
        "a_to_b_pkts": 0, "b_to_a_pkts": 0,
        "a_to_b_bytes": 0, "b_to_a_bytes": 0,
        # Welford for packet sizes
        "_sz_n": 0, "_sz_mean": 0.0, "_sz_M2": 0.0,
        # Welford for IAT (overall)
        "_iat_n": 0, "_iat_mean": 0.0, "_iat_M2": 0.0,
        "_last_ts": None,
        # TCP flags
        "syn": 0, "ack": 0, "fin": 0, "rst": 0, "psh": 0, "urg": 0,
    }
    agg = defaultdict(flow)

    count = 0
    with PcapReader(str(pcap)) as reader:
        for pkt in reader:
            count += 1
            if (count % 200000) == 0:
                print(f"  … {count:,} packets in {pcap.name}")

            ts = float(getattr(pkt, "time", time.time()))

            # endpoints
            if pkt.haslayer(IP):
                src_ip, dst_ip = pkt[IP].src, pkt[IP].dst
            elif pkt.haslayer(IPv6):
                src_ip, dst_ip = pkt[IPv6].src, pkt[IPv6].dst
            else:
                continue

            # transport (skip DNS-over-UDP)
            if pkt.haslayer(TCP):
                sport, dport, proto = pkt[TCP].sport, pkt[TCP].dport, "TCP"
                flags = int(pkt[TCP].flags)
            elif pkt.haslayer(UDP) and not pkt.haslayer(DNS):
                sport, dport, proto = pkt[UDP].sport, pkt[UDP].dport, "UDP"
                flags = 0
            else:
                continue

            (A_ip, A_port), (B_ip, B_port), fwd = canonicalize(src_ip, sport, dst_ip, dport)
            key = (proto, (A_ip, A_port), (B_ip, B_port))
            st = agg[key]

            # timing
            if st["_last_ts"] is not None:
                iat = ts - st["_last_ts"]
                if iat >= 0:
                    st["_iat_n"] += 1
                    d = iat - st["_iat_mean"]
                    st["_iat_mean"] += d / st["_iat_n"]
                    st["_iat_M2"]   += d * (iat - st["_iat_mean"])
            st["_last_ts"] = ts
            st["first_ts"] = ts if st["first_ts"] is None else st["first_ts"]
            st["last_ts"]  = ts

            size = int(len(pkt))
            st["total_packets"] += 1
            st["total_bytes"]   += size
            # Welford size
            st["_sz_n"] += 1
            delta = size - st["_sz_mean"]
            st["_sz_mean"] += delta / st["_sz_n"]
            st["_sz_M2"]   += delta * (size - st["_sz_mean"])

            if fwd:
                st["a_to_b_pkts"]  += 1
                st["a_to_b_bytes"] += size
            else:
                st["b_to_a_pkts"]  += 1
                st["b_to_a_bytes"] += size

            if proto == "TCP":
                st["syn"] += 1 if flags & 0x02 else 0
                st["ack"] += 1 if flags & 0x10 else 0
                st["fin"] += 1 if flags & 0x01 else 0
                st["rst"] += 1 if flags & 0x04 else 0
                st["psh"] += 1 if flags & 0x08 else 0
                st["urg"] += 1 if flags & 0x20 else 0

    # finalize -> rows
    rows = []
    for (proto, (A_ip, A_port), (B_ip, B_port)), st in agg.items():
        duration = max((st["last_ts"] - st["first_ts"]) if st["last_ts"] else 0.0, 0.0)
        avg_sz, std_sz    = finalize_mean_std(st["_sz_n"],  st["_sz_mean"],  st["_sz_M2"])
        mean_iat, std_iat = finalize_mean_std(st["_iat_n"], st["_iat_mean"], st["_iat_M2"])

        rows.append({
            "protocol": proto,
            "a_port": int(A_port),
            "b_port": int(B_port),
            "duration": float(duration),
            "total_packets": int(st["total_packets"]),
            "total_bytes": int(st["total_bytes"]),
            "packets_a_to_b": int(st["a_to_b_pkts"]),
            "packets_b_to_a": int(st["b_to_a_pkts"]),
            "bytes_a_to_b": int(st["a_to_b_bytes"]),
            "bytes_b_to_a": int(st["b_to_a_bytes"]),
            "avg_pkt_size": float(avg_sz),
            "std_pkt_size": float(std_sz),
            "mean_iat": float(mean_iat),
            "std_iat": float(std_iat),
            "syn": int(st["syn"]),
            "ack": int(st["ack"]),
            "fin": int(st["fin"]),
            "rst": int(st["rst"]),
            "psh": int(st["psh"]),
            "urg": int(st["urg"]),
            "application": app,
            "is_vpn": int(vpn_flag),
        })
    return rows

def append_rows(csv_path: Path, rows: list[dict]):
    header_needed = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        if header_needed:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, 0) for k in CSV_COLS})

def process_one(pcap: Path):
    app = infer_app_from_name(pcap.stem)
    if not app:
        print(f"[SKIP] {pcap.name}: cannot infer app from filename")
        return
    vpn_flag = infer_vpn_from_name(pcap.stem)

    out_csv = CSV_DIR / f"{app}_{'vpn' if vpn_flag else 'nonvpn'}.csv"
    print(f"[PROCESS] {pcap.name} -> {out_csv.name}")
    rows = process_pcap_to_rows(pcap, app, vpn_flag)
    append_rows(out_csv, rows)
    print(f"[OK] {len(rows)} flows appended to {out_csv.name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, help="Process just this single .pcap path (enables easy parallel).")
    args = ap.parse_args()

    if args.only:
        p = Path(args.only)
        if not p.exists():
            print(f"[ERROR] Not found: {p}")
            return
        process_one(p)
        return

    pcaps = sorted(PCAP_DIR.glob("*.pcap"))
    if not pcaps:
        print(f"[ERROR] No .pcap files in {PCAP_DIR.resolve()}")
        return

    for p in pcaps:
        process_one(p)

if __name__ == "__main__":
    main()
