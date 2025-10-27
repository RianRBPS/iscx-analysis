#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_flows_from_pcap.py

Reads PCAPs in ./datasets/pcaps and writes per-flow aggregates to ./datasets/flows/*.p
Each .p is a dict: {('TCP'|'UDP', (a_ip,a_port), (b_ip,b_port)) -> stats_dict}

The flow key is **bi-directional canonicalized**:
- We sort the two endpoints ((ip,port) tuples). The lexicographically smaller becomes A.
- For each packet we detect whether it was A->B or B->A and update the right counters.

stats_dict fields:
    duration               float (s)
    total_packets          int
    total_bytes            int
    packets_a_to_b         int
    packets_b_to_a         int
    bytes_a_to_b           int
    bytes_b_to_a           int
    avg_pkt_size           float
    std_pkt_size           float
    mean_iat               float   (overall inter-arrival, seconds)
    std_iat                float
    syn, ack, fin, rst, psh, urg   (TCP flag counts, overall)
"""

import logging
import pickle
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

from scapy.layers.dns import DNS
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6
from scapy.utils import PcapReader

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PCAP_DIR = Path("./datasets/pcaps")
OUT_DIR = Path("./datasets/flows")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def canonicalize(a_ip: str, a_port: int, b_ip: str, b_port: int):
    """Return canonical (A,B,is_forward) where A/B are ((ip,port), ...) and
    is_forward=True if original packet direction is A->B."""
    A = (a_ip, int(a_port))
    B = (b_ip, int(b_port))
    if A <= B:  # lexicographic
        return A, B, True
    else:
        return B, A, False

def process_pcap(pcap_path: Path) -> dict:
    """
    Returns a dict mapping flow_key -> stats_dict
    flow_key = (proto, A_endpoint, B_endpoint), where endpoints are (ip,port)
    """
    flow = lambda: {
        "first_ts": None, "last_ts": None,
        "total_packets": 0, "total_bytes": 0,
        "packets_a_to_b": 0, "packets_b_to_a": 0,
        "bytes_a_to_b": 0, "bytes_b_to_a": 0,
        "sizes": [],
        "iat_all": [],  # overall inter-arrival (regardless of direction)
        # TCP flags (overall)
        "syn": 0, "ack": 0, "fin": 0, "rst": 0, "psh": 0, "urg": 0,
        "_last_ts_for_iat": None,
    }
    agg = defaultdict(flow)

    with PcapReader(str(pcap_path)) as reader:
        for pkt in reader:
            ts = float(getattr(pkt, "time", time.time()))

            # Determine IP version and endpoints
            if pkt.haslayer(IP):
                src_ip, dst_ip = pkt[IP].src, pkt[IP].dst
            elif pkt.haslayer(IPv6):
                src_ip, dst_ip = pkt[IPv6].src, pkt[IPv6].dst
            else:
                continue

            # Transport & ports (skip DNS-over-UDP)
            if pkt.haslayer(TCP):
                sport, dport, proto = pkt[TCP].sport, pkt[TCP].dport, "TCP"
                flags = int(pkt[TCP].flags)
            elif pkt.haslayer(UDP) and not pkt.haslayer(DNS):
                sport, dport, proto = pkt[UDP].sport, pkt[UDP].dport, "UDP"
                flags = 0
            else:
                continue

            # Form canonical flow key and direction
            (A_ip, A_port), (B_ip, B_port), is_forward = canonicalize(src_ip, sport, dst_ip, dport)
            key = (proto, (A_ip, A_port), (B_ip, B_port))
            st = agg[key]

            # Timing (overall)
            if st["_last_ts_for_iat"] is not None:
                iat = ts - st["_last_ts_for_iat"]
                if iat >= 0:
                    st["iat_all"].append(iat)
            st["_last_ts_for_iat"] = ts
            st["first_ts"] = ts if st["first_ts"] is None else st["first_ts"]
            st["last_ts"] = ts

            # Sizes/bytes/packets
            size = int(len(pkt))
            st["total_packets"] += 1
            st["total_bytes"] += size
            st["sizes"].append(size)

            if is_forward:
                st["packets_a_to_b"] += 1
                st["bytes_a_to_b"] += size
            else:
                st["packets_b_to_a"] += 1
                st["bytes_b_to_a"] += size

            # TCP flags (overall)
            if proto == "TCP":
                st["syn"] += 1 if flags & 0x02 else 0
                st["ack"] += 1 if flags & 0x10 else 0
                st["fin"] += 1 if flags & 0x01 else 0
                st["rst"] += 1 if flags & 0x04 else 0
                st["psh"] += 1 if flags & 0x08 else 0
                st["urg"] += 1 if flags & 0x20 else 0

    # Finalize stats
    result = {}
    for key, st in agg.items():
        duration = max((st["last_ts"] - st["first_ts"]) if st["last_ts"] else 0.0, 0.0)
        sizes = st["sizes"]
        iats = st["iat_all"]

        avg_sz = mean(sizes) if sizes else 0.0
        std_sz = pstdev(sizes) if len(sizes) > 1 else 0.0
        mean_iat = mean(iats) if iats else 0.0
        std_iat = pstdev(iats) if len(iats) > 1 else 0.0

        result[key] = {
            "duration": duration,
            "total_packets": int(st["total_packets"]),
            "total_bytes": int(st["total_bytes"]),
            "packets_a_to_b": int(st["packets_a_to_b"]),
            "packets_b_to_a": int(st["packets_b_to_a"]),
            "bytes_a_to_b": int(st["bytes_a_to_b"]),
            "bytes_b_to_a": int(st["bytes_b_to_a"]),
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
        }

    return result

def main():
    pcaps = sorted(PCAP_DIR.glob("*.pcap"))
    if not pcaps:
        print(f"[ERROR] No .pcap files found in {PCAP_DIR.resolve()}")
        return

    for i, pcap in enumerate(pcaps, 1):
        out_path = OUT_DIR / f"{pcap.stem}_flows.p"
        if out_path.exists():
            print(f"[SKIP] {pcap.name} -> {out_path.name} (already exists)")
            continue

        print(f"[PROCESS] {i}/{len(pcaps)}: {pcap.name}")
        flows = process_pcap(pcap)
        print(f"[OK] {len(flows)} canonical bi-directional flows in {pcap.name}")

        with open(out_path, "wb") as f:
            pickle.dump(flows, f)

if __name__ == "__main__":
    main()
