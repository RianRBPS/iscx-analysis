"""Extrai features por fluxo a partir de arquivos .pcap e salva como CSV.

Features por fluxo incluem: duração, número total de pacotes, bytes, contagem por direção,
estatísticas de tamanho de pacote e de inter-arrival times, e flags TCP.

Uso: colocar arquivos .pcap em ./datasets/pcaps e rodar este script. Os arquivos CSV
serão gerados em ./datasets/flows_features/{pcap_stem}_flow_features.csv

Este script processa os pcaps em streaming (usa PcapReader iterável) para evitar alto
uso de memória em arquivos grandes.
"""

from collections import defaultdict
import csv
from pathlib import Path
from typing import Dict, Tuple, List

from scapy.utils import PcapReader
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6


def canonical_flow_key(src_ip: str, src_port: int, dst_ip: str, dst_port: int, proto: str):
    """Retorna uma chave canônica (direção-agnóstica) e um boolean indicando se
    a tupla original já está na ordem canônica.
    """
    a = (src_ip, src_port)
    b = (dst_ip, dst_port)
    if a <= b:
        return (proto, a, b), True
    else:
        return (proto, b, a), False


def process_pcap(pcap_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{pcap_path.stem}_flow_features.csv"

    # Accumuladores por flow
    flows: Dict[Tuple, Dict] = defaultdict(lambda: {
        "first_ts": None,
        "last_ts": None,
        "pkt_times": [],
        "pkt_sizes": [],
        "total_packets": 0,
        "total_bytes": 0,
        "packets_a_to_b": 0,
        "packets_b_to_a": 0,
        "bytes_a_to_b": 0,
        "bytes_b_to_a": 0,
        "syn": 0,
        "ack": 0,
        "fin": 0,
        "rst": 0,
        "psh": 0,
        "urg": 0,
        "protocol": "",
        "a": None,
        "b": None,
    })

    print(f"Processing {pcap_path}")
    reader = PcapReader(str(pcap_path))
    for packet in reader:
        # timestamp
        ts = getattr(packet, "time", None)

        # IP layer extraction
        if packet.haslayer(IP):
            src_ip = packet[IP].src
            dst_ip = packet[IP].dst
        elif packet.haslayer(IPv6):
            src_ip = packet[IPv6].src
            dst_ip = packet[IPv6].dst
        else:
            continue

        # Determine transport
        if packet.haslayer(TCP):
            proto = "TCP"
            sport = packet[TCP].sport
            dport = packet[TCP].dport
        elif packet.haslayer(UDP):
            proto = "UDP"
            sport = packet[UDP].sport
            dport = packet[UDP].dport
        else:
            continue

        # canonicalize flow key (direction-agnostic)
        key, is_a_to_b = canonical_flow_key(src_ip, sport, dst_ip, dport, proto)
        meta = flows[key]

        # Initialize protocol and endpoint info
        if meta["protocol"] == "":
            meta["protocol"] = proto
            meta["a"] = key[1]
            meta["b"] = key[2]

        # packet size (use total packet len)
        try:
            pkt_size = len(packet)
        except Exception:
            pkt_size = 0

        # timestamps
        if ts is not None:
            if meta["first_ts"] is None:
                meta["first_ts"] = ts
            meta["last_ts"] = ts
            meta["pkt_times"].append(ts)

        meta["pkt_sizes"].append(pkt_size)
        meta["total_packets"] += 1
        meta["total_bytes"] += pkt_size

        # direction-specific counts
        if is_a_to_b:
            meta["packets_a_to_b"] += 1
            meta["bytes_a_to_b"] += pkt_size
        else:
            meta["packets_b_to_a"] += 1
            meta["bytes_b_to_a"] += pkt_size

        # TCP flags
        if proto == "TCP":
            try:
                flags = int(packet[TCP].flags)
            except Exception:
                flags = 0
            # bit masks: FIN=0x01, SYN=0x02, RST=0x04, PSH=0x08, ACK=0x10, URG=0x20
            if flags & 0x02:
                meta["syn"] += 1
            if flags & 0x10:
                meta["ack"] += 1
            if flags & 0x01:
                meta["fin"] += 1
            if flags & 0x04:
                meta["rst"] += 1
            if flags & 0x08:
                meta["psh"] += 1
            if flags & 0x20:
                meta["urg"] += 1

    # Finished reading pcap; aggregate and write CSV
    fieldnames = [
        "flow_id",
        "protocol",
        "a_ip",
        "a_port",
        "b_ip",
        "b_port",
        "duration",
        "total_packets",
        "total_bytes",
        "packets_a_to_b",
        "packets_b_to_a",
        "bytes_a_to_b",
        "bytes_b_to_a",
        "avg_pkt_size",
        "std_pkt_size",
        "mean_iat",
        "std_iat",
        "syn",
        "ack",
        "fin",
        "rst",
        "psh",
        "urg",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for key, meta in flows.items():
            a = meta["a"]
            b = meta["b"]

            # duration
            if meta["first_ts"] is None or meta["last_ts"] is None:
                duration = 0.0
            else:
                duration = float(meta["last_ts"] - meta["first_ts"]) if meta["last_ts"] >= meta["first_ts"] else 0.0

            # packet size stats
            sizes = meta["pkt_sizes"]
            if sizes:
                avg_size = sum(sizes) / len(sizes)
                var_size = sum((x - avg_size) ** 2 for x in sizes) / len(sizes)
                std_size = var_size ** 0.5
            else:
                avg_size = 0.0
                std_size = 0.0

            # inter-arrival times
            iats: List[float] = []
            times = meta["pkt_times"]
            if len(times) >= 2:
                times_sorted = sorted(times)
                iats = [t2 - t1 for t1, t2 in zip(times_sorted[:-1], times_sorted[1:])]
                mean_iat = sum(iats) / len(iats)
                var_iat = sum((x - mean_iat) ** 2 for x in iats) / len(iats)
                std_iat = var_iat ** 0.5
            else:
                mean_iat = 0.0
                std_iat = 0.0

            row = {
                "flow_id": str(key),
                "protocol": meta["protocol"],
                "a_ip": a[0],
                "a_port": a[1],
                "b_ip": b[0],
                "b_port": b[1],
                "duration": round(duration, 6),
                "total_packets": meta["total_packets"],
                "total_bytes": meta["total_bytes"],
                "packets_a_to_b": meta["packets_a_to_b"],
                "packets_b_to_a": meta["packets_b_to_a"],
                "bytes_a_to_b": meta["bytes_a_to_b"],
                "bytes_b_to_a": meta["bytes_b_to_a"],
                "avg_pkt_size": round(avg_size, 2),
                "std_pkt_size": round(std_size, 2),
                "mean_iat": round(mean_iat, 6),
                "std_iat": round(std_iat, 6),
                "syn": meta["syn"],
                "ack": meta["ack"],
                "fin": meta["fin"],
                "rst": meta["rst"],
                "psh": meta["psh"],
                "urg": meta["urg"],
            }

            writer.writerow(row)

    print(f"Wrote features to {out_csv}")


def main():
    pcap_path = Path("./datasets/pcaps")
    out_dir = Path("./datasets/flows_features")

    if not pcap_path.exists():
        print("No pcaps directory found at ./datasets/pcaps. Coloque seus .pcap lá e rode de novo.")
        return

    for pcap_file in pcap_path.glob("*.pcap"):
        try:
            process_pcap(pcap_file, out_dir)
        except Exception as e:
            print(f"Erro processando {pcap_file}: {e}")


if __name__ == "__main__":
    main()
