"""Read the logged history (train-loss, learning-rate, ...) out of an offline wandb run.

Usage: python scripts/wandb_history.py wandb/offline-run-XXXX [out.csv]

Newer wandb releases dropped the Python reader for .wandb files, so this parses the
LevelDB-style log directly: a 7-byte file header, then 32 KiB blocks of records,
each with a 7-byte header (crc32, length, type) and a protobuf `Record` payload.
"""

import csv
import glob
import json
import struct
import sys

from wandb.proto import wandb_internal_pb2

BLOCK_LEN = 32768
HEADER_LEN = 7
FULL, FIRST, MIDDLE, LAST = 1, 2, 3, 4


def iter_records(path: str):
    with open(path, "rb") as f:
        data = f.read()
    ident, _magic, _version = struct.unpack("<4sHB", data[:HEADER_LEN])
    assert ident == b":W&B", f"{path} is not a wandb log"

    pos, pending = HEADER_LEN, b""
    while pos + HEADER_LEN <= len(data):
        if BLOCK_LEN - pos % BLOCK_LEN < HEADER_LEN:  # block trailer padding
            pos += BLOCK_LEN - pos % BLOCK_LEN
            continue
        _crc, length, rtype = struct.unpack("<IHB", data[pos : pos + HEADER_LEN])
        pos += HEADER_LEN
        chunk, pos = data[pos : pos + length], pos + length
        if rtype == FULL:
            yield chunk
        elif rtype == FIRST:
            pending = chunk
        elif rtype == MIDDLE:
            pending += chunk
        elif rtype == LAST:
            yield pending + chunk
            pending = b""
        else:  # zero padding at the end of the file
            break


def read_history(run_dir: str) -> list[dict]:
    rows = []
    for payload in iter_records(glob.glob(f"{run_dir}/*.wandb")[0]):
        rec = wandb_internal_pb2.Record()
        rec.ParseFromString(payload)
        if rec.WhichOneof("record_type") == "history":
            rows.append(
                {
                    (item.key or "/".join(item.nested_key)): json.loads(item.value_json)
                    for item in rec.history.item
                }
            )
    return rows


if __name__ == "__main__":
    rows = read_history(sys.argv[1])
    keys = sorted({k for r in rows for k in r})
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    print(f"{len(rows)} rows, keys: {keys}")
