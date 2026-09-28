#!/usr/bin/env python3
"""
resolve_country.py — map a Vaxtor country code to its OCR weight records in ocr_data.bin.

Usage:
    python resolve_country.py ocr_data.bin 1125
    python resolve_country.py ocr_data.bin 1125 --extract out_dir
    python resolve_country.py ocr_data.bin --list           # list all groups + members

Background (see model_weights.md):
  * ocr_data.bin is a container of 87 obfuscated records (keyless position cipher).
  * libvaxtorocr holds a model catalog: 10 named OCR "country groups". Each group owns
    one country-list record enumerating the codes it serves (terminated by 9999),
    plus one or two MNN model records and a glyph-template record.
  * To resolve a code: find the group whose country-list contains it, then use that
    group's model record(s).

The group -> {country-list, models, glyph} mapping below was recovered by static analysis
(Ghidra) of the catalog initializer FUN_001358c0 in libvaxtorocr.so.13.
"""
import sys, os, struct, re, argparse

# entry name (representative country code) -> record indices
GROUPS = [
    {"name": "2033", "idlist": 24, "models": [50, 23], "glyph": 25},
    {"name": "1195", "idlist": 27, "models": [26],     "glyph": 28},
    {"name": "1044", "idlist": 45, "models": [44],     "glyph": 46},
    {"name": "2207", "idlist": 33, "models": [53, 32], "glyph": 34},
    {"name": "1187", "idlist": 30, "models": [57, 29], "glyph": 31},
    {"name": "1110", "idlist": 36, "models": [55, 35], "glyph": 37},
    {"name": "1106", "idlist": 39, "models": [60, 38], "glyph": 40},
    {"name": "4001", "idlist": 42, "models": [41],     "glyph": 43},
    {"name": "4101", "idlist": 83, "models": [82],     "glyph": 84},
    {"name": "4416", "idlist": 48, "models": [47],     "glyph": 49},
]


def load_container(path):
    """Return list of deobfuscated record byte strings."""
    data = open(path, "rb").read()
    n = struct.unpack_from("<I", data, 0x20)[0]
    sizes = list(struct.unpack_from("<%dI" % n, data, 0x24))
    off = 0x24 + n * 8
    recs = []
    for size in sizes:
        recs.append(deobf(data[off:off + size], size))
        off += size
    return recs


def deobf(buf, size):
    out = bytearray(buf)
    for i in range(size):
        if (i & 1) == 0:
            t = size - i
            out[i] ^= ((t + t // 255) & 0xff)
        else:
            out[i] = (~(out[i] ^ ((i + i // 255) & 0xff))) & 0xff
    return bytes(out)


def codes_in(rec_bytes):
    """Country codes listed in an id-list record (drops the 9999 sentinel)."""
    toks = re.findall(rb"\b\d{3,4}\b", rec_bytes)
    return [t.decode() for t in toks if t != b"9999"]


def sniff_format(b):
    if b[:1] == b"\x08" and (b"pytorch" in b[:64] or b"onnx" in b[:256]):
        return "onnx"
    if b[:4] in (b"\x20\x00\x00\x00", b"\x1c\x00\x00\x00"):
        return "mnn"
    return "bin"


def resolve(recs, code):
    """Return list of matching groups for a country code."""
    hits = []
    for g in GROUPS:
        if code in codes_in(recs[g["idlist"]]):
            hits.append(g)
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ocr_data")
    ap.add_argument("code", nargs="?", help="country code, e.g. 1125")
    ap.add_argument("--extract", metavar="DIR", help="write resolved model records to DIR")
    ap.add_argument("--list", action="store_true", help="list all groups and their members")
    args = ap.parse_args()

    recs = load_container(args.ocr_data)

    if args.list:
        for g in GROUPS:
            members = codes_in(recs[g["idlist"]])
            print("group %-5s  models=%s glyph=%d  (%d countries): %s"
                  % (g["name"], g["models"], g["glyph"], len(members), ",".join(members)))
        return

    if not args.code:
        ap.error("provide a country code, or use --list")

    hits = resolve(recs, args.code)
    if not hits:
        print("country code %s not found in any OCR group." % args.code)
        print("(use --list to see all supported codes)")
        sys.exit(1)

    for g in hits:
        print("country %s -> group '%s'" % (args.code, g["name"]))
        for r in g["models"]:
            b = recs[r]
            print("   OCR model  : record %2d  (%d bytes, %s)" % (r, len(b), sniff_format(b)))
        print("   glyph tmpl : record %2d  (%d bytes)" % (g["glyph"], len(recs[g["glyph"]])))
        print("   country set: record %2d" % g["idlist"])

        if args.extract:
            os.makedirs(args.extract, exist_ok=True)
            for r in g["models"]:
                b = recs[r]
                ext = {"onnx": "onnx", "mnn": "mnn"}.get(sniff_format(b), "bin")
                fn = os.path.join(args.extract, "group_%s_model_rec%02d.%s" % (g["name"], r, ext))
                open(fn, "wb").write(b)
                print("   wrote %s" % fn)


if __name__ == "__main__":
    main()
