"""Training crops for the CRNN: real plates cut out the way the pipeline does it, and rendered synthetic plates.

Real:  an image whose file name starts with the plate text (`UITM1776.jpeg`, `WXY123_2.jpg`) goes through the
       default pipeline (rec_72 detector -> rec_71 corners -> quad warp -> 96x48), with the box, corners and
       padding jittered so one photo gives many slightly different crops.
Synth: plate text in the Malaysian formats (including the special series the stock model was never taught:
       letters I and O, long word prefixes) rendered with system fonts and degraded like a CCTV frame.

Both return 96x48 RGB PIL images; `to_tensor` turns them into the CRNN input.
"""
import glob
import io
import os
import re
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

W, H = 96, 48
ALNUM = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_LEN = 12  # the longest plate the 18 CTC steps have been seen to read (PUTRAJAYA541)


def to_tensor(img):
    """96x48 RGB PIL -> (3, 48, 96) BGR in [0, 1], the CRNN input."""
    return np.ascontiguousarray(np.asarray(img, np.float32)[..., ::-1].transpose(2, 0, 1) / 255)


def label_of(path):
    """Plate text from a file name: the leading run of letters and digits, upper-cased."""
    m = re.match(r"[A-Za-z0-9]+", os.path.basename(path))
    return m.group(0).upper() if m else ""


# ----------------------------------------------------------------------------------------------- real plates

class Rectifier:
    """The pipeline's crop: detector box -> rec_71 corner quad -> 192x96 warp -> 96x48 (see make_goldens.py)."""

    def __init__(self):
        import decode_samples as ds
        import read_plate as rp
        self.det, self.corners = rp.Detector(), ds.Net("71")

    def box(self, img):
        return self.det(img)

    def crop(self, img, box, rng=None, pad=0.03, margin=(0.2, 0.5)):
        """rng=None gives exactly the pipeline's crop; with an rng the geometry is jittered."""
        x1, y1, x2, y2 = box[:4]
        bw, bh = x2 - x1, y2 - y1
        mx, my = margin
        if rng is not None:
            x1, x2 = x1 + rng.normal(0, 0.015) * bw, x2 + rng.normal(0, 0.015) * bw
            y1, y2 = y1 + rng.normal(0, 0.03) * bh, y2 + rng.normal(0, 0.03) * bh
            mx, my, pad = mx * rng.uniform(0.8, 1.2), my * rng.uniform(0.8, 1.2), rng.uniform(0.0, 0.08)
        cx1, cy1, cx2, cy2 = x1 - mx * bw, y1 - my * bh, x2 + mx * bw, y2 + my * bh
        ctx = img.crop((cx1, cy1, cx2, cy2)).resize((W, H), Image.BILINEAR)
        a = np.asarray(ctx, np.float32)[..., ::-1] / 127.5 - 1
        p = self.corners(np.ascontiguousarray(a.transpose(2, 0, 1)[None]))
        rx1, ry1 = round(cx1), round(cy1)
        rw, rh = round(cx2) - rx1, round(cy2) - ry1
        q = np.array([[rx1 + p[2 * i] * rw, ry1 + p[2 * i + 1] * rh] for i in range(4)])
        if rng is not None:
            q += rng.normal(0, 0.012, q.shape) * [bw, bh]
        q = q.mean(0) + (q - q.mean(0)) * (1 + pad)
        warped = img.transform((2 * W, 2 * H), Image.QUAD, tuple(float(v) for v in q.reshape(-1)), Image.BILINEAR)
        return warped.resize((W, H), Image.BILINEAR)


def load_real(folder, rectifier, log=print):
    """[(label, full image, detector box)] for every labelled image in a folder that has a plate."""
    out = []
    for path in sorted(glob.glob(os.path.join(folder, "*"))):
        if not path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            continue
        label = label_of(path)
        if not label or len(label) > MAX_LEN or any(c not in ALNUM for c in label):
            log(f"  skipped {os.path.basename(path)}: no usable label in the file name")
            continue
        img = Image.open(path).convert("RGB")
        box = rectifier.box(img)
        if box is None:
            log(f"  skipped {os.path.basename(path)}: the detector found no plate")
            continue
        out.append((label, img, box))
    return out


# ------------------------------------------------------------------------------------------ synthetic plates

FONT_DIRS = ["/System/Library/Fonts/Supplemental", "/System/Library/Fonts", "/Library/Fonts",
             "/usr/share/fonts", "/usr/local/share/fonts", "C:/Windows/Fonts"]
# bold sans faces close to plate lettering; the first names found on this machine are used
FONT_NAMES = ["Arial Bold.ttf", "Arial Narrow Bold.ttf", "Arial Black.ttf", "Arial Rounded Bold.ttf",
              "Arial Bold Italic.ttf", "Arial Narrow Bold Italic.ttf", "DIN Alternate Bold.ttf",
              "DIN Condensed Bold.ttf", "Tahoma Bold.ttf", "Verdana Bold.ttf", "Trebuchet MS Bold.ttf",
              "Impact.ttf", "Arial.ttf", "Arial Narrow.ttf",
              "arialbd.ttf", "arialnb.ttf", "ariblk.ttf", "tahomabd.ttf", "verdanab.ttf", "impact.ttf",
              "DejaVuSans-Bold.ttf", "DejaVuSansCondensed-Bold.ttf", "LiberationSans-Bold.ttf",
              "LiberationSansNarrow-Bold.ttf", "FreeSansBold.ttf"]

# special series: word prefixes the standard-format prior does not cover
SPECIAL = ["PUTRAJAYA", "UITM", "UKM", "UPM", "UTM", "USM", "UUM", "UNIMAS", "UMS", "UMK", "UMT", "UNISZA", "UPSI",
           "UTEM", "UMP", "UTHM", "UNIMAP", "UPNM", "USIM", "IIUM", "UIM", "UNITEN", "MMU", "UNIKL", "UTP", "UCSI",
           "PROTON", "PERODUA", "SUKOM", "BAMBEE", "PATRIOT", "PERFECT", "MALAYSIA", "RIMAU", "NAAM", "XOIC",
           "XIIINAM", "G1M", "1M4U", "IM4U", "T1M", "A1M", "K1M", "NBOS", "GIM", "VIP", "LIMO", "KLIA", "PETRA",
           "GOLD", "SAM", "CHANCELLOR", "PERSONA", "SATRIA", "TIARA", "WAJA", "KRISS", "JAGUH", "LOTUS", "FF",
           "GG", "GT", "GP", "FFF", "US", "YES", "SMS", "MADANI", "TTB", "ALFA", "EV", "ASEAN", "GTR", "KIA",
           "IQ", "IO", "OO", "II", "OK"]
STANDARD_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # the ordinary series skips I and O


def sample_text(rng, exclude=()):
    """A plausible Malaysian plate. `exclude` lists prefixes never to produce (to hold a series out)."""
    for _ in range(1000):
        r = rng.random()
        digits = str(rng.integers(1, 10 ** rng.choice([1, 2, 3, 4, 4, 4])))
        if rng.random() < 0.3:  # numbers full of 1s (811, 2111, 110): on these plates a 1 is a bare stroke
            digits = "".join("1" if rng.random() < 0.5 else c for c in digits)
        if r < 0.45:  # ordinary: 1-3 letters, 1-4 digits, sometimes a suffix letter
            letters = "".join(rng.choice(list(STANDARD_LETTERS), rng.choice([1, 2, 3, 3, 3])))
            text = letters + digits + (rng.choice(list(STANDARD_LETTERS)) if rng.random() < 0.15 else "")
        elif r < 0.65:  # special series
            text = SPECIAL[rng.integers(len(SPECIAL))] + digits
        elif r < 0.97:  # any letters, I and O as likely as the rest - breaks the "never I" prior. Still letters
            pool = list(ALNUM[10:]) + ["I", "O"] * 3  # first and number last: position is all that tells I from 1
            text = "".join(rng.choice(pool, rng.integers(1, 7))) + digits
        else:  # a little free-form, so foreign layouts are not unlearned
            text = "".join(rng.choice(list(ALNUM), rng.integers(2, 10)))
        if len(text) <= MAX_LEN and not any(text.startswith(p) for p in exclude):
            return text
    raise RuntimeError("could not sample a plate text")


def split_groups(text):
    """Where the gap goes on a plate: between the leading letters and the number."""
    m = re.match(r"([A-Z]+)(\d+)([A-Z]?)$", text)
    return [m.group(1), m.group(2) + m.group(3)] if m and m.group(1) else [text]


class Synth:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        found = {}
        for d in FONT_DIRS:
            for name in FONT_NAMES:
                for p in glob.glob(os.path.join(d, "**", name), recursive=True)[:1]:
                    found.setdefault(name, p)
        if not found:
            raise RuntimeError("no plate-like TrueType fonts found; add a font directory to FONT_DIRS")
        self.fonts = list(found.values())
        self._cache = {}

    def font(self, path, size):
        key = (path, size)
        if key not in self._cache:
            self._cache[key] = ImageFont.truetype(path, size)
        return self._cache[key]

    def render(self, text):
        """One degraded 96x48 crop of a plate showing `text`."""
        rng = self.rng
        S = 4  # supersampling
        cw, ch = W * S, H * S
        dark, light = rng.uniform(0, 70), rng.uniform(150, 255)
        if rng.random() < 0.08:  # a few black-on-white plates
            dark, light = light, dark
        path = self.fonts[rng.integers(len(self.fonts))]
        groups = split_groups(text)
        two_rows = len(groups) == 2 and rng.random() < 0.12
        lines = groups if two_rows else [(" " if rng.random() < 0.85 else "").join(groups)]
        # Malaysian plates draw 1 as a plain stroke, the same shape as I; fonts give it a flag, so borrow the I
        bar_one = rng.random() < 0.6

        # the plate itself fills most of the crop; the rest is bumper, as in the rectified real crops
        pw, ph = cw * rng.uniform(0.80, 1.0), ch * (rng.uniform(0.75, 1.0) if two_rows else rng.uniform(0.50, 1.0))
        plate = Image.new("L", (int(pw), int(ph)), int(dark))
        d = ImageDraw.Draw(plate)
        line_h = ph / len(lines)
        spacing = rng.uniform(-0.04, 0.10)
        for li, line in enumerate(lines):
            size = int(line_h * rng.uniform(0.55, 0.92))
            f = self.font(path, max(size, 8))
            shapes = [("I" if c == "1" and bar_one else c) for c in line]
            widths = [d.textlength(c, font=f) for c in shapes]
            gap = spacing * size
            total = sum(widths) + gap * (len(line) - 1)
            squeeze = min(1.0, pw * rng.uniform(0.80, 0.95) / max(total, 1))
            glyphs = Image.new("L", (int(total) + 8, int(size * 1.4)), 0)
            gd = ImageDraw.Draw(glyphs)
            x = 4.0
            for c, wd in zip(shapes, widths):
                gd.text((x, size * 0.1), c, font=f, fill=255)
                x += wd + gap
            bbox = glyphs.getbbox()
            if bbox is None:
                continue
            glyphs = glyphs.crop(bbox)
            gw = max(2, int(glyphs.width * squeeze * (rng.uniform(0.75, 1.0) if squeeze == 1.0 else 1.0)))
            gh = max(2, min(int(line_h * 0.94), glyphs.height))
            glyphs = glyphs.resize((gw, gh), Image.BILINEAR)
            ox = int((pw - gw) * rng.uniform(0.3, 0.7))
            oy = int(li * line_h + (line_h - gh) * rng.uniform(0.3, 0.7))
            plate.paste(Image.new("L", glyphs.size, int(light)), (ox, oy), glyphs)
        if rng.random() < 0.6:  # raised border
            t = int(rng.integers(2, 7))
            d.rectangle([t, t, plate.width - t - 1, plate.height - t - 1],
                        outline=int(dark + (light - dark) * rng.uniform(0.2, 0.8)), width=int(rng.integers(1, 4)))

        canvas = Image.new("L", (cw, ch), int(rng.uniform(20, 200)))
        canvas.paste(plate, (int((cw - plate.width) * rng.uniform(0.2, 0.8)), int((ch - plate.height) * rng.uniform(0.2, 0.8))))

        # residual skew after rectification: small rotation / italic shear / perspective
        sh, rot = rng.normal(0, 0.10), rng.normal(0, 0.03)
        if rng.random() < 0.25:
            sh += rng.uniform(0.15, 0.4)  # italic plate fonts lean right
        k1, k2 = rng.normal(0, 0.00008, 2)
        cx, cy = cw / 2, ch / 2
        coeffs = (1, -sh, sh * cy, rot, 1, -rot * cx, k1 / S, k2 / S)
        canvas = canvas.transform((cw, ch), Image.PERSPECTIVE, coeffs, Image.BILINEAR, fillcolor=int(rng.uniform(20, 200)))

        canvas = canvas.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.6) * S))
        # the real crops are upscaled from plates 60-300 px wide: lose that detail first
        small_w = int(rng.uniform(60, 220))
        img = canvas.resize((small_w, max(8, small_w // 2)), Image.BILINEAR).resize((W, H), Image.BILINEAR)
        a = np.asarray(img, np.float32)
        a = (a - 128) * rng.uniform(0.7, 1.3) + 128 + rng.uniform(-30, 30)
        a += rng.normal(0, rng.uniform(0, 8), a.shape)
        if rng.random() < 0.5:  # lighting gradient
            a *= np.linspace(rng.uniform(0.7, 1.0), rng.uniform(0.7, 1.0), W)[None, :]
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        if rng.random() < 0.7:
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=int(rng.integers(30, 95)))
            img = Image.open(buf)
        rgb = np.repeat(np.asarray(img.convert("L"))[..., None], 3, -1).astype(np.float32)
        if rng.random() < 0.3:  # colour cameras: a slight tint
            rgb *= rng.uniform(0.85, 1.0, 3)
        return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))

    def sample(self, exclude=()):
        text = sample_text(self.rng, exclude)
        return text, self.render(text)


def augment_real(img, rng):
    """Photometric jitter for real crops (the geometry is jittered by Rectifier.crop)."""
    a = np.asarray(img, np.float32)
    a = (a - 128) * rng.uniform(0.8, 1.25) + 128 + rng.uniform(-20, 20)
    a += rng.normal(0, rng.uniform(0, 5), a.shape)
    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if rng.random() < 0.4:
        out = out.filter(ImageFilter.GaussianBlur(rng.uniform(0, 0.8)))
    if rng.random() < 0.5:
        buf = io.BytesIO()
        out.save(buf, "JPEG", quality=int(rng.integers(40, 95)))
        out = Image.open(buf).convert("RGB")
    return out


if __name__ == "__main__":  # contact sheet of synthetic plates, to eyeball against real crops
    s = Synth(1)
    sheet = Image.new("RGB", (W * 2 * 6, H * 2 * 8))
    for i in range(48):
        text, img = s.sample()
        sheet.paste(img.resize((W * 2, H * 2), Image.NEAREST), (i % 6 * W * 2, i // 6 * H * 2))
    sheet.save(sys.argv[1] if len(sys.argv) > 1 else "synth_sheet.png")
