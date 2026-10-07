"""The SDK's CRNN OCR models (rec_50..69) as a trainable PyTorch module, loaded from and written back to MNN.

The .bin is an MNN flatbuffer (converted TF -> ONNX -> MNN 2.5.1, weights stored as fp16). MNN's own
converter cannot export it and its tools phone home (see HANDOVER.md), so this reads the flatbuffer directly:

  MnnCrnn(path)        builds the network from the file: MobileNetV3 backbone -> 3x6 grid of 128-d features
                       -> 18-step BiLSTM(64) -> Dense(C) -> softmax. Matches MNN to ~1e-4 (see `verify`).
  model.export(path)   overwrites the weight bytes in a copy of the original file. The graph, tensor names
                       and file size do not change, so the result is a drop-in rec_NN.bin for the Android
                       app and lpr-api.

Input is the same as the SDK's: 96x48 BGR / 255, NCHW.
"""
import struct

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ALNUM = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# MNN.fbs OpType values and OpParameter-independent field slots (declaration order in the schema)
OP_BINARY, OP_CONST, OP_CONV, OP_CONV_DW, OP_INPUT, OP_POOL, OP_RELU6, OP_SOFTMAX, OP_RASTER, OP_WHILE = (
    7, 11, 12, 13, 34, 47, 70, 85, 128, 600)
BINARY = {0: torch.add, 1: torch.sub, 2: torch.mul, 3: torch.div, 7: torch.div, 8: torch.minimum, 9: torch.maximum}
QUAN_FP16 = 3


class Table:
    """Just enough of a flatbuffer reader: fields by slot number, with the byte offset of every vector."""

    def __init__(self, buf, pos):
        self.buf, self.pos = buf, pos
        self.vt = pos - struct.unpack_from("<i", buf, pos)[0]
        self.slots = (struct.unpack_from("<H", buf, self.vt)[0] - 4) // 2

    def _at(self, slot):
        if slot >= self.slots:
            return 0
        off = struct.unpack_from("<H", self.buf, self.vt + 4 + 2 * slot)[0]
        return self.pos + off if off else 0

    def scalar(self, slot, fmt="<i", default=0):
        p = self._at(slot)
        return struct.unpack_from(fmt, self.buf, p)[0] if p else default

    def table(self, slot):
        p = self._at(slot)
        return Table(self.buf, p + struct.unpack_from("<I", self.buf, p)[0]) if p else None

    def vector(self, slot):
        """(byte offset of the first element, element count), or (0, 0) when absent."""
        p = self._at(slot)
        if not p:
            return 0, 0
        p += struct.unpack_from("<I", self.buf, p)[0]
        return p + 4, struct.unpack_from("<I", self.buf, p)[0]

    def array(self, slot, dtype):
        start, n = self.vector(slot)
        return np.frombuffer(self.buf, dtype, n, start) if n else np.zeros(0, dtype)

    def tables(self, slot):
        start, n = self.vector(slot)
        return [Table(self.buf, p + struct.unpack_from("<I", self.buf, p)[0]) for p in range(start, start + 4 * n, 4)]

    def string(self, slot):
        start, n = self.vector(slot)
        return bytes(self.buf[start:start + n]).decode("utf-8", "replace")


class Stored:
    """Where a parameter lives in the file, so export can write it back in the same encoding."""

    def __init__(self, offset, count, dtype):
        self.offset, self.count, self.dtype = offset, count, np.dtype(dtype)

    def read(self, buf):
        return np.frombuffer(buf, self.dtype, self.count, self.offset).astype(np.float32)

    def write(self, buf, values):
        raw = np.ascontiguousarray(values, np.float32).reshape(-1).astype(self.dtype).tobytes()
        assert len(raw) == self.count * self.dtype.itemsize
        buf[self.offset:self.offset + len(raw)] = raw


def _stored_conv(buf, conv):
    """(weight, bias) locations of a Convolution2D: fp16 in quanParameter.buffer or plain float32."""
    quan = conv.table(3)
    if quan is not None and quan.vector(0)[1]:
        if quan.scalar(2) != QUAN_FP16:
            raise ValueError(f"conv weights use quantisation type {quan.scalar(2)}; only fp16 (3) is handled")
        start, n = quan.vector(0)
        weight = Stored(start, n // 2, "<f2")
    else:
        start, n = conv.vector(1)
        weight = Stored(start, n, "<f4")
    start, n = conv.vector(2)
    return weight, Stored(start, n, "<f4")


class MnnCrnn(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.path = str(path)
        buf = self.buf = open(path, "rb").read()
        net = Table(buf, struct.unpack_from("<I", buf, 0)[0])
        ops = net.tables(3)
        kinds = [op.scalar(5) for op in ops]
        convs = [i for i, k in enumerate(kinds) if k in (OP_CONV, OP_CONV_DW)]
        if kinds[-1] != OP_SOFTMAX or len(convs) < 3:
            raise ValueError(f"{path} is not one of the CRNN OCR models")

        self.stored = {}  # parameter name -> [Stored, ...] (a value can be duplicated in the file)
        self.params = nn.ParameterDict()

        def add(name, value, stored):
            self.params[name] = nn.Parameter(torch.from_numpy(np.array(value, np.float32)))
            self.stored[name] = stored

        # --- backbone: every op up to the last-but-one convolution, executed in file order
        self.steps = []
        last_backbone = convs[-2]
        for i, op in enumerate(ops[:last_backbone + 1]):
            kind, ins, out = kinds[i], op.array(0, "<i4").tolist(), op.array(4, "<i4").tolist()
            main = op.table(2)
            if kind == OP_INPUT:
                self.input_index = out[0]
            elif kind == OP_CONST:
                if main.scalar(2, "<i", 1) == 1:  # scalar float constants of the hard-sigmoid (x + 3) * 1/6
                    self.steps.append(("const", ins, out, main.array(7, "<f4").copy()))
            elif kind in (OP_CONV, OP_CONV_DW):
                c = main.table(0)
                group, cout, cin = c.scalar(9, "<i", 1), c.scalar(10), c.scalar(11)
                k = (c.scalar(3, "<i", 1), c.scalar(2, "<i", 1))
                pads = c.array(14, "<i4").tolist() or [c.scalar(1), c.scalar(0)] * 2  # top, left, bottom, right
                if c.scalar(8, "<b") != 0 or c.scalar(6, "<i", 1) != 1 or c.scalar(7, "<i", 1) != 1:
                    raise ValueError("only explicitly padded, undilated convolutions are handled")
                w, b = _stored_conv(buf, main)
                add(f"w{i}", w.read(buf).reshape(cout, cin // group, *k), [w])
                add(f"b{i}", b.read(buf), [b])
                act = "relu6" if c.scalar(13, "<b") else "relu" if c.scalar(12, "<b") else None
                cfg = dict(stride=(c.scalar(5, "<i", 1), c.scalar(4, "<i", 1)), groups=group,
                           pad=(pads[1], pads[3], pads[0], pads[2]), act=act)
                self.steps.append(("conv", ins, out, (f"w{i}", f"b{i}", cfg)))
            elif kind == OP_POOL:
                if main.scalar(7, "<b") != 1:
                    raise ValueError("only average pooling is handled")
                cfg = ((main.scalar(4), main.scalar(3)), (main.scalar(6), main.scalar(5)))
                self.steps.append(("avgpool", ins, out, cfg))
            elif kind == OP_BINARY:
                if main.scalar(0) not in BINARY:
                    raise ValueError(f"unexpected binary op {main.scalar(0)} at position {i}")
                self.steps.append(("binary", ins, out, BINARY[main.scalar(0)]))
            elif kind == OP_RELU6:
                self.steps.append(("relu6", ins, out, None))
            elif kind == OP_RASTER and len(ins) == 1:  # layout conversions around the squeeze-excite multiply
                self.steps.append(("same", ins, out, None))
            elif kind == OP_WHILE and len(ins) == 2:  # the broadcast multiply of squeeze-excite
                self.steps.append(("binary", ins, out, torch.mul))
            else:
                raise ValueError(f"unexpected op type {kind} at position {i} in the backbone")
        self.feature_index = ops[last_backbone].array(4, "<i4")[0]

        # --- head: the LSTM was lowered to loops by MNN, so take its ONNX constants by shape instead
        blobs = [op.table(2) for op, k in zip(ops, kinds) if k == OP_CONST]
        floats = [(b.array(0, "<i4").tolist(), b) for b in blobs if b.scalar(2, "<i", 1) == 1]
        w_logits, b_conv = _stored_conv(buf, ops[convs[-1]].table(2))
        c_last = ops[convs[-1]].table(2).table(0)
        self.classes, feat = c_last.scalar(10), c_last.scalar(11)
        hidden = feat // 2
        self.hidden = hidden

        def blob(dims):
            hits = [b for d, b in floats if d == dims]
            if not hits:
                raise ValueError(f"no float constant of shape {dims}; the head is not the expected BiLSTM")
            return [Stored(*b.vector(7), "<f4") for b in hits]

        w_ih = blob([2, 4 * hidden, self.params[f"b{last_backbone}"].numel()])
        w_hh = blob([2, 4 * hidden, hidden])
        bias = blob([2, 4 * hidden])  # Wb, Rb, and the Wb + Rb the lowered loop actually reads
        packed = blob([2, 8 * hidden])  # the original ONNX [Wb, Rb]
        values = [s.read(buf).reshape(2, 4 * hidden) for s in bias]
        total = max(values, key=lambda v: np.abs(v).sum())
        if not any(np.allclose(v, 0) for v in values) or len(w_ih) != 1 or len(w_hh) != 1:
            raise ValueError("expected a single LSTM with a zero recurrent bias")
        self.lstm_in = self.params[f"b{last_backbone}"].numel()
        add("lstm_w", w_ih[0].read(buf).reshape(2, 4 * hidden, self.lstm_in), w_ih)
        add("lstm_r", w_hh[0].read(buf).reshape(2, 4 * hidden, hidden), w_hh)
        add("lstm_b", total, [s for s, v in zip(bias, values) if not np.allclose(v, 0)])
        self.packed_bias = packed
        add("logits_w", w_logits.read(buf).reshape(self.classes, feat), [w_logits])
        b_const = blob([self.classes])
        add("logits_b", b_const[0].read(buf) + b_conv.read(buf), b_const)
        self.logits_b_conv = b_conv.read(buf)

    # groups for freezing / per-group learning rates
    def head_parameters(self):
        return [p for n, p in self.params.items() if n.startswith(("lstm_", "logits_"))]

    def backbone_parameters(self):
        return [p for n, p in self.params.items() if not n.startswith(("lstm_", "logits_"))]

    def features(self, x):
        t = {self.input_index: x}
        for kind, ins, out, arg in self.steps:
            if kind == "const":
                y = torch.as_tensor(arg, device=x.device).reshape(1, -1, 1, 1)
            elif kind == "conv":
                w, b, cfg = arg
                y = F.conv2d(F.pad(t[ins[0]], cfg["pad"]), self.params[w], self.params[b], cfg["stride"], 0, 1, cfg["groups"])
                y = F.relu(y) if cfg["act"] == "relu" else F.relu6(y) if cfg["act"] == "relu6" else y
            elif kind == "avgpool":
                y = F.avg_pool2d(t[ins[0]], arg[0], arg[1])
            elif kind == "binary":
                y = arg(t[ins[0]], t[ins[1]])
            elif kind == "relu6":
                y = F.relu6(t[ins[0]])
            else:
                y = t[ins[0]]
            t[out[0]] = y
        return t[self.feature_index]  # (N, 128, 3, 6)

    def _lstm(self, seq, d):
        """One direction of the ONNX LSTM (gate order i, o, f, c); seq is (T, N, F)."""
        w, r, b = self.params["lstm_w"][d], self.params["lstm_r"][d], self.params["lstm_b"][d]
        n = seq.shape[1]
        h = seq.new_zeros(n, self.hidden)
        c = seq.new_zeros(n, self.hidden)
        proj = seq @ w.T + b
        out = [None] * len(seq)
        for t in (range(len(seq)) if d == 0 else reversed(range(len(seq)))):
            i, o, f, g = (proj[t] + h @ r.T).chunk(4, -1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(c)
            out[t] = h
        return torch.stack(out)

    def forward(self, x):
        """x: (N, 3, 48, 96) BGR in [0, 1]. Returns logits (N, 18, C); the file's own output is their softmax."""
        f = self.features(x)
        seq = f.permute(2, 3, 0, 1).reshape(-1, f.shape[0], f.shape[1])  # row-major grid cells -> time steps
        y = torch.cat([self._lstm(seq, 0), self._lstm(seq, 1)], -1)
        return (y @ self.params["logits_w"].T + self.params["logits_b"]).transpose(0, 1)

    def export(self, path):
        """Writes the current weights into a copy of the original file (conv weights rounded to fp16)."""
        buf = bytearray(self.buf)
        hidden = self.hidden
        for name, places in self.stored.items():
            v = self.params[name].detach().cpu().numpy()
            if name == "logits_b":
                v = v - self.logits_b_conv
            for s in places:
                s.write(buf, v)
        for s in self.packed_bias:  # keep the unused ONNX [Wb, Rb] constant consistent
            b = self.params["lstm_b"].detach().cpu().numpy()
            s.write(buf, np.concatenate([b, np.zeros_like(b)], 1).reshape(2, 8 * hidden))
        with open(path, "wb") as f:
            f.write(buf)


def decode(probs, alphabet=ALNUM):
    """Greedy CTC over (T, C) scores, blank = last class. Same rule as read_plate.decode."""
    blank = probs.shape[-1] - 1
    chars, prev = [], None
    for k in probs.argmax(-1).tolist():
        if k != prev and k != blank:
            chars.append(alphabet[k])
        prev = k
    return "".join(chars)


def mnn_run(path, x):
    """Runs an MNN file on a (1, 3, 48, 96) array with the MNN runtime; returns (18, C) probabilities."""
    import MNN
    net = MNN.Interpreter(str(path))
    sess = net.createSession()
    inp = net.getSessionInput(sess)
    out = next(iter(net.getSessionOutputAll(sess).values()))
    x = np.ascontiguousarray(x, np.float32)
    inp.copyFrom(MNN.Tensor(x.shape, MNN.Halide_Type_Float, x, MNN.Tensor_DimensionType_Caffe))
    net.runSession(sess)
    shape = out.getShape()
    host = MNN.Tensor(shape, MNN.Halide_Type_Float, np.zeros(shape, np.float32), MNN.Tensor_DimensionType_Caffe)
    out.copyToHostTensor(host)
    return np.array(host.getData(), np.float32).reshape(shape)[0]


def verify(model, mnn_path, xs):
    """Largest difference between the PyTorch module and an MNN file over a batch of inputs."""
    with torch.no_grad():
        ours = torch.softmax(model(torch.from_numpy(xs)), -1).numpy()
    return max(float(np.abs(ours[i] - mnn_run(mnn_path, xs[i:i + 1])).max()) for i in range(len(xs)))


if __name__ == "__main__":
    import sys
    rng = np.random.default_rng(0)
    xs = rng.random((4, 3, 48, 96), np.float32)
    for p in sys.argv[1:]:
        try:
            m = MnnCrnn(p).eval()
            print(f"{p}: classes={m.classes} params={sum(v.numel() for v in m.params.values())} "
                  f"max |torch - mnn| = {verify(m, p, xs):.2e}")
        except ValueError as e:
            print(f"{p}: {e}")
