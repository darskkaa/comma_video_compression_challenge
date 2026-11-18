"""Motion-compensated context for the token model.

Between two coded frames the car moves a few metres, so the lane markings of the previous token map slide down the
road, further than the +-1 pixel context warps can follow.  `composite` re-projects the road plane of the previous
map with integer arithmetic only: a pixel (u, v) below the horizon row v0 looks up the previous map at

    u' = u0 + (u - u0) / z,   v' = v0 + (v - v0) / z,   z = 1 + tau * (v - v0) / 400        (then shifted by dx, dy)

Only the road surface is repainted (road <-> lane marking); everything else is copied.  The result goes through one
more 3x3 integer convolution (`conv_mc`) that is added to the temporal context of the inherited model:

    past <- requantize(past + conv_mc(one_hot(composite)), 0)

The inherited weights are untouched; with a zero branch the model is the inherited one.

Extension record (decoder/extfmt.py, tag 0x40): u8 channels, u16 frames, u8 rho, five int16 parameter offsets, then
the adaptive binary coder: per output channel its exponent, bias and 45 weights; per frame (from frame 1 on) the five
parameters tau (1/20), v0 (half rows), u0 (columns), dx, dy as differences from (rho * previous) >> 2.
"""
import importlib
import struct

import numpy as np
import torch
from torch.nn import functional as F

ROAD, LANE = 0, 1
PARAMS = 5
PROB_BITS = 12
ONE = 1 << PROB_BITS


def composite(prev, tau_q, v0_q, u0_q, dx=0, dy=0):
    """prev [H,W] integer tensor -> [H,W] long tensor"""
    H, W = prev.shape
    prev = prev.long()
    v = torch.arange(H, device=prev.device, dtype=torch.int64)[:, None]
    u = torch.arange(W, device=prev.device, dtype=torch.int64)[None, :]
    a = 2 * v + 1 - int(v0_q)  # twice the distance of the pixel centre below the horizon
    d = 16000 + int(tau_q) * a.clamp_min(0)  # z = d / 16000
    vp = torch.div(int(v0_q) * d + 16000 * a, 2 * d, rounding_mode="floor") - int(dy)
    up = torch.div(2 * int(u0_q) * d + (2 * u + 1 - 2 * int(u0_q)) * 16000, 2 * d, rounding_mode="floor") - int(dx)
    src = prev[vp.clamp(0, H - 1).expand(H, W), up.clamp(0, W - 1)]
    ground = prev <= LANE
    lane = (src == LANE) & (a > 0)
    out = prev.clone()
    out[ground & lane] = LANE
    out[ground & ~lane & (prev == LANE)] = ROAD
    return out


def attach(model, codes, device):
    """Add the branch to the inherited model; codes = (weight [C,5,3,3], bias [C], exponent [C]) integers."""
    hpac_integer = importlib.import_module("hpac_integer")
    conv = hpac_integer.IntegerConv2d(model.num_classes, model.ch, 3, padding=1, weight_bound=127,
                                      use_weight_scales=True, exponent_min=-6)
    weight, bias, exponent = codes
    with torch.no_grad():
        conv.weight.copy_(torch.from_numpy(np.asarray(weight, dtype=np.float32)))
        conv.bias.copy_(torch.from_numpy(np.asarray(bias, dtype=np.float32)))
        conv.exponent.copy_(torch.from_numpy(np.asarray(exponent, dtype=np.float32)))
    model.conv_mc = conv.to(device)
    return model


def add_branch(model, context, composite_map):
    """context from model.prepare_frame_context; composite_map [B,H,W] long"""
    hpac_integer = importlib.import_module("hpac_integer")
    shift, past, scale, spm = context
    batch, height, width = composite_map.shape
    one_hot = F.one_hot(composite_map, num_classes=model.num_classes).permute(0, 3, 1, 2).float()
    full = model._from_patches(past, batch, height // model.P, width // model.P) + model.conv_mc(one_hot)
    past = model._to_patches(hpac_integer.requantize(full, 0, -model.activation_bound, model.activation_bound))
    return shift, past, scale, spm


# ---- adaptive binary range coder (integer arithmetic only: identical decisions on every machine) -----------------
class Bit:
    """probability of a zero bit from counts, halved at LIMIT"""
    __slots__ = ("n0", "n1")
    LIMIT = 1023

    def __init__(self):
        self.n0 = 0
        self.n1 = 0

    def p0(self):
        p = ((2 * self.n0 + 1) << PROB_BITS) // (2 * (self.n0 + self.n1) + 2)
        return min(max(p, 8), ONE - 8)

    def update(self, bit):
        if bit:
            self.n1 += 1
        else:
            self.n0 += 1
        if self.n0 + self.n1 > self.LIMIT:
            self.n0 = (self.n0 + 1) >> 1
            self.n1 = (self.n1 + 1) >> 1


class Models(dict):
    def __missing__(self, key):
        m = self[key] = Bit()
        return m


class Encoder:
    def __init__(self):
        self.low = 0
        self.range = 0xFFFFFFFF
        self.cache = 0
        self.cache_size = 1
        self.out = bytearray()

    def _shift_low(self):
        if self.low < 0xFF000000 or self.low >= (1 << 32):
            carry = self.low >> 32
            temp = self.cache
            while True:
                self.out.append((temp + carry) & 0xFF)
                temp = 0xFF
                self.cache_size -= 1
                if self.cache_size == 0:
                    break
            self.cache = (self.low >> 24) & 0xFF
        self.cache_size += 1
        self.low = (self.low & 0x00FFFFFF) << 8

    def bit(self, model, bit):
        bound = (self.range >> PROB_BITS) * model.p0()
        if bit:
            self.low += bound
            self.range -= bound
        else:
            self.range = bound
        model.update(bit)
        while self.range < (1 << 24):
            self.range = (self.range << 8) & 0xFFFFFFFF
            self._shift_low()
        return bit

    def finish(self):
        for _ in range(5):
            self._shift_low()
        return bytes(self.out[1:]).rstrip(b"\x00")  # the decoder reads zeros past the end


class Decoder:
    def __init__(self, data):
        self.data = data
        self.pos = 0
        self.range = 0xFFFFFFFF
        self.code = 0
        for _ in range(4):
            self.code = (self.code << 8) | self._byte()

    def _byte(self):
        b = self.data[self.pos] if self.pos < len(self.data) else 0
        self.pos += 1
        return b

    def bit(self, model, _bit=0):
        bound = (self.range >> PROB_BITS) * model.p0()
        if self.code < bound:
            self.range = bound
            bit = 0
        else:
            self.code -= bound
            self.range -= bound
            bit = 1
        model.update(bit)
        while self.range < (1 << 24):
            self.range = (self.range << 8) & 0xFFFFFFFF
            self.code = ((self.code << 8) | self._byte()) & 0xFFFFFFFF
        return bit


def _uint(coder, models, ctx, value):
    """Exp-Golomb style: unary bit length (adaptive per position), then mantissa (top two bits adaptive)"""
    n = (value + 1).bit_length() - 1
    k = 0
    while coder.bit(models[(ctx, "u", k)], 1 if k < n else 0):
        k += 1
    rest, out = (value + 1) - (1 << k), 0
    for i in range(k - 1, -1, -1):
        out = (out << 1) | coder.bit(models[(ctx, "m", k, i)] if i >= k - 2 else models[("raw",)], (rest >> i) & 1)
    return (1 << k) + out - 1


def _int(coder, models, ctx, value):
    if not coder.bit(models[(ctx, "z")], 1 if value else 0):
        return 0
    negative = coder.bit(models[(ctx, "s")], 1 if value < 0 else 0)
    magnitude = _uint(coder, models, ctx, abs(value) - 1) + 1
    return -magnitude if negative else magnitude


def _pass(coder, weight, bias, exponent, params, rho, encode):
    models = Models()
    for c in range(weight.shape[0]):
        exponent[c] = _int(coder, models, ("e",), int(exponent[c]) if encode else 0)
        bias[c] = _int(coder, models, ("b",), int(bias[c]) if encode else 0)
        for i in range(weight.shape[1]):
            for k in range(9):
                ctx = ("w", i, min(abs(int(exponent[c])), 6))
                weight[c, i, k // 3, k % 3] = _int(coder, models, ctx, int(weight[c, i, k // 3, k % 3]) if encode else 0)
    for t in range(1, params.shape[0]):
        for k in range(PARAMS):
            pred = (rho * int(params[t - 1, k])) >> 2 if t > 1 else 0
            params[t, k] = pred + _int(coder, models, ("p", k), int(params[t, k]) - pred if encode else 0)


def encode_record(weight, bias, exponent, params):
    weight, bias, exponent, params = (np.array(a, dtype=np.int64) for a in (weight, bias, exponent, params))
    base = params[1:].min(0)
    rel = params.copy()
    rel[1:] -= base
    best = None
    for rho in range(5):
        coder = Encoder()
        _pass(coder, weight.copy(), bias.copy(), exponent.copy(), rel.copy(), rho, True)
        body = coder.finish()
        if best is None or len(body) < len(best[1]):
            best = (rho, body)
    return struct.pack("<BHB", weight.shape[0], params.shape[0], best[0]) + base.astype("<i2").tobytes() + best[1]


def decode_record(blob):
    """-> (weight [C,5,3,3], bias [C], exponent [C], params [N,5]) int64"""
    channels, frames, rho = struct.unpack_from("<BHB", blob)
    base = np.frombuffer(blob[4:4 + 2 * PARAMS], dtype="<i2").astype(np.int64)
    weight = np.zeros((channels, 5, 3, 3), np.int64)
    bias = np.zeros(channels, np.int64)
    exponent = np.zeros(channels, np.int64)
    params = np.zeros((frames, PARAMS), np.int64)
    _pass(Decoder(blob[4 + 2 * PARAMS:]), weight, bias, exponent, params, rho, False)
    params[1:] += base
    return weight, bias, exponent, params
