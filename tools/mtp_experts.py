# mtp_experts.py GGUF OUT [TENSORS]: an Unsloth MTP head's 512 routed experts in the engine's native blob layout
# (per expert: gate rows, up rows, down rows - native_expert_layout), behind a 16-byte header
# "SMTPEXP1" int32 gate/up ggml type, int32 down type.  TENSORS (optional): the directory of Strata's BF16 MTP tensors
# (mtp/tensors in the model's data directory); a few experts are checked against them.
import sys, struct, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))   # tools/: gguf_reader.py
from gguf_reader import GGUFFile
src, out = sys.argv[1], sys.argv[2]
tensors = sys.argv[3] if len(sys.argv) > 3 else None
H, FF, NE = 2560, 640, 512
g = GGUFFile(pathlib.Path(src)); mm = np.memmap(src, dtype=np.uint8, mode="r")
T = {t.name: t for t in g.tensors}
gate, up, down = T["blk.48.ffn_gate_exps.weight"], T["blk.48.ffn_up_exps.weight"], T["blk.48.ffn_down_exps.weight"]
assert gate.shape == [H, FF, NE] and up.shape == [H, FF, NE] and down.shape == [FF, H, NE], (gate.shape, down.shape)
assert gate.type_id == up.type_id
BLK = {8: (32, 34), 12: (256, 144)}          # Q8_0, Q4_K: values and bytes per block
def row_bytes(t, n): q, b = BLK[t]; return n // q * b
gu_row, d_row = row_bytes(gate.type_id, H), row_bytes(down.type_id, FF)
e_gu, e_d = FF * gu_row, H * d_row
def part(t, e, nb): o = g.data_start + t.offset + e * nb; return mm[o:o + nb]
with open(out, "wb") as f:
    f.write(b"SMTPEXP1" + struct.pack("<ii", gate.type_id, down.type_id))
    for e in range(NE):
        f.write(part(gate, e, e_gu).tobytes()); f.write(part(up, e, e_gu).tobytes()); f.write(part(down, e, e_d).tobytes())
print(f"{out}: gate/up {gate.type_name} down {down.type_name}, {NE} x {2 * e_gu + e_d} B = {NE * (2 * e_gu + e_d) / 2**30:.2f} GiB")

def f16(a): return a.view(np.float16).astype(np.float32)
def deq(t, raw, n):
    if t == 8:
        b = raw.reshape(-1, 34); return (f16(b[:, :2].copy()) * b[:, 2:].copy().view(np.int8).astype(np.float32)).reshape(-1)[:n]
    b = raw.reshape(-1, 144); d = f16(b[:, 0:2].copy())[:, 0]; dm = f16(b[:, 2:4].copy())[:, 0]
    sc = b[:, 4:16].astype(np.int32); qs = b[:, 16:144]
    y = np.empty((b.shape[0], 256), np.float32)
    def sm(j):
        if j < 4: return sc[:, j] & 63, sc[:, j + 4] & 63
        return (sc[:, j + 4] & 0xF) | ((sc[:, j - 4] >> 6) << 4), (sc[:, j + 4] >> 4) | ((sc[:, j] >> 6) << 4)
    for k in range(4):
        s1, m1 = sm(2 * k); s2, m2 = sm(2 * k + 1); q = qs[:, 32 * k:32 * k + 32]
        y[:, 64 * k:64 * k + 32] = (d * s1)[:, None] * (q & 0xF) - (dm * m1)[:, None]
        y[:, 64 * k + 32:64 * k + 64] = (d * s2)[:, None] * (q >> 4) - (dm * m2)[:, None]
    return y.reshape(-1)[:n]
if tensors is None: sys.exit(0)
S = str(pathlib.Path(tensors) / "mtp.layers.0.mlp.experts.")
gu_bf = np.memmap(S + "gate_up_proj.bin", dtype=np.uint16, mode="r").reshape(NE, 2 * FF, H)
dn_bf = np.memmap(S + "down_proj.bin", dtype=np.uint16, mode="r").reshape(NE, H, FF)
def bf(a): return (a.astype(np.uint32) << 16).view(np.float32)
blob = np.memmap(out, dtype=np.uint8, mode="r", offset=16).reshape(NE, -1)
for e in (0, 7, 300, 511):
    gq = deq(gate.type_id, blob[e, :e_gu], FF * H).reshape(FF, H)
    uq = deq(gate.type_id, blob[e, e_gu:2 * e_gu], FF * H).reshape(FF, H)
    dq = deq(down.type_id, blob[e, 2 * e_gu:], H * FF).reshape(H, FF)
    for nm, a, r in (("gate", gq, bf(gu_bf[e, :FF])), ("up", uq, bf(gu_bf[e, FF:])), ("down", dq, bf(dn_bf[e]))):
        rel = np.sqrt(((a - r) ** 2).mean() / (r ** 2).mean()); c = np.corrcoef(a.ravel(), r.ravel())[0, 1]
        print(f"  expert {e:3d} {nm:4s}: rel RMS error {rel:.4f}  corr {c:.5f}")
