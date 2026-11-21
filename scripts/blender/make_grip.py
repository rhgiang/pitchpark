"""Build the viewer's throwing hand (grip.glb): a two-seam fastball grip, headless.

    blender --background --python scripts/blender/make_grip.py -- [preview_dir] [out_dir]
    blender --background --python scripts/blender/make_grip.py -- --open [preview_dir] [out_dir]   (open.glb)

(No --factory-startup: needs the MPFB extension, `blender -c extension install -s -e mpfb`.)

The hand is MakeHuman's male hand from MPFB (its exported meshes are CC0), posed with its own
finger skeleton. Image-to-3D hands (TRELLIS/Tripo) came out with fused or invented fingers; a
modelled hand bends cleanly. Units below are MPFB's metres until the export step.

Every joint is posed by rotating the bone in its own frame (X = flexion, probed; Z = spread across the
palm), so fingers bend only the way real fingers bend. No IK: IK solved the thumb tip's position but
twisted its metacarpal, which bulged the base of the thumb.

--open builds open.glb instead: the same hand relaxed (fingers loosely flexed, nothing in it), which the
viewer shows once the ball has left the hand, in place of the grip (an empty grip read as a claw).

Output frame (web/src/handModels.js): LEFT hand, feet, origin at the wrist joint, +x along the forearm
into the hand, +y the back of the hand, +z the thumb side, the ball's centre at (0.28, 0, 0).
"""
import math
import os
import sys

import bmesh
import bpy
from mathutils import Matrix, Vector
from bl_ext.blender_org.mpfb.services.humanservice import HumanService
from bl_ext.blender_org.mpfb.services.targetservice import TargetService

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
OPEN = "--open" in argv            # build open.glb (the relaxed hand after release) instead of grip.glb
argv = [a for a in argv if a != "--open"]
PREVIEW_DIR = argv[0] if argv else None
OUT_DIR = argv[1] if len(argv) > 1 else os.path.join(ROOT, "web", "public", "models")  # review builds: pass a scratch folder

R_BALL = 1.45 * 0.0254     # ball radius (OBR 3.01: 9-9.25 in circumference)
FIGURE_M = 6.15 * 0.3048   # our pitcher's height (make_pitcher.py HEIGHT); the hand is scaled to him later
# Grip, read off reference photos of a two-seam fastball (Ryan's refs, 2026-11-20) and checked by eye
# against candidates (ref/previews/24b, 25b; contact variants 27: E chosen, 7.7 cm2 of skin on the ball). Joint angles are anatomical (0 = straight).
G = dict(rf=0.0085,        # finger half-thickness at the pad
         rf_thumb=0.0105,  # thumb half-thickness at the pad
         top=(40, 60, 0.5),  # index/middle MCP, PIP, DIP/PIP ratio: wrapped over the top of the ball
         sep=0.026,        # index-to-middle pad spacing: close and parallel, each along the inside of a narrow seam
         gap_min=0.0,      # palm clearance: none, the ball sits down in the hand (looked lodged, not perched)
         give=0.0025,      # how far skin may press into the ball (finger pads squash under grip pressure)
         side_give=0.005,  # ring/pinky press harder: their sides hold the ball's side (squish flattens them on)
         squish=0.009,     # skin this close to the ball is drawn onto it: flat contact patches, not points
         ring=(45, 90),    # preferred MCP, PIP: curled, against the side of the ball
         pinky=(55, 95))   # curled beside the ring, also on the ball

for o in list(bpy.data.objects): bpy.data.objects.remove(o)
md = TargetService.get_default_macro_info_dict(); md["gender"] = 1.0; md["muscle"] = 0.6
body = HumanService.create_human(macro_detail_dict=md)
rig = HumanService.add_builtin_rig(body, "default")
# Pose round the ball as it will be once this hand is scaled up to our pitcher (MPFB's man is shorter):
# in MPFB units the ball is smaller by that scale.
SCALE = FIGURE_M / body.dimensions.z
R = R_BALL / SCALE
print("SCALE to figure", round(SCALE, 3), "pose ball radius mm", round(R * 1000, 1))
for m in body.modifiers:
    if m.type == "MASK": m.show_viewport = False   # vertex indices = base mesh while measuring
P = rig.pose.bones
names = [g.name for g in body.vertex_groups]
def owner(v):
    bones = [e for e in v.groups if names[e.group] in P]
    return names[max(bones, key=lambda e: e.weight).group] if bones else None
OWNER = [owner(v) for v in body.data.vertices]

def upd(): bpy.context.view_layer.update()
def rot(name, x=0.0, y=0.0, z=0.0):
    pb = P[name]; pb.rotation_mode = "XYZ"
    pb.rotation_euler = (math.radians(x), math.radians(y), math.radians(z))
def axis(name, k): return (P[name].matrix.to_3x3()).col[k].normalized()   # rig at the origin
def turn(name, ax, ang):
    pb = P[name]; hd = pb.head.copy()
    pb.matrix = Matrix.Translation(hd) @ Matrix.Rotation(ang, 4, ax) @ Matrix.Translation(-hd) @ pb.matrix; upd()

def frame():
    """Hand frame: W wrist, d along the hand, l toward the index side, n out of the palm."""
    W = P["wrist.R"].head.copy()
    K = (P["finger2-1.R"].head + P["finger3-1.R"].head) / 2
    d = (K - W).normalized()
    l = P["finger2-1.R"].head - P["finger5-1.R"].head; l = (l - l.dot(d) * d).normalized()
    n = d.cross(l)
    b = P["finger3-1.R"]
    if axis(b.name, 0).cross(b.tail - b.head).dot(n) < 0: n = -n   # +X flexion moves the tip palm-ward
    return W, d, l, n

# 1. Wrist straight: the MPFB rest hand is bent off the forearm line, which read as a kink at the wrist.
W, d, l, n = frame()
fa = (P["lowerarm02.R"].tail - P["lowerarm01.R"].head).normalized()
q = d.rotation_difference(fa)
turn("wrist.R", q.axis, q.angle)
print("WRIST straightened by deg", round(math.degrees(q.angle), 1))
W, d, l, n = frame()

# spread sign: which way +Z on the index MCP moves its tip (want index toward +l, the thumb side)
rot("finger2-1.R", z=10); upd(); t1 = P["finger2-3.R"].tail.copy()
rot("finger2-1.R"); upd(); t0 = P["finger2-3.R"].tail.copy()
S_AB = 1 if (t1 - t0).dot(l) > 0 else -1

def pad(name, rf, s=1, at=0.5):
    """Flexor-side skin point of a bone: under +X flexion (sign s) the tail moves along X x bone,
    and that direction is the pad side. Returns (point, outward pad normal)."""
    pb = P[name]; v = pb.tail - pb.head
    nrm = (s * axis(name, 0)).cross(v).normalized()
    return pb.head + v * at + nrm * rf, nrm

def place():
    """Ball centre with both top-finger pads on its surface."""
    (p2, z2), (p3, z3) = pad("finger2-3.R", G["rf"]), pad("finger3-3.R", G["rf"])
    a = p2 - p3; w = z2 + z3; w = (w - w.dot(a.normalized()) * a.normalized()).normalized()
    half = a.length / 2
    return (p2 + p3) / 2 + w * math.sqrt(max(R * R - half * half, 0.0)), w

TOPB = {"finger2-2.R", "finger2-3.R", "finger3-2.R", "finger3-3.R"}
TIPB = {"finger2-3.R", "finger3-3.R"}
def fit_ball():
    """Let the ball settle in the top fingers: free 3D pattern search for the centre that keeps all
    index/middle/palm skin out of it (<= `give` in: skin gives), keeps the palm `gap_min` clear, and
    sits as close as it can to each finger's tip and middle sections (pads and middle phalanges touch).
    Returns (centre, palm gap, worst fingertip gap, worst penetration): m, >0 = air."""
    c0, _ = place()
    ev = body.evaluated_get(bpy.context.evaluated_depsgraph_get()); me = ev.to_mesh()
    grp = {}
    for i, v in enumerate(me.vertices):
        grp.setdefault(OWNER[i], []).append(v.co.copy())
    ev.to_mesh_clear()
    hand = [v for b in PALM | {f"finger{f}-{j}.R" for f in "23" for j in "123"} for v in grp.get(b, [])]
    palm = [v for b in PALM for v in grp.get(b, [])]
    near = lambda vs, c: min((v - c).length for v in vs) - R
    def cost(c):
        pen = -near(hand, c)
        pg = near(palm, c)
        tips = sum(max(0.0, near(grp[f"finger{f}-3.R"], c)) for f in "23")
        mids = sum(max(0.0, near(grp[f"finger{f}-2.R"], c)) for f in "23")
        return 100 * max(0.0, pen - G["give"]) + 100 * max(0.0, G["gap_min"] - pg) + tips + 0.5 * mids
    c, cst = c0.copy(), cost(c0)
    for step in (0.004, 0.002, 0.001, 0.0005):
        improved = True
        while improved:
            improved = False
            for dv in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                c2 = c + Vector(dv) * step; c2c = cost(c2)
                if c2c < cst: c, cst, improved = c2, c2c, True
    return (c, near(palm, c), max(near(grp[f"finger{f}-3.R"], c) for f in "23"), -near(hand, c))

PALM = {"wrist.R", "metacarpal1.R", "metacarpal2.R", "metacarpal3.R", "metacarpal4.R", "lowerarm02.R"}
def depth_by_bone(c):
    """Deepest skin point inside the ball per owning bone (m; >0 = inside)."""
    ev = body.evaluated_get(bpy.context.evaluated_depsgraph_get()); me = ev.to_mesh()
    out_ = {}
    for i, v in enumerate(me.vertices):
        o = OWNER[i]
        if o is None: continue
        dd = R - (v.co - c).length
        if dd > out_.get(o, -1): out_[o] = dd
    ev.to_mesh_clear(); return out_

def rest_flex(name):
    """The bend this joint already has in the MPFB rest pose (deg, + = flexed), measured between the
    parent and child bone directions about the child's flex axis. The relaxed hand is curled, so joint
    angles below are anatomical (0 = straight) and the pose rotation is anatomical - rest."""
    X = axis(name, 0); pb = P[name]
    pv = pb.parent.tail - pb.parent.head; v = pb.tail - pb.head
    pv = (pv - pv.dot(X) * X).normalized(); v = (v - v.dot(X) * X).normalized()
    return math.degrees(math.atan2(v.dot(X.cross(pv)), v.dot(pv)))
REST = {f"finger{f}-{j}.R": rest_flex(f"finger{f}-{j}.R") for f in "2345" for j in "123"}
print("REST flex", {k: round(v) for k, v in REST.items() if k[6] in "23"})
def anat(name, deg, z=0.0): rot(name, x=deg - REST[name], z=z)
def set_top(mcp, pip, a):
    dip = pip * G["top"][2]   # DIP follows PIP (tendon coupling); photos: pads lie fairly flat
    anat("finger2-1.R", mcp, S_AB * a); anat("finger2-2.R", pip); anat("finger2-3.R", dip)
    anat("finger3-1.R", mcp, -S_AB * a * 0.5); anat("finger3-2.R", pip); anat("finger3-3.R", dip)
    upd()

def padsep(): return (pad("finger2-3.R", G["rf"])[0] - pad("finger3-3.R", G["rf"])[0]).length

R_FT = R_BALL / 0.3048
def bl(s): return Vector((s.x, -s.z, s.y))                # viewer (Y up) -> Blender (Z up)


def export_hand(name, C, fa, W, n, l, ball):
    """Step 7: cut the hand (plus a forearm stub that tucks inside the figure's own forearm), scale it
    to our pitcher, move it into the slot frame (C -> (0.28, 0, 0) ft, fa -> +x), export <name>.glb.
    With `ball`, skin near the ball is first drawn onto it (contact) and the checks measure it.
    Returns (object, to_slot, FT) for the previews."""
    STUB_M = 0.03                 # forearm kept behind the wrist joint, m (MPFB)
    TRI_BUDGET = 6000             # whole hand: five real fingers need more than the old 3-finger blob
    FT = SCALE / 0.3048           # MPFB m -> our pitcher's ft
    X = fa
    Y = -n + n.dot(X) * X; Y.normalize()                     # back of the hand
    Z = l - l.dot(X) * X - l.dot(Y) * Y; Z.normalize()       # thumb side (right hand -> left: a mirror)
    def to_slot(p):
        q = p - C
        return Vector((0.28 + FT * X.dot(q), FT * Y.dot(q), FT * Z.dot(q)))

    print(f"[make_grip] {name}: wrist joint at slot x {to_slot(W).x:.3f} ft")
    me = bpy.data.meshes.new_from_object(body.evaluated_get(bpy.context.evaluated_depsgraph_get()))  # posed, base-mesh indices
    gbody = body.vertex_groups["body"].index                  # MPFB: skin (not proxy helpers)
    HAND = {"wrist.R", "lowerarm02.R"} | {b.name for b in P if b.name.endswith(".R") and b.name.startswith(("finger", "metacarpal"))}
    keep = [any(e.group == gbody for e in v.groups) and OWNER[i] in HAND and (me.vertices[i].co - W).dot(X) > -STUB_M
            for i, v in enumerate(body.data.vertices)]
    bm = bmesh.new(); bm.from_mesh(me); bm.verts.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not keep[v.index]], context="VERTS")
    # Contact: skin that faces the ball and lies within `squish` outside it (or anywhere inside it) moves
    # onto the surface, blended so the patch edge stays smooth: pads flatten on the leather like real
    # soft tissue. Then report the contact area (faces with every corner within 1 mm of the surface).
    bm.normal_update()
    for v in (bm.verts if ball else []):
        r_ = v.co - C; dist = r_.length - R
        if dist < G["squish"] and v.normal.dot(r_) < 0:          # facing the ball (not a finger's far side)
            w = 1.0 if dist <= 0 else (1 - dist / G["squish"]) ** 2
            v.co = v.co - r_.normalized() * dist * w
    CONTACT_CM2 = 0 if not ball else sum(f.calc_area() for f in bm.faces if all(abs((v.co - C).length - R) < 0.001 for v in f.verts)) * SCALE ** 2 * 1e4
    print("CONTACT area cm2", round(CONTACT_CM2, 1))
    for v in bm.verts: v.co = bl(to_slot(v.co))
    if Matrix((X, Y, Z)).determinant() < 0:                   # mirrored: keep faces pointing outward
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
    bm.to_mesh(me); bm.free()
    for o in list(bpy.data.objects): bpy.data.objects.remove(o)
    grip = bpy.data.objects.new(name, me); bpy.context.scene.collection.objects.link(grip)
    mat = bpy.data.materials.new("skin"); mat.diffuse_color = (0.58, 0.27, 0.13, 1)  # figure picks colour by name
    me.materials.clear(); me.materials.append(mat)
    tris = sum(len(p.vertices) - 2 for p in me.polygons)
    if tris > TRI_BUDGET:
        dec = grip.modifiers.new("dec", "DECIMATE"); dec.ratio = TRI_BUDGET / tris * 0.98
        bpy.context.view_layer.objects.active = grip; bpy.ops.object.modifier_apply(modifier="dec")
    for p in me.polygons: p.use_smooth = True

    # measured checks (viewer ft): triangles, extent, skin inside the ball
    pts = [Vector((v.co.x, v.co.z, -v.co.y)) for v in me.vertices]
    tris = sum(len(p.vertices) - 2 for p in me.polygons)
    inside = max(R_FT - (p - Vector((0.28, 0, 0))).length for p in pts)
    lo = [min(p[i] for p in pts) for i in range(3)]; hi = [max(p[i] for p in pts) for i in range(3)]
    print(f"[make_grip] {name}.glb: {tris} tris, x {lo[0]:.3f}..{hi[0]:.3f} y {lo[1]:.3f}..{hi[1]:.3f} "
          f"z {lo[2]:.3f}..{hi[2]:.3f} ft, deepest skin inside the ball {inside * 304.8:.1f} mm")
    assert tris <= TRI_BUDGET

    os.makedirs(OUT_DIR, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT"); grip.select_set(True)
    bpy.ops.export_scene.gltf(filepath=os.path.join(OUT_DIR, f"{name}.glb"), export_format="GLB", use_selection=True,
                              export_yup=True, export_apply=True, export_texcoords=False, export_materials="EXPORT")
    return grip, to_slot, FT


if OPEN:
    # Relaxed hand after release (Ryan: the empty grip read as a claw): every finger loosely flexed
    # (anatomical MCP/PIP/DIP, the natural rest of a hand that has just let go), thumb at MPFB's rest.
    for f, (m_, p_, d_) in zip("2345", ((15, 20, 10), (18, 24, 12), (22, 28, 14), (26, 32, 16))):
        anat(f"finger{f}-1.R", m_); anat(f"finger{f}-2.R", p_); anat(f"finger{f}-3.R", d_)
    upd()
    W, d, l, n = frame()
    # The wrist joint where grip.glb's lands (it prints it: the grip's wrist is turned to seat the ball, so
    # its joint sits 0.088 ft behind the slot origin); at 0 the open hand stood ~1 in off the forearm.
    GRIP_WRIST_X = -0.088
    export_hand("open", W + fa * ((0.28 - GRIP_WRIST_X) * 0.3048 / SCALE), fa, W, n, l, ball=False)
    raise SystemExit

# 2. Index + middle: the photo bend, spread (bisection) to the seam spacing; the ball settles into them.
mcp, pip = G["top"][:2]
lo_, hi_ = -10.0, 35.0
for _ in range(14):
    a = (lo_ + hi_) / 2; set_top(mcp, pip, a)
    lo_, hi_ = (a, hi_) if padsep() < G["sep"] else (lo_, a)
set_top(mcp, pip, a); C, gap, tipgap, pen = fit_ball()
print("TOP final mcp", mcp, "pip", pip, "spread", round(a, 1), "palm gap mm", round(gap*1000, 1), "tip gap mm", round(tipgap*1000, 1))
print("PAD SEP mm", round(padsep() * 1000, 1))

def segdist(name):
    a_, b_ = P[name].head, P[name].tail; ab = b_ - a_
    t = max(0.0, min(1.0, (C - a_).dot(ab) / ab.length_squared))
    return (a_ + ab * t - C).length
# 3. Ring and pinky: as tight as the photos without entering the ball.
def depth_of(bones):
    db = depth_by_bone(C); return max(db.get(b, -1) for b in bones)
def finger_fit(f, want, give=G["give"]):
    """Grid over MCP / PIP (DIP = 0.55 PIP) and spread for one finger: skin no more than `give` into
    the ball; then as much of the finger lying on the ball as possible (vertices within 2 mm of its
    surface: the side and pad pressed against it, supporting the grip); then nearest the `want` curl."""
    bones = [f"finger{f}-{j}.R" for j in "123"]; ids = [i for i, o in enumerate(OWNER) if o in bones]; best = None
    for z_ in (-30, -24, -18, -12, -6, 0, 6):
        for m_ in range(0, 85, 10):
            for p_ in range(70, 115, 5):     # curled (photos): middle joint well bent, knuckles showing
                anat(bones[0], m_, S_AB * z_); anat(bones[1], p_); anat(bones[2], p_ * 0.55); upd()
                ev = body.evaluated_get(bpy.context.evaluated_depsgraph_get()); me = ev.to_mesh()
                dist = [(me.vertices[i].co - C).length - R for i in ids]
                ev.to_mesh_clear()
                pen = -min(dist); touch = sum(1 for x in dist if -give <= x < 0.002)
                score = 1000 * max(0, pen - give) - 0.01 * touch + 0.002 * (abs(m_ - want[0]) + abs(p_ - want[1])) / 10
                if best is None or score < best[0]: best = (score, z_, m_, p_, pen, touch)
    _, z_, m_, p_, pen, touch = best
    anat(bones[0], m_, S_AB * z_); anat(bones[1], p_); anat(bones[2], p_ * 0.55); upd()
    print("CURL", f, "spread", z_, "mcp", m_, "pip", p_, "press mm", round(pen * 1000, 1), "skin on ball (verts)", touch)
finger_fit("4", G["ring"], G["side_give"])      # ring: along the side of the ball, supporting it
finger_fit("5", G["pinky"], G["side_give"])     # pinky: beside it, also on the ball

# 4. Thumb: pad under the ball, opposite the two top pads, facing the ball. Coordinate descent over
#    CMC (x, y, z), MCP flex and IP flex, with a soft pull toward small angles (natural, untwisted).
top = (pad("finger2-3.R", G["rf"])[0] + pad("finger3-3.R", G["rf"])[0]) / 2
T = C - (top - C).normalized() * R                      # point on the ball opposite the top pads
TH = ("finger1-1.R", "finger1-2.R", "finger1-3.R")
BOUND = [(-50, 50), (-20, 20), (-50, 50), (0, 70), (0, 80)]   # CMC x/twist/z, MCP, IP: no hyperextension
def set_thumb(v):
    rot(TH[0], x=v[0], y=v[1], z=v[2]); rot(TH[1], x=S_T * v[3]); rot(TH[2], x=S_T * v[4]); upd()
def thumb_cost(v):
    set_thumb(v)
    p, nrm = pad(TH[2], G["rf_thumb"], S_T, at=0.55)
    c1 = (((p - C).length - R) / 0.003) ** 2                       # pad on the ball's surface
    c1 += 3 * (math.degrees((p - C).angle(T - C)) / 45) ** 2      # ...somewhere under it, near opposite the top pads
    c2 = 1 - nrm.dot((C - p).normalized())
    pen = sum(max(0.0, R + r - segdist(b)) for b, r in zip(TH, (0.013, G["rf_thumb"], G["rf_thumb"])))
    c3 = (pen / 0.003) ** 2
    c4 = 0.3 * (sum((x / 90) ** 2 for x in v[:3]) + 3 * (v[1] / 90) ** 2)   # twist (y) costs extra
    return c1 + 2 * c2 + c3 + c4
body.hide_viewport = True     # skip the mesh deform while searching bones only
best = None
for S_T in (1, -1):          # thumb flex sign: not probed reliably, so keep whichever fits
  for sx in (-40, 0, 40):
      for sz in (-40, 0, 40):
          v = [sx, 0, sz, 20, 20]; cst = thumb_cost(v)
          for step in (16, 8, 4, 2, 1):
              improved = True
              while improved:
                  improved = False
                  for i in range(5):
                      for sg in (1, -1):
                          w = v.copy(); w[i] = max(BOUND[i][0], min(BOUND[i][1], w[i] + sg * step))
                          cw = thumb_cost(w)
                          if cw < cst: v, cst, improved = w, cw, True
          if best is None or cst < best[0]: best = (cst, v, S_T)
body.hide_viewport = False
S_T = best[2]; set_thumb(best[1])
p, nrm = pad(TH[2], G["rf_thumb"], S_T, at=0.55)
print("THUMB sign", S_T, "angles", best[1], "cost", round(best[0], 3), "pad miss mm", round((p - T).length * 1000, 1),
      "facing", round(nrm.dot((C - p).normalized()), 2))

db = depth_by_bone(C)
print("PENETRATION mm", {k: round(x * 1000, 1) for k, x in sorted(db.items(), key=lambda kv: -kv[1]) if x > 0})

# 5. Two-seam: lay the seams' narrowest stretch (where the two seams run parallel) along the top
#    fingers, centred between them, so each finger runs along the inside of one seam. Seam curve =
#    ball.js seamCurve (tennis-ball curve, k = 0.3).
K_ = 0.3
seam = []
for i in range(600):
    t = 2 * math.pi * i / 600
    v = Vector((math.cos(t) + K_ * math.cos(3 * t), math.sin(t) - K_ * math.sin(3 * t),
                2 * math.sqrt(K_ * (1 - K_)) * math.sin(2 * t)))
    seam.append(v.normalized())
far = lambda i, j: min(abs(i - j), 600 - abs(i - j)) > 100
i0, j0 = min(((i, j) for i in range(0, 600, 2) for j in range(0, 600, 2) if far(i, j)),
             key=lambda ij: (seam[ij[0]] - seam[ij[1]]).length)
m0 = (seam[i0] + seam[j0]).normalized()
al0 = seam[(i0 + 1) % 600] - seam[i0 - 1]; al0 = (al0 - al0.dot(m0) * m0).normalized()
B0 = Matrix((m0, al0, m0.cross(al0))).transposed()          # columns: out, along, across (ball frame)
p2_, p3_ = pad("finger2-3.R", G["rf"])[0], pad("finger3-3.R", G["rf"])[0]
m1 = ((p2_ + p3_) / 2 - C).normalized()
fd = (P["finger2-3.R"].tail - P["finger2-2.R"].head) + (P["finger3-3.R"].tail - P["finger3-2.R"].head)
al1 = (fd - fd.dot(m1) * m1).normalized()
B1 = Matrix((m1, al1, m1.cross(al1))).transposed()          # same, at the fingers
M = B1 @ B0.transposed()
print("SEAM narrowest chord mm", round((seam[i0] - seam[j0]).length * R_BALL * 1000, 1))

# 6. Seat the ball on the forearm line. The viewer draws the ball at the skeleton's hand joint, on the
#    forearm axis 0.28 ft past the wrist (handModels.js), so flex the wrist until the ball centre lies
#    on the forearm line; step 7 then slides the hand back along that line to put it at 0.28 ft.
fa = (P["lowerarm02.R"].tail - P["lowerarm01.R"].head).normalized()
qw = (C - W).rotation_difference(fa)
turn("wrist.R", qw.axis, qw.angle)
C = W + qw @ (C - W); M = qw.to_matrix() @ M
W, d, l, n = frame()
print("WRIST turned to seat the ball, deg", round(math.degrees(qw.angle), 1))

# 7. Cut the hand, scale it, export it in the slot frame.
grip, to_slot, FT = export_hand("grip", C, fa, W, n, l, ball=True)

# ---- preview (review only): the hand in the slot frame with the ball (two-seam seams) and the
#      figure's forearm (figure.js: taper elbow -> wrist, radius 0.13 -> 0.095 ft) ----
if PREVIEW_DIR:
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    def obj(name, mesh_fn, rgb):
        mesh_fn(); o = bpy.context.object; o.name = name; o.color = (*rgb, 1); return o
    ball = obj("ball", lambda: bpy.ops.mesh.primitive_uv_sphere_add(radius=R_FT, location=bl(Vector((0.28, 0, 0))), segments=64, ring_count=32), (0.95, 0.94, 0.9))
    bpy.ops.object.shade_smooth()
    cu = bpy.data.curves.new("seam", "CURVE"); cu.dimensions = "3D"; cu.bevel_depth = 0.0011 * FT
    sp = cu.splines.new("POLY"); sp.points.add(len(seam) - 1); sp.use_cyclic_u = True
    for i, s_ in enumerate(seam): sp.points[i].co = (*bl(to_slot(C + (M @ s_) * R * 1.004)), 1)
    seam_ob = bpy.data.objects.new("seam", cu); bpy.context.scene.collection.objects.link(seam_ob); seam_ob.color = (0.75, 0.1, 0.12, 1)
    arm = obj("forearm", lambda: bpy.ops.mesh.primitive_cone_add(vertices=24, radius1=0.13, radius2=0.095, depth=0.85,
              location=bl(Vector((-0.425, 0, 0))), rotation=(0, math.pi / 2, 0)), (0.75, 0.52, 0.38))
    bpy.ops.object.shade_smooth()
    grip.color = (0.79, 0.55, 0.4, 1)
    sc = bpy.context.scene; sc.render.engine = "BLENDER_WORKBENCH"; sc.render.resolution_x = sc.render.resolution_y = 450
    sc.display.shading.light = "STUDIO"; sc.display.shading.color_type = "OBJECT"; sc.display.shading.show_object_outline = True
    cam = bpy.data.objects.new("c", bpy.data.cameras.new("c")); sc.collection.objects.link(cam); sc.camera = cam
    cam.data.type = "ORTHO"; cam.data.ortho_scale = 0.75
    mid = bl(Vector((0.18, 0, 0)))
    up = bl(Vector((1, 0, 0)))                     # fingers up in the image
    for k, v in {"back": (0, 1, 0), "palm": (0, -1, 0), "thumbside": (0, 0, 1), "pinkyside": (0, 0, -1),
                 "q1": (0.3, 0.7, 0.6), "q2": (0.3, -0.6, 0.7)}.items():
        dv = bl(Vector(v)).normalized(); pos = mid + dv * 3
        f = (mid - pos).normalized(); upv = up if abs(f.dot(up)) < 0.95 else bl(Vector((0, 1, 0)))
        r = f.cross(upv).normalized(); u = r.cross(f)
        cam.matrix_world = Matrix.Translation(pos) @ Matrix((r, u, -f)).transposed().to_4x4()
        sc.render.filepath = os.path.join(PREVIEW_DIR, f"grip_{k}.png"); bpy.ops.render.render(write_still=True)
