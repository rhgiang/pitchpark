"""Build the viewer's pitcher body (pitcher.glb) in Blender, headless, from a reference model.

    blender --background --factory-startup --python scripts/blender/make_pitcher.py -- <preview_dir> [out_dir]

Source: ref/pitcher.glb (gitignored), an image-to-3D model (TRELLIS, MIT licence) of a T-posed
ballplayer generated from a concept image. It gives the SHAPE (head, build, uniform); its texture
is thrown away (the back of it is black: the generator only saw the front) and every region is
repainted in the viewer's toon colours, so it matches the rest of Pitch Park.

Steps (plain terms):
  1. Import, turn into the viewer's axes (x = the figure's left, y up, z = the way he faces), scale
     to the rig's height.
  2. Rebuild the surface on a voxel grid (closes holes, evens out the triangles), smooth it (the
     generator's faceting would read as noise under toon shading), thin it to the budget.
  3. Paint each face by region (belt, socks, sleeve trim, skin, hair, shoes), from the heights the
     reference's own texture shows on its front.
  4. Fit it to the viewer's skeleton (web/src/skeleton.js rest T-pose). The model's joints are a
     little off ours (arms 0.26 ft lower, feet wider apart), so every vertex is moved by the joint
     displacements of its nearest bones, blended along each bone: the same "linear blend" idea as
     skinning (Magnenat-Thalmann et al. 1988; three.js SkinnedMesh), with translations only. The
     animation then rotates the limbs about the right pivots.
  5. Cut the arms at the wrist joints and close the ends: the glove (mitt.glb) and the throwing
     hand (grip.glb) are separate models the viewer puts on the wrists.
  6. Rig: the viewer's 24 bones at its rest pose, the body weighted to them automatically (bone
     heat), the cap rigid on the head; export web/public/models/pitcher.glb.
"""
import math
import os
import re
import sys

import bmesh
import bpy
from mathutils import Matrix, Vector

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
PREVIEW_DIR = argv[0] if argv else None
OUT_DIR = argv[1] if len(argv) > 1 else os.path.join(ROOT, "web", "public", "models")
REF = os.path.join(ROOT, "ref", "pitcher.glb")

# Viewer colours (web/src/look.js PALETTE, figure.js) for the previews only; the viewer picks its
# own colour from each material's name.
PREVIEW_RGB = {"jersey": 0xF3F0E8, "pants": 0xE6E1D4, "navy": 0x2A4A6B, "skin": 0xC98D66,
               "hair": 0x2B1D14, "shoe": 0x1C1C1C}
MATS = list(PREVIEW_RGB)


def read_joints():
    """{smpl name: (rest position, parent name)} from web/src/skeleton.js's JOINTS table."""
    src = open(os.path.join(ROOT, "web", "src", "skeleton.js"), encoding="utf-8").read()
    rows = re.findall(r'smpl: "(\w+)".*?parent: (-?\d+), rest: \[([^\]]+)\]', src)
    assert len(rows) == 24, f"expected 24 joints in skeleton.js, found {len(rows)}"
    names = [r[0] for r in rows]
    return {n: (Vector([float(v) for v in r.split(",")]), names[int(p)] if int(p) >= 0 else None) for n, p, r in rows}


JT = read_joints()
V = lambda x, y, z: Vector((x, y, z))
bl = lambda p: Vector((p[0], -p[2], p[1]))  # viewer (Y-up) -> Blender (Z-up)
vw = lambda p: Vector((p[0], p[2], -p[1]))  # Blender -> viewer

# ---------- tuning knobs (viewer ft, after scaling; feedback rounds should only touch these) ----------
HEIGHT = 6.15        # top of the head (hair); the cap goes on top, to the pitcher's height (6.25)
Z_SHIFT = 0.15       # the reference stands this far behind the skeleton's centre line
VOXEL = 0.022        # remesh grid (smaller = crisper, more triangles)
SMOOTH = 12          # smoothing passes after the remesh (fewer: bumps that toon shading turns into blotches)
BODY_TRIS = 9000     # triangle budget
# Region boundaries read off the reference's texture (front centre line, leg, arm, side of head).
SHOE_Y = 0.42        # cleats below
SOCK_Y = 1.38        # navy socks below, knicker pants above
BELT = (3.44, 3.58)  # navy belt
BELT_LOOPS = dict(front=(0.23, 0.28), back=(0.17, 0.21))  # |x| of the white belt loops (reference texture)
TRIM_X = (1.21, 1.29)  # sleeve trim |x|; skin (forearm) beyond
NECK_Y = 5.02        # skin above (front of the collar; the back sits a little higher)
HAIR_Y = 5.76        # hairline height at the sides (just behind the face)
NAPE = (-0.3, 5.45)  # hairline at the back: (z relative to the head's centre, y); it slopes between
FACE_Z = 0.12        # face: in front of this (relative to the head's centre) stays skin
HEAD_Y = 5.35       # all skin above this (ears), outside the hair
NECK_HALF = 0.3      # |x| of the neck opening (the shoulders either side stay jersey)
# Slimming (Ryan: the reference's torso and neck read too beefy, and out of proportion). Measured on
# the fitted body: chest ~0.9 ft deep (an athlete in a jersey ~0.75), neck base ~0.7 x 0.6 ft (a real
# neck ~0.4 x 0.4), and hips/seat 1.2 x 0.87 ft, wider and deeper than the chest (pear-shaped; an
# athlete's hips are about as wide as his chest, the shoulders wider).
SLIM = dict(
    torso=(1.0, 0.84),    # x (width) and z (depth) scale of the trunk, belt to shoulders
    hips=(0.88, 0.86),    # the same for the seat and hips
    hips_y=(2.7, 2.95, 3.35, 3.6),
    torso_y=(3.55, 3.85, 4.85, 5.1),  # ramps in over the first two heights, out over the last two
    neck=0.72,            # x/z scale of the neck and the trapezius slope round it
    neck_y=(4.95, 5.15, 5.3, 5.45),
    neck_r=(0.3, 0.5),    # |x| from the centre: full neck slimming inside, none outside (the shoulders)
    arm_x=(0.45, 0.72),   # |x|: trunk slimming fades out here, so the arms (T-pose) keep their shape
    trap=0.14,            # ft the ring round the neck (chest top, trapezius, upper back) is lowered: the
                          # reference's chest rose to the chin, hiding the neck in front
    trap_y=(4.75, 4.95, 5.15, 5.25),  # heights it applies to (stops below the jaw)
    trap_x=(0.16, 0.26, 0.42, 0.62),  # distance from the neck's axis: ramps in off the neck, out by the shoulders
)
# Reshaping (Ryan, after slimming: the torso still had parts "weirdly protruding" and no real neck).
# The reference's trunk surface is lumpy (image-to-3D), and scaling keeps the lumps. So below the
# armpits each horizontal slice is moved onto a clean cross-section with a designed taper, the area
# round the shoulders is smoothed, and the neck becomes a clean oval column.
TRUNK = dict(
    # (height, half-width, half-depth) ft: waist ~1.0 x 0.68, chest ~1.18 x 0.77, widening to the armpits
    # (Ryan, next round: "very bulky": chest/back narrowed ~6%, shoulders slimmed below.)
    profile=[(3.45, 0.49, 0.34), (3.8, 0.48, 0.33), (4.15, 0.51, 0.345), (4.5, 0.55, 0.365), (4.8, 0.58, 0.355)],
    power=2.6,            # superellipse exponent: 2 an ellipse, higher boxier (a torso is a rounded box)
    y=(3.38, 3.52, 4.72, 4.95),   # reshaped fully between the middle two heights, ramps outside them
    smooth_y=(4.6, 5.25),         # then smoothed between these heights (chest, shoulders, upper back)
    smooth_x=0.75,                # ... inside this |x| (not the arms)
    smooth_iters=15,
)
SHOULDERS = dict(x=0.92, y=(4.55, 4.8, 5.6, 5.7), fade_x=(0.75, 1.0))  # x scale above the chest; fades out along the arm
# Forearms: the reference's curves forward (palm forward), so at the wrist cut it sat ~0.2 ft in front of the
# wrist joint, where the hand models attach: the hands looked detached. Each slice is moved onto the elbow-wrist
# line and the end tapered to the hand's wrist (grip/open stubs ~0.14 x 0.19 ft; a little wider so they tuck in).
FOREARM = dict(x=(1.72, 1.9),       # |x|: the tube takes over between these (just past the elbow joint at 1.67)
               elbow=(0.12, 0.125),  # half-size (y up, z forward) just below the elbow, ft
               wrist=(0.075, 0.1),   # at the wrist cut: the hand models' wrist (grip/open stubs ~0.14 x 0.19)
               smooth=8)             # smoothing passes where the tube meets the elbow
NECK = dict(r=(0.2, 0.185), y=(4.88, 5.0, 5.24, 5.32), reach=0.34)  # oval half-axes; band; |dist| from its axis
ARM_CUT = 2.52       # |x| of the wrist joints (skeleton.js): the hand models (grip.glb, mitt.glb) start here


def material(name):
    """A material named for the viewer colour; preview colour from PREVIEW_RGB (sRGB -> linear)."""
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    h = PREVIEW_RGB[name]
    lin = lambda c: (c / 255) / 12.92 if c / 255 <= 0.04045 else (((c / 255) + 0.055) / 1.055) ** 2.4
    m.diffuse_color = (lin(h >> 16 & 255), lin(h >> 8 & 255), lin(h & 255), 1)
    return m


def apply_modifiers(ob):
    """Bake the object's modifier stack into its mesh."""
    bpy.context.view_layer.update()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(bpy.context.evaluated_depsgraph_get()))
    ob.modifiers.clear()
    ob.data = me


def tris(ob):
    return sum(len(p.vertices) - 2 for p in ob.data.polygons)


def import_ref():
    """The reference as one mesh object in Blender axes, scaled to HEIGHT, feet on the ground,
    centred on the skeleton (x, and z by Z_SHIFT)."""
    bpy.ops.import_scene.gltf(filepath=REF)
    obs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    bpy.context.view_layer.update()
    for o in obs:
        o.data.transform(o.matrix_world)
        o.parent = None
        o.matrix_world = Matrix.Identity(4)
    for o in [o for o in bpy.context.scene.objects if o.type != "MESH"]:
        bpy.data.objects.remove(o)
    bpy.ops.object.select_all(action="DESELECT")
    for o in obs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = obs[0]
    if len(obs) > 1:
        bpy.ops.object.join()
    ob = bpy.context.view_layer.objects.active
    ob.name = "pitcher_body"
    zs = [v.co.z for v in ob.data.vertices]
    xs = [v.co.x for v in ob.data.vertices]
    s = HEIGHT / (max(zs) - min(zs))
    cx = (max(xs) + min(xs)) / 2
    ob.data.transform(Matrix.Translation(bl(V(0, 0, Z_SHIFT))) @ Matrix.Diagonal((s, s, s, 1))
                      @ Matrix.Translation(Vector((-cx, 0, -min(zs)))))
    return ob


def clean(ob):
    """Voxel remesh (watertight, even triangles), smooth, thin to BODY_TRIS."""
    m = ob.modifiers.new("remesh", "REMESH")
    m.mode, m.voxel_size = "VOXEL", VOXEL
    m = ob.modifiers.new("smooth", "SMOOTH")
    m.factor, m.iterations = 0.5, SMOOTH
    apply_modifiers(ob)
    for _ in range(4):  # decimating lands a little high: repeat
        if tris(ob) <= BODY_TRIS:
            break
        m = ob.modifiers.new("decimate", "DECIMATE")
        m.ratio = BODY_TRIS / tris(ob) * 0.98
        apply_modifiers(ob)


def region(c, head):
    """Material for a face at viewer position c (before the fit to the skeleton)."""
    ax = abs(c.x)
    if c.y < SHOE_Y:
        return "shoe"
    if c.y < SOCK_Y:
        return "navy"
    if ax > TRIM_X[1]:
        return "skin"
    if ax > TRIM_X[0] and c.y > 4.4:
        return "navy"
    if c.y > HEAD_Y and (hair_side(c, head) > 0 and c.z - head.z < FACE_Z or c.y > HAIR_Y + 0.2):
        return "hair"  # above the sloped hairline (sides and back), or above the forehead
    if c.y > NECK_Y + (0.12 if c.z < head.z else 0) and (ax < NECK_HALF or c.y > HEAD_Y):
        return "skin"
    if BELT[0] < c.y < BELT[1]:
        lo, hi = BELT_LOOPS["front" if c.z > 0.05 else "back"]
        return "pants" if lo < ax < hi else "navy"
    return "pants" if c.y < BELT[0] else "jersey"


def hairline(head):
    """The hairline as a plane (point, normal; viewer axes): through HAIR_Y just behind the face
    and NAPE at the back, so the hair reaches down the back of the head to the neck."""
    a, b = V(0, HAIR_Y, head.z + FACE_Z), V(0, NAPE[1], head.z + NAPE[0])
    t = (a - b).normalized()
    return a, V(0, t.z, -t.y) * (1 if t.z > 0 else -1)  # perpendicular to the slope, pointing up


def hair_side(c, head):
    """> 0 above the hairline."""
    a, n = hairline(head)
    return (c - a).dot(n)


def split_at_boundaries(ob, head):
    """Cut the surface along each colour boundary (bisect), so the paint has straight edges
    instead of following the triangles' zigzag."""
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    cuts = [(V(0, y, 0), V(0, 1, 0)) for y in (SHOE_Y, SOCK_Y, *BELT, NECK_Y, NECK_Y + 0.12, HAIR_Y, HAIR_Y + 0.2)]
    cuts += [hairline(head), (V(0, 0, head.z + FACE_Z), V(0, 0, 1))]
    cuts += [(V(s * x, 0, 0), V(1, 0, 0)) for x in (*TRIM_X, NECK_HALF) for s in (1, -1)]
    for co, no in cuts:
        geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
        bmesh.ops.bisect_plane(bm, geom=geom, plane_co=bl(co), plane_no=bl(no))
    # Belt loop edges: vertical cuts through the belt only (through everything they cost ~4k triangles).
    for x in (*BELT_LOOPS["front"], *BELT_LOOPS["back"]):
        for sx in (1, -1):
            faces = [f for f in bm.faces if BELT[0] - 0.05 < vw(f.calc_center_median()).y < BELT[1] + 0.05]
            geom = list({e for f in faces for e in f.edges}) + list({v for f in faces for v in f.verts}) + faces
            bmesh.ops.bisect_plane(bm, geom=geom, plane_co=bl(V(sx * x, 0, 0)), plane_no=bl(V(1, 0, 0)))
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bm.to_mesh(ob.data)
    bm.free()


def paint(ob):
    ob.data.materials.clear()
    for name in MATS:
        ob.data.materials.append(material(name))
    head_pts = [vw(v.co) for v in ob.data.vertices if vw(v.co).y > 5.5]
    head = sum(head_pts, Vector()) / len(head_pts)
    split_at_boundaries(ob, head)
    for f in ob.data.polygons:
        f.material_index = MATS.index(region(vw(f.center), head))
    ob.data.shade_smooth()


def source_joints(ob):
    """Where the reference's joints are, measured from its shape: the skeleton's joints, with the
    arm chain moved to the arm's centre line and the knees/ankles to each leg's centre."""
    P = [vw(v.co) for v in ob.data.vertices]

    def centre(sel, axis):
        q = [getattr(p, axis) for p in P if sel(p)]
        return (min(q) + max(q)) / 2
    src = {n: p.copy() for n, (p, _) in JT.items()}
    for s, x in (("left", 1), ("right", -1)):
        dy = centre(lambda p: abs(p.x - x * 1.67) < 0.04, "y") - JT[f"{s}_elbow"][0].y
        for j in ("shoulder", "elbow", "wrist", "hand"):
            src[f"{s}_{j}"].y += dy
        src[f"{s}_collar"].y += dy / 2
        for j, y in (("knee", JT[f"{s}_knee"][0].y), ("ankle", 0.6), ("foot", 0.15)):
            src[f"{s}_{j}"].x = centre(lambda p: abs(p.y - y) < 0.04 and p.x * x > 0.05, "x")
        print(f"[make_pitcher] {s}: arm dy {dy:.3f}, knee x {src[f'{s}_knee'].x:.2f}, ankle x {src[f'{s}_ankle'].x:.2f}")
    return src


def fit(ob, src):
    """Move each vertex by the joint displacements (src -> skeleton rest) of its two nearest bones,
    blended along each bone and between the bones by inverse distance (d^-4)."""
    bones = [(c, p) for c, (_, p) in JT.items() if p]
    delta = {n: JT[n][0] - src[n] for n in JT}
    for v in ob.data.vertices:
        p = vw(v.co)
        near = []
        for c, par in bones:
            a, b = src[par], src[c]
            ab = b - a
            t = min(max((p - a).dot(ab) / max(ab.length_squared, 1e-9), 0), 1)
            near.append(((p - (a + ab * t)).length, delta[par].lerp(delta[c], t)))
        near.sort(key=lambda e: e[0])
        w = [1 / (d ** 4 + 1e-6) for d, _ in near[:2]]
        move = sum((d * wi for (_, d), wi in zip(near[:2], w)), Vector()) / sum(w)
        v.co = bl(p + move)


def slim(ob):
    """Scale the trunk and neck toward the body's centre line (x = 0, z = the spine's), each inside
    a band of heights with smooth ramps at both ends and fading out sideways, so no crease appears
    where the slimmed part meets the rest (the shoulders, arms, hips and head are untouched)."""
    k = SLIM
    zc = JT["spine3"][0].z

    def band(y, a, b, c, d):
        return smoothstep((y - a) / (b - a)) * (1 - smoothstep((y - c) / (d - c)))

    for v in ob.data.vertices:
        p = vw(v.co)
        ax = abs(p.x)
        wt = band(p.y, *k["torso_y"]) * (1 - smoothstep((ax - k["arm_x"][0]) / (k["arm_x"][1] - k["arm_x"][0])))
        wn = band(p.y, *k["neck_y"]) * (1 - smoothstep((ax - k["neck_r"][0]) / (k["neck_r"][1] - k["neck_r"][0])))
        wh = band(p.y, *k["hips_y"])
        sx = 1 - (1 - k["torso"][0]) * wt - (1 - k["neck"]) * wn - (1 - k["hips"][0]) * wh
        sz = 1 - (1 - k["torso"][1]) * wt - (1 - k["neck"]) * wn - (1 - k["hips"][1]) * wh
        dn = math.hypot(p.x - JT["neck"][0].x, p.z - JT["neck"][0].z)
        drop = k["trap"] * band(p.y, *k["trap_y"]) * band(dn, *k["trap_x"])
        v.co = bl(V(p.x * sx, p.y - drop, zc + (p.z - zc) * sz))


def _centre(pts, y0, y1, sel, x0=None, step=0.1):
    """Centre (x, z) of the selected points' horizontal slices as a function of height: bounding-box
    middles of 0.1 ft slices (thin slices hold too few vertices: their centres jump), interpolated
    linearly between slices. x0 fixes x (the trunk is symmetric about x = 0)."""
    box = {}
    for p in pts:
        if y0 - step <= p.y <= y1 + step and sel(p):
            k = math.floor(p.y / step)
            lo, hi = box.get(k, ((p.x, p.z), (p.x, p.z)))
            box[k] = ((min(lo[0], p.x), min(lo[1], p.z)), (max(hi[0], p.x), max(hi[1], p.z)))
    mid = {(k + 0.5) * step: ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2) for k, (lo, hi) in box.items()}
    ys = sorted(mid)

    def at(y):
        y = min(max(y, ys[0]), ys[-1])
        i = max(j for j in range(len(ys)) if ys[j] <= y)
        j = min(i + 1, len(ys) - 1)
        u = 0 if j == i else (y - ys[i]) / (ys[j] - ys[i])
        x = mid[ys[i]][0] + (mid[ys[j]][0] - mid[ys[i]][0]) * u
        return (x if x0 is None else x0), mid[ys[i]][1] + (mid[ys[j]][1] - mid[ys[i]][1]) * u
    return at


def reshape(ob):
    """Trunk and neck onto clean cross-sections (TRUNK, NECK), then smooth the shoulder area.

    Each vertex keeps its height and its angle round the slice's centre; only its distance from the
    centre changes, to the target shape's radius at that angle, blended in and out over the bands'
    ramps. Superellipse |x/a|^p + |z/b|^p = 1: radius at angle t (from +z toward +x) is
    1 / ((|sin t|/a)^p + (|cos t|/b)^p)^(1/p)."""
    k, nk = TRUNK, NECK
    band = lambda y, a, b, c, d: smoothstep((y - a) / (b - a)) * (1 - smoothstep((y - c) / (d - c)))
    prof = k["profile"]

    def ab(y):
        y = min(max(y, prof[0][0]), prof[-1][0])
        for (y0, a0, b0), (y1, a1, b1) in zip(prof, prof[1:]):
            if y <= y1:
                u = smoothstep((y - y0) / (y1 - y0))
                return a0 + (a1 - a0) * u, b0 + (b1 - b0) * u

    def sup(t, a, b, pw):
        return 1 / ((abs(math.sin(t)) / a) ** pw + (abs(math.cos(t)) / b) ** pw) ** (1 / pw)

    P = [vw(v.co) for v in ob.data.vertices]
    tc = _centre(P, k["y"][0], k["y"][3], lambda p: abs(p.x) < 0.7, x0=0.0)
    nx, nz = JT["neck"][0].x, JT["neck"][0].z
    nc = _centre(P, nk["y"][0], nk["y"][3], lambda p: math.hypot(p.x - nx, p.z - nz) < nk["reach"], x0=nx)
    for v, p in zip(ob.data.vertices, P):
        w = band(p.y, *k["y"]) * (1 - smoothstep((abs(p.x) - 0.7) / 0.15))
        if w > 0:
            cx, cz = tc(p.y)
            t = math.atan2(p.x - cx, p.z - cz)
            r0 = math.hypot(p.x - cx, p.z - cz)
            r = r0 + (sup(t, *ab(p.y), k["power"]) - r0) * w
            p = V(cx + r * math.sin(t), p.y, cz + r * math.cos(t))
        dn = math.hypot(p.x - nx, p.z - nz)
        w = band(p.y, *nk["y"]) * (1 - smoothstep((dn - nk["reach"] + 0.08) / 0.08))
        if w > 0:
            cx, cz = nc(p.y)
            t = math.atan2(p.x - cx, p.z - cz)
            r0 = math.hypot(p.x - cx, p.z - cz)
            r = r0 + (sup(t, *nk["r"], 2) - r0) * w
            p = V(cx + r * math.sin(t), p.y, cz + r * math.cos(t))
        v.co = bl(p)
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    verts = [v for v in bm.verts if k["smooth_y"][0] < vw(v.co).y < k["smooth_y"][1] and abs(vw(v.co).x) < k["smooth_x"]]
    for _ in range(k["smooth_iters"]):
        bmesh.ops.smooth_vert(bm, verts=verts, factor=0.5, use_axis_x=True, use_axis_y=True, use_axis_z=True)
    bm.to_mesh(ob.data)
    bm.free()


def forearms(ob):
    """Rebuild each forearm as a clean tapered oval tube round the elbow-wrist line (FOREARM). The
    reference's forearm curved forward and was lumpy; straightening it slice by slice kept the lumps
    and creased it (Ryan: "odd and unsmooth"). Each vertex keeps its position along the arm and its
    angle round the axis; its distance from the axis becomes the oval's radius there (half-sizes
    eased from the elbow's to the wrist's), blended in past the elbow. Then the region is smoothed."""
    k = FOREARM
    for side, sx in (("left", 1), ("right", -1)):
        ey, ez = JT[f"{side}_elbow"][0].y, JT[f"{side}_elbow"][0].z
        on_arm = lambda p: sx * p.x > k["x"][0] - 0.05 and abs(p.y - ey) < 0.4 and abs(p.z - ez) < 0.5
        for v in ob.data.vertices:
            p = vw(v.co)
            if not on_arm(p):
                continue
            u = smoothstep((sx * p.x - k["x"][0]) / (ARM_CUT - k["x"][0]))   # 0 elbow end .. 1 wrist
            hy = k["elbow"][0] + (k["wrist"][0] - k["elbow"][0]) * u
            hz = k["elbow"][1] + (k["wrist"][1] - k["elbow"][1]) * u
            t = math.atan2(p.y - ey, p.z - ez)
            r = 1 / math.hypot(math.sin(t) / hy, math.cos(t) / hz)
            r0 = math.hypot(p.y - ey, p.z - ez)
            w = smoothstep((sx * p.x - k["x"][0]) / (k["x"][1] - k["x"][0]))
            rr = r0 + (r - r0) * w
            v.co = bl(V(p.x, ey + rr * math.sin(t), ez + rr * math.cos(t)))
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    verts = [v for v in bm.verts if k["x"][0] - 0.15 < abs(vw(v.co).x) < k["x"][1] + 0.1]
    for _ in range(k["smooth"]):
        bmesh.ops.smooth_vert(bm, verts=verts, factor=0.5, use_axis_x=True, use_axis_y=True, use_axis_z=True)
    bm.to_mesh(ob.data)
    bm.free()


def shoulders(ob):
    """Narrow the upper body (SHOULDERS x scale) above the chest, fading out along the arms."""
    k = SHOULDERS
    band = lambda y, a, b, c, d: smoothstep((y - a) / (b - a)) * (1 - smoothstep((y - c) / (d - c)))
    for v in ob.data.vertices:
        p = vw(v.co)
        w = band(p.y, *k["y"]) * (1 - smoothstep((abs(p.x) - k["fade_x"][0]) / (k["fade_x"][1] - k["fade_x"][0])))
        if w > 0:
            v.co = bl(V(p.x * (1 - (1 - k["x"]) * w), p.y, p.z))


def smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def cut(ob, keep_inner, name=None, weld=False, sides=(1, -1)):
    """Slice the arms at |x| = ARM_CUT (bisect: a clean edge loop), keeping the body side
    (keep_inner) or the hands. With `name`, work on a copy (returned). `weld` merges the duplicate
    vertices the glTF import makes at texture seams (a subdivision would tear them apart)."""
    if name:
        ob = bpy.data.objects.new(name, ob.data.copy())
        bpy.context.scene.collection.objects.link(ob)
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    if weld:
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.002)
    for sx in sides:
        geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
        bmesh.ops.bisect_plane(bm, geom=geom, plane_co=V(sx * ARM_CUT, 0, 0), plane_no=V(sx, 0, 0),
                               clear_outer=keep_inner, clear_inner=not keep_inner)
    bm.to_mesh(ob.data)
    bm.free()
    return ob


# ---------- the cap ----------
# A structured six-panel ball cap (after Ryan's reference photos; no logos): crown of six panels
# meeting at a top button (centre seam at the front), an eyelet in each panel, a pre-curved brim
# with rows of stitching, and a twisted rope across the front. Viewer axes; phi = angle round the
# head from the front (+z), theta = 0 at the cap's rim up to 90 deg at the button.
CAP = dict(
    centre=V(0.01, 0, 0.045),  # x/z of the head's centre (measured: head x +-0.31, z -0.35..0.44)
    rim_y=(5.80, 5.70),        # rim height at the front and back (the hair shows below at the back)
    top_y=6.22,                # crown top; the button reaches the pitcher's height (6.25)
    r=(0.335, 0.44, 0.40),     # crown radii at the rim: across, to the front, to the back
    front_flat=0.8,            # front panels: < 1 keeps them upright longer (a structured front)
    clear=0.008,               # gap left between the hair (pressed down where needed) and the crown's inside
    thick=0.015,
    seam_r=0.007, eyelet=(0.018, 0.62), button=(0.035, 0.018),
    brim=dict(arc=70, length=0.26, side_drop=0.17, front_drop=0.05, t=0.022, rows=6),
    rope=dict(arc=72, r=0.011, twists=26, lift=0.025),
)
PREVIEW_RGB.update(capDark=0x1B3049, rope=0xE5604B)  # seams/stitching/eyelets; PALETTE.coral
MATS = list(PREVIEW_RGB)


def grid_mesh(name, rows, mat, closed=False):
    """A surface through rows of viewer points (same length each); closed: rows are loops."""
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    vs = [[bm.verts.new(bl(p)) for p in r] for r in rows]
    n = len(rows[0])
    for a, b in zip(vs, vs[1:]):
        for k in range(n if closed else n - 1):
            bm.faces.new((a[k], a[(k + 1) % n], b[(k + 1) % n], b[k]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(material(mat))
    me.shade_smooth()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def tube(name, pts, r, mat, sides=6):
    """A thin round tube along a polyline (viewer points): seams, stitch rows, rope strands."""
    rings = []
    for i, p in enumerate(pts):
        t = (pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]).normalized()
        u = t.orthogonal().normalized()
        w = t.cross(u)
        rings.append([p + (u * math.cos(a) + w * math.sin(a)) * r for a in (2 * math.pi * k / sides for k in range(sides))])
    return grid_mesh(name, rings, mat, closed=True)


def cap_point(phi, theta, out=0.0):
    """A point on the ideal crown (degrees), `out` ft further from the head's centre."""
    k = CAP
    p, th = math.radians(phi), math.radians(theta)
    front = math.cos(p) > 0
    rz = k["r"][1] if front else k["r"][2]
    rim = k["rim_y"][0] + (k["rim_y"][1] - k["rim_y"][0]) * (1 - math.cos(p)) / 2
    # A structured front stays upright longer: its horizontal reach falls off as cos^front_flat.
    reach = math.cos(th) ** (k["front_flat"] if front else 1.0)
    c = V(k["centre"].x + (k["r"][0] + out) * reach * math.sin(p), rim + (k["top_y"] + out - rim) * math.sin(th),
          k["centre"].z + (rz + out) * reach * math.cos(p))
    return c


def press_hair(body, crown, o):
    """Like a real cap, flatten whatever is under the crown: each head vertex further out (from o,
    inside the head) than the crown's inside surface less CAP['clear'] is pulled back to it."""
    bpy.context.view_layer.update()
    lim = CAP["thick"] + CAP["clear"]
    for v in body.data.vertices:
        p = vw(v.co)
        if p.y < CAP["rim_y"][1] - 0.05 or abs(p.x) > 0.6:
            continue
        d = (p - o).normalized()
        ok, hit, _, _ = crown.ray_cast(bl(o), bl(d), distance=2.0)
        if ok and (p - o).length > (vw(hit) - o).length - lim:
            v.co = bl(o + d * ((vw(hit) - o).length - lim))


def build_cap(body):
    """Crown (the hair pressed down under it), panel seams, eyelets, button, brim with stitch rows, rope."""
    k = CAP
    o = V(k["centre"].x, 5.75, k["centre"].z)  # rays from inside the head
    bpy.context.view_layer.update()

    phis = [i * 7.5 for i in range(48)]
    thetas = [i * 6 for i in range(16)]  # 0..90
    crown = grid_mesh("cap_crown", [[cap_point(f, t) for f in phis] for t in thetas], "navy", closed=True)
    press_hair(body, crown, o)
    m = crown.modifiers.new("thick", "SOLIDIFY")
    m.thickness, m.offset = k["thick"], -1.0
    apply_modifiers(crown)
    surf = lambda f, t, out=0.0: cap_point(f, t) + (cap_point(f, t) - o).normalized() * out
    obs = [crown]
    for f in range(0, 360, 60):  # six panels: seams at 0 (front centre), 60, ... 300
        obs.append(tube("cap_seam", [surf(f, t, 0.003) for t in range(0, 91, 5)], k["seam_r"], "capDark"))
    for f in range(30, 360, 60):  # an eyelet in each panel
        e = surf(f, math.degrees(math.asin(k["eyelet"][1])), 0.002)
        n = (e - o).normalized()
        obs.append(tube("cap_eyelet", [e - n * 0.004, e + n * 0.004], k["eyelet"][0], "capDark", sides=10))
    top = surf(0, 90)
    obs.append(tube("cap_button", [top, top + V(0, k["button"][1], 0)], k["button"][0], "navy", sides=12))
    # Brim: from the crown's rim across the front, out along the rim's outward normal. Its length
    # tapers to the sides (a rounded "D"), its sides curve down (the pre-curve) and its front dips.
    b = k["brim"]
    us = [-b["arc"] + 2 * b["arc"] * i / 24 for i in range(25)]
    vs = [i / 8 for i in range(9)]

    def brim_pt(u, v):
        rim = surf(u, 0, -0.01)
        out = V(math.sin(math.radians(u)) * 0.4, 0, math.cos(math.radians(u))).normalized()
        reach = b["length"] * math.cos(math.radians(u * 90 / b["arc"])) ** 0.6  # to nothing at its ends
        s = (u / b["arc"]) ** 2
        f = reach * v / b["length"]  # how far out, 0..1: the drop grows with it
        return rim + out * reach * v - V(0, b["side_drop"] * s * f + b["front_drop"] * f ** 1.5, 0)
    brim = grid_mesh("cap_brim", [[brim_pt(u, v) for u in us] for v in vs], "navy")
    m = brim.modifiers.new("thick", "SOLIDIFY")
    m.thickness, m.offset = b["t"], 0.0
    apply_modifiers(brim)
    obs.append(brim)
    for j in range(b["rows"]):  # stitch rows following the brim's outline
        v = 0.25 + 0.7 * j / (b["rows"] - 1)
        obs.append(tube("cap_stitch", [brim_pt(u, v) + V(0, b["t"] / 2 + 0.002, 0) for u in us[1:-1]], 0.004, "capDark", sides=4))
    # Rope: two strands twisted round a line just above where the brim meets the crown.
    rp = k["rope"]
    line = [surf(-rp["arc"] + 2 * rp["arc"] * i / 120, 0, 0.012) + V(0, rp["lift"], 0) for i in range(121)]
    for strand in (0, math.pi):
        pts = []
        for i, c in enumerate(line):
            t = (line[min(i + 1, 120)] - line[max(i - 1, 0)]).normalized()
            n1 = (c - o).normalized()
            n1 = (n1 - t * n1.dot(t)).normalized()
            n2 = t.cross(n1)
            a = rp["twists"] * 2 * math.pi * i / 120 + strand
            pts.append(c + (n1 * math.cos(a) + n2 * math.sin(a)) * rp["r"] * 0.8)
        obs.append(tube("cap_rope", pts, rp["r"], "rope"))
    return obs


def render(name):
    """Workbench (flat studio-lit) previews: front, side, back, head, three-quarter; review only."""
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.light, sc.display.shading.color_type = "STUDIO", "MATERIAL"
    sc.display.shading.show_cavity = True
    sc.display.shading.show_object_outline = True
    sc.view_settings.view_transform = "Standard"
    sc.render.resolution_x, sc.render.resolution_y = 500, 700
    sc.render.film_transparent = True
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.data.lens = 50
    for view, d, target, dist in (("front", V(0.25, 0.1, 1), V(0, 3.2, 0), 13), ("side", V(1, 0.1, 0), V(0, 3.2, 0), 13),
                                  ("back", V(-0.25, 0.1, -1), V(0, 3.2, 0), 13), ("head", V(0.5, 0.15, 1), V(0, 5.75, 0.05), 4.2),
                                  ("three_q", V(0.7, 0.2, 1), V(0, 4.4, 0), 8)):
        dd = bl(d).normalized()
        cam.location = bl(target) + dd * dist
        xc = Vector((0, 0, 1)).cross(dd).normalized()
        cam.rotation_euler = Matrix((xc, dd.cross(xc), dd)).transposed().to_euler()
        sc.render.filepath = os.path.join(PREVIEW_DIR, f"{name}_{view}.png")
        bpy.ops.render.render(write_still=True)


def close_wrists(ob):
    """Fill each wrist's open edge loop (left by cut) with a flat cap, so the arm is a closed
    surface: bone heat weighting needs one, and no hole shows if a hand model sits off-centre."""
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    for sx in (1, -1):
        rim = [e for e in bm.edges if e.is_boundary and all(abs(v.co.x - sx * ARM_CUT) < 1e-3 for v in e.verts)]
        out = bmesh.ops.holes_fill(bm, edges=rim, sides=0)
        bmesh.ops.triangulate(bm, faces=out["faces"])
        for f in out["faces"]:
            f.material_index = MATS.index("skin")
    bm.to_mesh(ob.data)
    bm.free()


# Each bone runs from its joint to this child (skeleton.js parents); a leaf runs this way (viewer ft).
BONE_TO = {"pelvis": "spine1", "spine3": "neck", "left_collar": "left_shoulder", "right_collar": "right_shoulder"}
LEAF = {"head": V(0, 0.45, 0), "left_hand": V(0.15, 0, 0), "right_hand": V(-0.15, 0, 0),
        "left_foot": V(0, 0, 0.15), "right_foot": V(0, 0, 0.15)}


# Jersey piping (Ryan: the repaint lost the uniform's details). Read off the reference's textured
# front (render: 0.0037 ft per pixel): navy piping round a crew neck and down a button placket to the
# belt (just left of centre). The reference also has a "Y" from the neck to the placket; in toon
# shading it read as a necklace (Ryan: "weird neckline"), so it is left out. Viewer ft, fitted body.
PIPING = dict(
    r=0.016,                  # tube radius
    neck=(5.07, 5.22),        # crew-neck plane height at the front and at the back (low in front: more neck)
    neck_reach=0.42,          # horizontal distance from the neck joint searched for the neck loop
    placket=(-0.04, 4.82, 3.6),  # x, from y, to y
)


def surface_line(body, pts, d):
    """Project viewer points onto the body along direction d (a ray from 1 ft back), lifted off the
    surface a little; points whose ray misses are dropped."""
    out = []
    for p in pts:
        ok, hit, nrm, _ = body.ray_cast(bl(p - d * 1.0), bl(d), distance=3.0)
        if ok:
            out.append(vw(hit) + vw(nrm) * PIPING["r"])
    return out


def piping(body):
    """The jersey's navy piping as thin tubes on the surface, with a clean crew neckline.

    Neckline: the neck is cut by a plane tilted low in front and higher at the back (like a real crew
    neck); the cut is an exact closed loop round the neck (the piping goes on it) and a straight paint
    edge: skin above the plane, jersey below, near the neck. Only faces near the neck are cut.
    Placket: a vertical line sampled densely and projected onto the chest from
    the front, so the tube follows the surface instead of cutting under it between points."""
    k = PIPING
    nx, nz = JT["neck"][0].x, JT["neck"][0].z
    yf, yb = k["neck"]
    rz = 0.25  # the plane's slope runs over the neck's depth
    slope = (yf - yb) / (2 * rz)  # dy/dz: higher toward the back (-z)
    co, no = V(nx, (yf + yb) / 2, nz), V(0, 1, -slope).normalized()
    near = lambda q: math.hypot(q.x - nx, q.z - nz) < k["neck_reach"] and 4.85 < q.y < 5.5
    bm = bmesh.new()
    bm.from_mesh(body.data)
    faces = [f for f in bm.faces if near(vw(f.calc_center_median()))]
    geom = list({e for f in faces for e in f.edges}) + list({v for f in faces for v in f.verts}) + faces
    cut = bmesh.ops.bisect_plane(bm, geom=geom, plane_co=bl(co), plane_no=bl(no))
    edges = [e for e in cut["geom_cut"] if isinstance(e, bmesh.types.BMEdge)]
    # The neck's contour: in each angle bin round the neck joint, the innermost point of the cut
    # (the cut also crosses the shoulders and the edge of the cut region further out).
    n = 48
    ang = lambda q: math.atan2(q.x - nx, q.z - nz)
    rad = lambda q: math.hypot(q.x - nx, q.z - nz)
    bins = [None] * n
    for v in {v for e in edges for v in e.verts}:
        q = vw(v.co)
        i = int((ang(q) / (2 * math.pi) % 1) * n) % n
        if bins[i] is None or rad(q) < rad(bins[i]):
            bins[i] = q
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])
    bm.to_mesh(body.data)
    bm.free()
    # The loop follows the real neck surface: at each angle a ray goes out from the neck's axis, on the
    # plane, to the first surface it meets (the neck). An ideal oval sank into the back of the neck,
    # which sits further out (Ryan: "whatever is happening at the back of the neck"). Smoothed, so a
    # stray hit does not kink it; a miss falls back to the oval.
    plane_y = lambda z: co.y + slope * (z - nz)
    a_, b_ = NECK["r"]
    oval = lambda t: 1 / math.hypot(math.sin(t) / a_, math.cos(t) / b_)
    ts = [2 * math.pi * i / n for i in range(n)]
    rs = []
    for t in ts:
        d = V(math.sin(t), slope * math.cos(t), math.cos(t)).normalized()   # outward, along the plane
        ok, hit, _, _ = body.ray_cast(bl(V(nx, co.y, nz)), bl(d), distance=0.6)
        r = math.hypot(vw(hit).x - nx, vw(hit).z - nz) if ok else 0
        rs.append(r if 0.12 < r < 0.4 else oval(t))
    rs = [sorted(rs[(i + j) % n] for j in range(-2, 3))[2] for i in range(n)]
    for _ in range(3):
        rs = [(rs[i - 1] + 2 * rs[i] + rs[(i + 1) % n]) / 4 for i in range(n)]
    loop = [V(nx + r * math.sin(t), plane_y(nz + r * math.cos(t)), nz + r * math.cos(t)) for t, r in zip(ts, rs)]
    print(f"[make_pitcher] neck loop r {min(rs):.2f}..{max(rs):.2f}")
    r_at = lambda q: rs[round((ang(q) / (2 * math.pi) % 1) * n) % n]
    skin, jersey = MATS.index("skin"), MATS.index("jersey")
    for f in body.data.polygons:  # skin above the plane and within the neck, jersey otherwise
        q = vw(f.center)
        if near(q) and f.material_index in (skin, jersey):
            above = (q - co).dot(no)
            chin = math.cos(ang(q)) > 0.6 and above > 0.1   # the jaw overhangs the neck in front
            # Above the plane is skin (the nape curves out past the neck's radius toward the hairline),
            # except the shoulder slope just above it, off the neck.
            shoulder = rad(q) >= r_at(q) + 0.03 and above < 0.06 and not chin
            f.material_index = skin if above > 0 and not shoulder else jersey
    # On the surface, not in it; the back half further out and up a hair, where the shoulder slope
    # rises round the neck and hid all but a stub of the trim.
    def lift(p):
        back = max(0.0, -math.cos(ang(p)))
        return p + (p - V(nx, p.y, nz)).normalized() * k["r"] * (0.8 + 1.6 * back) + V(0, 0.02 * back, 0)
    loop = [lift(p) for p in loop]
    tubes = [tube("pipe_neck", loop + loop[:1], k["r"], "navy", sides=6)]
    front = V(0, 0, -1)
    x0, _, y1 = k["placket"]
    y0 = max(loop, key=lambda p: p.z).y  # the placket runs from the collar's front, straight down
    tubes.append(tube("pipe_placket", surface_line(body, [V(x0, y0 + (y1 - y0) * i / 30, 0) for i in range(31)], front), k["r"], "navy", sides=6))
    return tubes


def rig_and_bind(body, cap, pipes):
    """The viewer's 24-bone skeleton (skeleton.js rest T-pose), with the body skinned to it by
    Blender's automatic weights and the cap rigid on the head.

    Automatic weights = "bone heat" (Baran & Popovic 2007, "Automatic rigging and animation of 3D
    characters"): each bone heats the surface, heat diffuses along it, and a vertex's weight for a
    bone is its share of the heat. Unlike nearest-bone weights it follows the surface, so the armpit
    blends arm and torso smoothly and the inner thighs do not grab the other leg.
    Only the weights reach the viewer: it binds the mesh to its own bones at the same rest pose."""
    kids = {}
    for n, (_, par) in JT.items():
        kids.setdefault(par, []).append(n)
    arm = bpy.data.objects.new("rig", bpy.data.armatures.new("rig"))
    bpy.context.scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    for n, (p, _) in JT.items():
        eb = arm.data.edit_bones.new(n)
        to = BONE_TO.get(n) or (kids[n][0] if len(kids.get(n, [])) == 1 else None)
        eb.head, eb.tail = bl(p), bl(JT[to][0] if to else p + LEAF[n])
    for n, (_, par) in JT.items():
        if par:
            arm.data.edit_bones[n].parent = arm.data.edit_bones[par]
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    body.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    unweighted = sum(1 for v in body.data.vertices if not any(g.weight > 0 for g in v.groups))
    assert unweighted == 0, f"bone heat left {unweighted} vertices unweighted"
    for o in pipes:  # piping bends with the jersey under it: copy the nearest surface's weights
        bpy.context.view_layer.objects.active = o
        for g in body.vertex_groups:
            o.vertex_groups.new(name=g.name)
        m = o.modifiers.new("weights", "DATA_TRANSFER")
        m.object, m.use_vert_data, m.data_types_verts = body, True, {"VGROUP_WEIGHTS"}
        m.vert_mapping, m.layers_vgroup_select_src, m.layers_vgroup_select_dst = "POLYINTERP_NEAREST", "ALL", "NAME"
        bpy.ops.object.modifier_apply(modifier=m.name)
        o.parent = arm
        o.modifiers.new("rig", "ARMATURE").object = arm
    for o in cap:
        o.vertex_groups.new(name="head").add(list(range(len(o.data.vertices))), 1.0, "REPLACE")
        o.parent = arm
        o.modifiers.new("rig", "ARMATURE").object = arm
    return arm


def export(body, cap, rig):
    """pitcher.glb: the body and cap in the rest pose (viewer axes, ft), skinned to the 24 bones by
    name, material names = viewer colours."""
    bpy.ops.object.select_all(action="DESELECT")
    for o in [body, rig, *cap]:
        o.select_set(True)
    path = os.path.join(OUT_DIR, "pitcher.glb")
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", use_selection=True, export_yup=True,
                              export_skins=True, export_animations=False, export_texcoords=False,
                              export_materials="EXPORT")
    print(f"[make_pitcher] wrote {path} ({os.path.getsize(path) // 1024} KB)")


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    body = import_ref()
    clean(body)
    paint(body)
    fit(body, source_joints(body))
    slim(body)
    forearms(body)
    shoulders(body)
    reshape(body)
    cut(body, True)
    close_wrists(body)
    pipes = piping(body)
    cap = build_cap(body)
    rig = rig_and_bind(body, cap, pipes)
    print(f"[make_pitcher] body {tris(body)}, cap {sum(tris(o) for o in cap)} triangles")
    if PREVIEW_DIR:
        os.makedirs(PREVIEW_DIR, exist_ok=True)
        render("pitcher")
    export(body, cap + pipes, rig)


main()
