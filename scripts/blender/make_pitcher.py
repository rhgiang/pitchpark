"""Build the viewer's pitcher body (pitcher.glb) in Blender, headless.

    blender --background --factory-startup --python scripts/blender/make_pitcher.py -- <preview_dir> [out_dir]

How it works (plain terms): the body is ONE continuous surface grown round a stick figure. The stick
figure's points sit on the viewer's skeleton joints (read from web/src/skeleton.js, so model and
animation share one source of truth) plus a few extra points that shape it (chest, mid-thigh, calf).
Each point gets a thickness (two radii: across and front-to-back), and Blender's Skin modifier wraps
one closed skin over the whole stick figure, joints and branches included. A Subdivision Surface
modifier then rounds it off. That is the classic "base mesh" way to block out a character: real
shoulders, elbows and knees instead of separate shapes stuck together.

The uniform is part of that same body (its thickness is the clothed silhouette: baggy knicker
pants, a jersey over a broad chest) and is coloured by region; a few separate pieces sit on top
where cloth really is a separate layer: sleeves, belt, socks' tops, shoes. Head, cap and hair are
modelled separately and overlap the neck.

Frame: written in the viewer's axes (three.js, Y up), feet, rest T-pose for a 75 in pitcher: x = the
figure's left, y up, z = the way he faces. bl() converts to Blender's Z-up axes; the glTF exporter
converts back. Each mesh's material name picks the viewer colour (figure.js COLORS).
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

# Viewer colours (web/src/look.js PALETTE, figure.js) for the previews only; the viewer picks its
# own colour from each material's name.
PREVIEW_RGB = {"jersey": 0xF3F0E8, "pants": 0xE6E1D4, "navy": 0x2A4A6B, "skin": 0xC98D66,
               "hair": 0x2B1D14, "shoe": 0x1C1C1C}


def read_joints():
    """{smpl name: rest position (viewer axes, ft)} from web/src/skeleton.js's JOINTS table."""
    src = open(os.path.join(ROOT, "web", "src", "skeleton.js"), encoding="utf-8").read()
    rows = re.findall(r'smpl: "(\w+)".*?rest: \[([^\]]+)\]', src)
    assert len(rows) == 24, f"expected 24 joints in skeleton.js, found {len(rows)}"
    return {n: Vector([float(v) for v in r.split(",")]) for n, r in rows}


JT = read_joints()
V = lambda x, y, z: Vector((x, y, z))


def bl(p):
    """Viewer (Y-up) -> Blender (Z-up): (x, y, z) -> (x, -z, y)."""
    return Vector((p[0], -p[2], p[1]))


# ---------- tuning knobs (ft): feedback rounds should only touch these ----------
# The stick figure: (name, position, (radius across, radius front-to-back)). Positions use the
# skeleton's joints where one exists; the rest are shaping points between them. The radii are the
# CLOTHED silhouette (jersey over the chest, baggy knicker pants).
def stick_points():
    j = JT
    pts = [
        ("pelvis", j["pelvis"] + V(0, -0.05, 0), (0.42, 0.29)),
        ("waist", j["spine1"] + V(0, 0, 0.02), (0.34, 0.24)),
        ("belly", j["spine2"], (0.37, 0.25)),
        ("chest", j["spine3"] + V(0, 0.3, 0.0), (0.56, 0.36)),
        ("upchest", V(0, 4.98, -0.02), (0.5, 0.33)),
        ("neck0", V(0, 5.2, -0.04), (0.21, 0.19)),
        ("neck", j["head"] + V(0, 0.05, -0.02), (0.165, 0.165)),  # up into the head: no gap
    ]
    for s, x in (("left", 1), ("right", -1)):
        P = lambda n: j[f"{s}_{n}"]
        pts += [
            (f"{s}_collar", P("collar") + V(0, -0.02, 0), (0.22, 0.2)),
            (f"{s}_shoulder", P("shoulder"), (0.2, 0.19)),
            (f"{s}_bicep", P("shoulder").lerp(P("elbow"), 0.45), (0.17, 0.16)),
            (f"{s}_elbow", P("elbow"), (0.125, 0.125)),
            (f"{s}_forearm", P("elbow").lerp(P("wrist"), 0.3), (0.14, 0.13)),
            (f"{s}_wrist", P("wrist"), (0.085, 0.07)),
            (f"{s}_hand", P("wrist").lerp(P("hand"), 0.55), (0.115, 0.055)),
            (f"{s}_hip", P("hip") + V(0, -0.08, 0), (0.28, 0.3)),
            (f"{s}_thigh", P("hip").lerp(P("knee"), 0.5) + V(0, 0, 0.02), (0.27, 0.29)),
            (f"{s}_knee", P("knee"), (0.22, 0.23)),
            (f"{s}_calf", P("knee").lerp(P("ankle"), 0.33) + V(0, 0, -0.02), (0.2, 0.22)),
            (f"{s}_shin", P("knee").lerp(P("ankle"), 0.7), (0.13, 0.14)),
            (f"{s}_ankle", P("ankle") + V(0, -0.06, 0.02), (0.1, 0.11)),  # down into the shoe
        ]
    return pts


STICK_EDGES = [("pelvis", "waist"), ("waist", "belly"), ("belly", "chest"), ("chest", "upchest"),
               ("upchest", "neck0"), ("neck0", "neck")]
for _s in ("left", "right"):
    STICK_EDGES += [("upchest", f"{_s}_collar"), (f"{_s}_collar", f"{_s}_shoulder"),
                    (f"{_s}_shoulder", f"{_s}_bicep"), (f"{_s}_bicep", f"{_s}_elbow"),
                    (f"{_s}_elbow", f"{_s}_forearm"), (f"{_s}_forearm", f"{_s}_wrist"),
                    (f"{_s}_wrist", f"{_s}_hand"),
                    ("pelvis", f"{_s}_hip"), (f"{_s}_hip", f"{_s}_thigh"), (f"{_s}_thigh", f"{_s}_knee"),
                    (f"{_s}_knee", f"{_s}_calf"), (f"{_s}_calf", f"{_s}_shin"), (f"{_s}_shin", f"{_s}_ankle")]
SUBDIV = 2                    # subdivision levels on the body (each level x4 faces)
SKIN_GROW = 1.42              # Skin + Subdivision shrink the surface to ~70% of the radii: compensate

BODY_VOXEL = 0.022            # remesh grid, ft (smaller = crisper shapes, more triangles)
BODY_SMOOTH = 6               # smoothing passes after the remesh
BODY_TRIS = 9000              # body triangle budget after thinning


def shapes(x):
    """Muscle and cloth volumes melted into the body, for the side x = +1 (left) / -1 (right):
    (centre, semi-axes (across, up, front-to-back)), viewer axes."""
    return [
        (V(x * 0.27, 5.2, -0.08), (0.32, 0.2, 0.22)),     # trapezius: shoulders up into the neck
        (V(x * 0.64, 5.08, -0.04), (0.27, 0.26, 0.26)),   # deltoid (under the sleeve)
        (V(x * 0.03, 4.62, 0.0), (0.6, 0.52, 0.39)),      # ribcage: the broad chest (both halves overlap)
        (V(x * 0.2, 4.72, 0.15), (0.3, 0.26, 0.24)),      # chest
        (V(x * 0.25, 4.45, -0.15), (0.27, 0.42, 0.18)),   # back / lats
        (V(x * 0.15, 3.68, 0.0), (0.33, 0.2, 0.27)),      # jersey bloused over the belt
        (V(x * 0.17, 3.12, -0.15), (0.23, 0.27, 0.22)),    # seat
        (V(x * 0.29, 2.55, 0.06), (0.24, 0.44, 0.24)),    # thigh (quads, baggy pants)
        (V(x * 0.3, 1.4, -0.02), (0.25, 0.2, 0.26)),      # knicker pants ballooning over the sock top
        (V(x * 1.92, 5.1, -0.05), (0.3, 0.15, 0.145)),    # forearm muscle below the elbow
        (V(x * 2.62, 5.08, 0.06), (0.1, 0.045, 0.055)),    # thumb (T-pose: palm down, thumb forward)
    ]


# Colour regions on the body (viewer axes): the jersey ends at the belt and at the sleeve; the
# pants run from the belt to just below the calf (knicker length); navy socks below; skin from the
# sleeve to the hand and on the neck.
BELT_Y = 3.47                 # belt line height
SLEEVE_X = 1.0                # |x| where the body's jersey colour ends (the sleeve covers past it)
PANTS_END_Y = 1.27            # knicker pants end (below the calf), socks below
NECK_Y = 5.19                 # skin above this (under the collar)

# Cloth layers: sleeve end |x| and radii (at the shoulder, at the end), belt height, collar height.
CLOTH = dict(sleeve_end=1.18, sleeve_r=(0.3, 0.215), belt_h=0.1, collar_y=5.2)

# Head: an egg (taller than wide) with a fuller jaw; centre and semi-axes.
HEAD_C, HEAD_R = V(0, 5.86, 0.04), (0.33, 0.37, 0.35)
JAW = dict(c=V(0, 5.66, 0.12), r=(0.25, 0.17, 0.23))
# Cap (fitted ball cap): crown of six panels over the head, taller at the front; a curved brim.
CAP = dict(c=V(0, 5.9, 0.03), r=(0.375, 0.33, 0.39), front_lift=0.1, brim_len=0.3, brim_w=0.29,
           brim_drop=0.035, brim_t=0.025, button_r=0.035)


def material(name):
    """A material named for the viewer colour; preview colour from PREVIEW_RGB (sRGB -> linear)."""
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    h = PREVIEW_RGB.get(name, 0xFF00FF)
    lin = lambda c: (c / 255) / 12.92 if c / 255 <= 0.04045 else (((c / 255) + 0.055) / 1.055) ** 2.4
    m.diffuse_color = (lin(h >> 16 & 255), lin(h >> 8 & 255), lin(h & 255), 1)
    return m


def link(name, me):
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def apply_modifiers(ob):
    """Bake the object's modifier stack into its mesh."""
    bpy.context.view_layer.update()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(bpy.context.evaluated_depsgraph_get()))
    ob.modifiers.clear()
    ob.data = me


def build_body():
    """The clothed body: Skin modifier round the stick figure, subdivided, coloured by region."""
    pts = stick_points()
    names = [p[0] for p in pts]
    me = bpy.data.meshes.new("body")
    me.from_pydata([bl(p[1]) for p in pts], [(names.index(a), names.index(b)) for a, b in STICK_EDGES], [])
    ob = link("pitcher_body", me)
    ob.modifiers.new("skin", "SKIN")
    sk = me.skin_vertices[0].data
    for i, (_, _, (rx, rz)) in enumerate(pts):
        sk[i].radius = (rx * SKIN_GROW, rz * SKIN_GROW)
    sk[names.index("pelvis")].use_root = True
    m = ob.modifiers.new("subsurf", "SUBSURF")
    m.levels = m.render_levels = SUBDIV
    apply_modifiers(ob)
    # Melt the muscle and cloth shapes into it: join, rebuild as one surface on a grid (voxel
    # remesh), smooth the joins, thin to the budget.
    parts = [ob] + [ellipsoid("shape", c, r, "jersey", 24, 16) for side in (1, -1) for c, r in shapes(side)]
    bpy.ops.object.select_all(action="DESELECT")
    for o in parts:
        o.select_set(True)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.join()
    m = ob.modifiers.new("remesh", "REMESH")
    m.mode, m.voxel_size = "VOXEL", BODY_VOXEL
    m = ob.modifiers.new("smooth", "SMOOTH")
    m.factor, m.iterations = 0.6, BODY_SMOOTH
    apply_modifiers(ob)
    have = sum(len(p.vertices) - 2 for p in ob.data.polygons)
    if have > BODY_TRIS:
        m = ob.modifiers.new("decimate", "DECIMATE")
        m.ratio = BODY_TRIS / have
        apply_modifiers(ob)
    ob.data.materials.clear()
    for name in ("jersey", "pants", "navy", "skin"):
        ob.data.materials.append(material(name))
    for f in ob.data.polygons:
        c = Vector((f.center.x, f.center.z, -f.center.y))  # viewer axes
        if c.y > NECK_Y or abs(c.x) > SLEEVE_X:
            k = "skin"
        elif c.y > BELT_Y:
            k = "jersey"
        elif c.y > PANTS_END_Y:
            k = "pants"
        else:
            k = "navy"
        f.material_index = ["jersey", "pants", "navy", "skin"].index(k)
    ob.data.shade_smooth()
    return ob


def ellipsoid(name, c, r, mat, seg=32, rings=20):
    """A UV-sphere ellipsoid (viewer centre c, semi-axes r = (x, y, z))."""
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=rings, radius=1.0)
    bmesh.ops.scale(bm, vec=Vector((r[0], r[2], r[1])), verts=bm.verts)  # Blender (x, y=-z, z=y)
    bmesh.ops.translate(bm, vec=bl(c), verts=bm.verts)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(material(mat))
    me.shade_smooth()
    return link(name, me)


def vw(p):
    """Blender (Z-up) -> viewer (Y-up); inverse of bl()."""
    return Vector((p[0], p[2], -p[1]))


def sweep(name, rings, mat, closed_ends=False):
    """A surface through `rings` (lists of viewer points, same length, each a closed loop)."""
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    vs = [[bm.verts.new(bl(p)) for p in r] for r in rings]
    n = len(rings[0])
    for a_, b_ in zip(vs, vs[1:]):
        for k in range(n):
            bm.faces.new((a_[k], a_[(k + 1) % n], b_[(k + 1) % n], b_[k]))
    if closed_ends:
        bm.faces.new(vs[0][::-1])
        bm.faces.new(vs[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(material(mat))
    me.shade_smooth()
    return link(name, me)


def contour(body, y, off, n=48, axis_c=V(0, 0, 0)):
    """The body's outline round a horizontal slice at height y, `off` ft outside its surface:
    rays cast inwards from all round (viewer axes), centred on axis_c's x/z."""
    out = []
    for k in range(n):
        a = 2 * math.pi * k / n
        d = V(math.sin(a), 0, math.cos(a))
        o = V(axis_c.x, y, axis_c.z) + d * 2.0
        ok, hit, _, _ = body.ray_cast(bl(o), bl(-d), distance=2.0)
        out.append((vw(hit) if ok else o - d * 1.5) + d * off)
    return out


def band(body, name, y, h, off, mat, axis_c=V(0, 0, 0)):
    """A cloth band (belt, collar) wrapped round the body at height y, h tall, off proud of it."""
    ob = sweep(name, [contour(body, y - h / 2, off, axis_c=axis_c), contour(body, y + h / 2, off, axis_c=axis_c)], mat)
    m = ob.modifiers.new("thick", "SOLIDIFY")
    m.thickness, m.offset = 0.012, -1.0
    apply_modifiers(ob)
    return ob


def piping(body, name, pts, along_dir, r=0.013, mat="navy"):
    """A thin cord pressed onto the body: each point of `pts` is moved onto the surface by a ray
    cast along -along_dir, then swept as a small round tube."""
    hits = []
    for p in pts:
        ok, hit, _, _ = body.ray_cast(bl(p + along_dir * 1.5), bl(-along_dir), distance=3.0)
        if ok:
            hits.append(vw(hit) + along_dir * (r * 0.6))
    rings = []
    for i, p in enumerate(hits):
        t = (hits[min(i + 1, len(hits) - 1)] - hits[max(i - 1, 0)]).normalized()
        u = along_dir.normalized()
        w = t.cross(u).normalized()
        u = w.cross(t).normalized()
        rings.append([p + (u * math.cos(q) + w * math.sin(q)) * r for q in (2 * math.pi * k / 8 for k in range(8))])
    return sweep(name, rings, mat, closed_ends=True)


def build_cloth(body):
    """Cloth layered over the body: short jersey sleeves (open, with navy trim), belt, collar,
    piping down the placket and the pants' outside seams."""
    bpy.context.view_layer.update()
    obs = []
    C = CLOTH
    for x in (1, -1):
        # sleeve: a flared tube from inside the shoulder out past the deltoid, open at its end
        sh = JT["left_shoulder" if x > 0 else "right_shoulder"]
        rings = []
        for k in range(7):
            t = k / 6
            cx = x * (0.5 + (C["sleeve_end"] - 0.5) * t)
            r = C["sleeve_r"][0] + (C["sleeve_r"][1] - C["sleeve_r"][0]) * t
            rings.append([V(cx, sh.y + math.sin(q) * r * 1.02, sh.z + math.cos(q) * r) for q in (2 * math.pi * j / 28 for j in range(28))])
        sl = sweep("sleeve", rings, "jersey")
        m = sl.modifiers.new("thick", "SOLIDIFY")
        m.thickness, m.offset = 0.02, -1.0
        apply_modifiers(sl)
        obs.append(sl)
        end = rings[-1]
        trim = sweep("sleeve_trim", [[p + (p - V(p.x, sh.y, sh.z)).normalized() * 0.006 + V(-x * 0.035, 0, 0) for p in end],
                                     [p + (p - V(p.x, sh.y, sh.z)).normalized() * 0.006 for p in end]], "navy")
        m = trim.modifiers.new("thick", "SOLIDIFY")
        m.thickness, m.offset = 0.012, -1.0
        apply_modifiers(trim)
        obs.append(trim)
        # pants: piping down the outside seam, hip to the knicker cuff
        obs.append(piping(body, "pants_piping", [V(x * 0.3, y, 0.0) for y in [3.42 - 0.2 * k for k in range(11)]], V(x, 0, 0)))
    obs.append(band(body, "belt", BELT_Y, C["belt_h"], 0.006, "navy"))
    for x in (1, -1):  # the knicker pants' cuff, over where pants meet sock
        obs.append(band(body, "pants_cuff", PANTS_END_Y + 0.02, 0.07, 0.008, "pants", axis_c=V(x * 0.3, 0, 0)))
    obs.append(band(body, "collar", C["collar_y"], 0.07, 0.004, "navy", axis_c=V(0, 0, -0.04)))
    obs.append(piping(body, "placket", [V(0.04, y, 0) for y in [3.52 + 0.16 * k for k in range(11)]], V(0, 0, 1)))
    return obs


def build_head():
    """Head (egg + jaw), ears, nose, and hair at the back and sides under the cap."""
    obs = [ellipsoid("head", HEAD_C, HEAD_R, "skin"), ellipsoid("jaw", JAW["c"], JAW["r"], "skin")]
    for x in (-0.32, 0.32):
        obs.append(ellipsoid("ear", V(x, 5.83, 0.02), (0.05, 0.09, 0.07), "skin", 12, 8))
    obs.append(ellipsoid("nose", V(0, 5.8, 0.39), (0.05, 0.065, 0.055), "skin", 12, 8))
    obs.append(ellipsoid("hair", V(0, 5.9, -0.09), (0.34, 0.2, 0.31), "hair"))
    return obs


def build_cap():
    """Fitted ball cap: six-panel crown (taller at the front), button on top, curved brim."""
    k = CAP
    me = bpy.data.meshes.new("cap")
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=36, v_segments=24, radius=1.0)
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.z < -0.02], context="VERTS")  # top half
    for v in bm.verts:  # shape: oval crown, the front panels standing taller
        x, y, z = v.co  # Blender: y = -forward
        front = max(-y, 0.0)
        v.co = Vector((x * k["r"][0], y * k["r"][2], z * k["r"][1] * (1 + k["front_lift"] * front)))
        v.co += bl(k["c"])
    bm.to_mesh(me)
    bm.free()
    crown = link("cap_crown", me)
    m = crown.modifiers.new("thick", "SOLIDIFY")
    m.thickness, m.offset = 0.02, 1.0
    apply_modifiers(crown)
    crown.data.materials.append(material("navy"))
    crown.data.shade_smooth()
    # Brim: a curved plate in front of the crown's rim: half-oval outline, bent down at its sides
    # and front edge (pre-curved, like a worn cap), with thickness.
    me = bpy.data.meshes.new("brim")
    bm = bmesh.new()
    nu, nv = 24, 8
    grid = []
    for i in range(nu + 1):
        a = math.pi * i / nu  # 0..pi round the half-oval
        row = []
        for jj in range(nv + 1):
            t = jj / nv  # 0 at the rim, 1 at the brim's edge
            ex, ez = math.cos(a) * k["brim_w"] * (0.92 + 0.08 * t), math.sin(a) * k["brim_len"] * t
            x = ex
            z = k["r"][2] * 0.92 * math.sin(a) + ez  # forward
            y = k["c"].y + 0.01 - k["brim_drop"] * (t ** 1.6) - 0.04 * abs(math.cos(a)) * t  # bends down
            row.append(bm.verts.new(bl(V(x, y, z + k["c"].z))))
        grid.append(row)
    for i in range(nu):
        for jj in range(nv):
            bm.faces.new((grid[i][jj], grid[i][jj + 1], grid[i + 1][jj + 1], grid[i + 1][jj]))
    bm.to_mesh(me)
    bm.free()
    brim = link("cap_brim", me)
    m = brim.modifiers.new("thick", "SOLIDIFY")
    m.thickness, m.offset = k["brim_t"], 0.0
    apply_modifiers(brim)
    brim.data.materials.append(material("navy"))
    brim.data.shade_smooth()
    button = ellipsoid("cap_button", k["c"] + V(0, k["r"][1] + 0.005, 0), (k["button_r"], 0.016, k["button_r"]), "navy", 12, 6)
    return [crown, brim, button]


def build_shoes():
    """Cleats: a rounded upper over the ankle and foot, on a flat sole."""
    obs = []
    for s in ("left", "right"):
        a, f = JT[f"{s}_ankle"], JT[f"{s}_foot"]
        c = V(f.x, 0.16, (a.z + f.z) / 2 + 0.02)
        obs.append(ellipsoid("shoe", c, (0.15, 0.13, 0.36), "shoe", 24, 14))
        obs.append(ellipsoid("sole", V(f.x, 0.04, c.z), (0.155, 0.04, 0.38), "jersey", 24, 8))
    return obs


def render(obs, name):
    """Workbench (flat studio-lit) previews: front, side, back, three-quarter; for review only."""
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
        up = Vector((0, 0, 1))
        xc = up.cross(dd).normalized()
        cam.rotation_euler = Matrix((xc, dd.cross(xc), dd)).transposed().to_euler()
        sc.render.filepath = os.path.join(PREVIEW_DIR, f"{name}_{view}.png")
        bpy.ops.render.render(write_still=True)


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    body = build_body()
    obs = [body] + build_cloth(body) + build_head() + build_cap() + build_shoes()
    tris = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in obs)
    print(f"[make_pitcher] {len(obs)} parts, {tris} triangles")
    if os.environ.get("PITCHER_MEASURE"):  # half-widths / depths of the body at a few heights
        for y in (5.1, 4.75, 4.4, 4.0, 3.6, 3.1, 2.5, 1.75, 1.4, 0.8):
            c = contour(body, y, 0.0, n=16)
            print(f"[measure] y {y}: x {min(p.x for p in c):.2f}..{max(p.x for p in c):.2f}  z {min(p.z for p in c):.2f}..{max(p.z for p in c):.2f}")
        for x in (0.64, 0.9, 1.3, 1.67, 1.92, 2.3):
            ok, hit, _, _ = body.ray_cast(bl(V(x, 6.5, -0.05)), bl(V(0, -1, 0)), distance=3)
            ok2, hit2, _, _ = body.ray_cast(bl(V(x, 3.5, -0.05)), bl(V(0, 1, 0)), distance=3)
            print(f"[measure] arm x {x}: y {vw(hit2).y if ok2 else 0:.2f}..{vw(hit).y if ok else 0:.2f}")
    if PREVIEW_DIR:
        os.makedirs(PREVIEW_DIR, exist_ok=True)
        render(obs, "pitcher")


main()
