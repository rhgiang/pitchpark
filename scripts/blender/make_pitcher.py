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
TRIM_X = (1.21, 1.29)  # sleeve trim |x|; skin (forearm) beyond
NECK_Y = 5.02        # skin above (front of the collar; the back sits a little higher)
HAIR_Y = 5.76        # hairline height at the sides (just behind the face)
NAPE = (-0.3, 5.45)  # hairline at the back: (z relative to the head's centre, y); it slopes between
FACE_Z = 0.12        # face: in front of this (relative to the head's centre) stays skin
HEAD_Y = 5.35       # all skin above this (ears), outside the hair
NECK_HALF = 0.3      # |x| of the neck opening (the shoulders either side stay jersey)
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
        return "navy"
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


def rig_and_bind(body, cap):
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
    cut(body, True)
    close_wrists(body)
    cap = build_cap(body)
    rig = rig_and_bind(body, cap)
    print(f"[make_pitcher] body {tris(body)}, cap {sum(tris(o) for o in cap)} triangles")
    if PREVIEW_DIR:
        os.makedirs(PREVIEW_DIR, exist_ok=True)
        render("pitcher")
    export(body, cap, rig)


main()
