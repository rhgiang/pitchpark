"""Build the viewer's sculpted glove (mitt.glb) in Blender, headless. (The throwing hand, grip.glb,
is make_grip.py's: a posed MakeHuman hand replaced the metaball blob hand that was built here.)

    blender --background --factory-startup --python scripts/blender/make_hands.py -- [preview_dir]

Frame (see web/src/handModels.js): everything here is written in the *viewer's* axes (glTF /
three.js, Y up), in feet, for the LEFT hand: origin at the wrist, +x wrist -> fingers, +y back of
the hand (palm faces -y), +z thumb side. `bl()` converts to Blender's Z-up axes; the glTF exporter
converts back, Blender (x, y, z) -> glTF (x, z, -y).

Parts that must keep their own colour (lacing, webbing, cuff) carry their own material name; the
viewer picks the colour from that name.
"""
import math
import os
import sys

import bmesh
import bpy
from mathutils import Matrix, Vector

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
PREVIEW_DIR = argv[0] if argv else None
OUT_DIR = argv[1] if len(argv) > 1 else os.path.join(ROOT, "web", "public", "models")  # review builds: pass a scratch folder

# ---------- tuning knobs (feet unless noted): feedback rounds should only touch these ----------
MITT_TRI_BUDGET = 8000     # glove: shell + detail (laces, web, back opening); checked at export
# Glove, after the reference pictures (a stylised game glove): a big open hand. The layout is drawn
# flat, in (x = heel -> finger tips, z = pinky side (-) -> thumb side (+)), then laid on a gently
# cupped surface: the further a point is from the middle of the palm, the more it bends towards the
# palm side (-y). That bend is the pocket, so the glove stays round with no hard edges.
CUP = dict(across=0.9, zc=-0.03, tip=0.7, tip_from=0.5, heel=1.2, heel_to=0.14)  # bend, ft / ft^2
AROUND = 12                          # vertices round each tube's cross-section
SQUARE = 2.4                         # cross-section shape: 2 = ellipse, higher = squarer (puffy leather)
# Leather body = palm + stalls + thumb + heel roll, melted into one shell (voxel remesh) then smoothed:
VOXEL = 0.007                        # remesh grid, ft (smaller = sharper grooves, more triangles)
SMOOTH_ITERS = 10                    # smoothing passes after the remesh (more = rounder, softer)
BODY_TRIS, DETAIL_TRIS = 4200, 3800  # triangle split: leather shell / laces, web, back opening
# Palm: x range, z centre and half-width, half-thickness; its heel end is rounded off.
PALM = dict(x0=0.03, x1=0.5, zc=-0.045, hw=0.265, t=0.045, round_to=0.17)
# finger stalls, pinky -> index: path through (x, z) points, half-width, half-thickness. The pinky
# starts low: it is also the rolled pinky-side edge.
STALLS = [
    ([(0.12, -0.27), (0.5, -0.275), (0.8, -0.3)], 0.074, 0.056),
    ([(0.36, -0.115), (0.65, -0.12), (0.9, -0.13)], 0.072, 0.052),
    ([(0.36, 0.035), (0.66, 0.04), (0.95, 0.045)], 0.072, 0.052),
    ([(0.36, 0.175), (0.64, 0.19), (0.91, 0.205)], 0.07, 0.05),
]
THUMB = ([(0.06, 0.2), (0.33, 0.37), (0.6, 0.5)], 0.074, 0.058)
HEEL = ([(0.24, -0.29), (0.07, -0.27), (0.02, -0.13), (0.02, 0.04), (0.06, 0.17), (0.15, 0.23)], 0.045, 0.055)
# Back opening: a rolled rounded-rectangle loop on the back below the stalls, dark lining inside.
BACK_OPEN = dict(x=0.21, z=-0.03, rx=0.075, rz=0.13, r=0.02)
LACE_W, LACE_T = 0.017, 0.006        # laces are flat leather straps: half-width, half-thickness
PINKY_LACES = (0.1, 0.26, 0.42, 0.58)  # bands round the pinky roll, fraction along it
THUMB_LACES = (0.2, 0.4)
HEEL_LACES = 5                       # bands wrapped round the heel roll
WELT_R = 0.008                       # light piping in the grooves between finger stalls
# H-web between index and thumb: rows from these fractions up each tube; two posts across the gap,
# crossbars at the top and middle, X laces tying it to both stalls, the top bar wrapped in lace.
WEB_FROM_INDEX, WEB_FROM_THUMB, WEB_ROWS = 0.35, 0.45, 7
WEB_POSTS = (0.33, 0.67)
WEB_X_ROWS = (1, 3)
WEB_STRAP = (0.04, 0.012)            # post / bar half-width, half-thickness (flat leather strips)
WEB_CUP = 0.03                       # how far the web sags into the pocket at its middle
WEB_TOP_LACES = 6                    # lace wraps along the web's top bar

# Preview-only colours (the viewer's own: web/src/look.js, figure.js); not used by the viewer.
PREVIEW_RGB = {"glove": 0x8A5A2B, "gloveDark": 0x5B3A1C, "gloveLight": 0xB27A43, "skin": 0xC98D66, "ball": 0xF2EFE6}

def bl(p):
    """Viewer (glTF, Y-up) coordinates -> Blender (Z-up)."""
    return Vector((p[0], -p[2], p[1]))


def material(name):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    c = PREVIEW_RGB.get(name, PREVIEW_RGB["glove"])
    lin = lambda v: (v / 255 / 12.92) if v <= 10 else ((v / 255 + 0.055) / 1.055) ** 2.4  # sRGB -> linear
    m.diffuse_color = (lin((c >> 16) & 255), lin((c >> 8) & 255), lin(c & 255), 1)
    return m


def tidy_paint(ob, passes=3):
    """Majority filter: a triangle takes the material most of its edge neighbours have, which
    shaves the spikes off painted region edges left by the big decimated triangles."""
    me = ob.data
    nbrs = {}
    for f in me.polygons:
        for k in f.edge_keys:
            nbrs.setdefault(k, []).append(f.index)
    adj = [[g for k in f.edge_keys for g in nbrs[k] if g != f.index] for f in me.polygons]
    for _ in range(passes):
        mats = [f.material_index for f in me.polygons]
        for f, ns in zip(me.polygons, adj):
            votes = [mats[g] for g in ns]
            best = max(set(votes), key=votes.count) if votes else mats[f.index]
            if votes.count(best) >= 2:
                f.material_index = best


# ---------- glove: tubes swept along paths ----------
# Each part is a tube: a cross-section (rounded square, SQUARE) carried along a path of points,
# turned at every point so its "up" follows a reference direction (usually outward from the glove).
# Ends get a dome cap. Faces are tagged with a material as they are made, so colour edges follow
# the tube's own rows (clean stripes, no painting afterwards).


def spline(ctrl, n):
    """n points on a Catmull-Rom curve through `ctrl` (a smooth path from a few control points)."""
    c = [ctrl[0]] + list(ctrl) + [ctrl[-1]]
    out = []
    for k in range(n):
        u = k / (n - 1) * (len(ctrl) - 1)
        i = min(int(u), len(ctrl) - 2)
        t = u - i
        p0, p1, p2, p3 = c[i], c[i + 1], c[i + 2], c[i + 3]
        out.append(0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                          + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3))
    return out


class Builder:
    """Collects tube geometry (viewer axes) with a material per face."""
    MATS = ["glove", "gloveLight", "gloveDark"]

    def __init__(self):
        self.verts, self.faces, self.mats = [], [], []

    def tube(self, pts, w, h, mat="glove", inner=None, ups=None, around=AROUND, square=SQUARE,
             closed=False, caps=True):
        """Sweep a w x h (half sizes, ft; numbers or per-point lists) rounded-square section along
        `pts`. `ups[i]` is the section's 'up' (h) direction; default outward from the glove. Faces on
        the -up side (the pocket side) get material `inner` if given. Returns the frames
        [(point, tangent, lateral, up, w, h)] so other parts (laces) can wrap round it."""
        n = len(pts)
        w = w if isinstance(w, list) else [w] * n
        h = h if isinstance(h, list) else [h] * n
        ups = ups or [radial(p) for p in pts]
        frames = []
        for i, p in enumerate(pts):
            if closed:
                t = pts[(i + 1) % n] - pts[i - 1]
            else:
                t = pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]
            t.normalize()
            u = (ups[i] - t * ups[i].dot(t)).normalized()
            frames.append((p, t, u.cross(t), u, w[i], h[i]))
        e = 2 / square
        sp = lambda v: math.copysign(abs(v) ** e, v)
        ang = [2 * math.pi * (a + 0.5) / around for a in range(around)]

        def ring(p, l, u, ww, hh):
            base = len(self.verts)
            self.verts += [p + l * (ww * sp(math.cos(t))) + u * (hh * sp(math.sin(t))) for t in ang]
            return base

        def mat_of(a):  # the face between section vertices a and a+1
            mid = 2 * math.pi * (a + 1) / around
            return inner if inner and math.sin(mid) < -0.3 else mat

        rings = [ring(p, l, u, ww, hh) for p, t, l, u, ww, hh in frames]
        if caps and not closed:  # dome: shrinking rings past each end, then a pole
            for end, sign in ((0, -1), (n - 1, 1)):
                p, t, l, u, ww, hh = frames[end]
                r = min(ww, hh) * 0.95
                prev = rings[end]
                for j in (1, 2, 3):
                    phi = math.pi / 2 * j / 4
                    cur = ring(p + t * (sign * r * math.sin(phi)), l, u, ww * math.cos(phi), hh * math.cos(phi))
                    self._bridge(prev, cur, around, mat_of)
                    prev = cur
                pole = len(self.verts)
                self.verts.append(p + t * (sign * r))
                for a in range(around):
                    self.faces.append((prev + a, prev + (a + 1) % around, pole))
                    self.mats.append(mat_of(a))
        for i in range(n - 1 + (1 if closed else 0)):
            self._bridge(rings[i], rings[(i + 1) % n], around, mat_of)
        return frames

    def _bridge(self, r0, r1, around, mat_of):
        for a in range(around):
            b = (a + 1) % around
            self.faces.append((r0 + a, r0 + b, r1 + b, r1 + a))
            self.mats.append(mat_of(a))

    def slab(self, at, top, bottom, mat="glove", under=None, nx=12, nz=10):
        """A thin closed slab: grid points at(u, v, lift) for u, v in [0, 1], one layer at lift
        `top` and one at `bottom`, joined round the rim. Its bottom layer gets material `under`."""
        sides = []
        for lift in (top, bottom):
            base = len(self.verts)
            g = [[base + i * nz + j for j in range(nz)] for i in range(nx)]
            self.verts += [at(i / (nx - 1), j / (nz - 1), lift) for i in range(nx) for j in range(nz)]
            for i in range(nx - 1):
                for j in range(nz - 1):
                    self.faces.append((g[i][j], g[i + 1][j], g[i + 1][j + 1], g[i][j + 1]))
                    self.mats.append(under if lift == bottom and under else mat)
            sides.append([g[i][0] for i in range(nx)] + [g[nx - 1][j] for j in range(1, nz)]
                         + [g[i][nz - 1] for i in range(nx - 2, -1, -1)] + [g[0][j] for j in range(nz - 2, 0, -1)])
        a, b = sides
        for k in range(len(a)):  # the rim joining top and bottom
            self.faces.append((a[k], a[(k + 1) % len(a)], b[(k + 1) % len(b)], b[k]))
            self.mats.append(mat)

    def to_object(self, name):
        me = bpy.data.meshes.new(name)
        me.from_pydata([bl(v) for v in self.verts], [], self.faces)
        for m in self.MATS:
            me.materials.append(material(m))
        for poly, m in zip(me.polygons, self.mats):
            poly.material_index = self.MATS.index(m)
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)  # every tube is closed: point normals out
        bm.to_mesh(me)
        bm.free()
        me.shade_smooth()
        ob = bpy.data.objects.new(name, me)
        bpy.context.scene.collection.objects.link(ob)
        return ob


def vw(p):
    """Blender (Z-up) coordinates -> viewer (glTF, Y-up); inverse of bl()."""
    return Vector((p[0], p[2], -p[1]))


def apply_modifiers(ob):
    """Bake the object's modifier stack into its mesh."""
    bpy.context.view_layer.update()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(bpy.context.evaluated_depsgraph_get()))
    ob.modifiers.clear()
    ob.data = me


def cup_y(x, z):
    """Height (viewer y) of the cupped glove surface at layout point (x, z): highest in the middle of
    the palm, bending towards the palm side across the width, at the finger tips and at the heel."""
    c = CUP
    return -(c["across"] * (z - c["zc"]) ** 2 + c["tip"] * max(0.0, x - c["tip_from"]) ** 2
             + c["heel"] * max(0.0, c["heel_to"] - x) ** 2)


def nrm(x, z, e=1e-3):
    """Outward (back-of-hand side) normal of the cupped surface at (x, z): for a height field
    y = f(x, z) it is (-df/dx, 1, -df/dz), normalised."""
    fx = (cup_y(x + e, z) - cup_y(x - e, z)) / (2 * e)
    fz = (cup_y(x, z + e) - cup_y(x, z - e)) / (2 * e)
    return Vector((-fx, 1.0, -fz)).normalized()


def sp(x, z, lift=0.0):
    """Point on the cupped surface at layout (x, z), `lift` ft out along its normal."""
    return Vector((x, cup_y(x, z), z)) + nrm(x, z) * lift


def radial(p):
    """Outward direction at p (the 'up' for parts lying on the glove)."""
    return nrm(p.x, p.z)


def stall(B, path, w, h, n=16, widen=0.1):
    """A finger stall (or the thumb, or the heel roll): a puffy tube along `path` ((x, z) layout
    points) on the cupped surface, a little wider and flatter towards the tip like a real glove's."""
    pts = [sp(q.x, q.z) for q in spline([Vector((x, 0.0, z)) for x, z in path], n)]
    return B.tube(pts, [w * (0.95 + widen * k / (n - 1)) for k in range(n)],
                  [h * (1.0 - 0.1 * k / (n - 1)) for k in range(n)], "glove", ups=[radial(p) for p in pts])


def leather_shell(B):
    """Melt the overlapping leather pieces into one closed surface (voxel remesh: rebuild the
    union on a grid), round every join (smoothing), thin to BODY_TRIS, then colour it: faces
    looking into the pocket (against the surface normal) get the lighter palm leather."""
    ob = B.to_object("mitt_body")
    m = ob.modifiers.new("remesh", "REMESH")
    m.mode, m.voxel_size = "VOXEL", VOXEL
    m = ob.modifiers.new("smooth", "SMOOTH")
    m.factor, m.iterations = 0.6, SMOOTH_ITERS
    apply_modifiers(ob)
    tris = sum(len(p.vertices) - 2 for p in ob.data.polygons)
    if tris > BODY_TRIS:
        m = ob.modifiers.new("decimate", "DECIMATE")
        m.ratio = BODY_TRIS / tris
        apply_modifiers(ob)
    ob.data.materials.clear()
    for name in Builder.MATS:
        ob.data.materials.append(material(name))
    for f in ob.data.polygons:
        c, n = vw(f.center), vw(f.normal)
        f.material_index = 1 if n.dot(-radial(c)) > 0.35 else 0
    tidy_paint(ob)
    ob.data.shade_smooth()
    return ob


def build_mitt():
    """Pitcher's glove after the stylised reference: a big rounded, cupped hand. Palm, four puffy
    finger stalls (the pinky one doubling as the laced pinky-side roll), a long thumb and a rolled
    heel melted into one shell; welting between the stalls, X laces across the tip grooves, stitch
    dashes in the pocket, lace bands round the pinky, thumb and heel, an H-web laced between index
    and thumb, and a rolled back opening with dark lining. ~0.95 ft heel to tips."""
    B = Builder()
    P = PALM

    def palm_at(u, v, lift):
        x = P["x0"] + (P["x1"] - P["x0"]) * u
        k = max(0.0, (P["round_to"] - x) / (P["round_to"] - P["x0"]))  # 0 above the heel, 1 at its bottom
        hw = P["hw"] * math.sqrt(max(1 - k * k, 0.12))
        return sp(x, P["zc"] + hw * (2 * v - 1), lift)

    B.slab(palm_at, P["t"], -P["t"], nx=14, nz=12)
    fr = [stall(B, *s) for s in STALLS]
    th = stall(B, *THUMB, widen=0.04)
    heel = stall(B, *HEEL, n=20, widen=0.0)
    body = leather_shell(B)
    bpy.context.view_layer.update()

    def snap(p, d, off, reach=0.3):
        """Press point p onto the shell: cast a ray from outside along -d, land `off` above the hit."""
        d = d.normalized()
        ok, hit, _, _ = body.ray_cast(bl(p + d * reach), bl(-d), distance=2 * reach)
        return vw(hit) + d * off if ok else p + d * off

    D = Builder()

    def strap(pts, w=LACE_W, th_=LACE_T, mat="gloveDark", ups=None, closed=False, caps=False):
        D.tube(pts, w, th_, mat, ups=ups, around=4, square=3, closed=closed, caps=caps)

    def wrap(frame, n=10, pad=0.0, slant=0.012, snap_it=True):
        """A lace band round a tube at one of its frames, pressed onto the shell."""
        p, tt, l, u, ww, hh = frame
        ring = []
        for a in (2 * math.pi * k / n for k in range(n)):
            d = l * ((ww + pad) * math.cos(a)) + u * ((hh + pad) * math.sin(a)) + tt * (slant * math.cos(a))
            ring.append(snap(p, d, LACE_T * 1.3, reach=0.25) if snap_it else p + d)
        strap(ring, ups=[(q - p).normalized() for q in ring], closed=True)

    near = lambda frames, x: min(frames, key=lambda f: abs(f[0].x - x))
    # Welting: light piping pressed into the groove between neighbouring stalls, on the back.
    for a, b in zip(fr, fr[1:]):
        x0, x1 = BACK_OPEN["x"] + BACK_OPEN["rx"] + 0.06, min(a[-1][0].x, b[-1][0].x) - 0.1
        pts = []
        for k in range(8):
            fa, fb = near(a, x0 + (x1 - x0) * k / 7), near(b, x0 + (x1 - x0) * k / 7)
            mid = (fa[0] + fb[0]) / 2
            pts.append(snap(mid, radial(mid), WELT_R * 0.5))
        D.tube(pts, WELT_R, WELT_R, "gloveLight", around=6, square=2)
    # Lace bands: round the pinky roll, the thumb and the heel roll.
    for frames, fracs in ((fr[0], PINKY_LACES), (th, THUMB_LACES), (heel, [(k + 0.5) / HEEL_LACES for k in range(HEEL_LACES)])):
        for f in fracs:
            wrap(frames[round(f * (len(frames) - 1))])
    # X laces across the top of each groove, on the palm side (as on the reference's finger tips).
    for a, b in zip(fr, fr[1:]):
        x = min(a[-1][0].x, b[-1][0].x) - 0.1
        fa, fb = near(a, x), near(b, x)
        n_in = -radial((fa[0] + fb[0]) / 2)
        for da, db in ((-0.03, 0.03), (0.03, -0.03)):
            p0 = snap(fa[0] + fa[1] * da, n_in, LACE_T * 1.3)
            p1 = snap(fb[0] + fb[1] * db, n_in, LACE_T * 1.3)
            strap([p0, (p0 + p1) / 2 + n_in * 0.01, p1], ups=[n_in] * 3)
    # Back-of-tip laces: a short strap across the back of each stall near its tip.
    for f in fr:
        p, tt, l, u, ww, hh = near(f, f[-1][0].x - 0.08)
        strap([snap(p + l * (ww * s), u, LACE_T * 1.3) for s in (-0.75, -0.25, 0.25, 0.75)], ups=[u] * 4)
    # Stitch dashes in the pocket: two up each finger, and a row across the palm.
    dash = lambda x0, z0, x1, z1: strap([snap(sp(x0, z0), -nrm(x0, z0), LACE_T), snap(sp(x1, z1), -nrm(x1, z1), LACE_T)],
                                        ups=[-nrm(x0, z0)] * 2)
    for path, w, h in STALLS[1:]:
        (xa, za), (xb, zb), (xc, zc) = path
        for s0 in (0.25, 0.55):
            x0, z0 = xb + (xc - xb) * s0, zb + (zc - zb) * s0
            dash(x0, z0, x0 + 0.06, z0)
    for z in (-0.2, -0.09, 0.02, 0.13):
        dash(0.2, z - 0.035, 0.2, z + 0.035)
    # H-web: a lattice spanning the gap between the index stall's thumb-side edge and the thumb's
    # index-side edge, over the top part of both, cupped into the pocket.
    ix = fr[-1]

    def edge(frames, towards, u):
        p, tt, l, uu, ww, hh = frames[round(u * (len(frames) - 1))]
        side = l if l.dot(towards) > 0 else -l
        return p + side * (ww * 0.8)

    rows = []
    for r in range(WEB_ROWS):
        ui = WEB_FROM_INDEX + (1 - WEB_FROM_INDEX) * r / (WEB_ROWS - 1)
        ut = WEB_FROM_THUMB + (1 - WEB_FROM_THUMB) * r / (WEB_ROWS - 1)
        pi_, pt_ = ix[round(ui * (len(ix) - 1))][0], th[round(ut * (len(th) - 1))][0]
        e0, e1 = edge(ix, pt_ - pi_, ui), edge(th, pi_ - pt_, ut)
        cup = -(radial(e0) + radial(e1)).normalized() * WEB_CUP
        rows.append(lambda s, e0=e0, e1=e1, cup=cup: e0 + (e1 - e0) * s + cup * math.sin(math.pi * s))
    up_at = lambda s, r: (rows[min(r + 1, WEB_ROWS - 1)](s) - rows[max(r - 1, 0)](s)).cross(rows[r](1) - rows[r](0)).normalized()
    for s in WEB_POSTS:  # posts along the web
        D.tube([rows[r](s) for r in range(WEB_ROWS)], *WEB_STRAP, "gloveDark", ups=[up_at(s, r) for r in range(WEB_ROWS)],
               around=6, square=3)
    top = None
    for r in (WEB_ROWS - 1, WEB_ROWS // 2):  # crossbars: the top one spans the whole gap
        s0, s1 = (-0.05, 1.05) if r == WEB_ROWS - 1 else WEB_POSTS
        pts = [rows[r](s0 + (s1 - s0) * j / 9) for j in range(10)]
        frames = D.tube(pts, *WEB_STRAP, "gloveDark", ups=[up_at(0.5, r)] * 10, around=6, square=3)
        top = top or frames
    for k in range(WEB_TOP_LACES):  # the top bar wrapped in lace
        wrap(top[1 + round(k * (len(top) - 3) / (WEB_TOP_LACES - 1))], n=8, pad=0.004, slant=0.01, snap_it=False)
    for r in WEB_X_ROWS:  # X-crossed laces tying the posts to the index and the thumb
        for s0, s1 in ((-0.05, WEB_POSTS[0]), (WEB_POSTS[1], 1.05)):
            for a, b in ((r, r + 1), (r + 1, r)):
                strap([rows[a](s0), rows[b](s1)], ups=[up_at(0.5, r)] * 2)
    # Back opening: a rolled rounded-rectangle loop pressed onto the back, dark lining inside.
    o = BACK_OPEN
    sq = lambda v: math.copysign(abs(v) ** 0.5, v)  # superellipse, exponent 4: a rounded rectangle
    back_at = lambda x, z, off: snap(sp(x, z), nrm(x, z), off, reach=0.4)
    loop = [back_at(o["x"] + o["rx"] * sq(math.cos(q)), o["z"] + o["rz"] * sq(math.sin(q)), o["r"] * 0.5)
            for q in (2 * math.pi * k / 24 for k in range(24))]
    D.tube(loop, o["r"], o["r"], "glove", ups=[radial(p) for p in loop], around=7, square=2, closed=True)
    D.slab(lambda u, v, lift: back_at(o["x"] + o["rx"] * 0.85 * (2 * u - 1), o["z"] + o["rz"] * 0.85 * (2 * v - 1), lift),
           0.006, -0.012, "gloveDark", nx=8, nz=10)
    det = D.to_object("mitt_details")
    tris = sum(len(p.vertices) - 2 for p in det.data.polygons)
    assert tris <= DETAIL_TRIS, f"glove details: {tris} triangles > {DETAIL_TRIS}"
    return [body, det]


# ---------- glove restyled from a reference model ----------
# If ref/glove/ holds a .glb (an AI-generated glove exported from Tripo; not committed, the folder
# is gitignored), the glove is built from it instead of from the code above: its leather parts are
# melted into one smooth shell (voxel remesh + smoothing: this also irons out the lumpy generated
# surface and turns the small laces into soft bumps, the cartoon look), thinned to the budget,
# scaled, turned into the viewer's frame and painted in Pitch Park's flat colours.
REF_DIR = os.path.join(ROOT, "ref", "glove")
REF_DARK_PARTS = (1, 2, 4)        # tripo_part_N painted gloveDark: web, heel binding, wrist lining
REF_LEN = 0.95                    # heel -> finger tips, ft
REF_HEEL_X = -0.03                # where the heel sits along the hand (viewer x), ft
REF_OPENING = (-0.03, 0.0)        # (viewer y, z) of the middle of the hand opening at the heel: the
REF_OPENING_PART = 2              # forearm goes in here (the tripo part ringing it: the heel binding)
REF_VOXEL = 0.007                 # leather remesh grid, ft (bigger = softer and more cartoon)
REF_SHELL = 0.022                 # the reference is paper-thin open surfaces: thickened to this first
                                  # (solidify) so the remesh sees closed volumes; keep > 2 voxels
REF_SMOOTH = 6                    # smoothing passes after the remesh (fewer = more leather texture)
REF_CUT = dict(z=-0.15, y=-0.23)  # cut away everything past both (viewer axes): a loose lace tail
REF_BODY_TRIS, REF_DARK_TRIS = 4550, 1250   # shell / web, binding, lining; laces take the rest
REF_WEB_PART = 1                  # which tripo part is the web (its top edge gets lace loops)
REF_DARK_SMOOTH = 2               # light smoothing on the dark parts: the web keeps its creases
# Laces added on top: flat straps (half-width w, half-thickness t), loops sitting `inset` in from
# the rim, `ring` = their reach outwards and towards palm/back before hugging the surface; (lo, hi, count) along the pinky edge (x) and the heel (z); loops on the web's top; bars
# tying neighbouring fingers across the gaps between them (at x = gaps_x, z = gaps_z), +-tip_span.
REF_LACE = dict(w=0.016, t=0.006, inset=0.02, ring=(0.04, 0.07), pinky=(0.12, 0.75, 9), heel=(-0.24, 0.2, 8), web=9,
                gaps_x=0.8, gaps_z=(-0.17, -0.035, 0.07), tip_span=0.045,
                net_rows=(0.3, 0.5, 0.7), net_posts=(0.35, 0.65))  # web net: fractions up / across it
# The reference's axes: fingers +z, palm (pocket) +y, thumb and web +x. This turns them into the
# build axes (bl of the viewer frame: fingers +x, back of hand +z, thumb -y): (x, y, z) -> (z, -x, -y).
REF_TURN = Matrix(((0, 0, 1), (-1, 0, 0), (0, -1, 0))).to_4x4()


def ref_glove_path():
    """The reference glove .glb in ref/glove/, or None (then the code-built glove is used)."""
    if not os.path.isdir(REF_DIR):
        return None
    found = sorted(f for f in os.listdir(REF_DIR) if f.lower().endswith(".glb"))
    return os.path.join(REF_DIR, found[0]) if found else None


def join(obs, name):
    """Merge meshes into one object."""
    bpy.ops.object.select_all(action="DESELECT")
    for o in obs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = obs[0]
    bpy.ops.object.join()
    ob = bpy.context.view_layer.objects.active
    ob.name = ob.data.name = name
    return ob


def thin(ob, tris):
    """Decimate to at most `tris` triangles (decimating quads lands a little high: repeat)."""
    for _ in range(4):
        have = sum(len(p.vertices) - 2 for p in ob.data.polygons)
        if have <= tris:
            return
        m = ob.modifiers.new("decimate", "DECIMATE")
        m.ratio = tris / have * 0.98
        apply_modifiers(ob)


def paint(ob, pick):
    """Give the object the glove materials; pick(centre, normal) (viewer axes) names each face's."""
    ob.data.materials.clear()
    for name in Builder.MATS:
        ob.data.materials.append(material(name))
    for f in ob.data.polygons:
        f.material_index = Builder.MATS.index(pick(vw(f.center), vw(f.normal)))
    ob.data.shade_smooth()


def build_mitt_from_ref(path):
    """Glove from the reference model: [leather shell, dark parts (web, binding, lining)]."""
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    new = [o for o in bpy.data.objects if o not in before]
    bpy.context.view_layer.update()
    parts = []
    for o in new:
        if o.type == "MESH":
            o.data.transform(REF_TURN @ o.matrix_world)  # bake its placement, then turn it
            o.parent = None
            o.matrix_world = Matrix.Identity(4)
            parts.append(o)
        else:
            bpy.data.objects.remove(o)
    # scale and place: heel -> tips along +x from REF_HEEL_X; across, see the opening below
    pts = [v.co for o in parts for v in o.data.vertices]
    lo = [min(p[i] for p in pts) for i in range(3)]
    hi = [max(p[i] for p in pts) for i in range(3)]
    s = REF_LEN / (hi[0] - lo[0])
    move = Vector((REF_HEEL_X - lo[0] * s, 0.0, 0.0))
    for o in parts:
        o.data.transform(Matrix.Translation(move) @ Matrix.Scale(s, 4))
    # centre the hand opening on the wrist, so the forearm runs into the glove through it
    op = [vw(v.co) for o in parts if part_no(o) == REF_OPENING_PART for v in o.data.vertices]
    mid = lambda a: (min(getattr(p, a) for p in op) + max(getattr(p, a) for p in op)) / 2
    shift = bl((0.0, REF_OPENING[0] - mid("y"), REF_OPENING[1] - mid("z")))
    for o in parts:
        o.data.transform(Matrix.Translation(shift))
        bm = bmesh.new()  # cut the lace tail hanging off the pinky side, out in front of the pocket
        bm.from_mesh(o.data)
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if -v.co.y < REF_CUT["z"] and v.co.z < REF_CUT["y"]], context="VERTS")
        bm.to_mesh(o.data)
        bm.free()
    dark = [o for o in parts if int(o.name.split("_")[-1].split(".")[0]) in REF_DARK_PARTS]
    body = join([o for o in parts if o not in dark], "mitt_body")
    m = body.modifiers.new("thick", "SOLIDIFY")
    m.thickness, m.offset = REF_SHELL, 0.0
    m = body.modifiers.new("remesh", "REMESH")
    m.mode, m.voxel_size = "VOXEL", REF_VOXEL
    m = body.modifiers.new("smooth", "SMOOTH")
    m.factor, m.iterations = 0.6, REF_SMOOTH
    apply_modifiers(body)
    thin(body, REF_BODY_TRIS)
    paint(body, lambda c, n: "gloveLight" if n.y < -0.35 else "glove")  # facing the palm side: pocket
    tidy_paint(body, passes=6)
    web_pts = [vw(v.co) for o in dark if part_no(o) == REF_WEB_PART for v in o.data.vertices]
    for o in dark:  # where each dark part sits (viewer axes): the lining marks the hand's opening
        q = [vw(v.co) for v in o.data.vertices]
        print("[make_hands] part", part_no(o), "x %.3f..%.3f y %.3f..%.3f z %.3f..%.3f" % tuple(
            f(getattr(v, a) for v in q) for a in "xyz" for f in (min, max)))
    det = join(dark, "mitt_details")  # many small open pieces: remesh them closed so they can be thinned
    m = det.modifiers.new("thick", "SOLIDIFY")
    m.thickness, m.offset = REF_SHELL * 0.75, 0.0
    m = det.modifiers.new("remesh", "REMESH")
    m.mode, m.voxel_size = "VOXEL", REF_VOXEL * 0.75
    m = det.modifiers.new("smooth", "SMOOTH")
    m.factor, m.iterations = 0.5, REF_DARK_SMOOTH
    apply_modifiers(det)
    thin(det, REF_DARK_TRIS)
    paint(det, lambda c, n: "gloveDark")
    laces = ref_laces(body, det, web_pts)
    print("[make_hands] ref glove:", *(f"{o.name} {sum(len(p.vertices) - 2 for p in o.data.polygons)} tris" for o in (body, det, laces)))
    return [body, det, laces]


def part_no(ob):
    """N of an imported 'tripo_part_N' object."""
    return int(ob.name.split("_")[-1].split(".")[0])


def rim(pts, along, lo, hi, n, side, axis):
    """n points on the glove's outline, as seen from the palm: for n slices of `along` (0 = x,
    2 = z) between lo and hi, the vertex furthest along `axis` in direction `side` (+1 / -1)."""
    out = []
    w = (hi - lo) / n
    for k in range(n):
        c = lo + (k + 0.5) * w
        sl = [p for p in pts if abs(p[along] - c) < w * 0.6]
        if sl:
            out.append(max(sl, key=lambda p: side * p[axis]))
    return out


def ref_laces(body, det, web_pts):
    """Laces on the restyled glove, pressed onto its surface: loops wrapping the pinky edge, the
    heel rim and the web's top edge, and short bars tying neighbouring finger tips together."""
    bpy.context.view_layer.update()
    D = Builder()
    pts = [vw(v.co) for v in body.data.vertices]

    def snap(target, c, d, off, reach=0.3):
        """Cast a ray at point c from outside along -d onto `target`; land `off` above the hit."""
        d = d.normalized()
        ok, hit, _, _ = target.ray_cast(bl(c + d * reach), bl(-d), distance=2 * reach)
        return vw(hit) + d * off if ok else None

    def hug(target, p, off):
        """Move p to the nearest point of `target`'s surface, `off` out along its normal."""
        ok, loc, nor, _ = target.closest_point_on_mesh(bl(p))
        return vw(loc) + vw(nor) * off

    def loops(target, edge, mat, n=8):
        """A lace loop round the glove's rim at each edge point: a ring in the plane of the rim's
        outward direction and the palm/back axis (y), each point moved onto the nearest surface."""
        L = REF_LACE
        for i, p in enumerate(edge):
            a, b = edge[max(i - 1, 0)], edge[min(i + 1, len(edge) - 1)]
            t = (b - a).normalized()
            o = t.cross(Vector((0, 1, 0))).normalized()  # outward in the palm-view plane
            near = pts_near(p)
            if near and o.dot(p - sum(near, Vector()) / len(near)) < 0:
                o = -o
            c = p - o * L["inset"]
            ring = [hug(target, c + o * (L["ring"][0] * math.cos(q)) + Vector((0, L["ring"][1] * math.sin(q), 0)), L["t"] * 1.3)
                    for q in (2 * math.pi * k / n for k in range(n))]
            D.tube(ring, L["w"], L["t"], mat, ups=[(r - c).normalized() for r in ring], around=4, square=3, closed=True)

    def pts_near(p, r=0.12):
        return [q for q in pts[::7] if (q - p).length < r]

    L = REF_LACE
    loops(body, rim(pts, 0, *L["pinky"], -1, 2), "gloveDark")       # pinky edge: furthest -z
    loops(body, rim(pts, 2, *L["heel"], -1, 0), "gloveDark")        # heel rim: lowest x
    if web_pts:
        z0, z1 = min(p.z for p in web_pts), max(p.z for p in web_pts)
        loops(det, rim(web_pts, 2, z0 + 0.02, z1 - 0.02, L["web"], 1, 0), "glove")  # web top: highest x
        # the net: lace strips woven across both faces of the web, rows across and two posts up it
        x0, x1 = min(p.x for p in web_pts), max(p.x for p in web_pts)
        ym = sum(p.y for p in web_pts) / len(web_pts)
        for side in (1, -1):
            for f in L["net_rows"]:
                x = x0 + (x1 - x0) * f
                sl = [p for p in web_pts if abs(p.x - x) < 0.015]
                if len(sl) < 4:
                    continue
                za, zb = min(p.z for p in sl) + 0.025, max(p.z for p in sl) - 0.025
                row = [hug(det, Vector((x, ym + side * 0.1, za + (zb - za) * k / 5)), L["t"] * 1.3) for k in range(6)]
                D.tube(row, L["w"], L["t"], "glove", ups=[Vector((0, side, 0))] * 6, around=4, square=3, caps=False)
            for f in L["net_posts"]:
                z = z0 + (z1 - z0) * f
                sl = [p for p in web_pts if abs(p.z - z) < 0.015]
                if len(sl) < 4:
                    continue
                xa, xb = min(p.x for p in sl) + 0.03, max(p.x for p in sl) - 0.03
                post = [hug(det, Vector((xa + (xb - xa) * k / 5, ym + side * 0.1, z)), L["t"] * 1.3) for k in range(6)]
                D.tube(post, L["w"], L["t"], "glove", ups=[Vector((0, side, 0))] * 6, around=4, square=3, caps=False)
    # finger tips: a short lace bar across each gap between neighbouring fingers, on the back and
    # on the palm, pressed on from straight above / below
    for z in L["gaps_z"]:
        for side in (1, -1):
            d = Vector((0, side, 0))
            bar = [snap(body, Vector((L["gaps_x"], 0.0, z + dz)), d, L["t"] * 1.3, reach=1.0)
                   for dz in (-L["tip_span"], -L["tip_span"] / 3, L["tip_span"] / 3, L["tip_span"])]
            if all(bar):
                D.tube(bar, L["w"], L["t"], "gloveDark", ups=[d] * 4, around=4, square=3, caps=False)
    return D.to_object("mitt_laces")


def export(obs, name, budget):
    bpy.ops.object.select_all(action="DESELECT")
    for o in obs:
        o.select_set(True)
    path = os.path.join(OUT_DIR, name + ".glb")
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", use_selection=True, export_yup=True,
                              export_apply=True, export_texcoords=False, export_materials="EXPORT")
    tris = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in obs)
    lo = [min((o.matrix_world @ v.co)[i] for o in obs for v in o.data.vertices) for i in range(3)]
    hi = [max((o.matrix_world @ v.co)[i] for o in obs for v in o.data.vertices) for i in range(3)]
    # report in viewer axes: x, y (= Blender z), z (= -Blender y)
    print(f"[make_hands] {name}.glb: {tris} tris, viewer x {lo[0]:.3f}..{hi[0]:.3f} "
          f"y {lo[2]:.3f}..{hi[2]:.3f} z {-hi[1]:.3f}..{-lo[1]:.3f} ft")
    assert tris <= budget, f"{name}: {tris} triangles > budget {budget}"


def render(obs, name, extra=()):
    """Workbench (flat studio-lit) previews from a few angles, for review only."""
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.light, sc.display.shading.color_type = "STUDIO", "MATERIAL"
    sc.display.shading.show_cavity = True
    sc.view_settings.view_transform = "Standard"  # true colours (the default AgX dulls them)
    sc.display.shading.show_object_outline = True
    sc.render.resolution_x, sc.render.resolution_y = 640, 480
    sc.render.film_transparent = True  # composited onto white for the sheets
    bpy.context.view_layer.update()  # fresh bounding boxes for the camera target
    for o in sc.objects:
        o.hide_render = o not in obs and o not in extra
    target = sum((o.matrix_world @ Vector(o.bound_box[0]) + o.matrix_world @ Vector(o.bound_box[6]) for o in obs),
                 Vector()) / (2 * len(obs))
    cam_data = bpy.data.cameras.get("cam") or bpy.data.cameras.new("cam")
    cam_data.lens = 70
    cam = bpy.data.objects.get("cam") or bpy.data.objects.new("cam", cam_data)
    if cam.name not in sc.collection.objects:
        sc.collection.objects.link(cam)
    sc.camera = cam
    dist = 3.8 if name == "mitt" else 1.5
    # camera directions (viewer axes, from the model to the camera); fingers point up in the image
    views = {"back": (0.0, 1, -0.08), "palm": (0.0, -1, 0.15), "side": (-0.75, -0.25, -0.6), "three_q": (0.25, -0.75, 0.6)}
    up = bl((1, 0, 0))
    for view, d in views.items():
        d = bl(d).normalized()
        cam.location = target + d * dist
        xc = up.cross(d).normalized()  # camera right; camera looks along -d with +x of the model up
        cam.rotation_euler = Matrix((xc, d.cross(xc), d)).transposed().to_euler()
        sc.render.filepath = os.path.join(PREVIEW_DIR, f"{name}_{view}.png")
        bpy.ops.render.render(write_still=True)


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    ref = ref_glove_path()
    mitt = build_mitt_from_ref(ref) if ref else build_mitt()
    export(mitt, "mitt", MITT_TRI_BUDGET)
    if PREVIEW_DIR:
        os.makedirs(PREVIEW_DIR, exist_ok=True)
        render(mitt, "mitt")


main()
