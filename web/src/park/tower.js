import * as THREE from "three";
import { PALETTE, makeToonMaterial, lampMaterial, makeGlowSprite, noOutline, timeOfDay } from "../look.js";
import { makeBall, spinBall, setSeams, seamStyle, BALL_RADIUS_FT } from "../ball.js";
import { box, place } from "./build.js";
import { wordmarkTexture } from "./brand.js";
import { fenceDistFt, BATTERS_EYE_GAP_FT } from "./layout.js";
import { TOWER, EYE_PANEL, rpmLabel } from "./signature.js";

/*
 * The Spin Tower: a giant baseball on a pylon behind centre field, between twin light towers. It is the
 * viewer's own ball (ball.js makeBall, scaled up), turned by the same spinBall() and setSeams() from the
 * same spin axis, spin rate and slowed-down clock as the ball in flight, so it spins on THIS pitch's axis
 * (the blue rod through it is the axis, as on the real ball). The pitch's spin rate shows on a readout
 * panel on the pylon and, because the umpire and batter see only the sky above the batter's eye wall
 * (3 degrees of it), again on the wall's face. Scene coordinates (x, y up, z = -Statcast y).
 */

const ink = 0x22303a;

/** Paint the readout: a dark LED board, "SPIN" small, the rate big, "RPM" small. */
function drawPanel(g, w, h, text) {
  g.fillStyle = "#10161d"; g.fillRect(0, 0, w, h);
  g.strokeStyle = "#f2c14e"; g.lineWidth = h * 0.04;
  g.strokeRect(h * 0.05, h * 0.05, w - h * 0.1, h - h * 0.1);
  g.textBaseline = "middle";
  // Big, bold labels: the whole board is only ~130 px wide in the umpire view, so thin small text would mush away.
  g.fillStyle = "#f4ead2"; g.font = `bold ${h * 0.3}px sans-serif`; g.textAlign = "left";
  g.fillText("SPIN", w * 0.055, h * 0.32);
  g.fillText("RATE", w * 0.055, h * 0.68);
  g.fillStyle = "#ffcf5a"; g.font = `bold ${h * 0.6}px monospace`; g.textAlign = "center";
  g.fillText(text, w * 0.52, h * 0.52);
  g.fillStyle = "#f4ead2"; g.font = `bold ${h * 0.3}px sans-serif`; g.textAlign = "right";
  g.fillText("RPM", w * 0.965, h * 0.5);
}

/** A readout plane (or, with arcRadius, a strip of a cylinder about the plate, facing in) showing `text`; update(text) repaints only when the text changed. */
function makePanel(w, h, arcRadius = 0) {
  const c = document.createElement("canvas");
  c.width = 512; c.height = Math.round((512 * h) / w);
  const g = c.getContext("2d");
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  let shown = null;
  const update = (text) => {
    if (text === shown) return;
    shown = text;
    drawPanel(g, c.width, c.height, text);
    tex.needsUpdate = true;
  };
  update("---");
  let geo = new THREE.PlaneGeometry(w, h);
  if (arcRadius) { // the eye wall is an arc about home plate: bend the panel with it, or its ends sink into the wall
    // The wall's radius from home shrinks with bearing (it follows the fence), so lay the strip on that curve, not on a circle.
    geo = new THREE.PlaneGeometry(w, h, 40, 1);
    const p = geo.attributes.position;
    for (let i = 0; i < p.count; i++) {
      const a = (p.getX(i) / arcRadius); // bearing (rad) of this column, about home
      const r = fenceDistFt((a * 180) / Math.PI) + BATTERS_EYE_GAP_FT - 1.0;
      p.setXYZ(i, r * Math.sin(a), p.getY(i), -r * Math.cos(a));
    }
  }
  const mesh = noOutline(new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide })));
  return { mesh, update };
}

/** A tapered lattice-free mast with bracing rings, up to `h` ft. */
function mast(steel, h, r0, r1) {
  const g = new THREE.Group();
  const m = new THREE.Mesh(new THREE.CylinderGeometry(r1, r0, h, 4).rotateY(Math.PI / 4).translate(0, h / 2, 0), steel);
  m.castShadow = true;
  g.add(m);
  for (let y = 12; y < h - 6; y += 14) g.add(box(steel, (r0 + (r1 - r0) * (y / h)) * 2.5, 0.9, (r0 + (r1 - r0) * (y / h)) * 2.5, 0, y, 0));
  return g;
}

/** One light tower: mast, a lamp bank facing home, pennants on top. Lamps are written to lampSites for one instanced mesh. */
function lightTower(x, y, flags, poles, lampSites, glows) {
  const g = new THREE.Group();
  g.position.set(x, 0, -y);
  const steel = makeToonMaterial(PALETTE.steel);
  g.add(mast(steel, TOWER.mastH, 4.2, 2.3));
  g.add(box(makeToonMaterial(PALETTE.navy), 9, 3, 9, 0, 0, 0)); // footing
  const head = new THREE.Group();
  head.position.set(0, TOWER.mastH, 0);
  head.rotation.x = 0.1; // tilted down toward the field
  g.add(head);
  head.add(box(steel, 28, 15, 2.2, 0, -7.5, 0));
  head.add(box(makeToonMaterial(PALETTE.coral), 28.6, 1.4, 2.8, 0, 7.5, 0));
  head.add(box(makeToonMaterial(PALETTE.coral), 28.6, 1.1, 2.8, 0, -8.6, 0));
  for (let r = 0; r < 3; r++) for (let c = 0; c < 6; c++) lampSites.push({ parent: head, p: [-11.5 + c * 4.6, 3.8 - r * 4.6, 1.5] });
  const glow = makeGlowSprite(46);
  glow.position.set(0, 0, 4);
  head.add(glow);
  glows.push(glow);
  // Pennants on three short masts along the top.
  const colours = [PALETTE.coral, PALETTE.mustard, PALETTE.teal];
  [-11, 0, 11].forEach((dx, i) => {
    poles.push({ p: [x + dx, y], y: TOWER.mastH + 7.5, s: [1, 9, 1] });
    flags.push({ base: [x + dx, TOWER.mastH + 7.5 + 9 - 0.3, -y], yaw: 0, len: 9, h: 3.6, color: colours[(i + (x > 0 ? 1 : 0)) % 3], pennant: true });
  });
  return g;
}

export function buildTower(flags, poles) {
  const g = new THREE.Group();
  g.name = "spin-tower";
  const { x, y, ballR, ballZ, twinX } = TOWER;

  // The pylon: a tapered navy shaft with cream bands, a plinth, and a collar under the ball.
  const navy = makeToonMaterial(PALETTE.navy), cream = makeToonMaterial(PALETTE.cream), coral = makeToonMaterial(PALETTE.coral);
  const shaftH = ballZ - ballR - 3;
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(3.6, 6, shaftH, 8).translate(0, shaftH / 2, 0), navy);
  shaft.position.set(x, 3, -y);
  shaft.castShadow = shaft.receiveShadow = true;
  g.add(shaft);
  g.add(box(cream, 18, 3, 18, x, 0, -y));
  for (const h of [14, 28, 42]) {
    const t = h / shaftH;
    const band = new THREE.Mesh(new THREE.CylinderGeometry(6 - 2.4 * t - 0.05, 6 - 2.4 * t - 0.05 + 0.5, 1.6, 8), coral);
    band.position.set(x, 3 + h, -y);
    g.add(band);
  }
  const collar = new THREE.Mesh(new THREE.CylinderGeometry(7, 4, 4, 8), cream);
  collar.position.set(x, ballZ - ballR - 1, -y);
  collar.castShadow = true;
  g.add(collar);

  // Crest: the park's wordmark on a board between the twin towers, in front of the pylon.
  const crest = noOutline(new THREE.Mesh(new THREE.PlaneGeometry(56, 14), new THREE.MeshBasicMaterial({ map: wordmarkTexture() })));
  crest.position.set(x, 30, -y + 7.2);
  g.add(crest);
  const frame = box(navy, 58, 16, 1.2, x, 22, -y + 6.4);
  g.add(frame);
  for (const sx of [-26, 26]) g.add(box(navy, 2, 22, 2, x + sx, 0, -y + 6.4));

  // The readout under the ball (a copy of the one on the batter's eye wall: one texture each, same text).
  const pylonPanel = makePanel(24, 6.5);
  pylonPanel.mesh.position.set(x, ballZ - ballR - 3.5, -y + 6.4);
  pylonPanel.mesh.rotation.x = -0.12;
  g.add(pylonPanel.mesh);

  // The big panel on the face of the batter's eye wall, where the umpire's and batter's frames can see it.
  const eyeR = fenceDistFt(0) + BATTERS_EYE_GAP_FT - 1.0; // 1 ft off the wall face: depth resolution out here is a few tenths of a foot
  const eyePanel = makePanel(EYE_PANEL.w, EYE_PANEL.h, eyeR);
  eyePanel.mesh.position.set(0, EYE_PANEL.z, 0);
  g.add(eyePanel.mesh);

  // The ball: ball.js's, 120x bigger. Outline = a slightly bigger ink-coloured back-face shell (the viewer's
  // ball is not outlined by the post pass, and neither is this: it would scribble over the seams).
  const k = ballR / BALL_RADIUS_FT;
  const ball = noOutline(makeBall());
  ball.scale.setScalar(k);
  ball.children[0].scale.y = 0.62; // the blue axis rod: stub ends 0.3 radius out of the ball (it is 2 diameters long on the real-size ball)
  ball.position.set(x, ballZ, -y);
  ball.castShadow = true;
  const shell = noOutline(new THREE.Mesh(new THREE.SphereGeometry(BALL_RADIUS_FT * 1.035, 32, 20), new THREE.MeshBasicMaterial({ color: ink, side: THREE.BackSide })));
  ball.add(shell); // children [rod, seams, shell]: spinBall and setSeams use only the first two
  g.add(ball);

  // Twin light towers, lamps as one instanced mesh.
  const lampSites = [], glows = [];
  for (const s of [-1, 1]) g.add(lightTower(x + s * twinX, y, flags, poles, lampSites, glows));
  g.updateMatrixWorld(true);
  const lamps = new THREE.InstancedMesh(new THREE.BoxGeometry(3.8, 3.2, 0.7), lampMaterial, lampSites.length);
  const m = new THREE.Matrix4();
  lampSites.forEach((s, i) => {
    m.compose(new THREE.Vector3(...s.p), new THREE.Quaternion(), new THREE.Vector3(1, 1, 1)).premultiply(s.parent.matrixWorld);
    lamps.setMatrixAt(i, m);
  });
  g.add(noOutline(lamps));
  // Small up-lights round the pylon's collar: the ball is "softly lit" at night (lampMaterial glows then).
  const ups = new THREE.InstancedMesh(new THREE.BoxGeometry(2.2, 1.4, 2.2), lampMaterial, 6);
  for (let i = 0; i < 6; i++) {
    const a = (i / 6) * Math.PI * 2;
    place(ups, i, [x + Math.cos(a) * 9, 0.6, -y + Math.sin(a) * 9], 0, 1);
  }
  g.add(noOutline(ups));

  // Night: a warm wash on the ball (emissive), nothing by day. Checked every few frames, not each one.
  const ballMat = ball.material;
  let lastTod = null;
  const tickObj = new THREE.Object3D(); // the tick sits on its own empty, so mergeStatic still merges the tower
  g.add(tickObj);
  tickObj.userData.tick = () => {
    const tod = timeOfDay();
    if (tod === lastTod) return;
    lastTod = tod;
    ballMat.emissive.set(tod === "night" ? 0xd9c9a0 : tod === "sunset" ? 0x553a20 : 0x000000);
    ballMat.emissiveIntensity = tod === "night" ? 0.5 : 1;
  };

  /** spin: {rpm, axisDeg, angleRad, pitchType} or null. Turns the ball and updates the readouts (only when the text changes). */
  g.userData.setSpin = (spin) => {
    if (!spin) return;
    spinBall(ball, Number.isFinite(spin.rpm) ? spin.axisDeg : null, spin.angleRad);
    setSeams(ball, seamStyle(spin.pitchType));
    const text = rpmLabel(spin.rpm);
    pylonPanel.update(text);
    eyePanel.update(text);
  };
  return g;
}
