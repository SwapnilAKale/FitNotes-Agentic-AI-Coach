/* A muscular arm curling a dumbbell, drawn live with three.js in a flat blue cartoon style
 * on a transparent background. Used by vendor/pull-reload.js for the pull-to-reload badge.
 *
 *   const arm = await createCurlArm(box);   // fills the box and keeps fitting it
 *   arm.lift(p, dt);   // hold the arm p of the way up the curl (0 = hanging, 1 = top);
 *                      // dt (seconds since the last call) keeps the effort shake moving
 *   arm.play();        // full reps with their own timing (resistance, squeeze, pause)
 *   arm.pause();
 *
 * The model, its muscle-detail texture and three.js (release 170, MIT, in ./three) sit next to
 * this file. For visitors with reduced motion turned on, play() shows a still pose.
 */
import * as THREE from './three/three.module.min.js';
import { GLTFLoader } from './three/GLTFLoader.js';
import { OutlineEffect } from './three/OutlineEffect.js';

/* =====================================================================
   SETTINGS — everything you might want to change is here.
   ===================================================================== */
const DEFAULTS = {
  arm: 'athletic',                     // 'athletic' (the only model shipped here)
  muscleDetail: true,                  // muscle striations from the detail texture
  detailStrength: 0.8,                 // how strongly the muscle detail shows

  // Colours
  skin:        '#6eb6e2',
  shadowLevel: 0.62,                   // how dark the shaded side is (0 = black, 1 = no shadow)
  outline:     '#1d3349',
  muscleLines: '#2c6a99',
  weight:      '#5a5e66',

  // Lines
  outlineThickness: 0.008,             // outer outline
  showMuscleLines:  true,
  muscleLineWidth:  1.3,               // in screen pixels
  muscleLineAmount: 1.2,               // lower = more lines, higher = fewer

  // Light (direction it comes from: x = towards camera, y = up, z = in front of the arm)
  light: [-2, 3, 3],

  // Dumbbell (in model units, roughly centimetres)
  plateRadius: 7.5,
  plateThickness: 3.4,
  handleLength: 15,
  handleRadius: 1.6,
  gripOffset: 2.5,                     // moves the bar deeper into the palm

  // Motion
  speed: 1.0,                          // 0.5 = half speed, 2 = double speed
  pauseAtBottom: 0.35,                 // extra seconds resting at the bottom of each rep
  // Effort effects, all building up as the biceps contracts (0 = off, 1 = normal, higher = stronger)
  effort: { resistance: 1, shake: 1, veins: 1, strainMarks: 1, pumpFlush: 1 },
  strainGap: 1.6,                     // how far out from the arm the biceps strain marks start (model units)
  topHold: 0.3,                        // seconds the squeeze is held at the top (part of the resistance)
  flushColor: '#8f99ea',               // tint of the pump flush on the working muscles
  veinColor: '#8ccbee',               // veins read as slightly lighter raised ridges
  bicepsPump: 1.0,                     // how much the biceps bunches up into a peak as the elbow bends (0 = off)
  maxElbowBend: 105,                   // how far the forearm can fold up (degrees from the start pose).
                                       // Lower = biceps stays more visible at the top; null = no limit

  // Framing
  padding: 0.04,                       // empty space around the arm (fraction of the frame)
  armThickness: 7,                     // used to frame the arm (model units)
  shoulderFade: true,                  // soft fade where the arm is cut off at the shoulder
  fadeStart: -6, fadeLength: 5,        // where the fade begins above the shoulder joint, and its length (model units)
  fadeFront: 11, fadeBack: 30,         // how far in front of / behind the arm the fade reaches (model units)
  shoulderCurve: { width: 7.4, height: 6, joinAt: 2 },  // smooth rounded front of the shoulder top (model units)
};

export async function createCurlArm(host, settings = {}) {
const CONFIG = { ...DEFAULTS, ...settings, effort: { ...DEFAULTS.effort, ...settings.effort } };

/* ===================================================================== */
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setClearColor(0x000000, 0);
host.appendChild(renderer.domElement);
// strain marks are drawn on an off-screen 2D canvas and blended into the final image by the post shader
const overlay = document.createElement('canvas');
const octx = overlay.getContext('2d');
const overlayTex = new THREE.CanvasTexture(overlay);
overlayTex.colorSpace = THREE.NoColorSpace;
overlayTex.minFilter = overlayTex.magFilter = THREE.LinearFilter;
overlayTex.generateMipmaps = false;

const effect = new OutlineEffect(renderer, {
  defaultThickness: CONFIG.outlineThickness,
  defaultColor: new THREE.Color(CONFIG.outline).toArray(),
  defaultAlpha: 1,
});

const scene = new THREE.Scene();
const light = new THREE.DirectionalLight(0xffffff, Math.PI);
light.position.set(...CONFIG.light);
scene.add(light);

// two-tone toon ramp
const ramp = new Uint8Array([Math.round(CONFIG.shadowLevel * 255), 255]);
const gradientMap = new THREE.DataTexture(ramp, 2, 1, THREE.RedFormat);
gradientMap.minFilter = gradientMap.magFilter = THREE.NearestFilter;
gradientMap.needsUpdate = true;
const skinMat = new THREE.MeshToonMaterial({ color: CONFIG.skin, gradientMap });
const metalMat = new THREE.MeshToonMaterial({ color: CONFIG.weight, gradientMap });
// pump flush: the working muscles (marked in the model) take on a tint as they contract
const flushAmount = { value: 0 };
skinMat.onBeforeCompile = (sh) => {
  sh.uniforms.uFlush = flushAmount;
  sh.uniforms.flushColor = { value: new THREE.Color(CONFIG.flushColor) };
  sh.vertexShader = 'attribute float _flush;\nvarying float vFlush;\n' + sh.vertexShader.replace('#include <begin_vertex>', '#include <begin_vertex>\n  vFlush = _flush;');
  sh.fragmentShader = 'uniform float uFlush;\nuniform vec3 flushColor;\nvarying float vFlush;\n' +
    sh.fragmentShader.replace('vec4 diffuseColor = vec4( diffuse, opacity );', 'vec4 diffuseColor = vec4( mix( diffuse, flushColor, clamp( vFlush * uFlush, 0.0, 1.0 ) ), opacity );');
};
const veinMat = new THREE.MeshToonMaterial({ color: CONFIG.veinColor, gradientMap });
veinMat.userData.outlineParameters = { visible: false };       // veins read as raised ridges with muscle-line edges


// ---- load the arm ----
const asset = name => new URL(name, import.meta.url).href;
const [gltf, detail] = await Promise.all([
  new GLTFLoader().loadAsync(asset(`arm-${CONFIG.arm}.glb`)),
  CONFIG.muscleDetail ? new THREE.TextureLoader().loadAsync(asset(`arm-${CONFIG.arm}-normal.webp`)) : Promise.resolve(null),
]);
const normalMats = new Map();               // per-mesh material for the line-detection pass
const plainNormal = new THREE.MeshNormalMaterial();
if (detail) {
  detail.colorSpace = THREE.NoColorSpace;
  detail.anisotropy = renderer.capabilities.getMaxAnisotropy();
  skinMat.normalMap = detail;
  skinMat.normalScale.setScalar(CONFIG.detailStrength);
}
const skinNormalMat = new THREE.MeshNormalMaterial(detail ? { normalMap: detail, normalScale: new THREE.Vector2(CONFIG.detailStrength, CONFIG.detailStrength) } : {});
const arm = gltf.scene;
arm.traverse(o => {
  if (!o.isMesh) return;
  o.frustumCulled = false;
  if (o.name.includes('veins')) o.material = veinMat;
  else { o.material = skinMat; normalMats.set(o, skinNormalMat); }
});
scene.add(arm);
const bones = {};
arm.traverse(o => { if (o.isBone) bones[o.name.replace('mixamorig', '').replace(':', '')] = o; });
const boneWorld = n => bones[n].getWorldPosition(new THREE.Vector3());

const mixer = new THREE.AnimationMixer(arm);
const clip = gltf.animations[0];
mixer.clipAction(clip).play();

// ---- dumbbell, kept in the fist every frame ----
const dumbbell = new THREE.Group();
{
  // plates are lathed with rounded edges so the outline stays clean
  const R = CONFIG.plateRadius, T = CONFIG.plateThickness, b = Math.min(1.2, T * 0.3);
  const prof = [new THREE.Vector2(0.01, -T / 2)];
  for (let i = 0; i <= 6; i++) { const a = -Math.PI / 2 + (Math.PI / 2) * i / 6; prof.push(new THREE.Vector2(R - b + Math.cos(a) * b, -T / 2 + b + Math.sin(a) * b)); }
  for (let i = 0; i <= 6; i++) { const a = (Math.PI / 2) * i / 6; prof.push(new THREE.Vector2(R - b + Math.cos(a) * b, T / 2 - b + Math.sin(a) * b)); }
  prof.push(new THREE.Vector2(0.01, T / 2));
  const plateGeo = new THREE.LatheGeometry(prof, 48);
  const bar = new THREE.Mesh(new THREE.CylinderGeometry(CONFIG.handleRadius, CONFIG.handleRadius, CONFIG.handleLength + 2 * T, 24, 1, true), metalMat);
  dumbbell.add(bar);
  for (const s of [-1, 1]) {
    const plate = new THREE.Mesh(plateGeo, metalMat);
    plate.position.y = s * (CONFIG.handleLength / 2 + T / 2 - 1.2);
    dumbbell.add(plate);
  }
}
scene.add(dumbbell);
function placeDumbbell() {
  const h = boneWorld('RightHand'), idx = boneWorld('RightHandIndex1'), pin = boneWorld('RightHandPinky1'), mid = boneWorld('RightHandMiddle1');
  const axis = idx.clone().sub(pin).normalize();
  const along = mid.clone().sub(h).normalize();
  const palm = new THREE.Vector3().crossVectors(axis, along).normalize();
  dumbbell.position.copy(idx.clone().add(pin).multiplyScalar(0.5).lerp(h, 0.15).add(palm.multiplyScalar(CONFIG.gripOffset)));
  dumbbell.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), axis);
  dumbbell.updateMatrixWorld(true);
}
// Elbow limit: the forearm eases into a maximum bend instead of folding onto the biceps
mixer.setTime(0);
const forearmStart = bones['RightForeArm'].quaternion.clone();
const forearmNow = new THREE.Quaternion();
function limitElbow() {
  if (CONFIG.maxElbowBend == null) return;
  const fa = bones['RightForeArm'];
  forearmNow.copy(fa.quaternion);
  const bend = THREE.MathUtils.radToDeg(forearmStart.angleTo(forearmNow));
  const max = CONFIG.maxElbowBend, soft = 25, knee = max - soft;
  if (bend <= knee) return;
  const limited = knee + soft * Math.tanh((bend - knee) / soft);   // smooth, never passes max
  fa.quaternion.slerpQuaternions(forearmStart, forearmNow, limited / bend);
}
// ---- effort: how hard the arm is working (0 relaxed .. 1 fully contracted), from the elbow bend ----
const fx = CONFIG.effort;
const effortMeshes = { arm: [], veins: [] };
arm.traverse(o => { if (o.isMesh && o.morphTargetInfluences && o.morphTargetInfluences.length)
  (o.name.includes('veins') ? effortMeshes.veins : effortMeshes.arm).push(o); });
let effort = 0, shakeTime = 0;
function measureEffort() {
  const bend = THREE.MathUtils.radToDeg(forearmStart.angleTo(bones['RightForeArm'].quaternion));
  effort = THREE.MathUtils.smoothstep(bend, 15, 95);
}
// Shake: a fine tremble of the forearm and the weight under load
const _q = new THREE.Quaternion(), _q2 = new THREE.Quaternion(), _ax = new THREE.Vector3(), handTwistAxis = new THREE.Vector3(0, 1, 0);
function applyShake() {
  const amp = THREE.MathUtils.degToRad(1.4) * fx.shake * Math.pow(effort, 1.5);
  if (amp <= 0) return;
  const T = shakeTime * Math.PI * 2;
  const n1 = Math.sin(T * 9.3) * 0.55 + Math.sin(T * 13.7 + 1.3) * 0.3 + Math.sin(T * 5.9 + 2.2) * 0.25;
  const n2 = Math.sin(T * 11.1 + 0.7) * 0.5 + Math.sin(T * 7.3 + 2.9) * 0.35 + Math.sin(T * 16.9 + 0.4) * 0.15;
  const fa = bones['RightForeArm'];
  _q.copy(forearmStart).invert().multiply(fa.quaternion);
  _ax.set(_q.x, _q.y, _q.z); if (_ax.lengthSq() < 1e-10) _ax.set(1, 0, 0); _ax.normalize();
  fa.quaternion.multiply(_q2.setFromAxisAngle(_ax, amp * n1));                       // elbow trembles
  bones['RightHand'].quaternion.multiply(_q2.setFromAxisAngle(handTwistAxis, amp * 1.6 * n2));  // weight wobbles in the fist (no wrist bend)
}
function pumpBiceps() {
  const t = effort;
  for (const m of effortMeshes.arm) m.morphTargetInfluences[0] = t * CONFIG.bicepsPump;
  const vein = THREE.MathUtils.smoothstep(t, 0.25, 0.9) * fx.veins;
  for (const m of effortMeshes.veins) { m.morphTargetInfluences[0] = t * CONFIG.bicepsPump; m.morphTargetInfluences[1] = Math.min(1.5, vein); }
  flushAmount.value = 0.38 * t * fx.pumpFlush;
}
// The animation player only rewrites a joint when its value changes, so restore the clean
// animation pose first; otherwise holding still (e.g. the squeeze at the top) would re-apply the elbow limit and shake.
const rawFA = bones['RightForeArm'].quaternion.clone(), rawHand = bones['RightHand'].quaternion.clone();
function poseAt(t) {
  bones['RightForeArm'].quaternion.copy(rawFA); bones['RightHand'].quaternion.copy(rawHand);
  mixer.setTime(t);
  rawFA.copy(bones['RightForeArm'].quaternion); rawHand.copy(bones['RightHand'].quaternion);
  limitElbow(); measureEffort(); applyShake(); pumpBiceps(); arm.updateMatrixWorld(true); placeDumbbell();
}

// ---- side camera framed around the whole movement ----
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 1, 5000);
const bounds = { y0: Infinity, y1: -Infinity, z0: Infinity, z1: -Infinity };
const tmp = new THREE.Box3();
const grow = (v, r) => { bounds.y0 = Math.min(bounds.y0, v.y - r); bounds.y1 = Math.max(bounds.y1, v.y + r); bounds.z0 = Math.min(bounds.z0, v.z - r); bounds.z1 = Math.max(bounds.z1, v.z + r); };
for (let i = 0; i <= 40; i++) {
  poseAt(clip.duration * i / 40);
  grow(boneWorld('RightArm'), CONFIG.armThickness * 1.15);
  grow(boneWorld('RightForeArm'), CONFIG.armThickness);
  grow(boneWorld('RightHand'), CONFIG.armThickness * 0.8);
  grow(boneWorld('RightHandMiddle1'), CONFIG.armThickness * 0.8);
  tmp.setFromObject(dumbbell); grow(tmp.min, 0); grow(tmp.max, 0);
}
const centerY = (bounds.y0 + bounds.y1) / 2, centerZ = (bounds.z0 + bounds.z1) / 2;
camera.position.set(-1000, centerY, centerZ);
camera.lookAt(0, centerY, centerZ);

function fitCamera(w, h) {
  const bw = (bounds.z1 - bounds.z0), bh = (bounds.y1 - bounds.y0);
  const aspect = w / h;
  let halfH = bh / 2, halfW = bw / 2;
  if (halfW / halfH > aspect) halfH = halfW / aspect; else halfW = halfH * aspect;
  const pad = 1 / (1 - 2 * CONFIG.padding);
  camera.left = -halfW * pad; camera.right = halfW * pad; camera.top = halfH * pad; camera.bottom = -halfH * pad;
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld();
}

// ---- post-processing: muscle lines (from surface normals) + shoulder fade ----
const rtNormal = new THREE.WebGLRenderTarget(1, 1, { samples: 4 });
const rtColor = new THREE.WebGLRenderTarget(1, 1, { samples: 4 });
rtColor.texture.colorSpace = THREE.SRGBColorSpace;
const post = new THREE.ShaderMaterial({
  uniforms: {
    tColor: { value: rtColor.texture }, tNormal: { value: rtNormal.texture }, tOverlay: { value: overlayTex }, overlayOn: { value: 0 },
    px: { value: new THREE.Vector2() }, lineColor: { value: new THREE.Color(CONFIG.muscleLines) },
    threshold: { value: CONFIG.muscleLineAmount }, showLines: { value: CONFIG.showMuscleLines ? 1 : 0 },
    fadeP: { value: new THREE.Vector2() }, fadeD: { value: new THREE.Vector2(0, -1) },
    fadeStart: { value: 0 }, fadeLen: { value: 0 }, fadeFront: { value: 0 }, fadeBack: { value: 0 }, scRx: { value: 1 }, scRy: { value: 1 }, scC: { value: 0 }, olColor: { value: new THREE.Color(CONFIG.outline) }, aspect: { value: 1 },
    fadeOn: { value: CONFIG.shoulderFade ? 1 : 0 },
  },
  vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }',
  fragmentShader: `
    uniform sampler2D tColor, tNormal, tOverlay; uniform vec2 px, fadeP, fadeD; uniform vec3 lineColor;
    uniform float overlayOn;
    uniform float threshold, showLines, fadeStart, fadeLen, fadeFront, fadeBack, fadeOn, aspect, scRx, scRy, scC; uniform vec3 olColor;
    varying vec2 vUv;
    vec3 n(vec2 o){ return texture2D(tNormal, vUv + o * px).rgb * 2.0 - 1.0; }
    float a(vec2 o){ return texture2D(tNormal, vUv + o * px).a; }
    void main(){
      vec4 c = texture2D(tColor, vUv);
      vec3 gx = -n(vec2(-1,-1)) - 2.0*n(vec2(-1,0)) - n(vec2(-1,1)) + n(vec2(1,-1)) + 2.0*n(vec2(1,0)) + n(vec2(1,1));
      vec3 gy = -n(vec2(-1,-1)) - 2.0*n(vec2(0,-1)) - n(vec2(1,-1)) + n(vec2(-1,1)) + 2.0*n(vec2(0,1)) + n(vec2(1,1));
      float g = sqrt(dot(gx,gx) + dot(gy,gy));
      float inside = a(vec2(-3.5,-3.5)) * a(vec2(3.5,3.5)) * a(vec2(-3.5,3.5)) * a(vec2(3.5,-3.5)) * a(vec2(0,4.5)) * a(vec2(0,-4.5)) * a(vec2(4.5,0)) * a(vec2(-4.5,0));
      float e = smoothstep(threshold, threshold * 1.6, g) * inside * showLines;
      // smooth rounded front for the top of the shoulder (only above the point where it joins the arm)
      vec2 relS = (vUv - fadeP) * vec2(aspect, 1.0);
      float alS = dot(relS, fadeD), acS = dot(relS, vec2(-fadeD.y, fadeD.x));
      float inCurve = 1.0;
      if (fadeOn > 0.5 && acS > 0.0 && alS < scC && acS < fadeFront) {
        float r = length(vec2(acS / scRx, (alS - scC) / scRy));
        inCurve = 1.0 - smoothstep(0.97, 1.0, r);
        // fade the outer outline in along the curve so it doesn't start with a step
        vec3 cl = c.rgb / max(c.a, 1e-4);
        float isOutline = 1.0 - smoothstep(0.08, 0.22, distance(cl, olColor));
        c *= 1.0 - isOutline * (1.0 - smoothstep(0.85, 1.0, (scC - alS) < 0.0 ? 1.0 : 1.0 - (scC - alS) / scRy));
      }
      gl_FragColor = vec4(mix(c.rgb, lineColor, e * c.a), c.a);
      #include <colorspace_fragment>
      gl_FragColor *= inCurve;
      vec2 rel = (vUv - fadeP) * vec2(aspect, 1.0);
      float across = dot(rel, vec2(-fadeD.y, fadeD.x));      // + = in front of the arm
      float lim = across > 0.0 ? fadeFront : fadeBack;
      float f = max(smoothstep(fadeStart, fadeStart + fadeLen, dot(rel, fadeD)), smoothstep(lim, lim + 0.02, abs(across)));
      gl_FragColor *= mix(1.0, f, fadeOn);
      vec4 ov = texture2D(tOverlay, vUv) * overlayOn;                // strain marks, on top of everything
      gl_FragColor = gl_FragColor * (1.0 - ov.a) + vec4(ov.rgb * ov.a, ov.a);
    }`,
});
const postScene = new THREE.Scene();
postScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), post));
const postCam = new THREE.Camera();

function toScreen(v) { const p = v.clone().project(camera); return new THREE.Vector2(p.x * 0.5 + 0.5, p.y * 0.5 + 0.5); }

let overlayHasMarks = false;
function drawStrainMarks() {
  const W = overlay.width, H = overlay.height;
  const k = THREE.MathUtils.smoothstep(effort, 0.4, 0.95) * fx.strainMarks;
  if (k <= 0.01) {                                   // nothing to draw: clear once and stop updating
    if (overlayHasMarks) { octx.clearRect(0, 0, W, H); overlayHasMarks = false; overlayTex.needsUpdate = true; }
    post.uniforms.overlayOn.value = 0;               // no marks: don't show the marks layer at all
    return;
  }
  octx.clearRect(0, 0, W, H);
  overlayHasMarks = true; overlayTex.needsUpdate = true;
  post.uniforms.overlayOn.value = 1;
  const unitPx = H / (camera.top - camera.bottom);
  const scr = v => { const p = v.clone().project(camera); return new THREE.Vector2((p.x * 0.5 + 0.5) * W, (1 - (p.y * 0.5 + 0.5)) * H); };
  const S = boneWorld('RightArm'), E = boneWorld('RightForeArm'), Hd = boneWorld('RightHand');
  const up = E.clone().sub(S).normalize(), fo = Hd.clone().sub(E).normalize();
  const front = fo.clone().sub(up.clone().multiplyScalar(fo.dot(up))).normalize();
  octx.save();
  octx.strokeStyle = CONFIG.outline; octx.lineCap = 'round'; octx.globalAlpha = Math.min(1, k);
  octx.lineWidth = 0.34 * unitPx;
  const flick = (i) => 0.82 + 0.18 * Math.sin(shakeTime * 17 + i * 2.1);
  // 1. effort strokes fanning out from the biceps peak
  // aim out of the crook of the arm: between the biceps' front and straight up the upper arm
  const outW = front.clone().multiplyScalar(0.75).sub(up.clone().multiplyScalar(0.66)).normalize();
  const peakW = S.clone().lerp(E, 0.5).add(front.clone().multiplyScalar(7.4 + 2.4 * effort * CONFIG.bicepsPump)).add(outW.clone().multiplyScalar(CONFIG.strainGap));
  const pk = scr(peakW), dirOut = scr(peakW.clone().add(outW)).sub(pk).normalize();
  for (let i = -1; i <= 1; i++) {
    const a = i * 0.42, d = dirOut.clone().rotateAround(new THREE.Vector2(), a);
    const startOff = pk.clone().add(new THREE.Vector2(-dirOut.y, dirOut.x).multiplyScalar(i * 1.1 * unitPx));
    const len = (3.2 - Math.abs(i) * 0.7) * unitPx * flick(i) * (0.6 + 0.4 * k);
    const p0 = startOff.clone().add(d.clone().multiplyScalar(0.4 * unitPx)), p1 = startOff.clone().add(d.clone().multiplyScalar(0.4 * unitPx + len));
    const bend = new THREE.Vector2(-d.y, d.x).multiplyScalar(0.18 * len * (i === 0 ? 0.4 : i));
    octx.beginPath(); octx.moveTo(p0.x, p0.y); octx.quadraticCurveTo((p0.x + p1.x) / 2 + bend.x, (p0.y + p1.y) / 2 + bend.y, p1.x, p1.y); octx.stroke();
  }
  // 2. tremble marks beside the dumbbell, on the side away from the elbow
  const dc = scr(dumbbell.position), away = scr(dumbbell.position.clone().add(fo)).sub(dc).normalize();
  const base = Math.atan2(away.y, away.x), jitter = Math.sin(shakeTime * 40) * 0.06 * unitPx;
  octx.lineWidth = 0.28 * unitPx;
  for (let j = 0; j < 2; j++) {
    const r = (CONFIG.plateRadius + 1.6 + j * 1.3) * unitPx + jitter;
    for (const side of [-1, 1]) {
      const c = base + side * 0.95;
      octx.beginPath(); octx.arc(dc.x, dc.y, r, c - 0.32, c + 0.32); octx.stroke();
    }
  }
  octx.restore();
}

function render() {
  if (CONFIG.shoulderFade) {
    const s = toScreen(boneWorld('RightArm')), e = toScreen(boneWorld('RightForeArm'));
    const aspect = (camera.right - camera.left) / (camera.top - camera.bottom);
    const unit = 1 / (camera.top - camera.bottom);          // model units -> screen fraction (height)
    post.uniforms.aspect.value = aspect;
    post.uniforms.fadeP.value.copy(s);
    post.uniforms.fadeD.value.copy(e.sub(s).multiply(new THREE.Vector2(aspect, 1)).normalize());
    post.uniforms.fadeStart.value = CONFIG.fadeStart * unit;
    post.uniforms.fadeLen.value = CONFIG.fadeLength * unit;
    post.uniforms.fadeFront.value = CONFIG.fadeFront * unit;
    post.uniforms.fadeBack.value = CONFIG.fadeBack * unit;
    post.uniforms.scRx.value = CONFIG.shoulderCurve.width * unit;
    post.uniforms.scRy.value = CONFIG.shoulderCurve.height * unit;
    post.uniforms.scC.value = CONFIG.shoulderCurve.joinAt * unit;
  }
  const saved = [];
  scene.traverse(o => { if (o.isMesh) { saved.push([o, o.material]); o.material = normalMats.get(o) || plainNormal; } });
  renderer.setRenderTarget(rtNormal); renderer.clear(); renderer.render(scene, camera);
  for (const [o, m] of saved) o.material = m;
  renderer.setRenderTarget(rtColor); renderer.clear(); effect.render(scene, camera);
  drawStrainMarks();
  renderer.setRenderTarget(null); renderer.clear(); renderer.render(postScene, postCam);
}

function resize() {
  const w = Math.max(1, host.clientWidth), h = Math.max(1, host.clientHeight);
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  renderer.setPixelRatio(dpr);
  renderer.setSize(w, h, false);
  rtNormal.setSize(w * dpr, h * dpr); rtColor.setSize(w * dpr, h * dpr);
  overlay.width = Math.round(w * dpr); overlay.height = Math.round(h * dpr);
  overlayTex.dispose();                              // new size: let the texture be re-created at the new size
  overlayHasMarks = true; overlayTex.needsUpdate = true;
  post.uniforms.px.value.set(CONFIG.muscleLineWidth / w, CONFIG.muscleLineWidth / h);
  fitCamera(w, h);
}
new ResizeObserver(() => { resize(); render(); }).observe(host);
resize();

// ---- timeline: the curl slows through the hard part, holds the squeeze, lowers under control, rests ----
// find where the lift starts and where the top of the curl is
let tLift0 = 0, tPeak = 0;
{ let best = -1;
  for (let i = 0; i <= 120; i++) { const t = clip.duration * i / 120; mixer.setTime(t);
    const b = THREE.MathUtils.radToDeg(forearmStart.angleTo(bones['RightForeArm'].quaternion));
    if (!tLift0 && b > 8) tLift0 = t; if (b > best) { best = b; tPeak = t; } } }
const still = matchMedia('(prefers-reduced-motion: reduce)');
let playing = false, clipT = 0, holdT = 0, pauseT = 0, heldThisRep = false, last = 0, frame = 0;

function advance(dt) {
  const R = fx.resistance;
  if (pauseT > 0) { pauseT -= dt; if (pauseT <= 0) { clipT = 0; heldThisRep = false; } return; }
  if (holdT > 0) { holdT -= dt; return; }
  let rate = 1;
  if (clipT < tPeak) {                       // lifting: slowest through the middle (the sticking point)
    const u = THREE.MathUtils.clamp((clipT - tLift0) / Math.max(1e-3, tPeak - tLift0), 0, 1);
    rate = 1 - R * 0.42 * Math.sin(Math.PI * Math.min(1, u * 1.08));
  } else rate = 1 - R * 0.12;                // lowering under control
  rate = Math.max(0.2, rate);
  const next = clipT + dt * rate;
  if (!heldThisRep && clipT < tPeak && next >= tPeak && R > 0 && CONFIG.topHold > 0) {
    clipT = tPeak; holdT = CONFIG.topHold * Math.min(1, R); heldThisRep = true; return;
  }
  clipT = next;
  if (clipT >= clip.duration) { clipT = clip.duration; pauseT = CONFIG.pauseAtBottom; heldThisRep = true; if (pauseT <= 0) { clipT = 0; heldThisRep = false; } }
}

function tick(now) {
  if (!playing) return;
  const dt = Math.min((now - last) / 1000, 0.1); last = now;
  shakeTime += dt;
  advance(dt * CONFIG.speed);
  poseAt(clipT);
  render();
  frame = requestAnimationFrame(tick);
}
poseAt(0);
render();

return {
  // Hold the arm p of the way from hanging (0) to the top of the curl (1).
  lift(p, dt = 0) {
    playing = false; cancelAnimationFrame(frame);
    shakeTime += dt;
    clipT = THREE.MathUtils.clamp(p, 0, 1) * tPeak; holdT = pauseT = 0; heldThisRep = p >= 1;
    poseAt(clipT); render();
  },
  // Full reps from wherever the arm is now; a still pose for reduced motion.
  play() {
    if (still.matches || playing) return;
    playing = true; last = performance.now();
    frame = requestAnimationFrame(tick);
  },
  pause() { playing = false; cancelAnimationFrame(frame); },
};
}
