// parsimony viewer — three.js renderer for parsimony.pack.v1 files.
// Reads our native pack format (compartments + ingredients + per-
// instance placements with quaternion rotations + mesh URLs). For
// each ingredient type we build an InstancedMesh: sphere or
// multi-sphere ingredients render from a sphere primitive, mesh
// ingredients fetch their OBJ via OBJLoader and swap the geometry
// in once available (with a sphere placeholder during the fetch).
// Standard PBR view + an opt-in Goodsell post-pass with cel
// shading + depth-Sobel outlines.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { ShaderPass } from "three/addons/postprocessing/ShaderPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { mergeGeometries, mergeVertices } from "three/addons/utils/BufferGeometryUtils.js";

// ───── DOM refs ─────────────────────────────────────────────────────
const canvasWrap = document.getElementById("canvas-wrap");
const fileInput = document.getElementById("file-input");
const resetBtn = document.getElementById("reset-view");
const dropOverlay = document.getElementById("drop-overlay");
const legendEl = document.getElementById("legend");
const statusEl = document.getElementById("status");
const placementsStat = document.getElementById("stat-placements");
const typesStat = document.getElementById("stat-types");
const fpsStat = document.getElementById("stat-fps");
const toggleBbox = document.getElementById("toggle-bbox");
const toggleAxes = document.getElementById("toggle-axes");
const toggleGrid = document.getElementById("toggle-bg-grid");
const toggleSpin = document.getElementById("toggle-spin");
const sliceAxis = document.getElementById("slice-axis");
const slicePos = document.getElementById("slice-pos");
const slicePosValue = document.getElementById("slice-pos-value");
const sliceFlip = document.getElementById("slice-flip");

// ───── three.js scene ───────────────────────────────────────────────
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0e1116);
scene.fog = new THREE.Fog(0x0e1116, 1500, 6000);

const camera = new THREE.PerspectiveCamera(40, 1, 0.5, 100000);
camera.position.set(150, 120, 200);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.localClippingEnabled = true;
canvasWrap.appendChild(renderer.domElement);

// Style + outline state. Declared here so the composer setup below
// (which reads `outlinePixels`) doesn't trip the TDZ.
let style = "standard";
let outlinePixels = 1.5;

// ───── post-processing (Goodsell edge pass) ────────────────────────
// All the EffectComposer machinery is wrapped in try/catch — if any
// part of it fails (postprocessing addon mis-resolved, GL2 feature
// missing, depth-texture attachment quirk), the module still finishes
// loading and standard rendering keeps working. Goodsell mode is
// then disabled in the UI.
let composer = null;
let goodsellPass = null;
let goodsellAvailable = false;
try {
  // three.js canonical depth-post pattern: nearest-filtered colour
  // RT with an explicit DepthTexture(DepthFormat / UnsignedShortType).
  // We don't go through `composer.setPixelRatio` either — letting the
  // composer size match the renderer at resize time avoids a mis-
  // configured render target that some drivers reject.
  const depthTexture = new THREE.DepthTexture();
  depthTexture.format = THREE.DepthFormat;
  depthTexture.type = THREE.UnsignedShortType;
  const renderTarget = new THREE.WebGLRenderTarget(
    window.innerWidth,
    window.innerHeight,
    {
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      stencilBuffer: false,
      depthBuffer: true,
      depthTexture: depthTexture,
    }
  );

  composer = new EffectComposer(renderer, renderTarget);
  composer.setPixelRatio(window.devicePixelRatio);
  // EffectComposer clones the supplied renderTarget for its second
  // ping-pong buffer via WebGLRenderTarget.clone(), which *shares*
  // the same DepthTexture object. That creates an illegal feedback
  // loop: while ShaderPass writes to rt2, the shared depth texture
  // is bound as both DEPTH_ATTACHMENT (write) and TEXTURE_2D (read)
  // → "drawArraysInstanced: Texture level 0 would be read by
  // TEXTURE_2D unit 1, but written by framebuffer attachment
  // DEPTH_ATTACHMENT, which would be illegal feedback". Detach rt2
  // from the depth side entirely; only rt1 carries depth (which
  // RenderPass writes first), and downstream passes write colour
  // only to rt2.
  composer.renderTarget2.depthTexture = null;
  composer.renderTarget2.depthBuffer = false;
  const renderPass = new RenderPass(scene, camera);
  composer.addPass(renderPass);

  // Depth-edge outline pass. Linearizes the depth buffer, runs a 3×3
  // Sobel kernel, soft-thresholds, multiplies the input colour by
  // (1 − edge). Thickness is in screen pixels (independent of world
  // scale), so big and small objects both get a 1–2 px black silhouette.
  const GoodsellEdgeShader = {
  uniforms: {
    tDiffuse: { value: null },
    tDepth: { value: depthTexture },
    resolution: { value: new THREE.Vector2(1, 1) },
    cameraNear: { value: camera.near },
    cameraFar: { value: camera.far },
    outlineThickness: { value: outlinePixels },
    // The outline pass darkens the colour where the depth Sobel says
    // "edge here". Keep strength moderate so silhouettes are visible
    // but flat faces aren't crushed to black; threshold is a per-frag
    // depth-gradient cutoff (units are world-Å per pixel after the
    // centre-depth normalisation in the fragment shader).
    outlineStrength: { value: 0.7 },
    edgeThreshold: { value: 4.0 },
  },
  vertexShader: `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: `
    uniform sampler2D tDiffuse;
    uniform sampler2D tDepth;
    uniform vec2 resolution;
    uniform float cameraNear;
    uniform float cameraFar;
    uniform float outlineThickness;
    uniform float outlineStrength;
    uniform float edgeThreshold;
    varying vec2 vUv;

    float linearize(float d) {
      // Maps buffer-space depth d in [0,1] (what texture2D returns
      // from a DepthTexture) to view-space distance. The textbook
      // 2*n*f / (f+n - d*(f-n)) formula is for NDC depth in [-1,1],
      // not the buffer value; using it on the buffer value gives
      // about 2x the actual distance with non-linear scaling that
      // produced huge Sobel gradients on smooth surfaces — which
      // painted the whole scene black.
      return (cameraNear * cameraFar) /
             (cameraFar - d * (cameraFar - cameraNear));
    }

    float sampleD(vec2 uv) {
      return linearize(texture2D(tDepth, uv).x);
    }

    void main() {
      vec4 col = texture2D(tDiffuse, vUv);
      vec2 px = outlineThickness / resolution;

      // Sobel kernel over linearized depth — uses the corrected
      // buffer-to-view-depth mapping (see linearize() above).
      float d00 = sampleD(vUv + px * vec2(-1.0, -1.0));
      float d10 = sampleD(vUv + px * vec2( 0.0, -1.0));
      float d20 = sampleD(vUv + px * vec2( 1.0, -1.0));
      float d01 = sampleD(vUv + px * vec2(-1.0,  0.0));
      float d21 = sampleD(vUv + px * vec2( 1.0,  0.0));
      float d02 = sampleD(vUv + px * vec2(-1.0,  1.0));
      float d12 = sampleD(vUv + px * vec2( 0.0,  1.0));
      float d22 = sampleD(vUv + px * vec2( 1.0,  1.0));
      float gx = (d20 + 2.0*d21 + d22) - (d00 + 2.0*d01 + d02);
      float gy = (d02 + 2.0*d12 + d22) - (d00 + 2.0*d10 + d20);
      float g  = sqrt(gx*gx + gy*gy);

      // Normalize by centre depth so the same threshold works at all
      // distances. Edge magnitude is roughly "ratio of depth-jump to
      // surface-depth"; silhouettes against the sky give edge ≫ 1
      // (depth jumps from object to far plane), interior shading
      // gives edge ≪ 0.1 (smooth surface gradients).
      float centre = max(sampleD(vUv), 0.01);
      float edge = g / centre;

      float t = smoothstep(edgeThreshold, edgeThreshold * 1.5, edge) * outlineStrength;
      gl_FragColor = mix(col, vec4(0.0, 0.0, 0.0, col.a), t);
    }
  `,
  };

  goodsellPass = new ShaderPass(GoodsellEdgeShader);
  goodsellPass.enabled = false; // off until style switches to goodsell
  composer.addPass(goodsellPass);
  composer.addPass(new OutputPass());
  goodsellAvailable = true;
} catch (e) {
  console.warn("[viewer] Goodsell post-pass init failed; falling back to "
               + "standard rendering only:", e);
  composer = null;
  goodsellPass = null;
  // Stash the error so the sidebar can surface it (the user can't
  // see the console-warn unless they open dev tools).
  window.__goodsellInitError = e?.message || String(e) || "unknown";
}

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
// Each interaction may move the camera into / out of zoom levels
// where a different LOD is appropriate per type. Debounced to one
// reassess per animation frame.
controls.addEventListener("change", () => {
  // Function may be defined later in the module; guard.
  if (typeof scheduleReassess === "function") scheduleReassess();
});

// Lights — ambient + three directionals. Intensities bumped so
// MeshStandardMaterial (which absorbs a lot at default roughness)
// reads bright on the dark backdrop.
scene.add(new THREE.AmbientLight(0xffffff, 0.85));
const keyLight = new THREE.DirectionalLight(0xfff5e6, 1.6);
keyLight.position.set(200, 300, 150);
scene.add(keyLight);
const fillLight = new THREE.DirectionalLight(0xa6c8ff, 0.7);
fillLight.position.set(-150, -100, 100);
scene.add(fillLight);
const rimLight = new THREE.DirectionalLight(0xffffff, 0.5);
rimLight.position.set(0, -200, -300);
scene.add(rimLight);

// Scene helpers.
const axesHelper = new THREE.AxesHelper(50);
scene.add(axesHelper);
const gridHelper = new THREE.GridHelper(1000, 20, 0x445566, 0x223344);
gridHelper.position.y = 0;
scene.add(gridHelper);

let bboxLines = null;
// Each entry tracks one ingredient type and all of its LOD slots.
// `fallbackSphere` is the placeholder used before any OBJ arrives or
// when no LOD is loaded for an instance's desired level. `lods[i]`
// holds one InstancedMesh-pair per OBJ resolution (coarse → fine),
// populated lazily as the camera zoom triggers a load. Per-instance
// LOD + frustum culling re-partition placements across these meshes
// on every camera change so far/off-screen instances stay coarse
// (or aren't drawn) and only the visible neighborhood gets fine.
let instancedMeshes = [];
let dataBounds = null; // { min: [x,y,z], max: [x,y,z] }
let clippingPlane = null;
let autoSpin = false;
// Per-instance LOD bookkeeping (display-only).
let meshLoadingTotal = 0;
let meshLoadingDone = 0;
// Reused per-reassess scratch — keeps the per-placement loop allocation-free.
const _tmpFrustum = new THREE.Frustum();
const _tmpProjView = new THREE.Matrix4();
const _tmpMat = new THREE.Matrix4();
const _tmpQuat = new THREE.Quaternion();
const _tmpPos = new THREE.Vector3();
const _tmpSphere = new THREE.Sphere(new THREE.Vector3(), 0);
const _tmpScaleOne = new THREE.Vector3(1, 1, 1);
const _tmpScaleVec = new THREE.Vector3(1, 1, 1);
// (style + outlinePixels are declared earlier so they're in scope for
// the composer-setup block above.)

// Resize handling. setSize's third arg `updateStyle` defaults to true
// — leave it that way so the canvas's CSS dimensions match the
// drawing buffer (otherwise the canvas stays at the HTML default
// 300×150 and the rendered scene shows up clipped to the top-left).
function resize() {
  const r = canvasWrap.getBoundingClientRect();
  if (r.width === 0 || r.height === 0) return;
  camera.aspect = r.width / r.height;
  camera.updateProjectionMatrix();
  renderer.setSize(r.width, r.height);
  if (composer) {
    composer.setSize(r.width, r.height);
  }
  if (goodsellPass) {
    goodsellPass.uniforms.resolution.value.set(r.width, r.height);
    goodsellPass.uniforms.cameraNear.value = camera.near;
    goodsellPass.uniforms.cameraFar.value = camera.far;
  }
}
// ResizeObserver picks up grid-layout settling, splitter drags, and
// fullscreen toggles without relying on the window-resize event.
const resizeObserver = new ResizeObserver(resize);
resizeObserver.observe(canvasWrap);
window.addEventListener("resize", resize);
resize();

// ───── data loading ─────────────────────────────────────────────────

async function loadSimulariumFile(file) {
  const text = await file.text();
  const doc = JSON.parse(text);
  buildScene(doc, file.name);
}

function disposeMeshPair(pair) {
  if (!pair) return;
  for (const m of [pair.standardMesh, pair.celMesh]) {
    if (!m) continue;
    scene.remove(m);
    m.geometry.dispose();
    m.material.dispose();
  }
}

function clearPacking() {
  for (const e of instancedMeshes) {
    disposeMeshPair(e.fallbackSphere);
    if (e.lods) {
      for (const lvl of e.lods) {
        disposeMeshPair(lvl);
      }
    }
  }
  instancedMeshes = [];
  if (bboxLines) {
    scene.remove(bboxLines);
    bboxLines.geometry.dispose();
    bboxLines.material.dispose();
    bboxLines = null;
  }
}

// ───── Goodsell shaders ─────────────────────────────────────────────
// Cel-shading + inverted-hull outlines. Cel: quantize NdotL into a
// few bands per type, multiplied by the per-type colour. Outline:
// render BackSide of slightly-scaled geometry as solid black so it
// peeks out around the front-face silhouette.

const CEL_VERTEX_SHADER = `
varying vec3 vNormalW;
varying vec3 vWorldPos;
void main() {
  vec4 worldPos = modelMatrix * instanceMatrix * vec4(position, 1.0);
  vNormalW = normalize(mat3(modelMatrix) * mat3(instanceMatrix) * normal);
  vWorldPos = worldPos.xyz;
  gl_Position = projectionMatrix * viewMatrix * worldPos;
}
`;

const CEL_FRAGMENT_SHADER = `
varying vec3 vNormalW;
varying vec3 vWorldPos;
uniform vec3 uColor;
uniform vec3 uLightDir;
void main() {
  vec3 N = normalize(vNormalW);
  vec3 V = normalize(cameraPosition - vWorldPos);
  float NdotL = dot(N, normalize(uLightDir));
  // Wide-smoothstep gradient — Goodsell's illustrations are
  // painterly, not flat-cel, so this gradient matches the source
  // aesthetic better than a stark step function while also hiding
  // the triangle facets of a coarse sphere.
  float band = mix(0.45, 1.0, smoothstep(-0.4, 0.6, NdotL));
  // View-aligned silhouette: when the surface normal is nearly
  // perpendicular to the view direction (NdotV ≈ 0), we're at a
  // silhouette edge — fade toward black. This produces "ink line"
  // outlines without any depth-buffer post-pass, so it works at any
  // scale and isn't subject to the Sobel false-positive blowups the
  // depth-Sobel approach hit on dense scenes.
  float NdotV = abs(dot(N, V));
  float silhouette = smoothstep(0.0, 0.25, NdotV);
  vec3 baseColor = uColor * band;
  gl_FragColor = vec4(baseColor * silhouette, 1.0);
}
`;

function makeCelMaterial(color) {
  return new THREE.ShaderMaterial({
    uniforms: {
      uColor: { value: color.clone() },
      uLightDir: { value: new THREE.Vector3(0.5, 1.0, 0.8).normalize() },
    },
    vertexShader: CEL_VERTEX_SHADER,
    fragmentShader: CEL_FRAGMENT_SHADER,
    side: THREE.FrontSide,
    depthTest: true,
    depthWrite: true,
    transparent: false,
  });
}

// OBJLoader is shared across the session. Returns the first
// BufferGeometry inside the parsed Group; ingredients OBJs we emit
// are single-mesh so the first child is always the mesh.
//
// OBJ parsing happens on the main thread — a 200k-vertex fine LOD
// can block the event loop for a second or two. When the user zooms
// into a dense scene, dozens of fine LODs can be wanted at once; if
// we let them all parse in parallel, the tab freezes for tens of
// seconds. We throttle to one parse-in-flight at a time. There is
// no materialized FIFO queue — `drainObjLoadQueue` scans
// `entry.lods[*].wantLoad` / `wantPriority` (rewritten every
// reassessLODs pass) and picks the highest-priority wanted level
// each time. A level the user navigated away from simply isn't in
// `wantLoad` next frame, so it drops out of consideration without
// us having to track or cancel stale queue entries.
const objLoader = new OBJLoader();
let objLoadInFlight = false;
let _drainScheduled = false;

// ───── IndexedDB-backed mesh cache ──────────────────────────────────
// Stores already-parsed BufferGeometries (as raw typed-array dumps)
// across page reloads, so a regular refresh doesn't re-fetch + re-
// parse hundreds of OBJs. The browser HTTP cache handles bytes, but
// OBJ→BufferGeometry parsing is the actually expensive step here.
//
// Keyed by URL. A hard refresh that bypasses HTTP cache doesn't
// clear IndexedDB on its own — clear it via DevTools → Application
// → IndexedDB → "parsimony-viewer" if you regenerate meshes and
// want to bust the cache.
// Bump the suffix any time the shape of cached geometry data (or
// any pre-processing that mutates it before storage) changes. v3
// stored aggressively-smoothed-and-collapsed geometry from a pure
// Laplacian pass; v4 stores volume-preserving Taubin-smoothed
// geometry instead, so refresh re-fetches and re-smooths cleanly.
const IDB_NAME = "parsimony-viewer-v7";
const IDB_STORE = "geoms";
let _idbPromise = null;

function openIdb() {
  if (_idbPromise) return _idbPromise;
  _idbPromise = new Promise((resolve, reject) => {
    if (typeof indexedDB === "undefined") {
      reject(new Error("indexedDB unavailable"));
      return;
    }
    const req = indexedDB.open(IDB_NAME, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(IDB_STORE)) {
        db.createObjectStore(IDB_STORE);
      }
    };
    req.onerror = () => reject(req.error);
    req.onsuccess = () => resolve(req.result);
  }).catch((err) => {
    console.warn("[viewer] IndexedDB unavailable; mesh cache disabled:", err);
    return null;
  });
  return _idbPromise;
}

// Diagnostics — surface IDB activity so the user can verify caching
// is actually happening on subsequent refreshes. Counters reset on
// each buildScene; updateMeshLoadingStatus reads them so the sidebar
// shows e.g. "423/640 cached".
let cacheHits = 0;
let cacheMisses = 0;
let cacheWriteErrors = 0;

async function idbGet(key) {
  const db = await openIdb();
  if (!db) return null;
  return new Promise((resolve) => {
    let tx;
    try {
      tx = db.transaction(IDB_STORE, "readonly");
    } catch (err) {
      console.warn("[viewer] IDB readonly transaction failed:", err);
      resolve(null);
      return;
    }
    const req = tx.objectStore(IDB_STORE).get(key);
    req.onerror = () => {
      console.warn(`[viewer] IDB get error for ${key}:`, req.error);
      resolve(null);
    };
    req.onsuccess = () => resolve(req.result || null);
  });
}

async function idbPut(key, value) {
  const db = await openIdb();
  if (!db) return;
  await new Promise((resolve) => {
    let tx;
    try {
      tx = db.transaction(IDB_STORE, "readwrite");
    } catch (err) {
      cacheWriteErrors++;
      console.warn("[viewer] IDB readwrite transaction failed:", err);
      resolve();
      return;
    }
    const req = tx.objectStore(IDB_STORE).put(value, key);
    req.onerror = () => {
      cacheWriteErrors++;
      console.warn(`[viewer] IDB put error for ${key}:`, req.error);
      resolve();
    };
    req.onsuccess = () => resolve();
  });
}

function geometryToCacheEntry(geom) {
  const entry = {
    positions: geom.attributes.position.array,
  };
  if (geom.attributes.normal) entry.normals = geom.attributes.normal.array;
  if (geom.index) entry.indices = geom.index.array;
  return entry;
}

function cacheEntryToGeometry(entry) {
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(entry.positions, 3));
  if (entry.normals) {
    geom.setAttribute("normal", new THREE.BufferAttribute(entry.normals, 3));
  }
  if (entry.indices) {
    geom.setIndex(new THREE.BufferAttribute(entry.indices, 1));
  }
  return geom;
}

// Taubin λ-μ smoothing — alternates a Laplacian-style inward step
// (λ > 0) with an outward step (μ < 0, |μ| slightly larger than λ).
// The net effect rounds polyhedral silhouettes without the volume
// collapse that pure Laplacian iterates produce: each inward step
// shrinks the mesh slightly toward the centroid of its neighbours,
// the following outward step pushes it back. Critical for low-
// vertex coarse LODs — pure Laplacian with 5 passes at λ=0.4 on a
// 68-vertex mesh shrinks it to ~8% of original size, which our
// geomScale clamp can't fully recover (it would have to scale by
// ~22× to match enclosing_radius). Taubin's λμ pair keeps the
// vertex count's volume essentially fixed while still smoothing
// the high-frequency surface ripples.
//
// Classic values: λ=0.33, μ=-0.34. We use a slightly stronger pair
// for visible rounding in 3 iterations.
function smoothGeometryInPlace(geom, iterations = 3, lambda = 0.5, mu = -0.53) {
  if (!geom.index) return;
  const indices = geom.index.array;
  const positions = geom.attributes.position.array;
  const nv = geom.attributes.position.count;
  // Build adjacency (per-vertex set of neighbour indices).
  const neighbours = new Array(nv);
  for (let i = 0; i < nv; i++) neighbours[i] = new Set();
  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i], b = indices[i + 1], c = indices[i + 2];
    neighbours[a].add(b); neighbours[a].add(c);
    neighbours[b].add(a); neighbours[b].add(c);
    neighbours[c].add(a); neighbours[c].add(b);
  }
  const next = new Float32Array(positions.length);
  function step(factor) {
    for (let i = 0; i < nv; i++) {
      const nbrs = neighbours[i];
      let cx = 0, cy = 0, cz = 0, n = 0;
      for (const j of nbrs) {
        cx += positions[j * 3];
        cy += positions[j * 3 + 1];
        cz += positions[j * 3 + 2];
        n++;
      }
      const ix = i * 3;
      if (n > 0) {
        cx /= n; cy /= n; cz /= n;
        next[ix]     = positions[ix]     + factor * (cx - positions[ix]);
        next[ix + 1] = positions[ix + 1] + factor * (cy - positions[ix + 1]);
        next[ix + 2] = positions[ix + 2] + factor * (cz - positions[ix + 2]);
      } else {
        next[ix] = positions[ix];
        next[ix + 1] = positions[ix + 1];
        next[ix + 2] = positions[ix + 2];
      }
    }
    positions.set(next);
  }
  for (let iter = 0; iter < iterations; iter++) {
    step(lambda); // inward (Laplacian)
    step(mu);     // outward (anti-Laplacian)
  }
  geom.attributes.position.needsUpdate = true;
  geom.computeVertexNormals();
}

// Robust bounding-sphere radius: the 99th-percentile distance from
// the geometric centroid. Marching-cubes meshes can have outlier
// vertices far from the main body (stray cells where the SDF noise
// crosses zero); the max-distance bounding sphere over-reports the
// mesh's apparent size in those cases. The 99% cut keeps the size
// estimate aligned with what the eye perceives as the molecule's
// extent.
function robustBoundingRadius(geom) {
  const pos = geom.attributes.position.array;
  const n = geom.attributes.position.count;
  // Centroid.
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < n; i++) {
    cx += pos[i * 3]; cy += pos[i * 3 + 1]; cz += pos[i * 3 + 2];
  }
  cx /= n; cy /= n; cz /= n;
  // Squared distances → sort → percentile.
  const d2 = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const dx = pos[i * 3] - cx;
    const dy = pos[i * 3 + 1] - cy;
    const dz = pos[i * 3 + 2] - cz;
    d2[i] = dx * dx + dy * dy + dz * dz;
  }
  d2.sort();
  const cut = Math.min(n - 1, Math.max(0, Math.floor(n * 0.99)));
  return Math.sqrt(d2[cut]);
}

/// Resolve an OBJ URL from the pack to something the browser can
/// fetch. The pack stores project-root-relative paths (e.g.
/// `examples/pdb_meshes/foo.obj`). The viewer's index.html lives at
/// `/viewer/index.html` — relative fetches would look in
/// `/viewer/examples/...`. We prepend `/` so the URL is rooted at
/// whatever the http server is serving (which is the project root
/// when launched via `view_pack.sh`). Absolute URLs and protocol
/// URLs pass through unchanged.
function resolveMeshUrl(url) {
  if (!url) return url;
  if (url.startsWith("http://") || url.startsWith("https://") || url.startsWith("/")) {
    return url;
  }
  return "/" + url;
}

// Fetch + parse one OBJ, with IDB cache around it. Pure function:
// no priority logic, no scheduling — just "give me the geometry for
// this URL". The drain decides WHICH URL to pull next.
async function fetchAndParseObj(url) {
  const resolved = resolveMeshUrl(url);
  const cached = await idbGet(resolved);
  if (cached && cached.positions && cached.positions.length > 0) {
    cacheHits++;
    return cacheEntryToGeometry(cached);
  }
  cacheMisses++;
  const rawGeom = await new Promise((resolve, reject) => {
    objLoader.load(
      resolved,
      (group) => {
        let g = null;
        group.traverse((child) => {
          if (g == null && child.isMesh) g = child.geometry;
        });
        if (!g) {
          reject(new Error(`OBJ ${resolved} contained no mesh`));
          return;
        }
        resolve(g);
      },
      undefined,
      (err) => reject(err),
    );
  });
  // OBJLoader emits non-indexed geometry (every triangle has its own
  // copy of each vertex). mergeVertices deduplicates positions and
  // builds an index buffer — both the Laplacian smoother and the
  // IDB cache benefit: smoothing needs an index to walk neighbours,
  // and indexed geometry is ~3× smaller to store.
  const geom = mergeVertices(rawGeom, 1e-4);
  rawGeom.dispose();
  geom.computeVertexNormals();
  // (No runtime mesh smoothing in this build — the SDF-level
  // Gaussian smoothing in the python regen does the rounding work
  // already, and runtime smoothing of low-vertex coarse LODs was
  // pushing their robust radius under the degenerate threshold and
  // forcing the picker into sphere fallback. The smoother is kept
  // above as a callable utility if we want to enable it later.)
  idbPut(resolved, geometryToCacheEntry(geom)).catch((err) => {
    cacheWriteErrors++;
    console.warn(`[viewer] cache write failed for ${resolved}:`, err);
  });
  return geom;
}

// Schedule a drain on the microtask queue so multiple reassesses in
// the same animation frame coalesce into a single pick.
function scheduleDrain() {
  if (_drainScheduled) return;
  _drainScheduled = true;
  Promise.resolve().then(() => {
    _drainScheduled = false;
    drainObjLoadQueue();
  });
}

// Priority-based drain: at the moment one parse slot frees up, scan
// every entry's lods[] for the highest-priority wanted level (lowest
// `wantPriority` distance) and load that next. There is no
// materialized queue — `wantLoad` + `wantPriority` are written fresh
// each reassessLODs pass, so a level the user navigated away from
// quietly drops out of consideration on the next frame without us
// having to track or cancel a stale queue entry. Truly in-flight
// loads still run to completion (we can't abort OBJLoader mid-parse),
// but they're a single ~one-second blip — the next pick comes from
// the *current* view.
async function drainObjLoadQueue() {
  if (objLoadInFlight) return;
  let best = null;
  let bestPriority = Infinity;
  for (const entry of instancedMeshes) {
    if (!entry.lods) continue;
    for (let i = 0; i < entry.lods.length; i++) {
      const lvl = entry.lods[i];
      if (lvl.loaded || lvl.loading || !lvl.wantLoad) continue;
      if (lvl.wantPriority < bestPriority) {
        bestPriority = lvl.wantPriority;
        best = { entry, levelIdx: i, lvl };
      }
    }
  }
  if (!best) return;

  best.lvl.loading = true;
  objLoadInFlight = true;
  try {
    const geom = await fetchAndParseObj(best.lvl.url);
    if (!best.lvl.loaded) {
      ensureLodMesh(best.entry, best.levelIdx, geom);
      updateMeshLoadingStatus();
      scheduleReassess();
    }
  } catch (err) {
    console.warn(`LOD ${best.levelIdx} load failed for ${best.entry.name}:`, err);
  } finally {
    best.lvl.loading = false;
    objLoadInFlight = false;
    // setTimeout(0) yields a macrotask between parses so the just-
    // installed InstancedMesh and any pending UI work get a turn
    // before the next 1-2 second OBJ parse blocks the main thread.
    setTimeout(drainObjLoadQueue, 0);
  }
}

// Build the scene from a `parsimony.pack.v1` document.
async function buildScene(doc, fileName) {
  clearPacking();
  cacheHits = 0;
  cacheMisses = 0;
  cacheWriteErrors = 0;

  if (doc.format !== "parsimony.pack.v1") {
    throw new Error(
      `unsupported format '${doc.format}' (expected parsimony.pack.v1)`,
    );
  }

  const ingredients = doc.ingredients || [];
  const placements = doc.placements || [];

  // Index ingredients by id for quick placement lookup. The id is
  // not necessarily the array position (recipes can have gaps), so
  // build a real map.
  const ingredientById = new Map();
  for (const ing of ingredients) {
    ingredientById.set(ing.id, ing);
  }

  // Group placements by ingredient id.
  const byType = new Map();
  let dataMin = [Infinity, Infinity, Infinity];
  let dataMax = [-Infinity, -Infinity, -Infinity];
  for (const p of placements) {
    if (!byType.has(p.ingredient)) byType.set(p.ingredient, []);
    byType.get(p.ingredient).push(p);
    const ing = ingredientById.get(p.ingredient);
    const r = ing ? (ing.shape.enclosing_radius || ing.shape.radius || 1.0) : 1.0;
    const [x, y, z] = p.position;
    if (x - r < dataMin[0]) dataMin[0] = x - r;
    if (y - r < dataMin[1]) dataMin[1] = y - r;
    if (z - r < dataMin[2]) dataMin[2] = z - r;
    if (x + r > dataMax[0]) dataMax[0] = x + r;
    if (y + r > dataMax[1]) dataMax[1] = y + r;
    if (z + r > dataMax[2]) dataMax[2] = z + r;
  }
  dataBounds = { min: dataMin, max: dataMax };

  // Build one entry per ingredient — sphere fallback + LOD slots.
  // The sphere is what gets drawn initially (and as the bucket for
  // placements whose desired LOD hasn't loaded yet); `reassessLODs`
  // partitions placements across the fallback and any loaded LOD
  // meshes on every camera change.
  for (const [tid, pts] of byType.entries()) {
    const ing = ingredientById.get(tid);
    if (!ing) continue;
    const colorArr = ing.color || [0.5, 0.5, 0.5];
    const color = new THREE.Color(colorArr[0], colorArr[1], colorArr[2]);
    const enc = ing.shape.enclosing_radius || ing.shape.radius || 1.0;
    addInstancedType(ing, color, enc, pts);
  }
  applyStyle();

  // Schedule the first LOD assessment now (loads coarse mesh per
  // type) and re-evaluate on each camera move.
  scheduleReassess();

  // Bounding box wireframe from the pack's bounds field; fall back
  // to data bounds when missing.
  let bbMin, bbMax;
  if (doc.bounds) {
    bbMin = doc.bounds.min;
    bbMax = doc.bounds.max;
  } else {
    bbMin = dataMin;
    bbMax = dataMax;
  }
  bboxLines = makeBoxLines(bbMin, bbMax, 0x556677);
  scene.add(bboxLines);

  framePacking(bbMin, bbMax);
  renderLegend();
  updateStatus(fileName, placements.length, instancedMeshes.length);
  applyVisibility();
  applyClippingPlane();
  updateHelpersFromBounds(bbMin, bbMax);
}

// Build the sphere fallback mesh-pair (standard + cel) used as a
// placeholder before any OBJ has loaded and as the bucket for
// placements whose desired LOD level isn't loaded yet. Allocated
// to `placementCount` slots; the actual draw count is set
// dynamically by `reassessLODs`.
// Build the fallback geometry for an ingredient. For sphere /
// single-sphere ingredients it's just a sphere at enclosing_radius.
// For multi_sphere shapes (which is how the Rust pack writer
// represents cubes, cylinders, multi-cylinders, and dumbbell-style
// sphere trees) we union all the proxy spheres into a single
// geometry so the rendered shape actually resembles the source
// primitive — a chain of spheres reads as a cylinder, eight corner
// spheres read as a cube, etc. For mesh ingredients it's still a
// single sphere; the LOD pipeline takes over once an OBJ loads.
function buildFallbackGeometry(ing, enclosingRadius) {
  if (ing.shape.kind === "multi_sphere"
      && Array.isArray(ing.shape.spheres) && ing.shape.spheres.length > 0) {
    const geoms = ing.shape.spheres.map((s) => {
      const r = s.radius || 0.5;
      const g = new THREE.SphereGeometry(r, 16, 8);
      g.translate(s.offset[0], s.offset[1], s.offset[2]);
      return g;
    });
    const merged = mergeGeometries(geoms, false);
    geoms.forEach((g) => g.dispose());
    if (merged) return merged;
  }
  return new THREE.SphereGeometry(enclosingRadius, 24, 12);
}

function makeFallbackSphere(ing, color, enclosingRadius, placementCount) {
  const sphereGeom = buildFallbackGeometry(ing, enclosingRadius);
  const standardMat = new THREE.MeshStandardMaterial({
    color: color,
    roughness: 0.55,
    metalness: 0.05,
    depthTest: true,
    depthWrite: true,
    transparent: false,
  });
  const standardMesh = new THREE.InstancedMesh(sphereGeom, standardMat, placementCount);
  const celMesh = new THREE.InstancedMesh(
    sphereGeom.clone(),
    makeCelMaterial(color),
    placementCount,
  );
  for (const mesh of [standardMesh, celMesh]) {
    mesh.frustumCulled = false; // we cull per-instance ourselves
    mesh.count = 0;             // filled by reassessLODs
    scene.add(mesh);
  }
  return { standardMesh, celMesh };
}

// Add a new ingredient entry to `instancedMeshes`. The entry owns a
// sphere fallback (always present) and one LOD slot per declared
// OBJ resolution. LOD slots start without an InstancedMesh; they're
// materialised by `ensureLodMesh` once the OBJ arrives.
function addInstancedType(ing, color, enclosingRadius, pts) {
  const fallbackSphere = makeFallbackSphere(ing, color, enclosingRadius, pts.length);
  const lods = (ing.shape.kind === "mesh" && Array.isArray(ing.shape.lods))
    ? ing.shape.lods.map((l) => ({
        url: l.url,
        voxelSize: l.voxel_size,
        standardMesh: null,
        celMesh: null,
        geom: null,
        loaded: false,
        loading: false,
        wantLoad: false,
        wantPriority: Infinity,
        geomScale: 1.0,
        degenerate: false,
      }))
    : null;
  instancedMeshes.push({
    typeId: ing.id,
    name: ing.name,
    color,
    enclosingRadius,
    placements: pts,
    visible: true,
    fallbackSphere,
    lods,
  });
}

// Materialise (or replace) the InstancedMesh-pair for one LOD level
// once its OBJ has finished loading. Count is left at 0; the next
// `reassessLODs` call fills it. We dispose any prior geometry so
// repeated calls (e.g. on reload) don't leak GPU buffers.
function ensureLodMesh(entry, levelIdx, geom) {
  const lvl = entry.lods[levelIdx];

  // Per-LOD bounding analysis: the robust radius (99th-percentile
  // distance from centroid) tells us this mesh's apparent extent
  // ignoring outliers. If it's much smaller than enclosing_radius
  // — which the Rust packer derives from the voxelised proxies of
  // the finest LOD — the marching-cubes pass at this voxel size
  // couldn't capture the molecule shape (e.g. elongated tRNA at
  // 16 Å is a tiny stub). We flag it `degenerate` and the LOD
  // picker skips over it, falling through to a finer level or the
  // sphere fallback. Otherwise we record the scale that will make
  // the rendered mesh match the canonical enclosing_radius, so
  // LOD ↔ LOD transitions don't pop size.
  // Size-invariance contract: every LOD that we *do* render must
  // present a bounding radius equal to enclosing_radius. Same for
  // the sphere fallback. That way every LOD ↔ LOD swap (and every
  // sphere ↔ LOD swap) produces zero perceived size change — only
  // detail level changes. Implementation: exact scale = target /
  // robust, no clamp. If the natural radius is so far off the
  // target that the scale would distort the mesh's shape (e.g. a
  // 14-radius coarse stub of a 49-radius tRNA would need scale
  // 3.5×, stretching the blob beyond recognition), we mark the
  // LOD degenerate and the picker walks past it instead of
  // rendering a distorted version.
  const robustR = robustBoundingRadius(geom);
  const targetR = entry.enclosingRadius;
  if (robustR > 1e-6) {
    const ratio = robustR / targetR;
    lvl.geomScale = targetR / robustR;
    // Degenerate band: natural radius outside [0.4, 2.5] × target
    // means the LOD's shape is too different from the canonical
    // bounding envelope to scale into without visible distortion.
    lvl.degenerate = ratio < 0.4 || ratio > 2.5;
  } else {
    lvl.geomScale = 1.0;
    lvl.degenerate = true;
  }

  if (lvl.standardMesh) {
    // Already created; replace geometry in place.
    lvl.standardMesh.geometry.dispose();
    lvl.celMesh.geometry.dispose();
    lvl.standardMesh.geometry = geom;
    lvl.celMesh.geometry = geom.clone();
  } else {
    const standardMat = new THREE.MeshStandardMaterial({
      color: entry.color,
      roughness: 0.55,
      metalness: 0.05,
    });
    const standard = new THREE.InstancedMesh(geom, standardMat, entry.placements.length);
    const cel = new THREE.InstancedMesh(
      geom.clone(),
      makeCelMaterial(entry.color),
      entry.placements.length,
    );
    for (const mesh of [standard, cel]) {
      mesh.frustumCulled = false;
      mesh.count = 0;
      if (clippingPlane) {
        mesh.material.clippingPlanes = [clippingPlane];
        mesh.material.needsUpdate = true;
      }
      scene.add(mesh);
    }
    lvl.standardMesh = standard;
    lvl.celMesh = cel;
  }
  lvl.geom = geom;
  lvl.loaded = true;
  lvl.loading = false;
  // Apply current visibility for the new pair.
  const styleStandard = entry.visible && style === "standard";
  const styleCel = entry.visible && style === "goodsell";
  lvl.standardMesh.visible = styleStandard;
  lvl.celMesh.visible = styleCel;
}

// Per-instance LOD threshold: we pick the coarsest LOD whose voxel
// size projects to ≤ this many screen pixels at the placement's
// depth. Smaller = pickier (more often the fine LOD wins); larger =
// stay coarse longer (cheap, less detail). Tuned for cell-scale
// recipes where flying through dense crowds is the worst case. The
// sidebar "LOD px" slider writes here at runtime.
let lodVoxelPixelTarget = 4.0;

// Below this projected radius (in pixels) we don't bother loading
// or drawing an OBJ — the 24-segment sphere fallback at the
// ingredient's enclosing_radius is indistinguishable from the
// high-poly mesh at that scale. 3 px is roughly the threshold
// where the smoothed OBJ silhouette becomes visually distinct from
// a sphere; small ingredients switch to OBJ as soon as the user
// gets close enough to perceive the shape difference.
const lodSphereBudgetPx = 3.0;

let reassessQueued = false;
function scheduleReassess() {
  if (reassessQueued) return;
  reassessQueued = true;
  // Debounce to next animation frame so a flurry of OrbitControls
  // changes (e.g. an active drag) collapses into one pass.
  requestAnimationFrame(() => {
    reassessQueued = false;
    reassessLODs();
  });
}

// Build the current camera frustum into the shared scratch object.
// Must happen after `camera.updateProjectionMatrix` and
// `controls.update` have run for this frame.
function refreshFrustum() {
  _tmpProjView.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  _tmpFrustum.setFromProjectionMatrix(_tmpProjView);
}

// Walk every placement of every ingredient and assign it to one of:
//   • a loaded LOD InstancedMesh (matched per-instance based on
//     projected voxel size + camera distance), or
//   • the sphere fallback (if the desired LOD isn't loaded yet, or
//     no LODs are declared), or
//   • nothing (out of camera frustum).
// Sets `count` and matrices on each renderable. Schedules demand-
// loads for LODs that any in-frustum placement wants but doesn't
// have. This is the only place per-instance matrices are written
// after the initial scene build, so the work scales linearly with
// total placement count (≈ 27k for mycoplasma_full).
function reassessLODs() {
  refreshFrustum();
  const camPos = camera.position;
  const vh = renderer.domElement.clientHeight || 1;
  const fovHalfTan = Math.tan((camera.fov * Math.PI / 180) / 2);
  const camX = camPos.x, camY = camPos.y, camZ = camPos.z;

  meshLoadingTotal = 0;
  meshLoadingDone = 0;

  for (const entry of instancedMeshes) {
    if (entry.lods) {
      meshLoadingTotal += entry.lods.length;
      for (const lvl of entry.lods) if (lvl.loaded) meshLoadingDone++;
    }

    const lods = entry.lods;
    const hasLods = !!lods && lods.length > 0;
    const sphereR = entry.enclosingRadius;

    // Reset bucket counts. lodCounts[i] is the per-level draw count
    // we'll fill in this pass; sphereCount is the fallback bucket.
    // wantLoad + wantPriority are rebuilt fresh each reassess — that
    // is what makes the priority drain naturally drop stale wants
    // when the camera moves on.
    let sphereCount = 0;
    const lodCounts = hasLods ? new Array(lods.length).fill(0) : null;
    if (hasLods) {
      for (const lvl of lods) {
        lvl.wantLoad = false;
        lvl.wantPriority = Infinity;
      }
    }

    for (let pi = 0; pi < entry.placements.length; pi++) {
      const p = entry.placements[pi];
      const px = p.position[0], py = p.position[1], pz = p.position[2];
      _tmpSphere.center.set(px, py, pz);
      _tmpSphere.radius = sphereR;
      if (!_tmpFrustum.intersectsSphere(_tmpSphere)) continue;

      const dx = px - camX, dy = py - camY, dz = pz - camZ;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const r = p.rotation || [1, 0, 0, 0];
      _tmpQuat.set(r[1], r[2], r[3], r[0]); // pack v1 stores [w,x,y,z]
      _tmpPos.set(px, py, pz);

      // Project ingredient enclosing radius onto the screen. Used
      // both for the sphere-budget shortcut (below) and as the
      // baseline scale for per-voxel LOD picking.
      const scale = vh / (2 * Math.max(dist, 1.0) * fovHalfTan);
      const projectedRadiusPx = entry.enclosingRadius * scale;

      // Sphere routing: applies to sphere/multi-sphere ingredients
      // unconditionally, and to mesh ingredients whose projected
      // size is below the sphere budget. Loading + drawing a 17k-
      // triangle OBJ for something that's 4 pixels across is wasted
      // GPU bandwidth — the sphere proxy is visually identical at
      // that scale.
      if (!hasLods || projectedRadiusPx < lodSphereBudgetPx) {
        _tmpMat.compose(_tmpPos, _tmpQuat, _tmpScaleOne);
        entry.fallbackSphere.standardMesh.setMatrixAt(sphereCount, _tmpMat);
        entry.fallbackSphere.celMesh.setMatrixAt(sphereCount, _tmpMat);
        sphereCount++;
        continue;
      }

      // Mesh ingredient at a useful projected size: pick the coarsest
      // OBJ LOD whose voxel size projects to ≤ target pixels, skipping
      // any LOD that loaded as `degenerate` (the voxel size was too
      // coarse to capture the molecule's shape at all).
      let desired = -1;
      for (let i = 0; i < lods.length; i++) {
        if (lods[i].degenerate) continue;
        if (lods[i].voxelSize * scale <= lodVoxelPixelTarget) {
          desired = i;
          break;
        }
      }
      // Nothing meets the pixel target — pick the finest non-degenerate
      // level we know about. If they're all degenerate or none have
      // been loaded yet, desired stays -1 and we'll either route to
      // the sphere fallback below or, for unloaded levels, kick off a
      // load via wantLoad so the picker can re-evaluate next pass.
      if (desired === -1) {
        for (let i = lods.length - 1; i >= 0; i--) {
          if (!lods[i].degenerate) { desired = i; break; }
        }
      }

      // Walk down to a loaded non-degenerate level for actual render.
      let actual = desired;
      while (actual >= 0 && (!lods[actual].loaded || lods[actual].degenerate)) {
        actual--;
      }

      // Mark desired as wanted (priority bid = distance). Drain picks
      // the level with the minimum bid across the whole scene.
      if (desired >= 0 && !lods[desired].loaded && !lods[desired].loading) {
        lods[desired].wantLoad = true;
        if (dist < lods[desired].wantPriority) {
          lods[desired].wantPriority = dist;
        }
      }

      if (actual < 0) {
        // No loaded non-degenerate LOD yet — sphere fallback.
        _tmpMat.compose(_tmpPos, _tmpQuat, _tmpScaleOne);
        entry.fallbackSphere.standardMesh.setMatrixAt(sphereCount, _tmpMat);
        entry.fallbackSphere.celMesh.setMatrixAt(sphereCount, _tmpMat);
        sphereCount++;
      } else {
        const lvl = lods[actual];
        // geomScale brings every LOD's bounding extent to the canonical
        // enclosing_radius so LOD-to-LOD pop-ins don't change size.
        _tmpScaleVec.setScalar(lvl.geomScale);
        _tmpMat.compose(_tmpPos, _tmpQuat, _tmpScaleVec);
        lvl.standardMesh.setMatrixAt(lodCounts[actual], _tmpMat);
        lvl.celMesh.setMatrixAt(lodCounts[actual], _tmpMat);
        lodCounts[actual]++;
      }
    }

    // Commit counts. count=0 makes the renderer skip the draw call
    // entirely, so empty buckets cost nothing.
    entry.fallbackSphere.standardMesh.count = sphereCount;
    entry.fallbackSphere.celMesh.count = sphereCount;
    entry.fallbackSphere.standardMesh.instanceMatrix.needsUpdate = true;
    entry.fallbackSphere.celMesh.instanceMatrix.needsUpdate = true;
    if (hasLods) {
      for (let i = 0; i < lods.length; i++) {
        if (lods[i].standardMesh) {
          lods[i].standardMesh.count = lodCounts[i];
          lods[i].celMesh.count = lodCounts[i];
          lods[i].standardMesh.instanceMatrix.needsUpdate = true;
          lods[i].celMesh.instanceMatrix.needsUpdate = true;
        }
      }
    }

  }

  // Hand off to the priority drain. It scans all entries' wantLoad
  // levels and picks the one whose closest in-frustum placement is
  // nearest the camera, then starts loading it (subject to the
  // one-parse-at-a-time throttle). Levels no longer in wantLoad
  // simply aren't picked — that's our cancellation story.
  scheduleDrain();
  updateMeshLoadingStatus();
}

function parseHexColor(hex) {
  return new THREE.Color(hex);
}

function makeBoxLines([x0, y0, z0], [x1, y1, z1], color) {
  const corners = [
    [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
    [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
  ];
  const edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
  ];
  const pts = new Float32Array(edges.length * 6);
  for (let i = 0; i < edges.length; i++) {
    const [a, b] = edges[i];
    pts.set([...corners[a], ...corners[b]], i * 6);
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(pts, 3));
  const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.45 });
  return new THREE.LineSegments(geom, mat);
}

// The home camera-pose for the current packing. Captured at the end
// of framePacking so the Space-key "ease back" can return here.
let homeCameraPos = new THREE.Vector3();
let homeTarget = new THREE.Vector3();

function framePacking(bbMin, bbMax) {
  const cx = (bbMin[0] + bbMax[0]) * 0.5;
  const cy = (bbMin[1] + bbMax[1]) * 0.5;
  const cz = (bbMin[2] + bbMax[2]) * 0.5;
  const extent = Math.max(bbMax[0] - bbMin[0], bbMax[1] - bbMin[1], bbMax[2] - bbMin[2]);
  controls.target.set(cx, cy, cz);
  const dist = extent * 1.6;
  camera.position.set(cx + dist * 0.6, cy + dist * 0.4, cz + dist * 0.7);
  // Tight near/far gives the depth buffer reasonable precision; the
  // previous `extent*50` far plane crushed the scene into the last
  // 0.1% of the depth range, which broke the Goodsell post-pass
  // (depth-Sobel gradients ended up huge on every smooth surface
  // because perspective compression amplified tiny NDC changes into
  // huge linearised-depth jumps). 5× the extent is enough headroom
  // to zoom out comfortably.
  camera.near = Math.max(0.01, extent * 0.01);
  camera.far = extent * 5;
  camera.updateProjectionMatrix();
  controls.update();
  // The Goodsell post-pass linearizes the depth buffer using
  // cameraNear / cameraFar uniforms. Those are picked up at resize
  // time, but framePacking moves the planes to fit the scene — the
  // shader has to follow or every depth sample uses the construction-
  // time defaults (0.5 / 100000) and the Sobel gradient explodes,
  // making the whole frame full black.
  if (goodsellPass) {
    goodsellPass.uniforms.cameraNear.value = camera.near;
    goodsellPass.uniforms.cameraFar.value = camera.far;
  }
  homeCameraPos.copy(camera.position);
  homeTarget.copy(controls.target);
}

function updateHelpersFromBounds(bbMin, bbMax) {
  // Scale axes/grid based on the world extent so they remain useful.
  const extent = Math.max(bbMax[0] - bbMin[0], bbMax[1] - bbMin[1], bbMax[2] - bbMin[2]);
  scene.remove(axesHelper);
  axesHelper.geometry.dispose();
  const a = new THREE.AxesHelper(extent * 0.6);
  axesHelper.geometry = a.geometry;
  scene.add(axesHelper);

  scene.remove(gridHelper);
  gridHelper.geometry.dispose();
  const gridSize = extent * 2.5;
  const gridDivs = 20;
  const g = new THREE.GridHelper(gridSize, gridDivs, 0x445566, 0x223344);
  g.position.set((bbMin[0] + bbMax[0]) * 0.5, bbMin[1] - extent * 0.02, (bbMin[2] + bbMax[2]) * 0.5);
  gridHelper.geometry = g.geometry;
  gridHelper.position.copy(g.position);
  scene.add(gridHelper);
}

// ───── UI ───────────────────────────────────────────────────────────

function renderLegend() {
  legendEl.innerHTML = "";
  // Sort by descending count for readability.
  const sorted = [...instancedMeshes].sort((a, b) => b.count - a.count);
  for (const entry of sorted) {
    const row = document.createElement("div");
    row.className = "legend-row";
    const swatch = document.createElement("div");
    swatch.className = "swatch";
    swatch.style.background = "#" + entry.color.getHexString();
    const name = document.createElement("div");
    name.className = "name";
    name.textContent = entry.name;
    const count = document.createElement("div");
    count.className = "count";
    count.textContent = entry.count;
    row.append(swatch, name, count);
    row.addEventListener("click", () => {
      entry.visible = !entry.visible;
      row.classList.toggle("off", !entry.visible);
      applyVisibility();
    });
    legendEl.appendChild(row);
  }
}

function updateStatus(fileName, n, types) {
  statusEl.innerHTML = `<div class="num">${fileName}</div><div>${n} placements · ${types} types</div>`;
  placementsStat.innerHTML = `<span class="num" style="color:var(--text)">${n.toLocaleString()}</span> placements`;
  typesStat.innerHTML = `<span class="num" style="color:var(--text)">${types}</span> types`;
}

function updateMeshLoadingStatus() {
  const el = document.getElementById("mesh-loading-status");
  if (!el) return;
  const cacheLine = (cacheHits + cacheMisses + cacheWriteErrors) > 0
    ? ` (cache ${cacheHits} hit · ${cacheMisses} miss${cacheWriteErrors ? ` · ${cacheWriteErrors} write err` : ""})`
    : "";
  if (meshLoadingTotal === 0) {
    el.textContent = "no mesh ingredients";
  } else if (meshLoadingDone < meshLoadingTotal) {
    el.textContent = `loading meshes: ${meshLoadingDone}/${meshLoadingTotal}${cacheLine}`;
  } else {
    el.textContent = `${meshLoadingTotal} meshes loaded${cacheLine}`;
  }
}

function applyVisibility() {
  for (const e of instancedMeshes) {
    const sOn = e.visible && style === "standard";
    const gOn = e.visible && style === "goodsell";
    e.fallbackSphere.standardMesh.visible = sOn;
    e.fallbackSphere.celMesh.visible = gOn;
    if (e.lods) {
      for (const lvl of e.lods) {
        if (!lvl.standardMesh) continue;
        lvl.standardMesh.visible = sOn;
        lvl.celMesh.visible = gOn;
      }
    }
  }
}

function applyStyle() {
  if (style === "goodsell" && goodsellAvailable) {
    // Cream-paper backdrop, faithful to Goodsell's illustrations.
    scene.background = new THREE.Color(0xefe7d0);
    goodsellPass.enabled = true;
  } else {
    scene.background = new THREE.Color(0x0e1116);
    if (goodsellPass) goodsellPass.enabled = false;
  }
  applyVisibility();
}

function applyOutlineWidth(pixels) {
  outlinePixels = pixels;
  if (goodsellPass) goodsellPass.uniforms.outlineThickness.value = pixels;
}

function applyClippingPlane() {
  const axis = sliceAxis.value;
  if (!axis) {
    renderer.clippingPlanes = [];
    clippingPlane = null;
    return;
  }
  const normal = new THREE.Vector3(
    axis === "x" ? 1 : 0,
    axis === "y" ? 1 : 0,
    axis === "z" ? 1 : 0,
  );
  if (sliceFlip.checked) normal.negate();
  const t = parseFloat(slicePos.value);
  if (!dataBounds) return;
  const bbMin = dataBounds.min, bbMax = dataBounds.max;
  const lo = axis === "x" ? bbMin[0] : axis === "y" ? bbMin[1] : bbMin[2];
  const hi = axis === "x" ? bbMax[0] : axis === "y" ? bbMax[1] : bbMax[2];
  const worldPos = lo + (hi - lo) * (t * 0.5 + 0.5);
  slicePosValue.textContent = worldPos.toFixed(1);
  // Plane equation: normal · p + constant = 0; we keep points where
  // dot(normal, p) ≤ -constant. So constant = -dot(normal, planePoint).
  const planePoint = new THREE.Vector3(
    axis === "x" ? worldPos : 0,
    axis === "y" ? worldPos : 0,
    axis === "z" ? worldPos : 0,
  );
  const constant = -normal.dot(planePoint);
  clippingPlane = new THREE.Plane(normal, constant);
  renderer.clippingPlanes = [clippingPlane];
  // Re-apply per-material clipping for every variant (fallback +
  // each loaded LOD pair).
  for (const e of instancedMeshes) {
    const variants = [e.fallbackSphere.standardMesh, e.fallbackSphere.celMesh];
    if (e.lods) {
      for (const lvl of e.lods) {
        if (lvl.standardMesh) variants.push(lvl.standardMesh, lvl.celMesh);
      }
    }
    for (const m of variants) {
      m.material.clippingPlanes = [clippingPlane];
      m.material.needsUpdate = true;
    }
  }
}

// ───── input ────────────────────────────────────────────────────────

fileInput.addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (f) loadSimulariumFile(f);
});

resetBtn.addEventListener("click", () => {
  if (dataBounds) framePacking(dataBounds.min, dataBounds.max);
});

// Sidebar collapse toggle.
const sidebarEl = document.getElementById("sidebar");
const sidebarToggle = document.getElementById("sidebar-toggle");
sidebarToggle.addEventListener("click", () => {
  const collapsed = sidebarEl.classList.toggle("collapsed");
  sidebarToggle.textContent = collapsed ? "show" : "hide";
  // Reposition the button when collapsed: it follows the right edge
  // of the viewport rather than the right edge of the (now off-
  // screen) sidebar.
  sidebarToggle.style.right = collapsed ? "12px" : "22px";
  sidebarToggle.style.top = collapsed ? "12px" : "18px";
  // Trigger a renderer resize since the canvas effective area
  // changes (slight: sidebar is floating, but a resize covers any
  // browser quirk).
  resize();
});

toggleBbox.addEventListener("change", () => {
  if (bboxLines) bboxLines.visible = toggleBbox.checked;
});
toggleAxes.addEventListener("change", () => {
  axesHelper.visible = toggleAxes.checked;
});
toggleGrid.addEventListener("change", () => {
  gridHelper.visible = toggleGrid.checked;
});
toggleSpin.addEventListener("change", () => {
  autoSpin = toggleSpin.checked;
});
sliceAxis.addEventListener("change", applyClippingPlane);
slicePos.addEventListener("input", applyClippingPlane);
sliceFlip.addEventListener("change", applyClippingPlane);

// Style radio buttons + outline width slider.
const outlineRow = document.getElementById("outline-row");
const outlineWidthSlider = document.getElementById("outline-width");
const outlineWidthValue = document.getElementById("outline-width-value");
for (const radio of document.querySelectorAll('input[name="style"]')) {
  if (radio.value === "goodsell" && !goodsellAvailable) {
    radio.disabled = true;
    const err = window.__goodsellInitError || "unknown init failure";
    radio.parentElement.title = `Goodsell post-pass unavailable: ${err}`;
    radio.parentElement.style.opacity = "0.5";
    // Surface the error in the sidebar so we don't have to open
    // dev tools every time.
    const styleHeader = [...document.querySelectorAll("#sidebar h2")]
      .find((h) => h.textContent.trim().toLowerCase() === "style");
    if (styleHeader) {
      const note = document.createElement("div");
      note.style.cssText = "color: #ff8b6c; font-size: 11px; margin-top: 4px; "
        + "padding: 4px 6px; background: rgba(255,100,80,0.08); "
        + "border-radius: 3px; word-break: break-word;";
      note.textContent = `Goodsell unavailable: ${err}`;
      styleHeader.parentElement.insertBefore(note, styleHeader.nextSibling);
    }
  }
  radio.addEventListener("change", () => {
    if (!radio.checked) return;
    style = radio.value;
    outlineRow.style.display = style === "goodsell" ? "flex" : "none";
    applyStyle();
  });
}
outlineWidthSlider.addEventListener("input", () => {
  const w = parseFloat(outlineWidthSlider.value);
  outlineWidthValue.textContent = w.toFixed(2);
  applyOutlineWidth(w);
});

// "LOD px" slider — writes `lodVoxelPixelTarget` and triggers a
// reassess. Smaller = finer LODs picked sooner (more detail, more
// memory). Larger = stay coarse longer (less detail, less memory).
const meshBudgetSlider = document.getElementById("mesh-budget");
const meshBudgetValue = document.getElementById("mesh-budget-value");
if (meshBudgetSlider) {
  meshBudgetSlider.addEventListener("input", () => {
    lodVoxelPixelTarget = parseFloat(meshBudgetSlider.value);
    meshBudgetValue.textContent = lodVoxelPixelTarget.toFixed(1);
    scheduleReassess();
  });
}

// ───── keyboard navigation ─────────────────────────────────────────
// All 6 DOF (3 translation + 3 rotation), velocity-smoothed.
//
// Translation:
//   ←  →    pan left / right in the screen plane
//   ↑  ↓    pan up / down in the screen plane
//   PgUp/Dn translate forward / backward along the camera→target
//           axis (fly-through; both camera and target move so you
//           can pass through the scene rather than getting pinned)
//
// Rotation (quaternion — no gimbal lock; can tumble through poles
// and over the top of the scene freely):
//   Q  E    yaw  (rotate around the camera's local up)
//   W  S    pitch (rotate around the camera's local right)
//   A  D    roll  (rotate around the camera's forward axis)
//
// Space    ease back to the home pose captured by framePacking
//
// All five axes are velocity-smoothed: holding a key ramps an
// internal velocity toward ±1 with an ~120 ms time constant, and
// releasing it ramps back to zero with the same constant. This
// hides per-frame dt jitter (variable browser frame timing makes
// raw "delta × dt" motion feel staccato) and gives the camera a
// small amount of inertia at start/stop, which matches the
// expectation people have for fly-through controls.
// Inputs and textareas opt out so typing in the file dialog /
// sliders isn't intercepted.

const heldKeys = new Set();
const keyboardKeys = new Set([
  "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
  "PageUp", "PageDown",
  "KeyQ", "KeyW", "KeyE", "KeyA", "KeyS", "KeyD",
  "Space",
]);

let tweenActive = false;
let tweenT = 0;
const tweenDuration = 0.6; // seconds
const _tweenStartCamPos = new THREE.Vector3();
const _tweenStartTarget = new THREE.Vector3();

function isTypingTarget(t) {
  if (!t) return false;
  const tag = t.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || t.isContentEditable;
}

window.addEventListener("keydown", (e) => {
  if (isTypingTarget(e.target)) return;
  if (!keyboardKeys.has(e.code)) return;
  e.preventDefault();
  if (e.code === "Space") {
    // One-shot: start the tween. Held-Space doesn't re-trigger.
    if (!tweenActive) {
      tweenActive = true;
      tweenT = 0;
      _tweenStartCamPos.copy(camera.position);
      _tweenStartTarget.copy(controls.target);
    }
  } else {
    heldKeys.add(e.code);
  }
});
window.addEventListener("keyup", (e) => {
  heldKeys.delete(e.code);
});
// Blur clears held keys — otherwise a key released over a different
// window stays "held" and the camera drifts forever.
window.addEventListener("blur", () => heldKeys.clear());

// Pan in the camera's screen plane. dx > 0 = move right, dy > 0 = up.
function panCamera(dx, dy) {
  const right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0);
  const up    = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 1);
  const offset = right.multiplyScalar(dx).add(up.multiplyScalar(dy));
  camera.position.add(offset);
  controls.target.add(offset);
}

// Orbit camera around target via quaternion rotations rather than
// spherical (theta, phi) Euler angles. The Euler form has gimbal
// lock at the polar caps (looking straight up or down) — the
// azimuth axis collapses and you can't rotate through the pole.
// Quaternions parameterise arbitrary 3D rotations without that
// singularity, so the user can tumble over the top of the scene
// and back around without getting stuck. Yaw rotates around the
// camera's *local* up axis; pitch rotates around the camera's
// *local* right. After enough tumbling the camera may roll relative
// to world up — that's intentional, it's the price of fly-through
// freedom, and matches how flight-sim / DCC orbit controls behave.
const _orbitQuat = new THREE.Quaternion();
const _orbitForward = new THREE.Vector3();
const _orbitRight = new THREE.Vector3();
const _orbitUp = new THREE.Vector3();
const _orbitOffset = new THREE.Vector3();

function rotateCamera(yaw, pitch, roll) {
  // FPS-style "rotate in place": camera position stays fixed; we
  // rotate the local frame (forward / right / up) and slide the
  // target along the new forward at the same distance. The user
  // sees the view direction turn without the camera being orbited
  // around a fixed pivot — far more natural for fly-through
  // navigation through a scene.
  _orbitForward.subVectors(controls.target, camera.position);
  const distance = _orbitForward.length();
  if (distance < 1e-6) return;
  _orbitForward.divideScalar(distance); // normalize
  _orbitRight.crossVectors(_orbitForward, camera.up).normalize();
  if (_orbitRight.lengthSq() < 1e-6) {
    _orbitRight.set(1, 0, 0);
    if (Math.abs(_orbitForward.dot(_orbitRight)) > 0.999) {
      _orbitRight.set(0, 0, 1);
    }
  }
  _orbitUp.crossVectors(_orbitRight, _orbitForward).normalize();

  if (Math.abs(yaw) > 1e-7) {
    _orbitQuat.setFromAxisAngle(_orbitUp, yaw);
    _orbitForward.applyQuaternion(_orbitQuat);
    _orbitRight.applyQuaternion(_orbitQuat);
  }
  if (Math.abs(pitch) > 1e-7) {
    _orbitQuat.setFromAxisAngle(_orbitRight, pitch);
    _orbitForward.applyQuaternion(_orbitQuat);
    _orbitUp.applyQuaternion(_orbitQuat);
  }
  if (Math.abs(roll) > 1e-7) {
    _orbitQuat.setFromAxisAngle(_orbitForward, roll);
    _orbitUp.applyQuaternion(_orbitQuat);
    _orbitRight.applyQuaternion(_orbitQuat);
  }

  // Slide the target along the rotated forward at the original
  // camera-target distance; camera.position itself is untouched.
  controls.target.copy(camera.position).addScaledVector(_orbitForward, distance);
  camera.up.copy(_orbitUp);
}

// Translate the camera + target along the camera-to-target axis.
// Distance to target stays constant — both points move together —
// so the user can fly through the scene and out the other side
// instead of getting asymptotically pinned at the origin.
function translateCamera(amount) {
  _orbitForward.subVectors(controls.target, camera.position).normalize();
  camera.position.addScaledVector(_orbitForward, amount);
  controls.target.addScaledVector(_orbitForward, amount);
}

function easeInOut(t) {
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}

// Smoothed velocity state for each motion axis. Ramps toward the
// per-frame "target" derived from heldKeys; once the user releases
// a key the velocity decays smoothly to zero rather than snapping.
let _panVelX = 0, _panVelY = 0;
let _yawVel = 0, _pitchVel = 0, _rollVel = 0;
let _zoomVel = 0;
// 1 / time-constant for the velocity ramp. 4.0 ≈ 250 ms acceleration
// — slow enough to feel gradual under fingertip control, fast enough
// to still respond promptly when a key is held.
const _velK = 4.0;
// Max steady-state rates (applied at velocity = 1.0). Halved from
// the previous set to make held-key motion calmer.
const _panMaxFracPerSec  = 0.35;         // pan covers 35% of camera-target dist / sec
const _orbitMaxRadPerSec = Math.PI / 3;  // 60° / sec for yaw / pitch / roll
const _translateMaxFracPerSec = 0.5;     // forward translate 50% of dist / sec

function _rampVel(v, target, dt) {
  return v + (target - v) * (1 - Math.exp(-_velK * dt));
}

function applyKeyboardMotion(dt) {
  // Compute per-axis target velocities in [-1, 1] from held keys.
  // Translation: arrows + PgUp/Dn (3 axes).
  const panXt   = (heldKeys.has("ArrowRight") ? 1 : 0) - (heldKeys.has("ArrowLeft") ? 1 : 0);
  const panYt   = (heldKeys.has("ArrowUp")    ? 1 : 0) - (heldKeys.has("ArrowDown") ? 1 : 0);
  const zoomt   = (heldKeys.has("PageUp")     ? 1 : 0) - (heldKeys.has("PageDown")  ? 1 : 0);
  // Rotation: Q/E yaw, W/S pitch, A/D roll (3 axes).
  const yawt    = (heldKeys.has("KeyE") ? 1 : 0) - (heldKeys.has("KeyQ") ? 1 : 0);
  const pitcht  = (heldKeys.has("KeyW") ? 1 : 0) - (heldKeys.has("KeyS") ? 1 : 0);
  const rollt   = (heldKeys.has("KeyD") ? 1 : 0) - (heldKeys.has("KeyA") ? 1 : 0);

  _panVelX  = _rampVel(_panVelX,  panXt,  dt);
  _panVelY  = _rampVel(_panVelY,  panYt,  dt);
  _yawVel   = _rampVel(_yawVel,   yawt,   dt);
  _pitchVel = _rampVel(_pitchVel, pitcht, dt);
  _rollVel  = _rampVel(_rollVel,  rollt,  dt);
  _zoomVel  = _rampVel(_zoomVel,  zoomt,  dt);

  const distance = Math.max(0.001, camera.position.distanceTo(controls.target));
  const panRate   = distance * _panMaxFracPerSec;

  const dx = _panVelX * panRate * dt;
  const dy = _panVelY * panRate * dt;
  if (Math.abs(dx) > 1e-5 || Math.abs(dy) > 1e-5) panCamera(dx, dy);

  const yaw   = _yawVel   * _orbitMaxRadPerSec * dt;
  const pitch = _pitchVel * _orbitMaxRadPerSec * dt;
  const roll  = _rollVel  * _orbitMaxRadPerSec * dt;
  if (Math.abs(yaw) > 1e-6 || Math.abs(pitch) > 1e-6 || Math.abs(roll) > 1e-6) {
    rotateCamera(yaw, pitch, roll);
  }

  const translateAmount = _zoomVel * distance * _translateMaxFracPerSec * dt;
  if (Math.abs(translateAmount) > 1e-5) translateCamera(translateAmount);

  if (tweenActive) {
    tweenT += dt / tweenDuration;
    if (tweenT >= 1) {
      camera.position.copy(homeCameraPos);
      controls.target.copy(homeTarget);
      tweenActive = false;
    } else {
      const e = easeInOut(tweenT);
      camera.position.copy(_tweenStartCamPos).lerp(homeCameraPos, e);
      controls.target.copy(_tweenStartTarget).lerp(homeTarget, e);
    }
  }

  // Reassess LOD partition while any motion is still in progress —
  // including the inertia tail after a key release, so the LOD pick
  // stays in sync with the visible camera pose.
  const moving = Math.abs(_panVelX) + Math.abs(_panVelY)
               + Math.abs(_yawVel)  + Math.abs(_pitchVel) + Math.abs(_rollVel)
               + Math.abs(_zoomVel) > 1e-4;
  if (moving || tweenActive) scheduleReassess();
}

// Drag-and-drop anywhere on the page.
window.addEventListener("dragenter", (e) => {
  e.preventDefault();
  dropOverlay.classList.add("active");
});
window.addEventListener("dragover", (e) => {
  e.preventDefault();
});
window.addEventListener("dragleave", (e) => {
  // Only deactivate when we've fully left the window.
  if (e.relatedTarget === null) dropOverlay.classList.remove("active");
});
window.addEventListener("drop", (e) => {
  e.preventDefault();
  dropOverlay.classList.remove("active");
  const f = e.dataTransfer.files[0];
  if (f) loadSimulariumFile(f);
});

// ───── animation loop ───────────────────────────────────────────────

let lastFrame = performance.now();
let fpsAccum = 0;
let fpsCount = 0;
let fpsUpdate = performance.now();

function tick() {
  const now = performance.now();
  // Clamp dt so a dropped frame (e.g. during a heavy OBJ parse) doesn't
  // cause a one-shot ten-frames-worth of motion that feels like a jerk.
  const dt = Math.min(0.05, (now - lastFrame) / 1000);
  lastFrame = now;
  if (autoSpin) {
    controls.target;
    const target = controls.target;
    const offset = camera.position.clone().sub(target);
    const ang = dt * 0.25;
    const cos = Math.cos(ang), sin = Math.sin(ang);
    const nx = offset.x * cos - offset.z * sin;
    const nz = offset.x * sin + offset.z * cos;
    offset.x = nx;
    offset.z = nz;
    camera.position.copy(target).add(offset);
    // OrbitControls only emits "change" on user input, not on our
    // own camera moves; reassess explicitly so spin keeps the LOD +
    // frustum partition in sync.
    scheduleReassess();
  }
  applyKeyboardMotion(dt);
  controls.update();
  // The cel shader now provides Goodsell-style silhouette outlines
  // intrinsically (via view-aligned NdotV darkening), so we render
  // directly in both modes. The composer machinery is kept above as
  // dead-but-loaded code — when we want true depth-Sobel ink lines
  // we'll flip this back, but the per-fragment fresnel approach is
  // more robust to scene scale and doesn't crater on dense crowds.
  renderer.render(scene, camera);

  fpsAccum += dt;
  fpsCount++;
  if (now - fpsUpdate > 400) {
    const fps = fpsCount / fpsAccum;
    fpsStat.innerHTML = `<span class="num" style="color:var(--text)">${fps.toFixed(0)}</span> fps`;
    fpsAccum = 0;
    fpsCount = 0;
    fpsUpdate = now;
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

// ───── recipe dropdown ─────────────────────────────────────────────
// `data/index.json` lists demo packings staged in `viewer/data/`.
// Populate the dropdown from it; selecting an entry fetches and
// loads the file. Also honours `?file=…` for deep-linking.

const demoPicker = document.getElementById("demo-picker");

async function loadByPath(path) {
  try {
    const resp = await fetch(path);
    if (!resp.ok) throw new Error(resp.statusText);
    const doc = JSON.parse(await resp.text());
    buildScene(doc, path.split("/").pop());
  } catch (e) {
    console.warn("load failed:", e);
    statusEl.textContent = `failed to load ${path}: ${e.message}`;
  }
}

demoPicker.addEventListener("change", () => {
  const file = demoPicker.value;
  if (!file) return;
  loadByPath(`data/${file}`);
});

(async () => {
  // Populate the demo dropdown.
  try {
    const resp = await fetch("data/index.json");
    if (resp.ok) {
      const idx = await resp.json();
      for (const d of idx.demos) {
        const opt = document.createElement("option");
        opt.value = d.file;
        opt.textContent = d.label;
        demoPicker.appendChild(opt);
      }
    }
  } catch (e) {
    console.warn("could not fetch data/index.json:", e);
  }

  // Auto-load a packing if a ?file=... query param is present
  // (`view_pack.sh` uses this). Falls back to the first demo in the
  // dropdown so the viewer never opens completely empty.
  const params = new URLSearchParams(window.location.search);
  const file = params.get("file");
  if (file) {
    await loadByPath(file);
  } else if (demoPicker.options.length > 1) {
    demoPicker.selectedIndex = 1;
    demoPicker.dispatchEvent(new Event("change"));
  }
})();
