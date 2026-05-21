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
const toggleMembrane = document.getElementById("toggle-membrane");
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
// Compartment surfaces (the "membrane"): translucent shells for sphere
// compartments (the cell envelope). Rebuilt per packing, disposed with it.
let compartmentMeshes = [];
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
  // Impostor point size is in pixels, so it depends on viewport height.
  for (const m of compartmentMeshes) {
    const u = m.material && m.material.uniforms;
    if (u && u.uViewportHeight) u.uViewportHeight.value = r.height;
  }
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
  disposeCompartments();
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
let objLoadsInFlight = 0;

// ───── OBJ parsing on a Web Worker pool ─────────────────────────────
// OBJ fetch+parse is the load bottleneck and JS is single-threaded, so
// we move it off the main thread onto a pool of workers (real threads).
// Each worker fetches an OBJ and parses it (our OBJs are plain indexed
// v/f triangle soup — see obj-worker.js), returning transferable typed
// arrays the main thread caches + uploads. The main thread keeps the
// memory + IndexedDB cache tiers; only misses go to a worker. Falls back
// to main-thread parsing (`fetchAndParseObj`) if Workers are unavailable.
const WORKER_COUNT = Math.max(2, Math.min(8, navigator.hardwareConcurrency || 4));
const MAX_CONCURRENT_LOADS = WORKER_COUNT; // load tasks in flight at once
let objWorkers = null; // array of Workers, or null → main-thread parsing
let _workerNext = 0; // round-robin cursor
let _workerReqId = 0;
const _workerPending = new Map(); // reqId → { resolve, reject }

function initObjWorkers() {
  try {
    const ws = [];
    for (let i = 0; i < WORKER_COUNT; i++) {
      const w = new Worker(new URL("./obj-worker.js", import.meta.url));
      w.onmessage = (e) => {
        const { id, positions, normals, indices, error } = e.data;
        const p = _workerPending.get(id);
        if (!p) return;
        _workerPending.delete(id);
        if (error) p.reject(new Error(error));
        else p.resolve({ positions, normals, indices });
      };
      w.onerror = (ev) => {
        // Worker script failed to load/run — disable the pool and let
        // in-flight + future loads fall back to the main thread.
        console.warn("[viewer] OBJ worker failed; using main-thread parsing:",
                     ev.message || ev);
        objWorkers = null;
        for (const [, p] of _workerPending) p.reject(new Error("worker failed"));
        _workerPending.clear();
      };
      ws.push(w);
    }
    objWorkers = ws;
    console.log(`[viewer] OBJ parsing on ${WORKER_COUNT} web workers`);
  } catch (e) {
    console.warn("[viewer] web workers unavailable; parsing on main thread:", e);
    objWorkers = null;
  }
}
initObjWorkers();

// Hand one OBJ URL to a worker; resolves with { positions, normals,
// indices } typed arrays. Round-robins across the pool.
function parseObjInWorker(url) {
  return new Promise((resolve, reject) => {
    const id = _workerReqId++;
    _workerPending.set(id, { resolve, reject });
    objWorkers[_workerNext++ % objWorkers.length].postMessage({ id, url });
  });
}
let _drainScheduled = false;

// In-memory parsed-geometry cache — the tier above IndexedDB. Maps a
// resolved OBJ URL to its merged, normal-computed BufferGeometry plus
// the robust radius we derived from it. Unlike the per-scene
// InstancedMeshes (which clearPacking disposes on every scene switch),
// this map is never cleared during a session: navigating away from a
// packing and back re-materialises its meshes straight from RAM — a
// cheap geometry clone — instead of re-fetching, re-parsing, and
// popping them in one-at-a-time through the load throttle. Entries are
// "masters": no InstancedMesh ever owns one (every mesh gets a clone),
// so they survive mesh disposal. Session-scoped; a page reload falls
// back to the IndexedDB tier below.
const geomMemCache = new Map(); // resolvedUrl -> { geom, robustR }

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
const IDB_NAME = "parsimony-viewer-v8";
const IDB_STORE = "geoms";
// Cache-busting token from the pack's `cache_version`, prefixed onto every IDB
// key so regenerating the pack/meshes invalidates the cached geometry instead
// of serving stale meshes by URL. Set per pack in buildScene.
let packCacheVersion = "0";
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
  key = packCacheVersion + ":" + key;
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
  key = packCacheVersion + ":" + key;
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
  // First tier: instantly satisfy every wanted level we already hold in
  // the in-memory cache. These cost only a geometry clone — no fetch,
  // no OBJ parse — so there's no reason to serialise them behind the
  // one-at-a-time throttle the real loads use. Materialise them all in
  // one synchronous batch, even while a real parse is in flight. This
  // is what makes navigating back to a previously-viewed packing snap
  // to full detail instead of popping in level-by-level.
  let primedAny = false;
  for (const entry of instancedMeshes) {
    if (!entry.lods) continue;
    for (let i = 0; i < entry.lods.length; i++) {
      const lvl = entry.lods[i];
      if (lvl.loaded || lvl.loading || !lvl.wantLoad) continue;
      const cached = geomMemCache.get(resolveMeshUrl(lvl.url));
      if (!cached) continue;
      ensureLodMesh(entry, i, cached.geom, cached.robustR);
      primedAny = true;
    }
  }
  if (primedAny) {
    updateMeshLoadingStatus();
    scheduleReassess();
  }

  // Second tier: fill a concurrency pool with real fetch+parse loads,
  // each dispatched to a Web Worker (cache misses parse off the main
  // thread, so the coarse sweep and big fine LODs both stream without
  // freezing the UI). Cached levels were already materialised above.
  while (objLoadsInFlight < MAX_CONCURRENT_LOADS) {
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
    if (!best) break;
    best.lvl.loading = true;
    objLoadsInFlight++;
    loadLevel(best); // fire-and-track; loads overlap across the pool
  }
}

// Load one LOD level — memory cache was already checked by the prime
// loop, so here we hit IndexedDB (cross-reload) and, on a miss, parse in
// a worker; then cache + materialise, free the pool slot, and refill.
async function loadLevel(best) {
  const resolved = resolveMeshUrl(best.lvl.url);
  try {
    let geom;
    if (objWorkers) {
      let entry = await idbGet(resolved);
      if (entry && entry.positions && entry.positions.length > 0) {
        cacheHits++;
      } else {
        cacheMisses++;
        entry = await parseObjInWorker(resolved);
        idbPut(resolved, entry).catch(() => {
          cacheWriteErrors++;
        });
      }
      geom = cacheEntryToGeometry(entry);
    } else {
      // No workers — parse on the main thread (also handles its own cache).
      geom = await fetchAndParseObj(best.lvl.url);
    }
    const robustR = robustBoundingRadius(geom);
    geomMemCache.set(resolved, { geom, robustR });
    if (!best.lvl.loaded) {
      ensureLodMesh(best.entry, best.levelIdx, geom, robustR);
      updateMeshLoadingStatus();
      scheduleReassess();
    }
  } catch (err) {
    console.warn(`LOD ${best.levelIdx} load failed for ${best.entry.name}:`, err);
  } finally {
    best.lvl.loading = false;
    objLoadsInFlight--;
    scheduleDrain(); // a slot freed — refill the pool (coalesced)
  }
}

// ───── membrane cleanup ─────────────────────────────────────────────
// The cell envelope used to be drawn here as a translucent, fresnel-rimmed
// shell; that was dropped in favour of the dense impostor lipid bilayer
// below (buildImpostorMembrane), which reads as the membrane on its own.
// `disposeCompartments` is kept — it now tears down that impostor mesh,
// which is still tracked in `compartmentMeshes` and toggled as "membrane".

function disposeCompartments() {
  for (const m of compartmentMeshes) {
    scene.remove(m);
    m.geometry.dispose();
    m.material.dispose();
  }
  compartmentMeshes = [];
}

// ───── impostor lipid bilayer ───────────────────────────────────────
// A real membrane is ~10^6 lipids at ~76 Å²/headgroup — far too many to
// store as pack placements or draw as meshes. So we render lipid heads
// as GPU sphere *impostors* (one camera-facing point sprite each, shaded
// round + lit in the fragment shader: ~free per lipid) and *generate*
// the bilayer in the viewer from the cell compartment — two evenly
// (jittered) tiled leaflet shells — instead of packing a million
// placements. This is how cellPACK/Maritan render dense membranes.
//
// Tunables (eyeball + adjust):
const MEMBRANE_SPACING = 8.5;      // Å between lipids in a leaflet (smaller = denser).
                                   // ~72 Å²/lipid ≈ a real membrane (~76); was 11
                                   // (≈1.6× sparse) back when a locked GPU looked like a
                                   // perf wall. Impostors are ~free, so this is the knob.
const MEMBRANE_THICKNESS = 40;     // Å total bilayer thickness (centred on the cell surface)
const MEMBRANE_HEAD_RADIUS = 5;    // Å headgroup impostor radius
const MEMBRANE_TAIL_RADIUS = 4.0;  // Å tail-bead impostor radius (fat enough to read as a continuous strand)
const MEMBRANE_MAX_POINTS = 5000000; // safety cap on impostor points. Headroom so it does
                                   // NOT throttle real density for the 2000 Å cell
                                   // (~4.2M points at spacing 8.5); only binds for bigger cells.

function makeImpostorMaterial(headColor) {
  return new THREE.ShaderMaterial({
    uniforms: {
      uViewportHeight: { value: renderer.domElement.clientHeight || 900 },
      uColorHead: { value: headColor.clone() },
      uColorTail: { value: headColor.clone().multiplyScalar(0.7) },
      uLightDir: { value: new THREE.Vector3(0.3, 0.5, 0.85).normalize() },
    },
    vertexShader: `
      attribute float aRadius;
      attribute float aHead;
      uniform float uViewportHeight;
      varying float vHead;
      void main() {
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        vHead = aHead;
        gl_Position = projectionMatrix * mv;
        // Projected sphere diameter in pixels (projectionMatrix[1][1] = 1/tan(fov/2)).
        gl_PointSize = aRadius * uViewportHeight * projectionMatrix[1][1] / max(-mv.z, 0.001);
      }
    `,
    fragmentShader: `
      uniform vec3 uColorHead;
      uniform vec3 uColorTail;
      uniform vec3 uLightDir;
      varying float vHead;
      void main() {
        vec2 c = gl_PointCoord * 2.0 - 1.0;
        float r2 = dot(c, c);
        if (r2 > 1.0) discard;                       // round the sprite
        vec3 n = vec3(c.x, -c.y, sqrt(1.0 - r2));    // view-space sphere normal
        float ndl = max(dot(n, normalize(uLightDir)), 0.0);
        vec3 base = mix(uColorTail, uColorHead, vHead);
        gl_FragColor = vec4(base * (0.35 + 0.7 * ndl), 1.0);
      }
    `,
    transparent: false,
    depthTest: true,
    depthWrite: true,
  });
}

function buildImpostorMembrane(doc) {
  const comps = (doc.compartments || []).filter(
    (c) => c.kind === "sphere" && Array.isArray(c.center) && typeof c.radius === "number",
  );
  if (comps.length === 0) return;
  const lip = (doc.ingredients || []).find((i) => i.name === "lipid");
  const col = lip && lip.color
    ? new THREE.Color(lip.color[0], lip.color[1], lip.color[2])
    : new THREE.Color(0.96, 0.86, 0.55);

  // One lipid = a head at the leaflet surface + two tail beads reaching
  // toward the midplane. Offsets are radial distances from the bilayer
  // midplane, which we centre on the compartment surface so embedded
  // proteins — placed with their origin on that surface — span it.
  const h = MEMBRANE_THICKNESS * 0.5;
  const beadOffset = [h, 0.55 * h, 0.18 * h]; // head, tail, tail (→ midplane)
  const beadRadius = [MEMBRANE_HEAD_RADIUS, MEMBRANE_TAIL_RADIUS, MEMBRANE_TAIL_RADIUS];
  const beadHead = [1, 0, 0];
  const perLipid = beadOffset.length;
  const golden = Math.PI * (3 - Math.sqrt(5));

  for (const c of comps) {
    const [cx, cy, cz] = c.center;
    const Rmid = c.radius;
    const cap = Math.floor(MEMBRANE_MAX_POINTS / (2 * perLipid));
    const nLeaflet = Math.min(cap,
      Math.round((4 * Math.PI * Rmid * Rmid) / (MEMBRANE_SPACING * MEMBRANE_SPACING)));
    const total = 2 * nLeaflet * perLipid;
    const pos = new Float32Array(total * 3);
    const rad = new Float32Array(total);
    const head = new Float32Array(total);
    const amp = 0.5 * (3.54 / Math.sqrt(Math.max(1, nLeaflet))); // ~half arc spacing
    let w = 0;
    for (const sign of [1, -1]) { // outer (+) then inner (−) leaflet
      for (let i = 0; i < nLeaflet; i++) {
        const y = 1 - 2 * (i + 0.5) / nLeaflet;
        const rr = Math.sqrt(Math.max(0, 1 - y * y));
        const th = golden * i;
        let dx = Math.cos(th) * rr + (Math.random() - 0.5) * 2 * amp;
        let dy = y + (Math.random() - 0.5) * 2 * amp;
        let dz = Math.sin(th) * rr + (Math.random() - 0.5) * 2 * amp;
        const len = Math.hypot(dx, dy, dz) || 1;
        dx /= len; dy /= len; dz /= len;
        for (let b = 0; b < perLipid; b++) {
          const R = Rmid + sign * beadOffset[b];
          pos[w * 3] = cx + dx * R;
          pos[w * 3 + 1] = cy + dy * R;
          pos[w * 3 + 2] = cz + dz * R;
          rad[w] = beadRadius[b];
          head[w] = beadHead[b];
          w++;
        }
      }
    }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geom.setAttribute("aRadius", new THREE.BufferAttribute(rad, 1));
    geom.setAttribute("aHead", new THREE.BufferAttribute(head, 1));
    const points = new THREE.Points(geom, makeImpostorMaterial(col));
    points.frustumCulled = false;
    points.visible = !toggleMembrane || toggleMembrane.checked;
    scene.add(points);
    compartmentMeshes.push(points); // disposed by disposeCompartments
  }
}

// ───── chromosome / genome ───────────────────────────────────────────
// The chromosome is no longer a bespoke tube. The packer emits it as tens of
// thousands of `dna_segment` mesh instances — one shared real-dsDNA mesh
// (1BNA) tiled every ~12 bp along the genome path and rolled by the duplex
// twist so the instances form a continuous, real-scale double helix. It is a
// plain mesh ingredient, so it renders through the same instanced-mesh +
// per-instance LOD + cel/outline path as every protein: molecular up close,
// Goodsell-shaded, and naturally LOD'd to a cigar at cell scale. No special
// fiber/helix code path remains in the viewer.

// Build the scene from a `parsimony.pack.v1` document.
async function buildScene(doc, fileName) {
  clearPacking();
  cacheHits = 0;
  cacheMisses = 0;
  cacheWriteErrors = 0;
  // Bust the IDB mesh cache when the pack (and thus possibly the meshes) was
  // regenerated. Falls back to the filename for packs without a version.
  packCacheVersion = String(doc.cache_version ?? fileName ?? "0");

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
      // Lower-poly proxy spheres — multi_sphere ingredients (rods, and
      // the lipid bilayer at tens of thousands of instances) don't need
      // high tessellation; keeps the dense membrane cheap to draw.
      const g = new THREE.SphereGeometry(r, 12, 6);
      g.translate(s.offset[0], s.offset[1], s.offset[2]);
      return g;
    });
    const merged = mergeGeometries(geoms, false);
    geoms.forEach((g) => g.dispose());
    if (merged) return merged;
  }
  // Mesh ingredients get an anisotropic ellipsoid placeholder: a unit
  // sphere baked into the principal-axis ellipsoid the packer fitted to
  // the molecule. Elongated species then read as cigars (and flat ones
  // as discs) instead of fat enclosing balls, both before their OBJ
  // LODs stream in and at the sub-3px far range that never loads a mesh.
  // The per-instance placement quaternion (applied in reassessLODs)
  // orients the whole ellipsoid, so the instance path is unchanged and
  // sphere↔mesh swaps keep apparent size. Older packs without the field
  // fall through to the plain enclosing sphere.
  const ell = ing.shape.kind === "mesh" ? ing.shape.ellipsoid : null;
  if (ell && Array.isArray(ell.semi_axes)) {
    const r = Array.isArray(ell.rotation) ? ell.rotation : [1, 0, 0, 0];
    const g = new THREE.SphereGeometry(1, 20, 12);
    const m = new THREE.Matrix4().compose(
      new THREE.Vector3(),
      new THREE.Quaternion(r[1], r[2], r[3], r[0]), // pack stores [w, x, y, z]
      new THREE.Vector3(ell.semi_axes[0], ell.semi_axes[1], ell.semi_axes[2]),
    );
    g.applyMatrix4(m);
    return g;
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
// A LOD whose thinnest bounding-box extent is below this (Å) collapsed to
// a near-2D sheet at its voxel size — treated as degenerate so the picker
// skips it (a real molecule is never sub-3 Å thin in any dimension).
const PLANAR_MIN_THICKNESS = 3.0;

function ensureLodMesh(entry, levelIdx, geom, robustR) {
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
  // robustR is supplied by the caller when the geometry came from the
  // in-memory cache (we recorded it at first parse); recompute only on
  // a true first load.
  if (robustR == null) robustR = robustBoundingRadius(geom);
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

  // Also reject LODs that marching-cubes collapsed to a near-2D sheet: a
  // protein dimension thinner than the voxel renders as a flat quad
  // ("the square"). Flagging it degenerate makes the picker fall to a
  // finer LOD or the 3D ellipsoid, so the molecule keeps its depth.
  // (Real thin filaments are tens of Å thick — well above the cutoff.)
  geom.computeBoundingBox();
  const _ext = new THREE.Vector3();
  geom.boundingBox.getSize(_ext);
  if (Math.min(_ext.x, _ext.y, _ext.z) < PLANAR_MIN_THICKNESS) {
    lvl.degenerate = true;
  }

  if (lvl.standardMesh) {
    // Already created; replace geometry in place.
    lvl.standardMesh.geometry.dispose();
    lvl.celMesh.geometry.dispose();
    lvl.standardMesh.geometry = geom.clone();
    lvl.celMesh.geometry = geom.clone();
  } else {
    const standardMat = new THREE.MeshStandardMaterial({
      color: entry.color,
      roughness: 0.55,
      metalness: 0.05,
    });
    // Both meshes own a clone; `geom` stays an unowned master in
    // geomMemCache so clearPacking (which disposes mesh geometries)
    // can't free the cached copy.
    const standard = new THREE.InstancedMesh(geom.clone(), standardMat, entry.placements.length);
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
//
// On a whole-cell recipe (~20k mesh instances), viewing the whole cell
// from far away puts every instance in the frustum at a few px each —
// at 3 px they ALL route to meshes (no culling, mass OBJ loads) and the
// view becomes unnavigable. Default to a higher budget so distant
// instances stay cheap proxies; the "Mesh @px" slider tunes it.
let lodSphereBudgetPx = 12.0;

// Fraction of each ingredient's instances actually drawn — a viz-only
// subsample (the pack order is random, so a prefix is a uniform random
// subset). Defaults to 100% (the full pack); the "Show %" slider can dial
// it down to declutter / cut render load without re-packing (the packed
// cell is numerically sparse but the oversized proxies read as crowded).
let interiorFraction = 1.0;

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

    const nShow = Math.round(entry.placements.length * interiorFraction);
    for (let pi = 0; pi < nShow; pi++) {
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

  // Coarsest-LOD preload: independent of the camera, queue every mesh
  // ingredient's coarsest level so the whole scene shows real (if
  // coarse) geometry from the start — not just the framed neighbourhood,
  // and not a minute of placeholder spheres popping in one by one. Top
  // priority (−1) so this breadth pass loads before the distance-ranked
  // refinement of nearby instances; the ellipsoid fallback covers each
  // ingredient until its coarse mesh lands, and `reassessLODs` then
  // upgrades the near ones to finer levels. Once a coarse level is
  // loaded the guard below makes this a no-op.
  for (const entry of instancedMeshes) {
    const lods = entry.lods;
    if (!lods || lods.length === 0) continue;
    const lvl0 = lods[0];
    if (!lvl0.loaded && !lvl0.loading) {
      lvl0.wantLoad = true;
      lvl0.wantPriority = -1;
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
if (toggleMembrane) {
  toggleMembrane.addEventListener("change", () => {
    for (const m of compartmentMeshes) m.visible = toggleMembrane.checked;
  });
}
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

// "Mesh @px" slider — writes `lodSphereBudgetPx`, the projected radius
// below which an ingredient draws as a cheap proxy instead of loading
// its OBJ. Higher keeps far/dense views as proxies (navigable); lower
// loads meshes sooner (more detail, slower).
const sphereBudgetSlider = document.getElementById("sphere-budget");
const sphereBudgetValue = document.getElementById("sphere-budget-value");
if (sphereBudgetSlider) {
  sphereBudgetSlider.value = String(lodSphereBudgetPx);
  sphereBudgetValue.textContent = lodSphereBudgetPx.toFixed(0);
  sphereBudgetSlider.addEventListener("input", () => {
    lodSphereBudgetPx = parseFloat(sphereBudgetSlider.value);
    sphereBudgetValue.textContent = lodSphereBudgetPx.toFixed(0);
    scheduleReassess();
  });
}

// "Show %" slider — fraction of interior instances drawn (viz subsample).
const showFractionSlider = document.getElementById("show-fraction");
const showFractionValue = document.getElementById("show-fraction-value");
if (showFractionSlider) {
  showFractionSlider.value = String(Math.round(interiorFraction * 100));
  showFractionValue.textContent = showFractionSlider.value;
  showFractionSlider.addEventListener("input", () => {
    interiorFraction = parseFloat(showFractionSlider.value) / 100;
    showFractionValue.textContent = showFractionSlider.value;
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
  const yawt    = (heldKeys.has("KeyQ") ? 1 : 0) - (heldKeys.has("KeyE") ? 1 : 0);
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
