// obj-worker.js — off-main-thread OBJ fetch + parse for the parsimony
// viewer. Our OBJs are plain indexed triangle soup: shared `v` vertices
// + `f` integer faces, no normals/uv (marching-cubes output). So a tiny
// hand parser beats pulling three.js into the worker — no module-worker
// or CDN/importmap juggling, just pure JS returning transferable typed
// arrays the main thread caches and uploads.
//
// Protocol: main posts { id, url }; we reply { id, positions, normals,
// indices } (buffers transferred) or { id, error }.

function parseObj(text) {
  const positions = [];
  const indices = [];
  const lines = text.split("\n");
  for (let li = 0; li < lines.length; li++) {
    const line = lines[li];
    if (line.length < 2) continue;
    const c0 = line[0];
    if (c0 === "v" && line[1] === " ") {
      const t = line.trim().split(/\s+/);
      positions.push(+t[1], +t[2], +t[3]);
    } else if (c0 === "f" && line[1] === " ") {
      const t = line.trim().split(/\s+/);
      // Each token is `v`, `v/vt`, `v//vn`, or `v/vt/vn` — take the
      // vertex index (1-based → 0-based). Fan-triangulate any polygon.
      const face = [];
      for (let k = 1; k < t.length; k++) {
        const tok = t[k];
        const s = tok.indexOf("/");
        const vi = s === -1 ? tok : tok.slice(0, s);
        face.push(parseInt(vi, 10) - 1);
      }
      for (let k = 1; k + 1 < face.length; k++) {
        indices.push(face[0], face[k], face[k + 1]);
      }
    }
  }
  return { positions, indices };
}

// Area-weighted vertex normals (faces' cross products accumulated per
// vertex, then normalized) — matches three's computeVertexNormals.
function computeNormals(pos, idx) {
  const normals = new Float32Array(pos.length);
  for (let i = 0; i < idx.length; i += 3) {
    const a = idx[i] * 3, b = idx[i + 1] * 3, c = idx[i + 2] * 3;
    const e1x = pos[b] - pos[a], e1y = pos[b + 1] - pos[a + 1], e1z = pos[b + 2] - pos[a + 2];
    const e2x = pos[c] - pos[a], e2y = pos[c + 1] - pos[a + 1], e2z = pos[c + 2] - pos[a + 2];
    const nx = e1y * e2z - e1z * e2y;
    const ny = e1z * e2x - e1x * e2z;
    const nz = e1x * e2y - e1y * e2x;
    normals[a] += nx; normals[a + 1] += ny; normals[a + 2] += nz;
    normals[b] += nx; normals[b + 1] += ny; normals[b + 2] += nz;
    normals[c] += nx; normals[c + 1] += ny; normals[c + 2] += nz;
  }
  for (let i = 0; i < normals.length; i += 3) {
    const x = normals[i], y = normals[i + 1], z = normals[i + 2];
    const len = Math.hypot(x, y, z) || 1;
    normals[i] = x / len; normals[i + 1] = y / len; normals[i + 2] = z / len;
  }
  return normals;
}

self.onmessage = async (e) => {
  const { id, url } = e.data;
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`fetch ${url}: ${res.status}`);
    const { positions, indices } = parseObj(await res.text());
    if (positions.length === 0 || indices.length === 0) {
      throw new Error(`empty mesh: ${url}`);
    }
    const pos = new Float32Array(positions);
    const vertCount = positions.length / 3;
    const idx = vertCount > 65535 ? new Uint32Array(indices) : new Uint16Array(indices);
    const nrm = computeNormals(pos, idx);
    self.postMessage({ id, positions: pos, normals: nrm, indices: idx }, [
      pos.buffer,
      nrm.buffer,
      idx.buffer,
    ]);
  } catch (err) {
    self.postMessage({ id, error: String((err && err.message) || err) });
  }
};
