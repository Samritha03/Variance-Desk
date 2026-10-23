const BASE = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");

// A random id the browser keeps, so documents you upload are searched only for you.
const rid = () => Math.random().toString(36).slice(2) + Date.now().toString(36);
let SID = "";
try { SID = localStorage.getItem("gve-sid") || ""; if (!SID) { SID = rid(); localStorage.setItem("gve-sid", SID); } } catch { SID = SID || rid(); }

async function request(path, opts = {}) {
  const res = await fetch(BASE + path, { ...opts, headers: { ...(opts.headers || {}), "X-GVE-Session": SID } });
  if (!res.ok) {
    let msg = res.statusText;
    try { const j = await res.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch {}
    throw new Error(msg);
  }
  return res;
}
const post = (path, body) =>
  request(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then((r) => r.json());

export const api = {
  meta: () => request("/api/meta").then((r) => r.json()),
  preflight: (p) => post("/api/preflight", p),
  variances: (p) => post("/api/variances", p),
  commentary: (b) => post("/api/commentary", b),
  validate: (b) => post("/api/validate", b),
  bench: () => request("/api/bench").then((r) => r.json()),
  documents: () => request("/api/documents").then((r) => r.json()),
  addDocument: (file, fields = {}) => {
    const f = new FormData(); f.append("file", file);
    Object.entries(fields).forEach(([k, v]) => v && f.append(k, v));
    return request("/api/documents", { method: "POST", body: f }).then((r) => r.json());
  },
  removeDocument: (id) => request(`/api/documents/${id}`, { method: "DELETE" }).then((r) => r.json()),
  mappingData: () => request("/api/mapping/data").then((r) => r.json()),
  mappingRun: (b) => post("/api/mapping/run", b),
  mappingEval: () => request("/api/mapping/eval").then((r) => r.json()),
  search: (b) => post("/api/search", b),
  document: (id) => request(`/api/documents/${id}`).then((r) => r.json()),
  upload: (gl, coa) => {
    const f = new FormData(); f.append("gl", gl); f.append("coa", coa);
    return request("/api/datasets", { method: "POST", body: f }).then((r) => r.json());
  },
  months: (id) => request(`/api/datasets/${id}/months`).then((r) => r.json()),
  // Streams NDJSON; calls onEvent for every line.
  async investigate(body, onEvent) {
    const res = await request("/api/investigate", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    const reader = res.body.getReader(); const dec = new TextDecoder(); let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1);
        if (line) onEvent(JSON.parse(line));
      }
    }
  },
};
