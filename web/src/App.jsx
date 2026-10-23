import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api.js";
import { Chip, Icon, Toast, lineKey } from "./ui.jsx";
import Overview from "./views/Overview.jsx";
import Workspace from "./views/Workspace.jsx";
import Commentary from "./views/Commentary.jsx";
import Documents from "./views/Documents.jsx";
import Mapping from "./views/Mapping.jsx";
import DataView from "./views/DataView.jsx";

const PREVIEW = typeof __PREVIEW__ !== "undefined";
const monthLabel = (m) => new Date(m + "-01T00:00:00").toLocaleString("en-US", { month: "long", year: "numeric" });

const STEP = { documents: "Step 1 of 5 · Add documents", mapping: "Step 2 of 5 · Map the accounts", data: "Step 3 of 5 · Check the data",
  workspace: "Step 4 of 5 · Find the reasons and accept", commentary: "Step 5 of 5 · Draft and verify" };

// Back / next buttons at the bottom of every page, in workflow order.
function Pager({ view, setView, nav }) {
  const i = nav.findIndex((n) => n[0] === view);
  const prev = nav[i - 1], next = nav[i + 1];
  if (view === "overview" && !next) return null;
  if (i < 0) return (<div className="pager"><div><button className="btn" onClick={() => { setView("overview"); try { window.scrollTo({ top: 0 }); } catch {} }}>← Back to Overview</button></div><span className="muted">{STEP[view] || ""}</span><div /></div>);
  const go = (id) => { setView(id); try { window.scrollTo({ top: 0 }); } catch {} };
  return (
    <div className="pager">
      <div>{prev && <button className="btn" onClick={() => go(prev[0])}>← {prev[1]}</button>}</div>
      <span className="muted">{STEP[view] || ""}</span>
      <div>{next ? <button className="btn primary" onClick={() => go(next[0])}>Next: {next[1]} →</button>
        : <button className="btn primary" onClick={() => go("overview")}>Back to Overview →</button>}</div>
    </div>
  );
}

export default function App() {
  const [meta, setMeta] = useState(null);
  const [view, setView] = useState("overview");
  const [visited, setVisited] = useState({});       // tabs the user has opened (drives the green "Done" on the workflow)
  useEffect(() => { setVisited((v) => (v[view] ? v : { ...v, [view]: true })); }, [view]);
  const [params, setParams] = useState({ dataset: "sample", month: "", min_dollars: 5000, min_pct: 5 });
  const [months, setMonths] = useState([]);
  const [data, setData] = useState(null);          // variances response
  const [pre, setPre] = useState(null);            // preflight response
  const [loading, setLoading] = useState(false);
  const [toast, setToast] = useState("");
  const [inv, setInv] = useState({ agent: "baseline", retrieval: "hybrid", date_filter: true });
  const [findings, setFindings] = useState({});    // line key -> finding
  const [runMeta, setRunMeta] = useState(null);
  const [progress, setProgress] = useState(null);  // {done,total} while running
  const [accepted, setAccepted] = useState({});    // line key -> true
  const [notes, setNotes] = useState({});          // line key -> text

  useEffect(() => {
    api.meta().then((m) => {
      setMeta(m);
      setMonths(m.sample_months);
      setParams((p) => ({ ...p, month: m.sample_months[m.sample_months.length - 1] }));
    }).catch((e) => setToast("Cannot reach the API: " + e.message));
  }, []);

  const q = useMemo(() => ({ dataset: params.dataset, month: params.month, min_dollars: +params.min_dollars || 0, min_pct: +params.min_pct || 0 }), [params]);

  // Reload analysis whenever the scope changes (debounced for the number inputs).
  useEffect(() => {
    if (!params.month) return;
    setLoading(true);
    const t = setTimeout(() => {
      Promise.all([api.variances(q), api.preflight(q)])
        .then(([v, p]) => { setData(v); setPre(p); })
        .catch((e) => setToast(e.message))
        .finally(() => setLoading(false));
    }, 250);
    return () => clearTimeout(t);
  }, [q]);

  // Investigation results belong to a month; clear them when the scope changes.
  const scopeRef = useRef("");
  useEffect(() => {
    const s = `${params.dataset}|${params.month}`;
    if (scopeRef.current && scopeRef.current !== s) { setFindings({}); setAccepted({}); setRunMeta(null); }
    scopeRef.current = s;
  }, [params.dataset, params.month]);

  const investigate = useCallback(async () => {
    setFindings({}); setAccepted({}); setProgress({ done: 0, total: data?.lines.length || 0 });
    const body = { ...q, ...inv };
    try {
      await api.investigate(body, (ev) => {
        if (ev.type === "progress") {
          setFindings((f) => ({ ...f, [ev.finding.line_key]: ev.finding }));
          setProgress({ done: ev.done, total: ev.total });
        } else if (ev.type === "error") setToast("Investigation failed: " + ev.message);
        else if (ev.type === "start") setRunMeta({ agent: ev.agent, retrieval: ev.retrieval, date_filter: inv.date_filter });
      });
    } catch (e) { setToast(e.message); }
    setProgress(null);
  }, [q, inv, data]);

  const onUpload = async (gl, coa) => {
    try {
      const r = await api.upload(gl, coa);
      setMonths(r.months);
      setFindings({}); setAccepted({}); setNotes({});
      setParams((p) => ({ ...p, dataset: r.dataset, month: r.months[r.months.length - 1] }));
      setToast("");
    } catch (e) { setToast(e.message); }
  };
  const useSample = () => {
    setMonths(meta.sample_months); setNotes({});
    setParams((p) => ({ ...p, dataset: "sample", month: meta.sample_months[meta.sample_months.length - 1] }));
  };

  const lines = data?.lines || [];
  const found = Object.keys(findings).length;
  const acceptedCount = Object.keys(accepted).filter((k) => accepted[k]).length;
  const nav = [
    ["overview", "Overview", "overview"], ["documents", "Documents", "doc"], ["mapping", "Mapping", "lines"], ["data", "Data checks", "data"],
    ["workspace", "Variances", "lines"], ["commentary", "Commentary", "doc"],
  ];
  const titles = { overview: "Overview", workspace: "Variance workspace", documents: "Documents", mapping: "Account mapping", commentary: "Commentary", data: "Data checks" };
  const shared = { visited, meta, params, setParams, q, data, pre, lines, findings, runMeta, progress, accepted, setAccepted, notes, setNotes, inv, setInv, investigate, setToast, setView, monthLabel };

  const sub = { overview: "This month at a glance and what to do next.", workspace: "Find the reason behind each large variance, with the evidence.", documents: "The emails, invoices and notes the agent searches. Add your own and try the search.", mapping: "Map each company's accounts to one standard chart. The AI proposes, you approve.", commentary: "Draft the write-up and let the verifier check every figure.", data: "Problems in the ledger, checked before any variance is computed." };
  return (
    <div className="app">
      <header className="hdr">
        <div className="brand">
          <div className="logo"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#280949" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><path d="M4 17l6-7 4 4 6-8" /></svg></div>
          <div><b>Variance Desk</b><small>Budget vs actual, auditable</small></div>
        </div>
        <nav className="tabs">
          {nav.map(([id, label]) => (
            <button key={id} className={view === id ? "on" : ""} onClick={() => setView(id)}>
              {label}
            </button>
          ))}
        </nav>
        <div className="hdr-r">
          <span>Retrieval: BM25 + {meta?.embedder || "…"}</span>
        </div>
      </header>
      <div className="ctx">
        <div className="entity"><small>Entity</small><b>{meta ? (params.dataset === "sample" ? meta.company : "Uploaded data") : "…"}</b></div>
        <div className="field"><label>Period</label>
          <select value={params.month} onChange={(e) => setParams({ ...params, month: e.target.value })}>
            {months.map((m) => <option key={m} value={m}>{monthLabel(m)}</option>)}
          </select></div>
        <div className="field"><label>Material if ≥ $</label>
          <input type="number" min="0" step="500" value={params.min_dollars} onChange={(e) => setParams({ ...params, min_dollars: e.target.value })} /></div>
        <div className="field"><label>and ≥ %</label>
          <input type="number" min="0" step="1" value={params.min_pct} onChange={(e) => setParams({ ...params, min_pct: e.target.value })} /></div>
        <span className="spacer" />
        <span className="muted" style={{ fontSize: 12 }}>Demo data is fictional</span>
      </div>
      <div className="main">
        <main className="page">
          <div className="pagehead"><h1>{titles[view]}</h1><p>{sub[view]}</p></div>
          {PREVIEW && <div className="callout warn"><b>Interactive preview.</b> Real numbers and a real run of the keyword-baseline agent, recorded from the Python backend. Drafting, editing and verifying work live here. The AI agent needs an API key, so it is off in this preview.</div>}
          {view === "overview" && <Overview {...shared} loading={loading} acceptedCount={acceptedCount} />}
          {view === "workspace" && <Workspace {...shared} loading={loading} />}
          {view === "documents" && <Documents {...shared} />}
          {view === "mapping" && <Mapping {...shared} />}
          {view === "commentary" && <Commentary {...shared} acceptedCount={acceptedCount} />}
          {view === "data" && <DataView {...shared} />}
          <Pager view={view} setView={setView} nav={nav} />
        </main>
      </div>
      <Toast msg={toast} onClose={() => setToast("")} />
    </div>
  );
}
