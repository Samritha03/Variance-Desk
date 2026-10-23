import React, { useEffect, useRef, useState } from "react";
import { api } from "../api.js";
import { Chip, DocModal, Spinner } from "../ui.jsx";

const TYPES = ["", "email", "invoice", "contract", "memo", "note", "report"];
const MODES = [["hybrid", "Hybrid"], ["bm25", "Keyword only"], ["dense", "Embeddings only"]];

const IS_PREVIEW = typeof __PREVIEW__ !== "undefined";

export default function Documents({ params, monthLabel, setToast }) {
  const [list, setList] = useState(null);
  const [files, setFiles] = useState([]);
  const [fields, setFields] = useState({ date: "", type: "" });
  const [busy, setBusy] = useState(false);
  const [results, setResults] = useState([]);
  const [show, setShow] = useState("all");
  const [open, setOpen] = useState(null);
  const input = useRef(null);
  const [q, setQ] = useState({ query: "", mode: "hybrid", filter: true });
  const [hits, setHits] = useState(null);
  const [searching, setSearching] = useState(false);

  const load = () => api.documents().then(setList).catch((e) => setToast(e.message));
  useEffect(() => { load(); }, []);

  const add = async () => {
    setBusy(true); const out = [];
    for (const f of files) {
      try {
        const d = await api.addDocument(f, files.length === 1 ? fields : { type: fields.type });
        out.push({ name: f.name, ok: true, text: `Added as ${d.id}, dated ${d.date} (${d.date_source}).` });
      } catch (e) { out.push({ name: f.name, ok: false, text: e.message.replace(`${f.name}: `, "") }); }
    }
    setResults(out); setBusy(false);
    if (out.some((r) => r.ok)) { setFiles(out.filter((r) => !r.ok).map((r) => files.find((f) => f.name === r.name))); if (input.current) input.current.value = ""; }
    load(); setHits(null);
  };
  const remove = async (id) => { try { await api.removeDocument(id); load(); setHits(null); } catch (e) { setToast(e.message); } };
  const run = async () => {
    setSearching(true);
    try { setHits((await api.search({ query: q.query, month: params.month, date_filter: q.filter, mode: q.mode, k: 4 })).hits); }
    catch (e) { setToast(e.message); } finally { setSearching(false); }
  };

  const rows = (list?.documents || []).filter((d) => show === "all" || d.source === show);
  const needsDate = results.some((r) => !r.ok && /no date found/.test(r.text));
  return (
    <>
      {IS_PREVIEW && <div className="callout warn">In this preview, documents you add stay in your browser tab and the search is keyword-only. The full app also reads PDFs and adds the embeddings half of the hybrid search.</div>}
      <div className="card">
        <div className="card-h"><h3>Add documents</h3><span className="muted">.txt, .md, .eml or text-based .pdf · up to 2 MB each</span></div>
        <div className="card-b stack">
          <div className="row">
            <input ref={input} type="file" multiple accept=".txt,.md,.eml,.pdf" onChange={(e) => { setFiles([...e.target.files]); setResults([]); }} />
            <div className="field"><label>Date{needsDate && " (needed)"}</label>
              <input type="text" placeholder="2026-09-14" style={{ width: 120, ...(needsDate ? { borderColor: "var(--warn, #b45309)" } : {}) }} disabled={files.length > 1}
                value={fields.date} onChange={(e) => setFields({ ...fields, date: e.target.value })} /></div>
            <div className="field"><label>Type</label>
              <select value={fields.type} onChange={(e) => setFields({ ...fields, type: e.target.value })}>
                {TYPES.map((t) => <option key={t} value={t}>{t || "detect"}</option>)}</select></div>
            <button className="btn primary" disabled={!files.length || busy} onClick={add}>{busy ? <Spinner /> : null}Add {files.length > 1 ? `${files.length} files` : "document"}</button>
          </div>
          {results.map((r, i) => (
            <div key={i} className={`callout ${r.ok ? "ok" : "bad"}`}><b>{r.name}</b> — {r.text}</div>
          ))}
          <div className="muted" style={{ fontSize: 12.5 }}>
            The date is read from the file. Use a date close to the month you are analysing, or the agent will not find it. Your files are private to you.
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-h"><h3>Try the search</h3><span className="muted">the same search the agent uses</span></div>
        <div className="card-b stack">
          <form className="row" onSubmit={(e) => { e.preventDefault(); if (q.query.trim().length > 1) run(); }}>
            <input type="text" style={{ flex: 1, minWidth: 220 }} placeholder="e.g. trade show costs moved to September" value={q.query} onChange={(e) => setQ({ ...q, query: e.target.value })} />
            <select value={q.mode} onChange={(e) => setQ({ ...q, mode: e.target.value })}>{MODES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select>
            <label className="row" style={{ gap: 6, fontSize: 13 }}><input type="checkbox" checked={q.filter} onChange={(e) => setQ({ ...q, filter: e.target.checked })} />Only around {monthLabel(params.month)}</label>
            <button className="btn primary" type="submit" disabled={q.query.trim().length < 2 || searching}>{searching ? <Spinner /> : null}Search</button>
          </form>
          {hits && hits.length === 0 && <div className="muted">Nothing matched. With the date filter on, only documents near {monthLabel(params.month)} are searched.</div>}
          {hits && hits.map((h, i) => (
            <div key={i} className="hit" onClick={() => setOpen({ id: h.doc_id, quote: "" })} style={{ cursor: "pointer" }}>
              <div className="row" style={{ gap: 8 }}><b>{h.title}</b><Chip kind="gray">{h.doc_id}</Chip>{h.uploaded && <Chip kind="ok">yours</Chip>}
                <span className="muted">{h.type} · {h.date}</span><span className="spacer" /><span className="muted num">score {h.score}</span></div>
              <div className="muted" style={{ fontSize: 13, marginTop: 4 }}>{h.text.replace(/\s+/g, " ").slice(0, 220)}…</div>
            </div>
          ))}
        </div>
      </div>

      <div className="card" style={{ overflow: "hidden" }}>
        <div className="card-h"><h3>All documents</h3>
          {list && <span className="muted">{list.documents.length} total · {list.uploaded} yours</span>}<span className="spacer" />
          <div className="seg">{[["all", "All"], ["uploaded", "Yours"], ["sample", "Sample"]].map(([v, l]) => <button key={v} className={show === v ? "on" : ""} onClick={() => setShow(v)}>{l}</button>)}</div></div>
        {!list ? <div className="card-b"><div className="skel" /></div> : rows.length === 0 ? <div className="empty"><h4>Nothing here yet</h4>Add a file above and it will show up in this list.</div> : (
          <div className="tblscroll"><table>
            <thead><tr><th style={{ width: 80 }}>ID</th><th style={{ width: 110 }}>Date</th><th style={{ width: 90 }}>Type</th><th>Title</th><th style={{ width: 90 }}>Source</th><th className="r" style={{ width: 150 }} /></tr></thead>
            <tbody>{rows.map((d) => (
              <tr key={d.id}><td className="mono">{d.id}</td><td className="num">{d.date}</td><td>{d.type}</td><td>{d.title}</td>
                <td>{d.source === "uploaded" ? <Chip kind="ok">yours</Chip> : <Chip kind="gray">sample</Chip>}</td>
                <td className="r"><button className="btn small" onClick={() => setOpen({ id: d.id, quote: "" })}>View</button>{d.source === "uploaded" && <> <button className="btn small" onClick={() => remove(d.id)}>Remove</button></>}</td></tr>
            ))}</tbody>
          </table></div>
        )}
      </div>
      <DocModal id={open?.id} quote={open?.quote} onClose={() => setOpen(null)} />
    </>
  );
}
