import React, { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api.js";
import { Chip, Spinner, lineKey } from "../ui.jsx";

// Render one line of the draft, wrapping any text the verifier flagged.
function highlight(line, snippets) {
  let parts = [line];
  for (const s of snippets) {
    if (!s) continue;
    parts = parts.flatMap((p) => {
      if (typeof p !== "string") return [p];
      const i = p.indexOf(s);
      return i < 0 ? [p] : [p.slice(0, i), { mark: s }, p.slice(i + s.length)];
    });
  }
  return parts.map((p, i) => (typeof p === "string" ? inline(p, i) : <mark className="bad" key={i}>{p.mark}</mark>));
}
function inline(t, key) {
  const bits = t.split(/(\*\*[^*]+\*\*)/g);
  return <React.Fragment key={key}>{bits.map((b, i) => (b.startsWith("**") ? <b key={i}>{b.slice(2, -2)}</b> : b))}</React.Fragment>;
}

function Preview({ text, findings }) {
  const snippets = findings.filter((f) => f.snippet).map((f) => f.snippet);
  const rows = text.split("\n").filter((l) => l.trim());
  return (
    <div className="annot">
      {rows.map((l, i) => /^\s*[-*]\s+/.test(l)
        ? <ul key={i} style={{ margin: "0 0 8px", paddingLeft: 20 }}><li>{highlight(l.replace(/^\s*[-*]\s+/, ""), snippets)}</li></ul>
        : <p key={i}>{highlight(l, snippets)}</p>)}
    </div>
  );
}

const PREVIEW = typeof __PREVIEW__ !== "undefined";

export default function Commentary({ q, meta, lines, findings, accepted, notes, acceptedCount, setToast, monthLabel, params, setView }) {
  const [mode, setMode] = useState("offline");
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);     // last server validation
  const [busy, setBusy] = useState(false);
  const [applied, setApplied] = useState([]);
  const [approved, setApproved] = useState(false);
  const [editing, setEditing] = useState(false);
  const seq = useRef(0);

  const body = useMemo(() => ({
    ...q,
    notes: lines.filter((l) => notes[lineKey(l)]?.trim()).map((l) => ({ department: l.department, account_code: l.account_code, text: notes[lineKey(l)] })),
    suggestions: lines.filter((l) => accepted[lineKey(l)] && findings[lineKey(l)]?.status === "supported")
      .map((l) => ({ department: l.department, account_code: l.account_code, text: findings[lineKey(l)].explanation, sources: findings[lineKey(l)].sources })),
  }), [q, lines, notes, accepted, findings]);

  const generate = async (plant) => {
    setBusy(true); setApproved(false);
    try {
      const r = await api.commentary({ ...body, mode, plant_errors: plant });
      setText(r.text); setResult(r); setApplied(r.applied); setEditing(false);
    } catch (e) { setToast(e.message); }
    setBusy(false);
  };

  // Live verification: any edit is re-checked against the data.
  useEffect(() => {
    if (!text) return;
    const my = ++seq.current;
    const t = setTimeout(() => {
      api.validate({ ...body, text }).then((r) => { if (my === seq.current) { setResult(r); setApproved(false); } }).catch(() => {});
    }, 400);
    return () => clearTimeout(t);
  }, [text]);

  // Reset when the month changes.
  useEffect(() => { setText(""); setResult(null); setApplied([]); setApproved(false); }, [params.month, params.dataset]);

  const fl = result?.findings || [];
  const stale = text && result && result.text !== text;
  const download = () => {
    const blob = new Blob([`# ${monthLabel(params.month)} variance commentary\n\n${text}\n`], { type: "text/markdown" });
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `variance-commentary-${params.month}.md`; a.click();
  };
  const copy = () => navigator.clipboard?.writeText(text).then(() => setToast("Copied to clipboard."));

  return (
    <div className="cm">
      <div className="stack">
        <div className="card">
          <div className="card-b row">
            <div className="seg">
              <button className={mode === "offline" ? "on" : ""} onClick={() => setMode("offline")}>Template writer</button>
              <button className={mode === "claude" ? "on" : ""} disabled={!meta?.claude_available} onClick={() => setMode("claude")}
                title={meta?.claude_available ? "" : "Needs an API key (Claude or Gemini) in the local app"}>AI writer</button>
            </div>
            <button className="btn primary" disabled={busy || !lines.length} onClick={() => generate(false)}>{busy ? <><Spinner /> Drafting</> : text ? "Redraft" : "Draft commentary"}</button>
            <span className="spacer" />
            <button className="btn small" disabled={!text || busy} onClick={() => generate(true)} title="Plants a wrong number, a flipped direction and an invented cause, so you can watch the verifier catch them">Plant 3 errors (demo)</button>
          </div>
          <div className="card-b" style={{ paddingTop: 0 }} >
            <span className="muted">Inputs: <b>{body.notes.length}</b> analyst note{body.notes.length === 1 ? "" : "s"}, <b>{body.suggestions.length}</b> accepted agent finding{body.suggestions.length === 1 ? "" : "s"}. Lines with neither are written as “driver not yet confirmed”.</span>
          </div>
        </div>

        {!text && !busy && (
          <div className="card"><div className="empty"><h4>No draft yet</h4>
            Press “Draft commentary”. {acceptedCount === 0 && <>Tip: run the investigation first and accept findings in <button className="link" onClick={() => setView("workspace")}>Variances</button>, so the draft can include source-backed reasons.</>}</div></div>
        )}

        {text && (
          <div className="card">
            <div className="card-h">
              <h3>{editing ? "Edit draft" : "Draft"}</h3>
              {result && !stale && (fl.length === 0 ? <Chip kind="ok" dot>All checks passed</Chip> : <Chip kind="bad" dot>{fl.length} problem{fl.length > 1 ? "s" : ""}</Chip>)}
              {stale && <Chip kind="gray"><Spinner /> checking</Chip>}
              <span className="spacer" />
              <button className="btn small" onClick={() => setEditing(!editing)}>{editing ? "Preview" : "Edit text"}</button>
              <button className="btn small" onClick={copy}>Copy</button>
              {!PREVIEW && <button className="btn small" onClick={download}>Download .md</button>}
            </div>
            <div className="card-b">
              {applied.length > 0 && <div className="callout warn" style={{ marginBottom: 12 }}><b>Demo:</b> errors were planted on purpose.<ul style={{ margin: "6px 0 0", paddingLeft: 18 }}>{applied.map((a, i) => <li key={i}>{a}</li>)}</ul></div>}
              {editing ? <textarea className="draft" value={text} onChange={(e) => setText(e.target.value)} /> : <Preview text={text} findings={fl} />}
            </div>
            <div className="card-h" style={{ borderTop: "1px solid var(--line)", borderBottom: 0 }}>
              <span className="muted">{approved ? "Approved for release." : fl.length ? "Fix the flagged problems before approving." : "A person signs off. The checker does not replace review."}</span>
              <span className="spacer" />
              <button className="btn primary" disabled={fl.length > 0 || stale || approved} onClick={() => setApproved(true)}>{approved ? "Approved" : "Approve commentary"}</button>
            </div>
          </div>
        )}
      </div>

      <div className="card" style={{ position: "sticky", top: 74 }}>
        <div className="card-h"><h3>Verifier</h3></div>
        <div className="card-b stack">
          <div className="muted">Checks every figure, direction and cause against the computed data. No AI is involved in this step.</div>
          {!result && <div className="muted">Waiting for a draft.</div>}
          {result && fl.length === 0 && <div className="callout ok"><b>Passed.</b> {result.facts_checked} variance lines checked: numbers match, directions match, no unsupported causes.</div>}
          {fl.map((f, i) => (
            <div className="finding" key={i}>
              <b>{f.kind}</b> · {f.where}<div>{f.detail}</div>
            </div>
          ))}
          <div className="muted" style={{ fontSize: 12.5 }}>Rules: figures must match; “over/under” and “favorable” must match; a cause needs an analyst note or an accepted source-backed finding; cited document ids must belong to that line.</div>
        </div>
      </div>
    </div>
  );
}
