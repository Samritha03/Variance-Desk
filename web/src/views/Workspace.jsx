import React, { useEffect, useState } from "react";
import { api } from "../api.js";
import { Chip, DocModal, STATUS, Spinner, lineKey } from "../ui.jsx";

function Seg({ value, onChange, options, disabled }) {
  return (
    <div className="seg">
      {options.map(([v, label, dis]) => (
        <button key={v} className={value === v ? "on" : ""} disabled={disabled || dis} onClick={() => onChange(v)}>{label}</button>
      ))}
    </div>
  );
}

function Detail({ line, f, accepted, setAccepted, notes, setNotes, openDoc, ask }) {
  const k = lineKey(line);
  return (
    <div className="stack">
      <div className="card">
        <div className="card-h">
          <h3>{line.department} / {line.account_name}</h3>
          <Chip kind={line.favorable ? "ok" : "bad"}>{line.favorable ? "Favorable" : "Unfavorable"}</Chip>
        </div>
        <div className="card-b numbers">
          <div><small>Budget</small><b className="num">{line.budget_str}</b></div>
          <div><small>Actual</small><b className="num">{line.actual_str}</b></div>
          <div><small>Variance</small><b className={`num ${line.favorable ? "pos" : "neg"}`}>{line.variance_str}</b></div>
          <div><small>% of budget</small><b className="num">{line.variance_pct_str}</b></div>
        </div>
      </div>

      {!f && (
        <div className="card"><div className="empty"><h4>Not investigated yet</h4>Run the investigation to have the agent look for the reason in invoices, emails and ledger entries.</div></div>
      )}

      {f && (
        <>
          <div className={`verdict ${f.status}`}>
            <h4><Chip kind={STATUS[f.status].kind} dot>{STATUS[f.status].label}</Chip>
              <span className="muted" style={{ fontWeight: 400 }}>confidence: {f.confidence}</span></h4>
            <div>{f.explanation}</div>
            {f.agent === "keyword-baseline" && f.status === "supported" && <div className="muted" style={{ marginTop: 8, fontSize: 12.5 }}>Baseline: quotes the best-matching passage and does not reason about whether it explains the amount.</div>}
            {f.problems.length > 0 && <div className="callout warn" style={{ marginTop: 10 }}><b>Why it was not verified:</b> {f.problems.join(" ")}</div>}
          </div>

          {f.evidence.length > 0 && (
            <div className="card">
              <div className="card-h"><h3>Evidence</h3><span className="muted">quotes are checked against what the agent retrieved</span></div>
              <div className="card-b stack">
                {f.evidence.map((e, i) => (
                  <div className="ev" key={i}>
                    <div className="row"><button className="link" onClick={() => openDoc(e.source_id, e.quote)}>{e.source_id}</button>
                      <span className="muted">{e.source_id.startsWith("T") ? "ledger transaction" : "open the source to see the quote in context"}</span></div>
                    {e.quote && <blockquote>“{e.quote}”</blockquote>}
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="card">
            <div className="card-h"><h3>What the agent did</h3><Chip kind="gray">{f.trace.length} tool call{f.trace.length === 1 ? "" : "s"}</Chip></div>
            <div className="card-b trace">
              {f.trace.length === 0 && <span className="muted">No tool calls recorded.</span>}
              {f.trace.map((t, i) => (
                <div className="step" key={i}>
                  <div className="n">{i + 1}</div>
                  <div><code>{t.tool}({Object.entries(t.args).map(([a, b]) => `${a}=${JSON.stringify(b)}`).join(", ")})</code>
                    <div className="o">→ {t.summary}</div></div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}

      {ask}
      <div className="card">
        <div className="card-h"><h3>Your decision</h3></div>
        <div className="card-b stack">
          {f?.status === "supported" ? (
            <label className="check">
              <input type="checkbox" checked={!!accepted[k]} onChange={(e) => setAccepted({ ...accepted, [k]: e.target.checked })} />
              Use this explanation in the commentary
            </label>
          ) : <span className="muted">Only a verified, source-backed explanation can be accepted.</span>}
          <div className="stack" style={{ gap: 6 }}>
            <label className="muted" style={{ fontSize: 12.5 }}>Or write your own business reason (always overrides the agent)</label>
            <input type="text" placeholder="e.g. Fall trade show landed in September" value={notes[k] || ""}
              onChange={(e) => setNotes({ ...notes, [k]: e.target.value })} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default function Workspace(p) {
  const { lines, findings, progress, inv, setInv, investigate, meta, data, loading, accepted, setAccepted, notes, setNotes, runMeta } = p;
  const [sel, setSel] = useState(null);
  const [doc, setDoc] = useState(null);
  useEffect(() => { if (lines.length && !lines.some((l) => lineKey(l) === sel)) setSel(lineKey(lines[0])); }, [lines]);
  const line = lines.find((l) => lineKey(l) === sel);
  const running = !!progress;
  const supportedOnly = data?.investigation_supported;
  const claudeOk = meta?.claude_available;

  return (
    <>
      <div className="card">
        <div className="card-b row" style={{ gap: 18 }}>
          <div className="row"><span className="muted">Agent</span>
            <Seg value={inv.agent} onChange={(v) => setInv({ ...inv, agent: v })} disabled={running}
              options={[["baseline", "Keyword baseline"], ["claude", claudeOk ? "AI agent" : "AI agent (needs key)", !claudeOk]]} /></div>
          <div className="row"><span className="muted">Search</span>
            <Seg value={inv.retrieval} onChange={(v) => setInv({ ...inv, retrieval: v })} disabled={running}
              options={[["bm25", "Keyword"], ["dense", "Embeddings"], ["hybrid", "Hybrid"]]} /></div>
          <label className="check" style={{ fontWeight: 500 }}>
            <input type="checkbox" checked={inv.date_filter} disabled={running} onChange={(e) => setInv({ ...inv, date_filter: e.target.checked })} />
            Only documents near the month</label>
          <span className="spacer" />
          <button className="btn primary" disabled={running || !lines.length || !supportedOnly} onClick={investigate}>
            {running ? <><Spinner /> Investigating {progress.done}/{progress.total}</> : Object.keys(findings).length ? "Re-run investigation" : "Investigate variances"}
          </button>
        </div>
        {running && <div className="progress" style={{ margin: "0 18px 14px" }}><i style={{ width: `${(progress.done / Math.max(progress.total, 1)) * 100}%` }} /></div>}
        {!supportedOnly && <div className="card-b" style={{ paddingTop: 0 }}><div className="callout warn">Investigation needs a documents folder and a transaction ledger. Only the sample company has them. You can still write notes and draft commentary.</div></div>}
        {runMeta && !running && <div className="card-b muted" style={{ paddingTop: 0, fontSize: 12.5 }}>
          Last run: {runMeta.agent === "claude" ? "AI agent" : "keyword baseline"} · {runMeta.retrieval} search · date filter {runMeta.date_filter ? "on" : "off"}</div>}
      </div>

      <div className="ws">
        <div className="card lines">
          <div className="card-h"><h3>Material variances</h3><span className="muted">{lines.length}</span></div>
          {loading && !lines.length && <div className="card-b"><div className="skel" /></div>}
          {!loading && !lines.length && <div className="empty"><h4>Nothing material</h4>Lower the thresholds in the top bar.</div>}
          {lines.map((l) => {
            const f = findings[lineKey(l)];
            return (
              <div key={l.id} className={`line ${sel === lineKey(l) ? "on" : ""}`} onClick={() => setSel(lineKey(l))}>
                <div><div className="t">{l.department} / {l.account_name}</div>
                  <div className="s">{l.actual_str} vs {l.budget_str} budget</div></div>
                <div className="r num"><span className={l.favorable ? "pos" : "neg"}>{l.variance_str}</span>
                  <div className="s muted" style={{ fontWeight: 400 }}>{l.variance_pct_str}</div></div>
                <div style={{ gridColumn: "1 / -1" }} className="row">
                  {f ? <Chip kind={STATUS[f.status].kind} dot>{STATUS[f.status].label}</Chip> : <Chip kind="gray">Not investigated</Chip>}
                  {accepted[lineKey(l)] && <Chip kind="ok">Accepted</Chip>}
                  {notes[lineKey(l)]?.trim() && <Chip kind="info">Note added</Chip>}
                </div>
              </div>
            );
          })}
        </div>
        <div>{line ? <Detail line={line} f={findings[lineKey(line)]} accepted={accepted} setAccepted={setAccepted}
          notes={notes} setNotes={setNotes} openDoc={(id, quote) => setDoc({ id, quote })}
          /> : null}</div>
      </div>
      <DocModal id={doc?.id} quote={doc?.quote} onClose={() => setDoc(null)} />
    </>
  );
}
