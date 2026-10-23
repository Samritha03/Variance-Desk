import React from "react";
import { Chip, lineKey } from "../ui.jsx";

function Kpi({ label, t, good, pl }) {
  if (!t) return <div className="card kpi"><div className="skel" style={{ width: 90 }} /><div className="skel" style={{ height: 30, marginTop: 10 }} /></div>;
  const fav = good === "up" ? t.variance >= 0 : t.variance <= 0;
  return (
    <div className="card kpi">
      <div className="l">{label}{pl && <span className={t.actual < 0 ? "neg" : "pos"} style={{ fontWeight: 700 }}> ({t.actual < 0 ? "Loss" : "Profit"})</span>}</div>
      <div className="v num">{t.actual_str}</div>
      {!pl && (<>
      <div className={`d num ${fav ? "pos" : "neg"}`}>{t.variance_str} ({t.variance_pct_str}) {fav ? "better" : "worse"} than budget</div>
      <div className="muted num" style={{ fontSize: 12.5, marginTop: 2 }}>Budget {t.budget < 0 ? `(${t.budget_str})` : t.budget_str}</div>
      </>)}
    </div>
  );
}

export default function Overview({ data, pre, lines, findings, accepted, acceptedCount, setView, monthLabel, params, meta, loading, visited }) {
  const t = data?.totals;
  const maxV = Math.max(1, ...lines.map((l) => Math.abs(l.variance)));
  const found = Object.keys(findings).length;
  const supported = Object.values(findings).filter((f) => f.status === "supported").length;
  const canInvestigate = data?.investigation_supported;
  const steps = [
    { n: 1, phase: 0, title: "Add documents", done: !!visited?.documents, view: "documents",
      body: "Add the invoices, emails and notes the agent will search, for any month. The documents are searchable.", cta: "Add documents" },
    { n: 2, phase: 0, title: "Map the accounts", done: !!visited?.mapping, view: "mapping",
      body: "Each company names accounts differently. The mapper proposes where each belongs in one standard chart; you approve.", cta: "Open mapping" },
    { n: 3, phase: 0, title: "Check the data", done: !!visited?.data, view: "data",
      body: "Scans the ledger for problems like unmapped spend or missing rows, so you know the numbers can be trusted.",
      tag: pre && pre.blockers + pre.warnings > 0 ? <Chip kind={pre.blockers ? "bad" : "warn"} dot>{pre.blockers + pre.warnings} issue{pre.blockers + pre.warnings === 1 ? "" : "s"} found</Chip> : null,
      cta: "Review checks" },
    { n: 4, phase: 1, title: "Find the reasons, accept what you trust", done: found > 0 && found === lines.length && acceptedCount > 0, view: "workspace",
      body: "The agent searches your documents for why each big variance moved and quotes its sources. You accept or reject each answer.",
      tag: lines.length > 0 ? <Chip kind="info">{found ? `${found} of ${lines.length} investigated` : `${lines.length} to review`}</Chip> : null,
      cta: found ? "Open results" : "Start" },
    { n: 5, phase: 2, title: "Draft and verify", done: !!visited?.commentary, view: "commentary",
      body: "A model or template writes the words from the explanations you accepted. Every number is then checked against the data.", cta: "Open commentary" },
  ];
  const phases = [["Prepare", "Get the data and documents ready"], ["Investigate", "The agent searches, you decide"], ["Report", "Write it up and check every figure"]];
  return (
    <>
      <div className="callout">
        Every month, finance teams explain why actuals differ from budget. Here the <b>code computes every number</b>, an <b>agent finds the reason</b> in documents and must <b>quote its sources</b>, and a <b>verifier</b> checks the final write-up before a person signs off.
      </div>
      <div className="kpis">
        <Kpi label={`Revenue · ${data ? data.label : ""}`} t={t?.revenue} good="up" />
        <Kpi label="Expenses" t={t?.expenses} good="down" />
        <Kpi label="Operating income" t={t?.operating_income} good="up" pl />
      </div>
      <div className="card">
        <div className="card-h noline"><h3>Your workflow for {data?.label || "…"}</h3></div>
        <div className="card-b phases">
          {phases.map(([name, sub], pi) => (
            <div key={name} className="phase">
              <div className="phase-h"><b>{pi + 1}. {name}</b><span className="muted">{sub}</span></div>
              {steps.filter((s) => s.phase === pi).map((s) => (
                <div key={s.n} className="stack stepcard">
                  <div className="row"><Chip kind={s.done ? "ok" : "gray"}>{s.done ? "Done" : `Step ${s.n}`}</Chip>{s.tag}</div>
                  <b>{s.title}</b>
                  <div className="muted">{s.body}</div>
                  <div><button className="btn small" onClick={() => setView(s.view)}>{s.cta}</button></div>
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
      <div className="card" style={{ overflow: "hidden" }}>
        <div className="card-h"><h3>Material variances</h3><span className="muted">ranked by size</span><span className="spacer" /><Chip kind="gray">{lines.length} material variance{lines.length === 1 ? "" : "s"}</Chip></div>
        <div className="tblscroll"><table>
          <thead><tr><th>Line</th><th className="r">Budget</th><th className="r">Actual</th><th className="r">Variance</th><th>Share of largest</th><th>Reason</th></tr></thead>
          <tbody>
            {loading && !lines.length && <tr><td colSpan="6"><div className="skel" /></td></tr>}
            {lines.map((l) => {
              const f = findings[lineKey(l)];
              return (
                <tr key={l.id}>
                  <td><b>{l.department}</b> / {l.account_name}</td>
                  <td className="r num">{l.budget_str}</td><td className="r num">{l.actual_str}</td>
                  <td className={`r num ${l.favorable ? "pos" : "neg"}`}><b>{l.variance_str}</b> <span className="muted">({l.variance_pct_str})</span></td>
                  <td><div className="vbar"><i className={`${l.variance >= 0 ? "pos" : "neg"} ${l.favorable ? "f" : "u"}`} style={{ width: `${(Math.abs(l.variance) / maxV) * 50}%` }} /></div></td>
                  <td>{f ? <Chip kind={f.status === "supported" ? "ok" : f.status === "no_evidence" ? "info" : "warn"} dot>{f.status === "supported" ? "Explained" : f.status === "no_evidence" ? "No evidence" : "Unverified"}</Chip> : <span className="muted">Not investigated</span>}</td>
                </tr>
              );
            })}
            {!loading && !lines.length && <tr><td colSpan="6" className="empty">No material variances at these thresholds.</td></tr>}
          </tbody>
        </table></div>
      </div>
    </>
  );
}
