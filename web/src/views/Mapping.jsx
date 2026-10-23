import React, { useEffect, useMemo, useState } from "react";
import { api } from "../api.js";
import { Chip, Spinner } from "../ui.jsx";

const KEEP = { decisions: {}, entity: null };      // survives switching tabs; lives only in this browser tab
const NONE = "NONE";
const pct = (a, b) => (b ? `${Math.round((a / b) * 100)}%` : "–");

function Seg({ value, onChange, options }) {
  return <div className="seg">{options.map(([v, l, dis]) => <button key={v} disabled={dis} className={value === v ? "on" : ""} onClick={() => onChange(v)}>{l}</button>)}</div>;
}

export default function Mapping({ meta, setToast }) {
  const [data, setData] = useState(null);
  const [entity, setEntity] = useState(KEEP.entity);
  const [agent, setAgent] = useState("baseline");
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);
  const [decisions, setDec] = useState(KEEP.decisions);
  const [filter, setFilter] = useState("all");
  const [ev, setEv] = useState(null);
  const claudeOk = meta?.claude_available;

  useEffect(() => { api.mappingData().then((d) => { setData(d); if (!entity) setEntity(Object.keys(d.entities)[0]); }).catch((e) => setToast(e.message)); api.mappingEval().then(setEv).catch(() => {}); }, []);
  useEffect(() => { KEEP.entity = entity; }, [entity]);
  const run = async (ent = entity, ag = agent) => {
    if (!ent) return;
    setBusy(true);
    try { setRes(await api.mappingRun({ entity: ent, agent: ag, rules: [] })); } catch (e) { setToast(e.message); } finally { setBusy(false); }
  };
  useEffect(() => { if (entity) run(entity, agent); }, [entity, agent]);

  const decide = (code, target) => { const d = { ...decisions, [`${entity}|${code}`]: target }; KEEP.decisions = d; setDec(d); };
  const undo = (code) => { const d = { ...decisions }; delete d[`${entity}|${code}`]; KEEP.decisions = d; setDec(d); };
  const std = data?.standard || [];
  const byCode = useMemo(() => Object.fromEntries(std.map((s) => [s.code, s])), [std]);

  const rows = (res?.rows || []).map((r) => {
    const d = decisions[`${entity}|${r.source_code}`];
    return d ? { ...r, decided: d, target_code: d === NONE ? null : d, target_name: d === NONE ? null : byCode[d]?.name, status: "rule" } : r;
  });
  const counts = { auto: rows.filter((r) => r.status === "auto").length, review: rows.filter((r) => r.status === "review").length, rule: rows.filter((r) => r.status === "rule").length };
  const shown = rows.filter((r) => filter === "all" || r.status === "review");

  return (
    <>
      <div className="callout">
        Every company names its accounts differently, so numbers can't be compared until each account is mapped to one standard chart.
        The mapper proposes a home for each account. Only confident proposals apply on their own; anything unsure, or with no good home, waits for you. When you approve, it becomes a rule.
      </div>

      <div className="card">
        <div className="card-b row" style={{ gap: 18 }}>
          <div className="row"><span className="muted">Entity</span>
            <Seg value={entity} onChange={setEntity} options={Object.keys(data?.entities || {}).map((k) => [k, k])} /></div>
          <div className="row"><span className="muted">Mapper</span>
            <Seg value={agent} onChange={setAgent} options={[["baseline", "Similarity baseline"], ["claude", claudeOk ? "AI mapper" : "AI mapper (needs key)", !claudeOk]]} /></div>
          <span className="spacer" />{busy && <Spinner />}
        </div>
      </div>

      <div className="kpis" style={{ gridTemplateColumns: "repeat(4, 1fr)" }}>
        {[["Accounts", rows.length, "in this entity"], ["Applied automatically", counts.auto, "confident enough to skip review"], ["Needs your review", counts.review, "unsure or no good home"], ["Approved by you", counts.rule, "now rules for the next run"]].map(([k, v, s]) => (
          <div className="card kpi" key={k}><div className="l">{k}</div><div className="v num">{v}</div><div className="muted" style={{ fontSize: 12.5 }}>{s}</div></div>))}
      </div>

      <div className="card" style={{ overflow: "hidden" }}>
        <div className="card-h"><h3>{entity || "…"}</h3><span className="muted">{res?.agent?.endsWith("-mapper") && res.agent !== "similarity-baseline" ? "AI mapper" : "similarity baseline, no model"}</span><span className="spacer" />
          <Seg value={filter} onChange={setFilter} options={[["all", "All"], ["review", "Needs review"]]} /></div>
        <div className="tblscroll"><table>
          <thead><tr><th style={{ width: 90 }}>Code</th><th>Source account</th><th style={{ width: 280 }}>Maps to (standard chart)</th><th style={{ width: 130 }}>Status</th><th>Why</th><th style={{ width: 90 }} /></tr></thead>
          <tbody>{shown.map((r) => (
            <tr key={r.source_code}>
              <td className="mono">{r.source_code}</td>
              <td><b>{r.source_name}</b> <span className="muted">{r.type}</span></td>
              <td><select style={{ maxWidth: 270 }} value={r.target_code || ""} onChange={(ev2) => decide(r.source_code, ev2.target.value || NONE)}>
                <option value="">No match, leave unmapped</option>
                {std.filter((s) => s.type === r.type).map((s) => <option key={s.code} value={s.code}>{s.code} · {s.name}</option>)}</select></td>
              <td>{r.status === "auto" ? <Chip kind="ok" dot>Auto-applied</Chip> : r.status === "rule" ? <Chip kind="info" dot>{r.decided === NONE ? "Left unmapped" : "Approved"}</Chip> : <Chip kind="warn" dot>Review{r.confidence === "low" ? " · low" : ""}</Chip>}</td>
              <td className="muted" style={{ fontSize: 12.5 }}>{r.reason}</td>
              <td className="r">{r.status === "rule" ? <button className="btn small" onClick={() => undo(r.source_code)}>Undo</button>
                : <button className="btn small" disabled={!r.target_code} onClick={() => decide(r.source_code, r.target_code)}>Approve</button>}</td>
            </tr>))}</tbody>
        </table></div>
        <div className="card-b muted" style={{ fontSize: 12.5 }}>
          Numbering differs between companies, so the mapper does not trust codes: Bayside's 5000 is hosting, not the standard 5000. Your decisions are kept in this browser tab only.
        </div>
      </div>

      {ev && (
        <div className="card" style={{ overflow: "hidden" }}>
          <div className="card-h"><h3>How good is it?</h3><span className="muted">{ev.accounts} accounts, answer key written by hand · {ev.no_home} have no home in the standard chart</span></div>
          <div className="tblscroll"><table>
            <thead><tr><th>Mapper</th><th className="r">Right proposal</th><th className="r">Applied automatically</th><th className="r">Auto-applied and right</th><th className="r">Wrong and applied</th><th className="r">Wrong, sent to review</th></tr></thead>
            <tbody>{ev.results.map((r) => (
              <tr key={r.mapper}><td><b>{r.mapper}</b>{r.model && <div className="muted" style={{ fontSize: 12 }}>{r.model} · {r.run}</div>}</td>
                <td className="r num">{r.correct}/{r.n} ({pct(r.correct, r.n)})</td><td className="r num">{r.auto}</td><td className="r num">{r.auto_correct}/{r.auto}</td>
                <td className="r num"><b>{r.wrong_auto}</b></td><td className="r num">{r.wrong_caught}/{r.wrong}</td></tr>))}</tbody>
          </table></div>
          <div className="tblscroll"><table>
            <thead><tr><th>Kind of account</th>{ev.results.map((r) => <th key={r.mapper} className="r">{r.mapper}</th>)}</tr></thead>
            <tbody>{Object.keys(ev.results[0].by_tag).map((k) => (
              <tr key={k}><td>{k}</td>{ev.results.map((r) => <td key={r.mapper} className="r num">{r.by_tag[k].correct}/{r.by_tag[k].n}</td>)}</tr>))}</tbody>
          </table></div>
          <div className="card-b muted" style={{ fontSize: 12.5 }}>
            “Wrong and applied” matters most: a wrong mapping nobody reviewed.
            {ev.results.some((r) => r.measured)
              ? " Scored on a small set of made-up accounts, so it shows the method works, not how it will do on real books."
              : " The AI mapper has not been scored yet."}
          </div>
        </div>
      )}
    </>
  );
}
