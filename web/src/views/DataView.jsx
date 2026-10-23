import React from "react";
import { Chip } from "../ui.jsx";

export default function DataView({ pre, loading }) {
  const issues = pre?.issues || [];
  return (
    <>
      <div className="card" style={{ overflow: "hidden" }}>
        <div className="card-h"><h3>Ledger checks</h3>
          {pre && <><Chip kind={pre.blockers ? "bad" : "ok"} dot>{pre.blockers} blocking</Chip><Chip kind={pre.warnings ? "warn" : "ok"} dot>{pre.warnings} warnings</Chip><span className="muted">{pre.rows.toLocaleString()} rows</span></>}</div>
        {!pre && loading && <div className="card-b"><div className="skel" /></div>}
        {pre && issues.length === 0 && <div className="empty"><h4>No problems found</h4>The ledger passed every check.</div>}
        {issues.length > 0 && (
          <table>
            <thead><tr><th style={{ width: 110 }}>Severity</th><th style={{ width: 150 }}>Check</th><th>What was found</th><th style={{ width: 110 }}>Rows</th></tr></thead>
            <tbody>{issues.map((i, n) => (
              <tr key={n}><td><Chip kind={i.severity === "blocker" ? "bad" : "warn"} dot>{i.severity === "blocker" ? "Blocking" : "Warning"}</Chip></td>
                <td>{i.check.replace(/_/g, " ")}</td><td>{i.message}</td><td className="num muted">{i.rows?.length || "–"}</td></tr>
            ))}</tbody>
          </table>
        )}
        <div className="card-b muted" style={{ fontSize: 12.5 }}>Duplicates, unmapped accounts, flipped signs, gaps and spikes are caught before any variance is computed, because bad input makes every later number wrong.</div>
      </div>
    </>
  );
}
