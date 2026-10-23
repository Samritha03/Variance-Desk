import React, { useEffect, useState } from "react";
import { api } from "./api.js";

export const lineKey = (l) => `${l.department}|${l.account_code}`;

export const Icon = ({ name, size = 17 }) => {
  const p = {
    overview: "M4 13h6V4H4v9zm0 7h6v-5H4v5zm9 0h7V11h-7v9zm0-16v5h7V4h-7z",
    lines: "M4 6h16M4 12h16M4 18h10",
    doc: "M7 3h7l5 5v13H7V3zm7 0v5h5M10 13h6M10 17h6",
    chart: "M4 20V10m6 10V4m6 16v-7m4 7H2",
    data: "M12 4c4.4 0 8 1.1 8 2.5S16.4 9 12 9 4 7.9 4 6.5 7.6 4 12 4zM4 6.5v5C4 12.9 7.6 14 12 14s8-1.1 8-2.5v-5M4 11.5v5C4 17.9 7.6 19 12 19s8-1.1 8-2.5v-5",
    check: "M5 12l5 5 9-10",
    search: "M11 4a7 7 0 100 14 7 7 0 000-14zm9 16l-4-4",
  }[name];
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d={p} />
    </svg>
  );
};

export const Chip = ({ kind = "gray", children, dot }) => (
  <span className={`chip ${kind}`}>{dot && <i className="dot" />}{children}</span>
);

export const STATUS = {
  supported: { label: "Supported", kind: "ok" },
  no_evidence: { label: "No evidence", kind: "info" },
  unverified: { label: "Could not verify", kind: "warn" },
};

export function Spinner() { return <span className="spin" />; }

export function Toast({ msg, kind = "bad", onClose }) {
  useEffect(() => { if (!msg) return; const t = setTimeout(onClose, 9000); return () => clearTimeout(t); }, [msg]);
  if (!msg) return null;
  return <div className="toast"><div className={`callout ${kind}`} onClick={onClose} role="alert">{msg}</div></div>;
}

// Modal showing a source document with the quoted passage highlighted.
export function DocModal({ id, quote, onClose }) {
  const [doc, setDoc] = useState(null);
  const [err, setErr] = useState("");
  useEffect(() => {
    if (!id) return;
    setDoc(null); setErr("");
    if (id.startsWith("T")) { setDoc({ id, title: "Ledger transaction", type: "ledger", date: "", text: "This is a row in the general-ledger transaction file. See the investigation trace for its vendor, amount and reference." }); return; }
    api.document(id).then(setDoc).catch((e) => setErr(e.message));
  }, [id]);
  if (!id) return null;
  const idx = doc && quote ? doc.text.toLowerCase().indexOf(quote.toLowerCase()) : -1;
  return (
    <div className="scrim" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="card-h">
          <h3>{doc ? `${doc.id} · ${doc.title}` : id}</h3>
          {doc && <Chip kind="gray">{doc.type}</Chip>}
          {doc?.date && <span className="muted">{doc.date}</span>}
          <span className="spacer" />
          <button className="btn small" onClick={onClose}>Close</button>
        </div>
        <div className="card-b">
          {err && <div className="callout bad">{err}</div>}
          {!doc && !err && <div className="skel" />}
          {doc && (idx >= 0 ? (
            <pre>{doc.text.slice(0, idx)}<mark>{doc.text.slice(idx, idx + quote.length)}</mark>{doc.text.slice(idx + quote.length)}</pre>
          ) : <pre>{doc.text}</pre>)}
        </div>
      </div>
    </div>
  );
}

export const money = (n) => (n == null ? "–" : (n < 0 ? "-" : "") + "$" + Math.abs(n).toLocaleString("en-US", { maximumFractionDigits: 0 }));
