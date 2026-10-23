import io

from fastapi.testclient import TestClient

from gve import api as api_mod
from gve.api import app
from gve.ingest import find_dates, parse_document

import pytest

H = {"X-GVE-Session": "test-a"}
client = TestClient(app)


def _up(name, body, headers=H, **form):
    return client.post("/api/documents", files={"file": (name, io.BytesIO(body))}, data=form, headers=headers)


@pytest.fixture(autouse=True)
def clean():
    api_mod._sessions.clear()
    yield
    api_mod._sessions.clear()


def test_date_reading():
    assert find_dates("Due 10/05/2026, sent 3 Sept 2026, renewed 2026-02-30") == ["2026-10-05", "2026-09-03"]
    d = parse_document("m.txt", b"From: A\nSubject: Retainer step-up\nDate: September 3, 2026\n\nBody")
    assert (d["date"], d["type"], d["title"], d["date_source"]) == ("2026-09-03", "email", "Retainer step-up", "Date: line")


def test_undated_file_is_refused_not_guessed():
    with pytest.raises(ValueError, match="no date found"):
        parse_document("n.txt", b"A note with no date at all.")
    assert _up("n.txt", b"A note with no date at all.").status_code == 422
    assert _up("n.txt", b"A note with no date at all.", date="2026-09-10").status_code == 200


def test_bad_files_get_plain_messages():
    assert "supported" in _up("x.exe", b"MZ").json()["detail"]
    assert "empty" in _up("e.txt", b"   ").json()["detail"]
    assert "date must look like" in _up("e.txt", b"hello", date="Sept 3").json()["detail"]


def test_upload_is_searchable_and_scoped_to_the_visitor():
    body = b"Subject: Zebrafish permit fee\nDate: 2026-09-05\n\nThe quarterly zebrafish permit fee rose to $12,400."
    r = _up("p.txt", body)
    assert r.status_code == 200 and r.json()["id"] == "D9001" and r.json()["source"] == "uploaded"
    q = {"query": "zebrafish permit fee", "month": "2026-09"}
    hits = client.post("/api/search", json=q, headers=H).json()["hits"]
    assert hits and hits[0]["doc_id"] == "D9001" and hits[0]["uploaded"]
    other = client.post("/api/search", json=q, headers={"X-GVE-Session": "test-b"}).json()["hits"]
    assert all(h["doc_id"] != "D9001" for h in other)
    assert client.get("/api/documents/D9001", headers=H).status_code == 200
    assert client.get("/api/documents/D9001", headers={"X-GVE-Session": "test-b"}).status_code == 404


def test_date_filter_applies_to_uploads():
    _up("old.txt", b"Subject: Zebrafish permit fee\n\nzebrafish permit fee", date="2025-01-10")
    hits = client.post("/api/search", json={"query": "zebrafish permit", "month": "2026-09"}, headers=H).json()["hits"]
    assert all(h["doc_id"] != "D9001" for h in hits)
    hits = client.post("/api/search", json={"query": "zebrafish permit", "month": "2026-09", "date_filter": False}, headers=H).json()["hits"]
    assert hits[0]["doc_id"] == "D9001"


def test_list_and_remove():
    _up("a.txt", b"hello world note", date="2026-09-01")
    rows = client.get("/api/documents", headers=H).json()
    assert rows["uploaded"] == 1 and any(d["source"] == "sample" for d in rows["documents"])
    assert client.delete("/api/documents/D01", headers=H).status_code == 404     # sample docs are not removable
    assert client.delete("/api/documents/D9001", headers=H).status_code == 200
    assert client.get("/api/documents", headers=H).json()["uploaded"] == 0


def test_cap_per_visitor(monkeypatch):
    monkeypatch.setattr(api_mod, "MAX_USER_DOCS", 2)
    for i in range(2):
        assert _up(f"{i}.txt", b"note text here", date="2026-09-01").status_code == 200
    assert "Limit" in _up("3.txt", b"note text here", date="2026-09-01").json()["detail"]


def test_pdf_text_is_extracted():
    from reportlab.pdfgen import canvas
    buf = io.BytesIO(); c = canvas.Canvas(buf); c.drawString(72, 750, "Invoice 2026-09-12 Quartz Audit LLP fee $9,100"); c.save()
    r = _up("inv.pdf", buf.getvalue())
    assert r.status_code == 200, r.text
    assert r.json()["date"] == "2026-09-12" and r.json()["type"] == "invoice"


def test_uploaded_document_changes_what_the_agent_finds():
    """A line with no documented cause becomes explainable once the right document is added."""
    import json
    month = "2026-09"
    base = {"dataset": "sample", "month": month, "agent": "baseline", "retrieval": "hybrid"}
    def run(headers):
        out = {}
        with client.stream("POST", "/api/investigate", json=base, headers=headers) as s:
            for line in s.iter_lines():
                e = json.loads(line)
                if e["type"] == "progress":
                    out[e["finding"]["line_key"]] = e["finding"]
        return out
    before = run(H)
    assert before["G&A|6800"]["status"] == "no_evidence"
    doc = (b"From: Elena Ortiz, CFO\nTo: Finance\nSubject: G&A professional fees, September\nDate: 2026-09-08\n\n"
           b"Professional fees for G&A ran over budget in September because outside counsel billed the "
           b"Harborline lease renegotiation, about $34,000, which was not budgeted.")
    assert _up("fees.txt", doc).status_code == 200
    after = run(H)["G&A|6800"]
    assert after["status"] == "supported" and "D9001" in after["sources"]
    assert run({"X-GVE-Session": "test-b"})["G&A|6800"]["status"] == "no_evidence"   # other visitors unaffected
