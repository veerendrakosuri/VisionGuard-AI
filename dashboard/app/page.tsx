"use client";

import { FormEvent, useCallback, useEffect, useState } from "react";

const API = "/api/backend";

type Inspection = {
  inspection_id: string;
  filename: string;
  decision: "PASS" | "FAIL";
  anomaly_score: number;
  inference_time_ms: number;
  threshold_source: string;
  heatmap_url: string;
  created_at: string;
};

type Summary = {
  total_inspections: number;
  pass_count: number;
  fail_count: number;
  fail_rate: number;
  average_inference_time_ms: number;
};

export default function Dashboard() {
  const [items, setItems] = useState<Inspection[]>([]);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("Connecting to inspection service…");

  const refresh = useCallback(async () => {
    try {
      const [history, metrics] = await Promise.all([
        fetch(`${API}/api/v1/inspections?limit=20`, { cache: "no-store" }),
        fetch(`${API}/api/v1/inspections/analytics/summary`, { cache: "no-store" }),
      ]);
      if (!history.ok || !metrics.ok) throw new Error("API unavailable");
      setItems((await history.json()).items);
      setSummary(await metrics.json());
      setMessage("Live · model decisions remain provisional");
    } catch {
      setMessage("API unavailable — start VisionGuard on port 8000");
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  async function inspect(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const input = event.currentTarget.elements.namedItem("image") as HTMLInputElement;
    if (!input.files?.[0]) return;
    setBusy(true);
    setMessage("Inspecting image…");
    const body = new FormData();
    body.append("image", input.files[0]);
    const response = await fetch(`${API}/api/v1/inspections`, { method: "POST", body });
    setMessage(response.ok ? "Inspection completed" : "Inspection failed");
    setBusy(false);
    if (response.ok) {
      event.currentTarget.reset();
      await refresh();
    }
  }

  return (
    <main>
      <header>
        <div><span className="eyebrow">INDUSTRIAL VISION OPERATIONS</span><h1>VisionGuard AI</h1></div>
        <span className="status">{message}</span>
      </header>

      <section className="hero">
        <div><p className="kicker">Inspection control</p><h2>See defects before they ship.</h2>
          <p>Submit a sheet-metal image, review the anomaly decision, and track line quality.</p></div>
        <form onSubmit={inspect}><input name="image" type="file" accept="image/png,image/jpeg" required />
          <button disabled={busy}>{busy ? "Inspecting…" : "Run inspection"}</button></form>
      </section>

      <section className="metrics">
        <article><span>Total inspections</span><strong>{summary?.total_inspections ?? "—"}</strong></article>
        <article><span>Passed</span><strong>{summary?.pass_count ?? "—"}</strong></article>
        <article><span>Failed</span><strong className="danger">{summary?.fail_count ?? "—"}</strong></article>
        <article><span>Failure rate</span><strong>{summary ? `${(summary.fail_rate * 100).toFixed(1)}%` : "—"}</strong></article>
        <article><span>Average latency</span><strong>{summary ? `${summary.average_inference_time_ms} ms` : "—"}</strong></article>
      </section>

      <section className="history"><div className="section-title"><div><p className="kicker">Recent activity</p><h2>Inspection history</h2></div><button className="quiet" onClick={() => void refresh()}>Refresh</button></div>
        <div className="table-wrap"><table><thead><tr><th>Result</th><th>File</th><th>Score</th><th>Latency</th><th>Time</th><th>Heatmap</th></tr></thead>
          <tbody>{items.length === 0 ? <tr><td colSpan={6} className="empty">No inspections recorded yet.</td></tr> : items.map((item) => <tr key={item.inspection_id}>
            <td><span className={`badge ${item.decision.toLowerCase()}`}>{item.decision}</span></td><td>{item.filename}</td><td>{item.anomaly_score.toFixed(4)}</td><td>{item.inference_time_ms.toFixed(0)} ms</td><td>{new Date(item.created_at).toLocaleString()}</td><td><a href={`${API}${item.heatmap_url}`} target="_blank">View</a></td>
          </tr>)}</tbody></table></div>
      </section>
      <footer>PatchCore · sheet_metal · Decisions are provisional until threshold calibration.</footer>
    </main>
  );
}
