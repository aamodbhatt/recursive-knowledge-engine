import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "/api";
const ACCEPTED_DOC_TYPES = [
  ".pdf",
  ".docx",
  ".txt",
  ".md",
  ".markdown",
  ".rst",
  ".csv",
  ".tsv",
  ".json",
  ".jsonl",
  ".yaml",
  ".yml",
  ".xml",
  ".html",
  ".htm",
  ".log",
  ".ini",
  ".cfg",
  ".conf",
  ".toml",
  ".env",
  ".rtf",
  ".sql",
  ".sh",
  ".bash",
  ".zsh",
  ".py",
  ".js",
  ".ts",
  ".jsx",
  ".tsx",
  ".java",
  ".go",
  ".rs",
  ".c",
  ".cpp",
  ".h",
  ".hpp",
];

function extensionOf(filename) {
  const parts = String(filename || "").toLowerCase().split(".");
  return parts.length > 1 ? `.${parts.pop()}` : "";
}

function attachmentKind(filename) {
  const ext = extensionOf(filename);
  if (ext === ".pdf") return { kind: "pdf", icon: "PDF" };
  if (ext === ".docx" || ext === ".doc" || ext === ".rtf") return { kind: "docx", icon: "DOC" };
  if (ext === ".txt" || ext === ".md" || ext === ".markdown" || ext === ".rst") {
    return { kind: "text", icon: "TXT" };
  }
  if (ext === ".csv" || ext === ".tsv") return { kind: "table", icon: "CSV" };
  if (ext === ".json" || ext === ".jsonl" || ext === ".yaml" || ext === ".yml") {
    return { kind: "data", icon: "DATA" };
  }
  if (ext === ".html" || ext === ".htm" || ext === ".xml") return { kind: "markup", icon: "HTML" };
  if (
    [
      ".py",
      ".js",
      ".ts",
      ".jsx",
      ".tsx",
      ".java",
      ".go",
      ".rs",
      ".c",
      ".cpp",
      ".h",
      ".hpp",
      ".sql",
      ".sh",
      ".bash",
      ".zsh",
    ].includes(ext)
  ) {
    return { kind: "code", icon: "CODE" };
  }
  return { kind: "file", icon: "FILE" };
}

function getSessionId() {
  const key = "rke_session_id";
  const existing = window.localStorage.getItem(key);
  if (existing) return existing;
  const sid = crypto.randomUUID();
  window.localStorage.setItem(key, sid);
  return sid;
}

function formatDuration(ms) {
  if (!Number.isFinite(ms)) return "-";
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

function formatPercent(value) {
  const safe = Math.max(0, Math.min(1, Number(value || 0)));
  return `${Math.round(safe * 100)}%`;
}

function formatBytes(bytes) {
  const size = Number(bytes || 0);
  if (size < 1024) return `${size} B`;
  if (size < 1024 ** 2) return `${(size / 1024).toFixed(1)} KB`;
  if (size < 1024 ** 3) return `${(size / 1024 ** 2).toFixed(1)} MB`;
  return `${(size / 1024 ** 3).toFixed(1)} GB`;
}

function parseWords(text) {
  return (text.toLowerCase().match(/[a-z0-9_-]+/g) || []).filter(Boolean);
}

function buildTokenDiff(previousText, currentText) {
  const tokenRegex = /[A-Za-z0-9_-]+|[^\sA-Za-z0-9_-]+/g;
  const previousWords = parseWords(previousText);
  const previousWordSet = new Set(previousWords);
  const currentWordSet = new Set(parseWords(currentText));

  const annotatedCurrent = (currentText.match(tokenRegex) || []).map((token) => {
    if (!/[A-Za-z0-9_-]/.test(token)) return { token, type: "neutral" };
    return {
      token,
      type: previousWordSet.has(token.toLowerCase()) ? "same" : "added",
    };
  });

  const removedWords = [...new Set(previousWords.filter((word) => !currentWordSet.has(word)))];

  return { annotatedCurrent, removedWords };
}

function MarkdownBlock({ text }) {
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]} className="markdown-body">
      {text || ""}
    </ReactMarkdown>
  );
}

function Toggle({ label, checked, onChange }) {
  return (
    <label className="toggle-chip">
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      <span>{label}</span>
    </label>
  );
}

function Skeleton({ lines = 4 }) {
  return (
    <div className="skeleton-block" aria-hidden="true">
      {Array.from({ length: lines }).map((_, idx) => (
        <div key={idx} className="skeleton-line" />
      ))}
    </div>
  );
}

function ConfidenceChart({ loops }) {
  if (!loops?.length) {
    return <p className="empty-note">Run a query to see confidence calibration.</p>;
  }

  const values = loops.map((loop) => Math.max(0, Math.min(1, Number(loop.confidence || 0))));
  const width = 340;
  const height = 120;
  const pad = 16;
  const step = values.length === 1 ? 0 : (width - pad * 2) / (values.length - 1);

  const points = values.map((value, idx) => {
    const x = pad + idx * step;
    const y = height - pad - value * (height - pad * 2);
    return { x, y };
  });

  const polylinePoints = points.map((p) => `${p.x},${p.y}`).join(" ");

  return (
    <div className="confidence-chart">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Confidence by iteration">
        <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} className="chart-axis" />
        <line x1={pad} y1={pad} x2={pad} y2={height - pad} className="chart-axis" />
        <polyline points={polylinePoints} className="chart-line" />
        {points.map((point, idx) => {
          const decreased = idx > 0 && values[idx] < values[idx - 1];
          return (
            <g key={`${point.x}-${point.y}`}>
              <circle cx={point.x} cy={point.y} r="4" className={decreased ? "chart-point down" : "chart-point"} />
              <text x={point.x} y={height - 4} className="chart-label" textAnchor="middle">
                {idx + 1}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function PaperclipIcon() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
      <path
        d="M8.5 12.5 14.7 6.3a3.5 3.5 0 1 1 5 5l-8.4 8.4a5.5 5.5 0 0 1-7.8-7.8l8.2-8.2"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function ChatIcon() {
  return (
    <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true">
      <path
        d="M5 6.5h14a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2H9l-4 3v-3H5a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2Z"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function ChartIcon() {
  return (
    <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true">
      <path
        d="M4 19.5h16M7 16V9m5 7V6m5 10v-4"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function RefreshIcon() {
  return (
    <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">
      <path
        d="M20 12a8 8 0 1 1-2.35-5.66M20 4v4h-4"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">
      <path
        d="M20 14.8A8.5 8.5 0 1 1 9.2 4a7 7 0 1 0 10.8 10.8Z"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function SunIcon() {
  return (
    <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">
      <circle cx="12" cy="12" r="3.5" fill="none" stroke="currentColor" strokeWidth="1.8" />
      <path
        d="M12 3v2.2M12 18.8V21M3 12h2.2M18.8 12H21M5.6 5.6 7.2 7.2M16.8 16.8l1.6 1.6M18.4 5.6 16.8 7.2M7.2 16.8 5.6 18.4"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
    </svg>
  );
}

function SparkIcon() {
  return (
    <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">
      <path
        d="m12 3 1.8 4.4L18 9.2l-4.2 1.8L12 15l-1.8-4L6 9.2l4.2-1.8L12 3Z"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function getInitialTheme() {
  const saved = window.localStorage.getItem("rke_theme");
  if (saved === "light" || saved === "dark") return saved;
  return window.matchMedia?.("(prefers-color-scheme: light)")?.matches ? "light" : "dark";
}

export default function App() {
  const sessionId = useMemo(() => getSessionId(), []);
  const uploadInputRef = useRef(null);
  const threadViewportRef = useRef(null);
  const [theme, setTheme] = useState(() => getInitialTheme());
  const [activeInfoTab, setActiveInfoTab] = useState("workspace");

  const [health, setHealth] = useState(null);
  const [history, setHistory] = useState([]);
  const [historySummary, setHistorySummary] = useState({ total_queries: 0, estimated_tokens_used: 0 });
  const [failureLogs, setFailureLogs] = useState([]);

  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(5);
  const [depth, setDepth] = useState(2);
  const [ablationMode, setAblationMode] = useState(false);
  const [ablationDepths, setAblationDepths] = useState([1, 2, 3]);
  const [fastMode, setFastMode] = useState(true);
  const [deterministicMode, setDeterministicMode] = useState(false);
  const [answerVerbosity, setAnswerVerbosity] = useState("normal");
  const [challengeMode, setChallengeMode] = useState(true);
  const [focusMode, setFocusMode] = useState(false);
  const [showTuning, setShowTuning] = useState(false);

  const [result, setResult] = useState(null);
  const [activeAblationDepth, setActiveAblationDepth] = useState(null);

  const [running, setRunning] = useState(false);
  const [loadingHealth, setLoadingHealth] = useState(true);
  const [uploadStatus, setUploadStatus] = useState("No file uploaded yet.");
  const [queryStatus, setQueryStatus] = useState("Ready.");
  const [attachments, setAttachments] = useState([]);
  const [threadMessages, setThreadMessages] = useState([]);
  const [feedbackByResponse, setFeedbackByResponse] = useState({});
  const [rlmStats, setRlmStats] = useState({
    total_feedback_count: 0,
    tracked_sources: 0,
    top_positive_sources: [],
    top_negative_sources: [],
  });
  const quickPrompts = useMemo(
    () => [
      "Give me a concise executive summary of the attached document.",
      "List the key claims with direct supporting evidence from sources.",
      "What are the limitations, caveats, and risks in this document?",
      "Create a structured brief with findings, assumptions, and open questions.",
    ],
    [],
  );

  async function refreshHealth() {
    try {
      setLoadingHealth(true);
      const res = await fetch(`${API_BASE}/health`);
      if (!res.ok) return;
      setHealth(await res.json());
    } catch {
      setHealth(null);
    } finally {
      setLoadingHealth(false);
    }
  }

  async function refreshHistory() {
    try {
      const res = await fetch(`${API_BASE}/history/${sessionId}`);
      if (!res.ok) return;
      const data = await res.json();
      setHistory(data.items || []);
      setHistorySummary({
        total_queries: data.total_queries || 0,
        estimated_tokens_used: data.estimated_tokens_used || 0,
      });
    } catch {
      setHistory([]);
      setHistorySummary({ total_queries: 0, estimated_tokens_used: 0 });
    }
  }

  async function refreshFailures() {
    try {
      const res = await fetch(`${API_BASE}/failures?session_id=${encodeURIComponent(sessionId)}`);
      if (!res.ok) return;
      const data = await res.json();
      setFailureLogs(data.items || []);
    } catch {
      setFailureLogs([]);
    }
  }

  async function refreshRLMStats() {
    try {
      const res = await fetch(`${API_BASE}/rlm/stats`);
      if (!res.ok) return;
      const data = await res.json();
      setRlmStats(data);
    } catch {
      setRlmStats({
        total_feedback_count: 0,
        tracked_sources: 0,
        top_positive_sources: [],
        top_negative_sources: [],
      });
    }
  }

  useEffect(() => {
    refreshHealth();
    refreshHistory();
    refreshFailures();
    refreshRLMStats();
  }, []);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    window.localStorage.setItem("rke_theme", theme);
  }, [theme]);

  useEffect(() => {
    if (!threadViewportRef.current) return;
    threadViewportRef.current.scrollTo({
      top: threadViewportRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [threadMessages, running]);

  async function uploadFile(file) {
    if (!file) return;
    const attachmentId = crypto.randomUUID();
    const meta = attachmentKind(file.name);
    setAttachments((current) => [
      ...current,
      {
        id: attachmentId,
        name: file.name,
        kind: meta.kind,
        icon: meta.icon,
        storedAs: null,
        status: "uploading",
        detail: "Uploading...",
      },
    ]);
    setUploadStatus(`Uploading ${file.name}...`);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Upload failed");
      const indexedDelta = Number(data.chunks_added || 0);
      setUploadStatus(
        indexedDelta > 0
          ? `Indexed ${data.filename}: +${indexedDelta} chunks (total ${data.total_chunks}).`
          : `${data.filename} already indexed (no new chunks added).`,
      );
      setAttachments((current) =>
        current.map((item) =>
          item.id === attachmentId
            ? {
                ...item,
                status: "indexed",
                storedAs: data.stored_as || null,
                detail: indexedDelta > 0 ? `Indexed (+${indexedDelta} chunks)` : "Already indexed",
              }
            : item,
        ),
      );
      await refreshHealth();
    } catch (error) {
      setUploadStatus(`Upload error: ${error.message}`);
      setAttachments((current) =>
        current.map((item) =>
          item.id === attachmentId
            ? {
                ...item,
                status: "error",
                detail: "Upload failed",
              }
            : item,
        ),
      );
    }
  }

  function onChooseUpload() {
    uploadInputRef.current?.click();
  }

  async function onUploadSelected(event) {
    const files = Array.from(event.target.files || []);
    if (!files.length) return;
    for (const file of files) {
      // Sequential indexing avoids CPU spikes on free-tier deployments.
      // eslint-disable-next-line no-await-in-loop
      await uploadFile(file);
    }
    event.target.value = "";
  }

  function removeDraftAttachment(attachmentId) {
    setAttachments((current) => current.filter((item) => item.id !== attachmentId));
  }

  function toggleAblationDepth(value) {
    setAblationDepths((current) => {
      if (current.includes(value)) {
        const next = current.filter((item) => item !== value);
        return next.length ? next : [value];
      }
      return [...current, value].sort((a, b) => a - b);
    });
  }

  async function onClearIndex() {
    const confirmed = window.confirm("Clear the full FAISS index and metadata?");
    if (!confirmed) return;

    try {
      const res = await fetch(`${API_BASE}/index/clear`, { method: "POST" });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to clear index");
      setUploadStatus("Index cleared.");
      setAttachments([]);
      setResult(null);
      await refreshHealth();
    } catch (error) {
      setUploadStatus(`Clear index error: ${error.message}`);
    }
  }

  async function runQuery() {
    if (!question.trim()) {
      setQueryStatus("Enter a question first.");
      return;
    }
    if (attachments.some((item) => item.status === "uploading")) {
      setQueryStatus("Wait for attachments to finish indexing before sending.");
      return;
    }

    const submittedQuestion = question.trim();
    const submittedAttachments = attachments.map((item) => ({
      id: item.id,
      name: item.name,
      kind: item.kind,
      icon: item.icon,
      storedAs: item.storedAs || null,
      status: item.status,
    }));
    const sourceFilters = submittedAttachments
      .filter((item) => item.status === "indexed" && item.storedAs)
      .map((item) => item.storedAs);

    setThreadMessages((current) => [
      ...current,
      {
        id: crypto.randomUUID(),
        role: "user",
        text: submittedQuestion,
        attachments: submittedAttachments,
      },
    ]);
    setQuestion("");
    setAttachments([]);

    const selectedDepths = ablationMode ? [...new Set(ablationDepths)].sort((a, b) => a - b) : [Number(depth)];
    const requestDepth = Math.max(...selectedDepths);

    setRunning(true);
    setQueryStatus("Running retrieval loop...");

    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: submittedQuestion,
          session_id: sessionId,
          top_k: Number(topK),
          max_iterations: requestDepth,
          fast_mode: fastMode,
          deterministic_mode: deterministicMode,
          ablation_mode: ablationMode,
          ablation_depths: ablationMode ? selectedDepths : undefined,
          source_filters: sourceFilters.length ? sourceFilters : undefined,
          answer_verbosity: answerVerbosity,
          challenge_mode: challengeMode,
        }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Query failed");

      const nonEmptyLoopAnswer = (data.loops || []).find((loop) => (loop.answer || "").trim())?.answer || "";
      const finalAnswer = (data.final_answer || "").trim() || nonEmptyLoopAnswer || "No answer generated. Try re-running.";
      const normalized = { ...data, final_answer: finalAnswer };

      setResult(normalized);
      setActiveAblationDepth(normalized.ablation_best_depth || null);
      setQueryStatus(`Done in ${formatDuration(normalized.total_duration_ms)}.`);
      setThreadMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          responseId: normalized.response_id,
          text: normalized.final_answer,
          warning: normalized.low_coverage_warning
            ? "Coverage is low, review retrieved evidence before trusting this answer."
            : null,
          meta: [
            `Confidence ${formatPercent(normalized.confidence)}`,
            `Latency ${formatDuration(normalized.total_duration_ms)}`,
            `Coverage ${formatPercent(normalized.answer_coverage)}`,
            normalized.reliability_grade ? `Reliability ${String(normalized.reliability_grade).toUpperCase()}` : null,
            normalized.challenge_mode ? `Challenge risk ${formatPercent(normalized.challenge_risk)}` : null,
            normalized.loops?.length
              ? `Groundedness ${formatPercent(
                  normalized.loops[normalized.best_iteration - 1]?.groundedness ?? 0,
                )}`
              : null,
            normalized.stopped_early ? `Early stop: ${normalized.stop_reason}` : "Full run",
          ].filter(Boolean),
        },
      ]);
      await Promise.all([refreshHealth(), refreshHistory(), refreshFailures(), refreshRLMStats()]);
    } catch (error) {
      setQueryStatus(`Query error: ${error.message}`);
      setThreadMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          text: `Query error: ${error.message}`,
          isError: true,
        },
      ]);
    } finally {
      setRunning(false);
    }
  }

  async function onQuery(event) {
    event.preventDefault();
    await runQuery();
  }

  async function exportFailureCases() {
    try {
      const res = await fetch(`${API_BASE}/failures/export?session_id=${encodeURIComponent(sessionId)}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Export failed");

      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `failure-cases-${sessionId}.json`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch (error) {
      setQueryStatus(`Export error: ${error.message}`);
    }
  }

  async function submitFeedback(responseId, rating) {
    if (!responseId || ![-1, 1].includes(Number(rating))) return;
    try {
      setFeedbackByResponse((current) => ({ ...current, [responseId]: "saving" }));
      const res = await fetch(`${API_BASE}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          response_id: responseId,
          session_id: sessionId,
          rating: Number(rating),
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Feedback failed");
      setFeedbackByResponse((current) => ({ ...current, [responseId]: rating > 0 ? "up" : "down" }));
      await refreshRLMStats();
    } catch (error) {
      setFeedbackByResponse((current) => ({ ...current, [responseId]: "error" }));
      setQueryStatus(`Feedback error: ${error.message}`);
    }
  }

  async function copyText(value) {
    try {
      await navigator.clipboard.writeText(value || "");
      setQueryStatus("Copied to clipboard.");
    } catch {
      setQueryStatus("Clipboard copy failed.");
    }
  }

  function reusePrompt(value) {
    setQuestion(value || "");
    setQueryStatus("Prompt loaded into composer.");
  }

  function clearThread() {
    setThreadMessages([]);
    setResult(null);
    setQueryStatus("Thread cleared.");
  }

  const hasIndexedDocs = Number(health?.documents_indexed || 0) > 0;
  const ablationResults = result?.ablation_results || [];

  const activeRun = useMemo(() => {
    if (!result) return null;

    if (ablationResults.length > 0) {
      if (activeAblationDepth) {
        const exact = ablationResults.find((item) => item.depth === activeAblationDepth);
        if (exact) return exact;
      }
      const best = ablationResults.find((item) => item.is_best);
      return best || ablationResults[0];
    }

    return {
      depth: result.max_iterations_used,
      loops: result.loops,
      confidence: result.confidence,
      total_duration_ms: result.total_duration_ms,
      answer_coverage: result.answer_coverage,
      best_iteration: result.best_iteration,
      is_failure: result.is_failure,
      stop_reason: result.stop_reason,
      stopped_early: result.stopped_early,
    };
  }, [result, ablationResults, activeAblationDepth]);

  const loops = activeRun?.loops || [];

  const confidenceDrops = useMemo(() => {
    const drops = [];
    for (let idx = 1; idx < loops.length; idx += 1) {
      if (Number(loops[idx].confidence || 0) < Number(loops[idx - 1].confidence || 0)) {
        drops.push(loops[idx].iteration);
      }
    }
    return drops;
  }, [loops]);

  const queryEvolution = useMemo(() => {
    if (!result || !loops.length) return [];
    let previous = result.question;

    return loops.map((loop) => {
      const current = loop.refined_query || loop.query;
      const diff = buildTokenDiff(previous, current);
      const row = {
        iteration: loop.iteration,
        previous,
        current,
        diff,
      };
      previous = current;
      return row;
    });
  }, [result, loops]);

  const promptWordCount = useMemo(
    () => (question.trim().match(/[A-Za-z0-9_-]+/g) || []).length,
    [question],
  );
  const promptTokenEstimate = useMemo(
    () => Math.max(0, Math.round(promptWordCount * 1.35)),
    [promptWordCount],
  );
  const scopedAttachmentCount = useMemo(
    () => attachments.filter((item) => item.status === "indexed").length,
    [attachments],
  );

  function onPromptKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (!running) {
        runQuery();
      }
    }
  }

  return (
    <div className={`workspace ${focusMode ? "focus" : ""}`}>
      <header className="topbar">
        <div className="app-title">
          <h1>Recursive Knowledge Engine</h1>
          <p>Calm workspace for recursive evidence retrieval</p>
        </div>
        <div className="topbar-right">
          <div className="status-chip">
            <span className={`status-dot ${health?.status === "ok" ? "ok" : "down"}`} />
            <span>{loadingHealth ? "Checking" : health?.status === "ok" ? "Healthy" : "Offline"}</span>
          </div>
          <div className="status-chip">
            {health?.model
              ? `${health?.provider ? `${String(health.provider).toUpperCase()} · ` : ""}${health.model}`
              : "Model unavailable"}
          </div>
          <button
            type="button"
            className="status-chip action-chip"
            onClick={() => setTheme((value) => (value === "dark" ? "light" : "dark"))}
            title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
          >
            {theme === "dark" ? <SunIcon /> : <MoonIcon />}
            <span>{theme === "dark" ? "Light mode" : "Dark mode"}</span>
          </button>
          <button type="button" className="status-chip action-chip" onClick={() => setFocusMode((value) => !value)}>
            {focusMode ? "Show workspace" : "Focus chat"}
          </button>
          <button type="button" className="status-chip action-chip" onClick={clearThread}>
            Clear thread
          </button>
        </div>
      </header>

      <div className="workspace-tabs" role="tablist" aria-label="Workspace sections">
        <button
          type="button"
          role="tab"
          aria-selected={activeInfoTab === "workspace"}
          className={`tab-btn ${activeInfoTab === "workspace" ? "active" : ""}`}
          onClick={() => setActiveInfoTab("workspace")}
        >
          <ChatIcon />
          <span>Ask + Tune</span>
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activeInfoTab === "stats"}
          className={`tab-btn ${activeInfoTab === "stats" ? "active" : ""}`}
          onClick={() => setActiveInfoTab("stats")}
        >
          <ChartIcon />
          <span>Research Stats</span>
        </button>
      </div>

      <main className="center-stage">
        <section className="chat-surface">
          <div className="chat-surface-head">
            <div className="head-metrics">
              <span>Docs in scope: {scopedAttachmentCount}</span>
              <span>Pass: {ablationMode ? `Ablation (${ablationDepths.join(", ")})` : depth}</span>
              <span>Status: {queryStatus}</span>
            </div>
            <div className="head-actions">
              <button type="button" className="mini-btn" onClick={refreshHealth}>
                <RefreshIcon />
                <span>Refresh health</span>
              </button>
            </div>
          </div>
          {!threadMessages.length && !running && (
            <div className="empty-state">
              <h2>Ask a question about your docs</h2>
              <p>
                Enter sends prompt, Shift+Enter adds a newline. Upload a file with the paperclip inside the composer.
              </p>
              <div className="quick-prompts">
                {quickPrompts.map((item) => (
                  <button key={item} type="button" className="quick-chip" onClick={() => reusePrompt(item)}>
                    {item}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="thread" ref={threadViewportRef}>
            {threadMessages.map((message) => (
              <article key={message.id} className={`bubble ${message.role} ${message.isError ? "error" : ""}`}>
                <MarkdownBlock text={message.text} />
                {!!message.attachments?.length && (
                  <div className="message-attachments">
                    {message.attachments.map((item) => (
                      <div key={item.id} className="attachment-chip indexed">
                        <span className={`attachment-icon ${item.kind || "file"}`}>{item.icon || "FILE"}</span>
                        <div className="attachment-info">
                          <span className="attachment-name" title={item.name}>{item.name}</span>
                          <span className="attachment-state">Attached</span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
                {!!message.meta?.length && (
                  <div className="meta-strip">
                    {message.meta.map((item) => (
                      <span key={item}>{item}</span>
                    ))}
                  </div>
                )}
                <div className="message-actions">
                  <button type="button" className="mini-btn" onClick={() => copyText(message.text)}>
                    Copy
                  </button>
                  {message.role === "user" && (
                    <button type="button" className="mini-btn" onClick={() => reusePrompt(message.text)}>
                      Reuse
                    </button>
                  )}
                </div>
                {message.role === "assistant" && message.responseId && (
                  <div className="feedback-row">
                    <button
                      type="button"
                      className={`feedback-btn ${feedbackByResponse[message.responseId] === "up" ? "active" : ""}`}
                      onClick={() => submitFeedback(message.responseId, 1)}
                      title="Helpful answer"
                    >
                      Helpful
                    </button>
                    <button
                      type="button"
                      className={`feedback-btn ${feedbackByResponse[message.responseId] === "down" ? "active" : ""}`}
                      onClick={() => submitFeedback(message.responseId, -1)}
                      title="Not helpful"
                    >
                      Needs work
                    </button>
                  </div>
                )}
                {!!message.warning && <p className="warning">{message.warning}</p>}
              </article>
            ))}
            {running && (
              <article className="bubble assistant">
                <Skeleton lines={8} />
              </article>
            )}
          </div>
        </section>

        {ablationResults.length > 0 && (
          <section className="ablation-row">
            {ablationResults.map((item) => (
              <button
                key={item.depth}
                type="button"
                className={`pass-card ${item.is_best ? "best" : ""} ${activeAblationDepth === item.depth ? "active" : ""}`}
                onClick={() => setActiveAblationDepth(item.depth)}
              >
                <strong>{item.depth} Pass{item.depth > 1 ? "es" : ""}</strong>
                <span>{formatPercent(item.confidence)}</span>
                <span>{formatDuration(item.total_duration_ms)}</span>
              </button>
            ))}
          </section>
        )}

        <section className="assistant-panels">
          {activeInfoTab === "workspace" && (
            <>
          <details className="panel" open>
            <summary>Confidence and timeline</summary>
            <div className="panel-body">
              <ConfidenceChart loops={loops} />
              {confidenceDrops.length > 0 && <p className="warning">Confidence dropped at step(s): {confidenceDrops.join(", ")}</p>}
              <div className="timeline-list">
                {loops.map((loop) => (
                  <article key={`${loop.iteration}-${loop.query}`} className="timeline-row">
                    <header>
                      <strong>Step {loop.iteration}</strong>
                      <span>{formatDuration(loop.duration_ms)}</span>
                      <span>{formatPercent(loop.confidence)}</span>
                    </header>
                    <p>Query: {loop.query}</p>
                    <p>Refined: {loop.refined_query}</p>
                  </article>
                ))}
              </div>
            </div>
          </details>
            </>
          )}

          {activeInfoTab === "stats" && (
            <>
          <details className="panel">
            <summary>Retrieved sources</summary>
            <div className="panel-body">
              {loops.map((loop) => (
                <section key={`sources-${loop.iteration}`} className="sources-group">
                  <h3>Step {loop.iteration}</h3>
                  <ul className="source-list">
                    {(loop.retrieved_chunks || []).map((chunk) => (
                      <li key={`chunk-${loop.iteration}-${chunk.chunk_id}`}>
                        <div className="source-head">
                          <span>{chunk.source}</span>
                          <span>{formatPercent(chunk.score)}</span>
                        </div>
                        <p>{chunk.text}</p>
                      </li>
                    ))}
                  </ul>
                </section>
              ))}
            </div>
          </details>

          <details className="panel">
            <summary>Query evolution</summary>
            <div className="panel-body">
              {!!result && (
                <article className="evolution-step">
                  <h3>Original query</h3>
                  <p>{result.question}</p>
                </article>
              )}
              {queryEvolution.map((step) => (
                <article className="evolution-step" key={`evo-${step.iteration}`}>
                  <h3>Step {step.iteration}</h3>
                  <p className="token-line">
                    {step.diff.annotatedCurrent.map((token, idx) => (
                      <span key={`${token.token}-${idx}`} className={`token ${token.type}`}>
                        {token.token}
                      </span>
                    ))}
                  </p>
                  {!!step.diff.removedWords.length && <p className="removed-line">Removed: {step.diff.removedWords.join(", ")}</p>}
                </article>
              ))}
            </div>
          </details>

          <details className="panel">
            <summary>Robustness audit</summary>
            <div className="panel-body stack-gap">
              {!result && <p className="empty-note">Run a query to compute robustness metrics.</p>}
              {!!result && (
                <>
                  <div className="stats-grid">
                    <span>Reliability: {String(result.reliability_grade || "unknown").toUpperCase()}</span>
                    <span>Challenge risk: {formatPercent(result.challenge_risk)}</span>
                    <span>Support redundancy: {formatPercent(result.support_redundancy)}</span>
                    <span>Single-source dependency: {formatPercent(result.single_source_dependency)}</span>
                  </div>
                  {!!result.audit_summary && <p className="subtle">{result.audit_summary}</p>}
                  {!!result.challenge_query && <p className="subtle">Challenge query: {result.challenge_query}</p>}
                  {!!result.challenge_chunks?.length && (
                    <ul className="source-list">
                      {result.challenge_chunks.slice(0, 4).map((chunk) => (
                        <li key={`challenge-${chunk.chunk_id}`}>
                          <div className="source-head">
                            <span>{chunk.source}</span>
                            <span>{formatPercent(chunk.score)}</span>
                          </div>
                          <p>{chunk.text}</p>
                        </li>
                      ))}
                    </ul>
                  )}
                </>
              )}
            </div>
          </details>

          <details className="panel">
            <summary>Workspace stats and tools</summary>
            <div className="panel-body stack-gap">
              <div className="stats-grid">
                <span>Docs: {health?.documents_indexed ?? 0}</span>
                <span>FAISS: {formatBytes(health?.faiss_index_size_bytes)}</span>
                <span>Queries: {historySummary.total_queries}</span>
                <span>Tokens: {historySummary.estimated_tokens_used}</span>
                <span>Feedback: {rlmStats.total_feedback_count}</span>
                <span>RLM sources: {rlmStats.tracked_sources}</span>
              </div>
              {!!rlmStats.top_positive_sources?.length && (
                <p className="subtle">Top helpful sources: {rlmStats.top_positive_sources.join(", ")}</p>
              )}
              {!!rlmStats.top_negative_sources?.length && (
                <p className="subtle">Low-quality sources: {rlmStats.top_negative_sources.join(", ")}</p>
              )}
              <p className="subtle">{uploadStatus}</p>
              <div className="tool-actions">
                <button type="button" className="button ghost" onClick={onClearIndex}>Clear index</button>
                <button type="button" className="button ghost" onClick={exportFailureCases}>Export failures JSON</button>
              </div>
              <div className="history-block">
                <h3>Recent prompts</h3>
                <ul className="history-list">
                  {[...history].reverse().slice(0, 6).map((item, idx) => (
                    <li key={`${item.created_at}-${idx}`}>
                      <button type="button" onClick={() => setQuestion(item.question)}>{item.question}</button>
                    </li>
                  ))}
                </ul>
              </div>
              {!!failureLogs.length && (
                <div className="failure-block">
                  <h3>Recent failures</h3>
                  <ul className="failure-list">
                    {[...failureLogs].reverse().slice(0, 4).map((failure) => (
                      <li key={failure.id}>{failure.reason}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </details>
            </>
          )}
        </section>
      </main>

      <section className="composer-wrap">
        {showTuning && (
          <div className="tuning-drawer">
            <div className="tuning-head">
              <span className="tuning-title">
                <SparkIcon />
                <span>Tuning controls</span>
              </span>
              <span className="tuning-note">Open</span>
            </div>

            <details className="panel tuning-help">
              <summary>What each control means</summary>
              <div className="panel-body explainer-grid">
                <article>
                  <h3>Passes</h3>
                  <p>A pass runs retrieve - answer - critique - refine. More passes can improve recall but add latency.</p>
                </article>
                <article>
                  <h3>Top-K</h3>
                  <p>How many chunks are retrieved each pass. Higher values widen context, but can add noise.</p>
                </article>
                <article>
                  <h3>Attached scope</h3>
                  <p>When docs are attached, retrieval is restricted to those files for that prompt.</p>
                </article>
                <article>
                  <h3>Fast mode</h3>
                  <p>Uses heuristic critique + early stop for lower latency. Disable for fully model-driven critique.</p>
                </article>
                <article>
                  <h3>Deterministic</h3>
                  <p>Uses lower-variance generation for reproducibility and debugging.</p>
                </article>
                <article>
                  <h3>Ablation</h3>
                  <p>Runs 1/2/3 pass depths side-by-side so you can compare quality vs speed directly.</p>
                </article>
                <article>
                  <h3>Answer length</h3>
                  <p>Short is concise, Normal is balanced, Long returns a more complete report.</p>
                </article>
                <article>
                  <h3>Robustness audit</h3>
                  <p>Looks for contradictory evidence and single-source dependency to flag fragile answers.</p>
                </article>
              </div>
            </details>

            <div className="tuning-grid">
              <label>
                Top-K
                <input type="number" min="1" max="12" value={topK} onChange={(event) => setTopK(Number(event.target.value))} />
              </label>
              <label>
                Pass depth
                <div className="segment">
                  {[1, 2, 3].map((value) => (
                    <button
                      key={value}
                      type="button"
                      className={`segment-btn ${Number(depth) === value ? "active" : ""}`}
                      onClick={() => setDepth(value)}
                    >
                      {value}
                    </button>
                  ))}
                </div>
              </label>
              <label>
                Answer length
                <div className="segment">
                  {[
                    { value: "short", label: "Short" },
                    { value: "normal", label: "Normal" },
                    { value: "long", label: "Long" },
                  ].map((option) => (
                    <button
                      key={option.value}
                      type="button"
                      className={`segment-btn ${answerVerbosity === option.value ? "active" : ""}`}
                      onClick={() => setAnswerVerbosity(option.value)}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
              </label>
              <Toggle label="Fast mode" checked={fastMode} onChange={setFastMode} />
              <Toggle label="Deterministic" checked={deterministicMode} onChange={setDeterministicMode} />
              <Toggle label="Ablation" checked={ablationMode} onChange={setAblationMode} />
              <Toggle label="Robustness audit" checked={challengeMode} onChange={setChallengeMode} />
            </div>

            {ablationMode && (
              <div className="ablation-selector">
                {[1, 2, 3].map((value) => (
                  <button
                    key={value}
                    type="button"
                    className={`pill ${ablationDepths.includes(value) ? "selected" : ""}`}
                    onClick={() => toggleAblationDepth(value)}
                  >
                    {value} pass
                  </button>
                ))}
              </div>
            )}

            <p className={`inline-status ${queryStatus.includes("error") ? "error" : ""}`}>{queryStatus}</p>
          </div>
        )}

        <form onSubmit={onQuery} className="composer-shell">
          <input
            ref={uploadInputRef}
            type="file"
            accept={ACCEPTED_DOC_TYPES.join(",")}
            multiple
            className="hidden-input"
            onChange={onUploadSelected}
          />

          {!!attachments.length && (
            <div className="composer-attachments">
              {attachments.slice(-6).map((item) => (
                <div key={item.id} className={`attachment-chip ${item.status}`}>
                  <span className={`attachment-icon ${item.kind || "file"}`}>{item.icon || "FILE"}</span>
                  <div className="attachment-info">
                    <span className="attachment-name" title={item.name}>{item.name}</span>
                    <span className="attachment-state">{item.detail}</span>
                  </div>
                  <button
                    type="button"
                    className="attachment-remove"
                    onClick={() => removeDraftAttachment(item.id)}
                    title="Remove attachment"
                    aria-label={`Remove ${item.name}`}
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          )}

          <div className="composer-input-row">
            <button
              type="button"
              className="icon-button"
              onClick={onChooseUpload}
              title="Attach and index one or more documents"
            >
              <PaperclipIcon />
            </button>

            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={onPromptKeyDown}
              placeholder="Ask about your indexed documents"
              rows={3}
            />

            <div className="composer-actions">
              <button
                type="button"
                className={`button tune-toggle ${showTuning ? "active" : ""}`}
                onClick={() => setShowTuning((value) => !value)}
                aria-expanded={showTuning}
              >
                {showTuning ? "Hide tuning" : "Tune"}
              </button>
              <button
                type="submit"
                className="button"
                disabled={running || !hasIndexedDocs || attachments.some((item) => item.status === "uploading")}
              >
                {running ? "Running..." : "Send"}
              </button>
            </div>
          </div>

          <div className="composer-footnote">
            <span>{promptWordCount} words</span>
            <span>~{promptTokenEstimate} tokens</span>
            <span>{scopedAttachmentCount} attached docs in scope</span>
            <span>Enter to send, Shift+Enter for newline</span>
          </div>
        </form>
      </section>
    </div>
  );
}
