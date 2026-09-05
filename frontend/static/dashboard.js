const API_BASE_STORAGE_KEY = "ssl_api_base";

function resolvedApiBase() {
  const params = new URLSearchParams(window.location.search);
  const override = params.get("api_base");
  const stored = (() => {
    try {
      return window.localStorage.getItem(API_BASE_STORAGE_KEY) || "";
    } catch {
      return "";
    }
  })();
  const requested = (override || window.API_BASE || stored || "").replace(/\/+$/, "");
  // Same-origin by default: vercel.json rewrites /api to the Space, so the
  // session cookie is first-party. A cross-site cookie from hf.space is
  // dropped by Safari and by Chrome's third-party cookie blocking, which made
  // every dashboard call 401 no matter how CORS was configured.
  const defaultApiBase = "";
  return requested === window.location.origin ? requested : defaultApiBase;
}

function dashboardDetailHref(eventId) {
  const hrefParams = new URLSearchParams();
  hrefParams.set("id", eventId || "");
  const apiBase = resolvedApiBase();
  if (apiBase) {
    hrefParams.set("api_base", apiBase);
  }
  return `/dashboard-detail.html?${hrefParams.toString()}`;
}

function dashboardHomeHref() {
  const hrefParams = new URLSearchParams();
  const apiBase = resolvedApiBase();
  if (apiBase) {
    hrefParams.set("api_base", apiBase);
  }
  const query = hrefParams.toString();
  return query ? `/dashboard.html?${query}` : "/dashboard.html";
}

function dashboardApiUrl(path) {
  const base = resolvedApiBase();
  if (!base) return path;
  return `${base}${path}`;
}

async function logoutAdmin() {
  await fetch(dashboardApiUrl("/api/admin/logout"), {
    method: "POST",
    credentials: "include",
  });
  const login = new URL("/admin-login.html", window.location.origin);
  const base = resolvedApiBase();
  if (base !== window.location.origin) login.searchParams.set("api_base", base);
  window.location.replace(login.toString());
}

function setText(id, value) {
  const node = document.getElementById(id);
  if (node) node.textContent = value;
}

function sanitizeDashboardUrl(value) {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  if (!trimmed || trimmed === "URL not provided") return null;
  try {
    const parsed = new URL(trimmed, window.location.origin);
    return parsed.protocol === "http:" || parsed.protocol === "https:" ? parsed.href : null;
  } catch {
    return null;
  }
}

function createStatusPill(status) {
  const pill = document.createElement("span");
  pill.className = `status-pill status-${status || "answered"}`;
  pill.textContent = status || "answered";
  return pill;
}

function createScorePill(score, isLowConfidence) {
  if (score === null || score === undefined) {
    const span = document.createElement("span");
    span.className = "muted-value";
    span.textContent = "N/A";
    return span;
  }

  const pill = document.createElement("span");
  pill.className = `score-pill${isLowConfidence ? " score-low" : ""}`;
  pill.textContent = String(score);
  return pill;
}

function formatNumber(value) {
  const number = Number(value || 0);
  return Number.isFinite(number) ? number.toLocaleString() : "0";
}

function formatSeconds(milliseconds) {
  const seconds = Number(milliseconds || 0) / 1000;
  if (seconds > 0 && seconds < 0.01) return "<0.01 s";
  return `${seconds.toFixed(2)} s`;
}

function formatCost(value) {
  if (value === null || value === undefined) return "unpriced";
  return `$${Number(value).toFixed(6)}`;
}

function createMetricCell(primary, secondary) {
  const cell = document.createElement("td");
  const main = document.createElement("span");
  main.className = "count-pair";
  main.textContent = primary;
  cell.appendChild(main);
  if (secondary) {
    const note = document.createElement("span");
    note.className = "muted-value";
    note.textContent = secondary;
    cell.appendChild(note);
  }
  return cell;
}

function renderMetricsTable(containerId, emptyId, headers, rows) {
  const container = document.getElementById(containerId);
  const empty = document.getElementById(emptyId);
  if (!container) return;
  container.innerHTML = "";
  if (!rows.length) {
    if (empty) empty.hidden = false;
    return;
  }
  if (empty) empty.hidden = true;
  const table = document.createElement("table");
  table.className = "metrics-table";
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  headers.forEach((header) => {
    const th = document.createElement("th");
    th.textContent = header;
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);
  const tbody = document.createElement("tbody");
  rows.forEach((cells) => {
    const tr = document.createElement("tr");
    cells.forEach((value) => {
      const td = document.createElement("td");
      if (value instanceof Node) td.appendChild(value);
      else td.textContent = value;
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.append(thead, tbody);
  container.appendChild(table);
}

function createKindPill(kind) {
  const pill = document.createElement("span");
  pill.className = `kind-pill kind-${kind || "step"}`;
  pill.textContent = kind || "step";
  return pill;
}

function renderEmptyRow(message) {
  const row = document.createElement("tr");
  const cell = document.createElement("td");
  cell.colSpan = 9;
  cell.className = "empty-state";
  cell.textContent = message;
  row.appendChild(cell);
  return row;
}

function renderRankedList(listId, emptyId, items) {
  const list = document.getElementById(listId);
  const empty = document.getElementById(emptyId);
  if (!list || !empty) return;

  list.innerHTML = "";
  if (!items || items.length === 0) {
    empty.hidden = false;
    return;
  }

  empty.hidden = true;
  items.forEach(([label, count]) => {
    const item = document.createElement("li");
    const left = document.createElement("span");
    left.textContent = label;
    const right = document.createElement("strong");
    right.textContent = String(count);
    item.append(left, right);
    list.appendChild(item);
  });
}

function renderProblemEvents(events) {
  const list = document.getElementById("problemEventsList");
  const empty = document.getElementById("problemEventsEmpty");
  if (!list || !empty) return;

  list.innerHTML = "";
  if (!events || events.length === 0) {
    empty.hidden = false;
    return;
  }

  empty.hidden = true;
  events.forEach((event) => {
    const link = document.createElement("a");
    link.href = dashboardDetailHref(event.id || "");
    link.appendChild(createStatusPill(event.status || "answered"));
    link.append(` ${event.display_label || event.id || "Interaction"}`);
    list.appendChild(link);
  });
}

function renderEval(evalPayload) {
  const model = document.getElementById("evalModel");
  const grid = document.getElementById("evalGrid");
  const empty = document.getElementById("evalEmpty");
  const problems = document.getElementById("evalProblems");
  if (!model || !grid || !empty || !problems) return;

  model.textContent = evalPayload?.model || "";
  problems.innerHTML = "";

  if (!evalPayload?.summary) {
    grid.hidden = true;
    empty.hidden = false;
    return;
  }

  grid.hidden = false;
  empty.hidden = true;
  setText("evalTotalCases", String(evalPayload.summary.total_cases ?? "-"));
  setText("evalCorrectness", String(evalPayload.summary.average_scores?.correctness_vs_corpus ?? "-"));
  setText("evalCitations", String(evalPayload.summary.average_scores?.citations ?? "-"));
  setText("evalHallucinated", String(evalPayload.summary.classification_counts?.hallucinated_yes ?? "-"));

  (evalPayload.problem_cases || []).slice(0, 5).forEach((item) => {
    const p = document.createElement("p");
    const strong = document.createElement("strong");
    strong.textContent = item.id || "case";
    p.append(strong, ` ${item.notes || ""}`);
    problems.appendChild(p);
  });
}

async function loadDashboardPage() {
  const body = document.getElementById("historyTableBody");
  if (!body) return;

  try {
    const response = await fetch(dashboardApiUrl("/api/dashboard"), { credentials: "include" });
    if (response.status === 401) {
      const login = new URL("/admin-login.html", window.location.origin);
      const base = resolvedApiBase();
      if (base !== window.location.origin) login.searchParams.set("api_base", base);
      window.location.replace(login.toString());
      return;
    }
    if (!response.ok) throw new Error(`Dashboard request failed (${response.status})`);
    const dashboard = await response.json();

    setText("metricTotal", String(dashboard.stats?.total ?? 0));
    setText("metricClarifications", String(dashboard.stats?.clarifications ?? 0));
    setText("metricLowConfidence", String(dashboard.stats?.low_confidence ?? 0));
    setText("metricBlocked", String(dashboard.stats?.blocked ?? 0));
    setText("metricErrors", String(dashboard.stats?.errors ?? 0));
    setText("metricAvgLatency", formatSeconds(dashboard.stats?.avg_latency_ms));
    setText("metricTotalTokens", formatNumber(dashboard.stats?.total_tokens));
    setText("metricAvgTokens", `${formatNumber(dashboard.stats?.avg_tokens)} avg / answer`);
    setText("metricTotalCost", `$${Number(dashboard.stats?.total_cost_usd || 0).toFixed(4)}`);
    setText("metricAvgCost", `${formatCost(dashboard.stats?.avg_cost_usd ?? 0)} avg / answer`);

    body.innerHTML = "";
    const history = dashboard.chat_history || [];
    if (history.length === 0) {
      body.appendChild(renderEmptyRow("No chat logs yet. Ask the chatbot a question and refresh this page."));
    } else {
      history.forEach((event) => {
        const row = document.createElement("tr");

        const statusCell = document.createElement("td");
        statusCell.appendChild(createStatusPill(event.status));

        const mappingCell = document.createElement("td");
        const link = document.createElement("a");
        link.className = "history-link";
        link.href = dashboardDetailHref(event.id || "");
        const strong = document.createElement("strong");
        strong.textContent = event.display_label || event.id || "Interaction";
        const preview = document.createElement("span");
        preview.textContent = event.preview_text || "";
        const timestamp = document.createElement("small");
        timestamp.textContent = event.timestamp || event.id || "";
        link.append(strong, preview);
        mappingCell.append(link, timestamp);

        const confidenceCell = document.createElement("td");
        confidenceCell.appendChild(createScorePill(event.confidence_score, event.is_low_confidence));

        const sourceCell = document.createElement("td");
        const shown = document.createElement("span");
        shown.className = "count-pair";
        shown.textContent = `${event.source_count || 0} shown`;
        const retrieved = document.createElement("span");
        retrieved.className = "muted-value";
        retrieved.textContent = `${event.retrieved_count || 0} retrieved`;
        sourceCell.append(shown, retrieved);

        const pathCell = document.createElement("td");
        const pathLabel = document.createElement("span");
        pathLabel.className = "path-label";
        pathLabel.textContent = event.path_label || event.response_mode || "direct";
        pathLabel.title = event.path_label || "";
        pathCell.appendChild(pathLabel);

        const retrievalCell = event.top_score === null || event.top_score === undefined
          ? createMetricCell("N/A")
          : createMetricCell(`top ${Number(event.top_score).toFixed(3)}`, `gap ${Number(event.score_gap || 0).toFixed(3)}`);

        const tokenCell = event.total_tokens
          ? createMetricCell(formatNumber(event.total_tokens), `${event.token_usage?.call_count || 0} call(s)`)
          : createMetricCell("N/A");

        const costCell = document.createElement("td");
        costCell.textContent = formatCost(event.cost_usd);

        const latencyCell = document.createElement("td");
        latencyCell.textContent = formatSeconds(event.latency_ms);
        if (event.latency_breakdown) {
          const note = document.createElement("span");
          note.className = "muted-value";
          note.textContent = `retrieval ${formatSeconds(event.latency_breakdown.retrieval_ms)} / llm ${formatSeconds(event.latency_breakdown.llm_ms)}`;
          latencyCell.appendChild(note);
        }

        row.append(statusCell, mappingCell, confidenceCell, pathCell, sourceCell, retrievalCell, tokenCell, costCell, latencyCell);
        body.appendChild(row);
      });
    }

    renderRankedList("sourceUsageList", "sourceUsageEmpty", dashboard.source_usage || []);
    renderRankedList("categoryUsageList", "categoryUsageEmpty", dashboard.category_usage || []);
    renderProblemEvents(dashboard.problem_events || []);
    renderEval(dashboard.eval || {});
  } catch (error) {
    body.innerHTML = "";
    body.appendChild(renderEmptyRow(error.message || "Unable to load dashboard."));
    const emptyIds = ["sourceUsageEmpty", "categoryUsageEmpty", "problemEventsEmpty", "evalEmpty"];
    emptyIds.forEach((id) => {
      const node = document.getElementById(id);
      if (node) {
        node.hidden = false;
        node.textContent = "Unable to load dashboard data from the backend.";
      }
    });
  }
}

function renderDetailStat(label, value) {
  const wrapper = document.createElement("div");
  const dt = document.createElement("dt");
  dt.textContent = label;
  const dd = document.createElement("dd");
  if (value instanceof Node) {
    dd.appendChild(value);
  } else {
    dd.textContent = value;
  }
  wrapper.append(dt, dd);
  return wrapper;
}

function renderConfidenceSection(event) {
  const section = document.getElementById("confidenceSection");
  if (!section) return;

  section.innerHTML = "";
  const confidence = event.trace?.confidence;
  if (!confidence) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "No confidence data was recorded for this interaction.";
    section.appendChild(empty);
    return;
  }

  const stats = document.createElement("dl");
  stats.className = "detail-stats single-column";
  stats.append(
    renderDetailStat("Score", event.confidence_score ?? "N/A"),
    renderDetailStat("Low Confidence", event.is_low_confidence ? "yes" : "no")
  );
  section.appendChild(stats);

  if (event.confidence_reasons?.length) {
    const list = document.createElement("ul");
    list.className = "reason-list";
    event.confidence_reasons.forEach((reason) => {
      const item = document.createElement("li");
      item.textContent = reason;
      list.appendChild(item);
    });
    section.appendChild(list);
  }
}

function renderSources(sources) {
  const list = document.getElementById("detailSources");
  const empty = document.getElementById("detailSourcesEmpty");
  if (!list || !empty) return;

  list.innerHTML = "";
  if (!sources || sources.length === 0) {
    empty.hidden = false;
    return;
  }

  empty.hidden = true;
  sources.forEach((source) => {
    const item = document.createElement("li");
    const title = document.createElement("strong");
    title.textContent = source.title || "Untitled source";
    item.appendChild(title);
    const safeUrl = sanitizeDashboardUrl(source.url);
    if (safeUrl) {
      const link = document.createElement("a");
      link.href = safeUrl;
      link.target = "_blank";
      link.rel = "noreferrer";
      link.textContent = safeUrl;
      item.appendChild(link);
    }
    list.appendChild(item);
  });
}

async function loadDashboardDetailPage() {
  const detailEmpty = document.getElementById("detailEmpty");
  if (!detailEmpty) return;

  const params = new URLSearchParams(window.location.search);
  const eventId = params.get("id");
  if (!eventId) {
    const text = document.getElementById("detailEmptyText");
    if (text) text.textContent = "No interaction id was provided.";
    return;
  }

  try {
    const response = await fetch(dashboardApiUrl(`/api/dashboard/interaction/${encodeURIComponent(eventId)}`), { credentials: "include" });
    if (response.status === 401) {
      const login = new URL("/admin-login.html", window.location.origin);
      const base = resolvedApiBase();
      if (base !== window.location.origin) login.searchParams.set("api_base", base);
      window.location.replace(login.toString());
      return;
    }
    if (response.status === 404) {
      const text = document.getElementById("detailEmptyText");
      if (text) text.textContent = "That log entry could not be found in the local JSONL file.";
      return;
    }
    if (!response.ok) throw new Error(`Interaction request failed (${response.status})`);
    const event = await response.json();

    detailEmpty.hidden = true;
    const detailLayout = document.getElementById("detailLayout");
    if (detailLayout) detailLayout.hidden = false;

    setText("detailLabel", event.display_label || event.id || "Interaction");
    setText("detailTimestamp", event.timestamp || "");
    setText("detailSourceSummary", `${event.source_count || 0} returned sources`);
    setText("detailPreviewText", event.preview_text || "");

    const stats = document.getElementById("detailStats");
    if (stats) {
      stats.innerHTML = "";
      stats.append(
        renderDetailStat("Status", createStatusPill(event.status || "answered")),
        renderDetailStat("Latency", formatSeconds(event.latency_ms)),
        renderDetailStat("Mode", event.response_mode || "unknown"),
        renderDetailStat("Clarification", event.needs_clarification ? "yes" : "no"),
        renderDetailStat("Blocked", event.blocked ? "yes" : "no"),
        renderDetailStat("Confidence", event.confidence_score ?? "N/A"),
        renderDetailStat("Sources", String(event.source_count || 0))
      );
    }

    renderConfidenceSection(event);
    renderSources(event.sources || []);
    setText("retrievalSummary", JSON.stringify(event.retrieval_summary || {}, null, 2));

    setText("detailPathLabel", event.path_label || event.response_mode || "direct");
    renderMetricsTable(
      "pathTable",
      "pathEmpty",
      ["#", "Step", "Kind", "Model", "Latency", "Tokens"],
      (event.path || []).map((step, index) => [
        String(index + 1),
        step.step || "",
        createKindPill(step.kind),
        step.model || "—",
        formatSeconds(step.latency_ms),
        step.total_tokens ? formatNumber(step.total_tokens) : "—",
      ])
    );

    const routeStats = document.getElementById("routeStats");
    if (routeStats) {
      routeStats.innerHTML = "";
      const route = event.route_summary || {};
      routeStats.append(
        renderDetailStat("Response mode", route.response_mode || event.response_mode || "unknown"),
        renderDetailStat("Routing mode", route.routing_mode || "unknown"),
        renderDetailStat("Question type", route.question_type || "unknown")
      );
    }

    const latencyStats = document.getElementById("latencyStats");
    if (latencyStats) {
      const breakdown = event.latency_breakdown || {};
      latencyStats.innerHTML = "";
      latencyStats.append(
        renderDetailStat("Retrieval", formatSeconds(breakdown.retrieval_ms)),
        renderDetailStat("LLM calls", formatSeconds(breakdown.llm_ms)),
        renderDetailStat("Other", formatSeconds(breakdown.other_ms))
      );
    }
    setText("detailLatencyTotal", `${formatSeconds(event.latency_ms)} total`);

    const usage = event.token_usage || {};
    setText("detailCostTotal", formatCost(usage.cost_usd));
    const tokenStats = document.getElementById("tokenStats");
    if (tokenStats) {
      tokenStats.innerHTML = "";
      tokenStats.append(
        renderDetailStat("Input", formatNumber(usage.input_tokens)),
        renderDetailStat("Output", formatNumber(usage.output_tokens)),
        renderDetailStat("Thinking", formatNumber(usage.thinking_tokens)),
        renderDetailStat("Cached", formatNumber(usage.cached_tokens)),
        renderDetailStat("Total", formatNumber(usage.total_tokens)),
        renderDetailStat("LLM calls", String(usage.call_count || 0))
      );
    }
    renderMetricsTable(
      "tokenTable",
      "tokenEmpty",
      ["Stage", "Model", "In", "Out", "Think", "Cost"],
      (event.llm_calls || []).map((call) => [
        call.step || "",
        call.model || "",
        formatNumber(call.input_tokens),
        formatNumber(call.output_tokens),
        formatNumber(call.thinking_tokens),
        formatCost(call.cost_usd),
      ])
    );
    const unpricedNote = document.getElementById("tokenUnpricedNote");
    if (unpricedNote) unpricedNote.hidden = usage.fully_priced !== false;

    renderMetricsTable(
      "retrievalScoreTable",
      "retrievalScoreEmpty",
      ["Rank", "Score", "Title", "Section", "Chunk", "Source"],
      (event.retrieval_scores || []).map((rowData) => [
        String(rowData.rank ?? ""),
        rowData.forced ? "pinned" : Number(rowData.score || 0).toFixed(4),
        rowData.title || "—",
        rowData.section_name || "—",
        rowData.chunk_index === null || rowData.chunk_index === undefined ? "—" : String(rowData.chunk_index),
        rowData.source_path || "—",
      ])
    );
  } catch (error) {
    const text = document.getElementById("detailEmptyText");
    if (text) text.textContent = error.message || "Unable to load interaction details.";
  }
}

if (document.getElementById("historyTableBody")) {
  document.getElementById("logoutButton")?.addEventListener("click", logoutAdmin);
  loadDashboardPage();
}

if (document.getElementById("detailEmpty")) {
  document.getElementById("logoutButton")?.addEventListener("click", logoutAdmin);
  const dashboardBackLink = document.querySelector('a[href="/dashboard.html"]');
  if (dashboardBackLink) {
    dashboardBackLink.href = dashboardHomeHref();
  }
  loadDashboardDetailPage();
}
