function resolvedApiBase() {
  const params = new URLSearchParams(window.location.search);
  const override = params.get("api_base");
  return override || window.API_BASE || "";
}

function dashboardDetailHref(eventId) {
  const params = new URLSearchParams(window.location.search);
  const hrefParams = new URLSearchParams();
  hrefParams.set("id", eventId || "");
  if (params.get("api_base")) {
    hrefParams.set("api_base", params.get("api_base"));
  }
  return `/dashboard-detail.html?${hrefParams.toString()}`;
}

function dashboardHomeHref() {
  const params = new URLSearchParams(window.location.search);
  const hrefParams = new URLSearchParams();
  if (params.get("api_base")) {
    hrefParams.set("api_base", params.get("api_base"));
  }
  const query = hrefParams.toString();
  return query ? `/dashboard.html?${query}` : "/dashboard.html";
}

function dashboardApiUrl(path) {
  return `${resolvedApiBase()}${path}`;
}

function setText(id, value) {
  const node = document.getElementById(id);
  if (node) node.textContent = value;
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

function renderEmptyRow(message) {
  const row = document.createElement("tr");
  const cell = document.createElement("td");
  cell.colSpan = 5;
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
    const response = await fetch(dashboardApiUrl("/api/dashboard"));
    if (!response.ok) throw new Error(`Dashboard request failed (${response.status})`);
    const dashboard = await response.json();

    setText("metricTotal", String(dashboard.stats?.total ?? 0));
    setText("metricClarifications", String(dashboard.stats?.clarifications ?? 0));
    setText("metricLowConfidence", String(dashboard.stats?.low_confidence ?? 0));
    setText("metricBlocked", String(dashboard.stats?.blocked ?? 0));
    setText("metricErrors", String(dashboard.stats?.errors ?? 0));

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

        const latencyCell = document.createElement("td");
        latencyCell.textContent = `${event.latency_ms || 0} ms`;

        row.append(statusCell, mappingCell, confidenceCell, sourceCell, latencyCell);
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
    if (source.url && source.url !== "URL not provided") {
      const link = document.createElement("a");
      link.href = source.url;
      link.target = "_blank";
      link.rel = "noreferrer";
      link.textContent = source.url;
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
    const response = await fetch(dashboardApiUrl(`/api/dashboard/interaction/${encodeURIComponent(eventId)}`));
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
        renderDetailStat("Latency", `${event.latency_ms || 0} ms`),
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
  } catch (error) {
    const text = document.getElementById("detailEmptyText");
    if (text) text.textContent = error.message || "Unable to load interaction details.";
  }
}

if (document.getElementById("historyTableBody")) {
  loadDashboardPage();
}

if (document.getElementById("detailEmpty")) {
  const dashboardBackLink = document.querySelector('a[href="/dashboard.html"]');
  if (dashboardBackLink) {
    dashboardBackLink.href = dashboardHomeHref();
  }
  loadDashboardDetailPage();
}
