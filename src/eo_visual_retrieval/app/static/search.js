"use strict";
const $ = id => document.getElementById(id);
let previewUrl;
let lastSearch;
function paragraph(parent, text) {
  const p = document.createElement("p"); p.textContent = text; parent.append(p);
}
function setExample(id) {
  $("image").value = "";
  $("item-id").value = id;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  $("example-preview").hidden = !id;
  if (id) $("example-preview").src = `/thumbnail?item_id=${encodeURIComponent(id)}`;
}
$("item-id").addEventListener("change", () => setExample($("item-id").value));
$("image").addEventListener("change", () => {
  $("item-id").value = "";
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  const file = $("image").files[0];
  $("example-preview").hidden = !file;
  if (file) { previewUrl = URL.createObjectURL(file); $("example-preview").src = previewUrl; }
});
$("weight").addEventListener("input", () => {
  $("weight-value").textContent = `${Math.round(Number($("weight").value) * 100)}%`;
});
function query() {
  const value = {
    text: $("text").value, item_id: $("item-id").value || null,
    text_weight: Number($("weight").value), interpret: $("interpret").checked,
    start_date: $("start-date").value || null, end_date: $("end-date").value || null,
    collection: $("collection").value.trim() || null,
    max_cloud_cover: $("cloud").value === "" ? null : Number($("cloud").value),
  };
  const bounds = $("bbox").value.trim();
  if (bounds) {
    value.bbox = bounds.split(/[\s,]+/).map(Number);
    if (value.bbox.length !== 4 || value.bbox.some(n => !Number.isFinite(n)))
      throw new Error("Enter four bounding-box coordinates: west, south, east, north.");
  }
  return value;
}
async function responseJson(response) {
  const data = await response.json().catch(() => ({error: "Request rejected. Check the image size and query."}));
  if (!response.ok) throw new Error(data.error || "Search failed. Check the input and try again.");
  return data;
}
function showPlan(plan) {
  $("plan").hidden = false;
  $("filters").replaceChildren(); $("notes").replaceChildren();
  const active = Object.entries(plan.filters).filter(([,v]) => v !== null);
  const fields = [["Embedded description", plan.text || "None (image only)"]];
  fields.push(...(active.length ? active : [["Scope", "All local index scenes"]]));
  for (const [key,value] of fields) {
    const dt = document.createElement("dt"), dd = document.createElement("dd");
    dt.textContent = key.replaceAll("_", " "); dd.textContent = Array.isArray(value) ? value.join(", ") : value;
    $("filters").append(dt, dd);
  }
  for (const note of plan.notes) {
    const li = document.createElement("li"); li.textContent = note; $("notes").append(li);
  }
}
function showResults(data) {
  lastSearch = data; $("export").hidden = false;
  const diagnostic = $("diagnostics"); diagnostic.replaceChildren();
  diagnostic.hidden = !data.diagnostics;
  if (data.diagnostics) {
    const d = data.diagnostics;
    paragraph(diagnostic, `${data.index_count} indexed → ${data.candidate_count} eligible → ${data.results.length} shown · ${d.elapsed_ms.toFixed(0)} ms engine time`);
    paragraph(diagnostic, `${d.excluded_example} example excluded · ${d.excluded_by_filters} scenes excluded by filters`);
    for (const [name, counts] of Object.entries(d.filter_counts))
      paragraph(diagnostic, `${name.replaceAll("_", " ")}: ${counts.pass} pass, ${counts.fail} fail, ${counts.missing} missing`);
    paragraph(diagnostic, d.filter_counts_scope);
    paragraph(diagnostic, d.timing_scope);
    const details = document.createElement("details"), summary = document.createElement("summary");
    summary.textContent = "Query and model provenance";
    const pre = document.createElement("pre"); pre.className = "provenance-json";
    pre.textContent = JSON.stringify({model: data.provenance, query: data.query_input, ranker: data.ranker, tie_break: d.tie_break}, null, 2);
    details.append(summary, pre); diagnostic.append(details);
  }
  $("results").replaceChildren(); $("score-help").hidden = !data.results.length;
  $("status").textContent = data.message || `${data.mode} search · ${data.results.length} results from ${data.candidate_count} eligible scenes`;
  for (const result of data.results) {
    const card = document.createElement("article"); card.className = "result";
    const img = document.createElement("img"); img.src = `/thumbnail?item_id=${encodeURIComponent(result.item_id)}`;
    img.alt = `RGB scene ${result.item_id}`; img.loading = "lazy";
    const name = document.createElement("h3"); name.textContent = result.item_id;
    card.append(img, name);
    const lines = [`Combined: ${result.score.toFixed(3)}`];
    if (result.text_score !== null) lines.push(`Text: ${result.text_score.toFixed(3)}`);
    if (result.image_score !== null) lines.push(`Image: ${result.image_score.toFixed(3)}`);
    lines.push(`Date: ${result.metadata.date || "unknown"}`,
      `Scene cloud: ${result.metadata.cloud_cover === null ? "unknown" : result.metadata.cloud_cover.toFixed(1) + "%"}`);
    for (const line of lines) { const p = document.createElement("p"); p.textContent = line; card.append(p); }
    if (result.explanation) {
      const why = document.createElement("details"), summary = document.createElement("summary");
      summary.textContent = `Why rank #${result.rank}?`; why.append(summary);
      const e = result.explanation;
      paragraph(why, `Weighted text ${e.text_contribution.toFixed(4)} + weighted image ${e.image_contribution.toFixed(4)} = ${result.score.toFixed(4)} (rounded).`);
      if (e.text_rank !== null) paragraph(why, `Text alone: #${e.text_rank}`);
      if (e.image_rank !== null) paragraph(why, `Image alone: #${e.image_rank}`);
      paragraph(why, e.rank_scope);
      for (const [name, status] of Object.entries(e.filter_checks)) paragraph(why, `${name.replaceAll("_", " ")}: ${status}`);
      paragraph(why, `Collection: ${result.metadata.collection || "unknown"}`);
      paragraph(why, `Chip center (lon, lat): ${result.metadata.centroid_lonlat?.join(", ") || "unknown"}`);
      paragraph(why, "This explains ranking arithmetic and metadata eligibility. It does not identify objects or prove change.");
      card.append(why);
    }
    const button = document.createElement("button"); button.type = "button";
    button.textContent = "Use as example";
    button.addEventListener("click", () => { setExample(result.item_id); $("text").focus(); });
    card.append(button); $("results").append(card);
  }
}
async function run(reviewOnly) {
  $("submit").disabled = true; $("review").disabled = true;
  $("status").textContent = reviewOnly ? "Interpreting filters…" : "Searching scenes…";
  $("results").replaceChildren(); $("score-help").hidden = true; $("plan").hidden = true;
  $("diagnostics").hidden = true; $("export").hidden = true; lastSearch = null;
  try {
    const value = query();
    const plan = await responseJson(await fetch("/api/plan", {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(value)
    }));
    showPlan(plan);
    if (reviewOnly) { $("status").textContent = "Review the filters, adjust the controls, then search."; return; }
    const form = new FormData(); form.append("query", JSON.stringify(value));
    const file = $("image").files[0]; if (file) form.append("image", file);
    const data = await responseJson(await fetch("/api/search", {method: "POST", body: form}));
    showPlan(data.plan); showResults(data);
  } catch (error) { $("status").textContent = error.message; }
  finally { $("submit").disabled = false; $("review").disabled = false; }
}
$("search-form").addEventListener("submit", event => { event.preventDefault(); run(false); });
$("review").addEventListener("click", () => run(true));
$("export").addEventListener("click", () => {
  if (!lastSearch) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(lastSearch, null, 2)], {type: "application/json"}));
  const link = document.createElement("a"); link.href = url; link.download = "eo-search-record.json";
  link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});
