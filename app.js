// DealShield MVP — EN-only. Runs fully in browser.
// AI: Named Entity Recognition via Transformers.js (ONNX Runtime in browser).
// Ref: https://huggingface.co/docs/transformers.js/en/index  | CDN examples via jsDelivr package page.
// NER model: Xenova/bert-base-NER (Transformers.js compatible).

// Lazy-load Transformers.js to avoid hard failure if a CDN/model is unavailable.
let _pipelineFn = null;
let _pipelineLoadPromise = null;

async function _getPipelineFn(){
  if (_pipelineFn) return _pipelineFn;
  if (_pipelineLoadPromise) return _pipelineLoadPromise;
  const sources = [
    "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2",
    "https://unpkg.com/@xenova/transformers@2.17.2?module",
    "https://esm.sh/@xenova/transformers@2.17.2"
  ];
  _pipelineLoadPromise = (async () => {
    let lastErr = null;
    for (const url of sources){
      try {
        const mod = await import(url);
        if (mod && typeof mod.pipeline === "function"){
          _pipelineFn = mod.pipeline;
          return _pipelineFn;
        }
      } catch (e) { lastErr = e; }
    }
    throw lastErr || new Error("Failed to load transformers pipeline");
  })();
  return _pipelineLoadPromise;
}


const $ = (id) => document.getElementById(id);

const aiStatus = $("aiStatus");
const input = $("input");
const analyzeBtn = $("analyzeBtn");

const scoreEl = $("score");
const riskPill = $("riskPill");
const reasonsEl = $("reasons");

const partiesEl = $("parties");
const amountEl = $("amount");
const deadlineEl = $("deadline");
const paymentEl = $("payment");
const linksEl = $("links");

const planEl = $("plan");
const replyBox = $("replyBox");
const highlightedEl = $("highlighted");

const demo1 = $("demo1");
const demo2 = $("demo2");
const demo3 = $("demo3");

const copyReplyBtn = $("copyReply");
const exportBtn = $("exportBtn");
const exportMenu = $("exportMenu");
const exportTxtBtn = $("exportTxt");
const exportPdfBtn = $("exportPdf");
let ner = null;

const DEMOS = {
  clean: `Hi Omar,
We’d like to hire you for a landing page redesign. Budget is $1,200, delivery in 10 days.
Payment: 50% upfront via bank transfer, 50% after delivery.
Please confirm the milestone breakdown and send an invoice.
— Sarah, Northwind Studio`,
  bank_change: `Hello,
Quick update: our bank details have changed. Please pay the invoice to the NEW account below today.
Account name: NW Trading Ltd
IBAN: XX00 0000 0000 0000
Also, keep this confidential and do not contact anyone else — we’re in a rush.
Thanks.`,
  advance_fee: `URGENT: We need to secure your service now.
To start, please pay the “activation fee” of $150 today. After that, we will release the full $3,000.
Use this link to confirm: bit.ly/pay-confirm
We only accept crypto. Don’t tell anyone about this deal.`
};

function setPill(level){
  riskPill.className = "pill " + (level === "LOW" ? "low" : level === "MEDIUM" ? "med" : "high");
  riskPill.textContent = level;
}

function escapeHtml(str){
  return str.replace(/[&<>"']/g, (m) => ({
    "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"
  }[m]));
}

function highlightMatches(text, matches){
  // matches: array of substrings (case-insensitive) to highlight
  let safe = escapeHtml(text);
  // highlight longer phrases first
  const uniq = Array.from(new Set(matches)).filter(Boolean).sort((a,b)=>b.length-a.length);
  for (const phrase of uniq){
    const re = new RegExp(phrase.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "ig");
    safe = safe.replace(re, (m)=>`<mark>${m}</mark>`);
  }
  return safe;
}

function extractLinks(text){
  const urlRe = /\bhttps?:\/\/[^\s)]+|\b(?:bit\.ly|t\.co|tinyurl\.com|goo\.gl)\/[^\s)]+|\b[a-z0-9.-]+\.[a-z]{2,}(?:\/[^\s)]*)?/ig;
  const out = (text.match(urlRe) || []).slice(0, 8);
  return Array.from(new Set(out));
}

function extractAmount(text){
  const re = /(\$|€|£)\s?\d{1,3}(?:[,\s]\d{3})*(?:\.\d{2})?|\b\d{1,3}(?:[,\s]\d{3})*(?:\.\d{2})?\s?(USD|EUR|GBP)\b/ig;
  const m = text.match(re);
  return m ? m[0] : null;
}

function extractDeadline(text){
  const re = /\b(in\s+\d{1,3}\s+(days?|weeks?)|by\s+\w+\s+\d{1,2}|\b\d{1,2}\/\d{1,2}\/\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b)\b/ig;
  const m = text.match(re);
  return m ? m[0] : null;
}

function detectPayment(text){
  const t = text.toLowerCase();
  const hits = [];
  if (/(iban|swift|bank transfer|wire)/i.test(text)) hits.push("Bank transfer");
  if (/(paypal)/i.test(text)) hits.push("PayPal");
  if (/(crypto|usdt|btc|eth|wallet)/i.test(text)) hits.push("Crypto");
  if (/(gift card|voucher)/i.test(text)) hits.push("Gift cards");
  if (!hits.length) return "Unspecified";
  return Array.from(new Set(hits)).join(", ");
}

function scoreRisk(text){
  const t = text.toLowerCase();
  const reasons = [];
  const plan = [];
  let score = 10;

  const add = (pts, label, triggerPhrases=[], actions=[]) => {
    score += pts;
    reasons.push({ label, pts, triggerPhrases });
    for (const a of actions) plan.push(a);
  };

  // Core signals
  if (/(urgent|asap|today|immediately|right now|rush)/i.test(text)){
    add(12, "Urgency pressure", ["urgent","asap","today","immediately","right now","rush"], [
      "Slow down: verify key terms before sending money.",
    ]);
  }
  if (/(confidential|don’t tell|keep this secret)/i.test(text)){
    add(18, "Secrecy request", ["confidential","don’t tell","keep this secret"], [
      "Treat secrecy requests as a red flag. Confirm identity via a second channel.",
    ]);
  }
  if (/(activation fee|processing fee|release the funds|advance fee|to start, pay)/i.test(text)){
    add(28, "Advance-fee / pay-first pattern", ["activation fee","processing fee","advance fee","to start, pay"], [
      "Do not pay fees upfront. Require clear contract + verifiable business identity.",
      "Ask for a standard invoice and verifiable company details.",
    ]);
  }
  if (/(bank details have changed|new account|pay to the new|updated payment details)/i.test(text)){
    add(30, "Payment details change request", ["bank details have changed","new account","updated payment details"], [
      "Freeze payments: confirm new payment details via a verified second channel (call / known contact).",
      "Compare the new details against previous invoices / contracts.",
    ]);
  }
  const links = extractLinks(text);
  if (links.some(l=>/(bit\.ly|t\.co|tinyurl\.com|goo\.gl)/i.test(l))){
    add(20, "Shortened link", ["bit.ly","t.co","tinyurl.com","goo.gl"], [
      "Avoid shortened links for payments. Request the full official domain.",
    ]);
  }
  if (/(only accept crypto|crypto only|usdt only)/i.test(text)){
    add(16, "Payment rail restriction (crypto-only)", ["only accept crypto","crypto only","usdt only"], [
      "Prefer standard invoicing and traceable business payment rails for first-time counterparties.",
    ]);
  }

  // Ambiguity / missing details
  const amt = extractAmount(text);
  if (!amt){
    add(10, "Missing or unclear amount", [], [
      "Clarify the exact amount, currency, and milestone schedule in writing.",
    ]);
  }
  const dl = extractDeadline(text);
  if (!dl){
    add(6, "Missing deadline / deliverables clarity", [], [
      "Confirm deliverables, acceptance criteria, and deadline.",
    ]);
  }

  // Normalize plan (unique + ordered)
  const uniqPlan = Array.from(new Set(plan));

  // Clamp score
  score = Math.max(0, Math.min(100, score));

  let level = "LOW";
  if (score >= 70) level = "HIGH";
  else if (score >= 40) level = "MEDIUM";

  return { score, level, reasons, plan: uniqPlan, links };
}

async function ensureNER(opts = {}){
  const silent = !!opts.silent;
  if (ner) return ner;
  if (!silent) aiStatus.textContent = "AI: loading…";

  // Use quantized model for browsers when available.
  const pipeline = await _getPipelineFn();
  // Smaller NER model = faster cold-start.
  const modelId = "Xenova/distilbert-base-cased-finetuned-conll03-english";

  try {
    ner = await pipeline("token-classification", modelId, { quantized: true });
  } catch (e) {
    // Some environments don't support quantized weights; retry.
    ner = await pipeline("token-classification", modelId);
  }

  if (!silent) aiStatus.textContent = "AI: on-device ✓";
  return ner;
}


function buildSafeReply(level, snapshot){
  const tone = level === "HIGH"
    ? "Before proceeding, I need to verify a few details for safety."
    : level === "MEDIUM"
      ? "Quick verification before we proceed:"
      : "Just confirming a couple of details to avoid misunderstandings:";

  const lines = [
    "Hi — thanks for the update.",
    tone,
    "",
    "1) Please confirm the exact amount + currency and the payment method.",
    "2) Please confirm the payment details via a second channel (call / known contact).",
    "3) Please share a standard invoice and your company details (legal name, website, address).",
  ];

  if (snapshot.links.length) lines.push("4) Please share the full official domain (no shortened links).");
  if (snapshot.payment.includes("Crypto")) lines.push("5) For first-time engagements, I prefer standard invoicing and traceable business payment rails.");
  lines.push("");
  lines.push("Once confirmed, I’m happy to proceed immediately.");
  return lines.join("\n");
}

function renderReasons(reasons){
  reasonsEl.innerHTML = "";
  if (!reasons.length){
    const li = document.createElement("li");
    li.textContent = "No major red flags detected in this message.";
    reasonsEl.appendChild(li);
    return;
  }
  for (const r of reasons.sort((a,b)=>b.pts-a.pts)){
    const li = document.createElement("li");
    li.textContent = `${r.label} (+${r.pts})`;
    reasonsEl.appendChild(li);
  }
}

function renderPlan(plan){
  planEl.innerHTML = "";
  for (const p of plan){
    const li = document.createElement("li");
    li.textContent = p;
    planEl.appendChild(li);
  }
  if (!plan.length){
    const li = document.createElement("li");
    li.textContent = "Proceed with standard invoicing and confirm details in writing.";
    planEl.appendChild(li);
  }
}

function renderSnapshot({ parties, amount, deadline, payment, links }){
  partiesEl.textContent = parties || "—";
  amountEl.textContent = amount || "—";
  deadlineEl.textContent = deadline || "—";
  paymentEl.textContent = payment || "—";
  linksEl.textContent = links.length ? links.join("  •  ") : "—";
}

function buildMarkdownSummary(result, snapshot, reply, rawText){
  const lines = [];
  lines.push(`# DealShield Summary`);
  lines.push(`**Risk:** ${result.level} (${result.score}/100)`);
  lines.push(``);
  lines.push(`## Deal Snapshot`);
  lines.push(`- Parties: ${snapshot.parties || "—"}`);
  lines.push(`- Amount: ${snapshot.amount || "—"}`);
  lines.push(`- Deadline: ${snapshot.deadline || "—"}`);
  lines.push(`- Payment: ${snapshot.payment || "—"}`);
  lines.push(`- Links: ${snapshot.links.length ? snapshot.links.join(", ") : "—"}`);
  lines.push(``);
  lines.push(`## Reasons`);
  if (result.reasons.length){
    for (const r of result.reasons.sort((a,b)=>b.pts-a.pts)) lines.push(`- ${r.label} (+${r.pts})`);
  } else {
    lines.push(`- No major red flags detected.`);
  }
  lines.push(``);
  lines.push(`## Verification Plan`);
  for (const p of result.plan) lines.push(`- ${p}`);
  lines.push(``);
  lines.push(`## Safe Reply`);
  lines.push("```");
  lines.push(reply);
  lines.push("```");
  lines.push(``);
  lines.push(`## Original input`);
  lines.push("```");
  lines.push(rawText);
  lines.push("```");
  return lines.join("\n");
}
function markdownToPlain(md){
  return md
    .replace(/^#{1,6}\s+/gm, "")
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/```\s*\n?/g, "")
    .replace(/^-\s+/gm, "• ")
    .trim();
}

function buildHtmlReport(result, snapshot, reply, rawText){
  const esc = (s) => (s || "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
  const reasons = (result.reasons && result.reasons.length)
    ? result.reasons.slice().sort((a,b)=>b.pts-a.pts).map(r => `<li>${esc(r.label)} <span style="opacity:.7;">(+${r.pts})</span></li>`).join("")
    : `<li>No major red flags detected.</li>`;
  const plan = (result.plan || []).map(p => `<li>${esc(p)}</li>`).join("");

  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>DealShield Report</title>
<style>
  body{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 32px; color:#111; }
  h1{ margin:0 0 8px; font-size: 22px; }
  .meta{ margin: 0 0 18px; font-size: 13px; color:#444; }
  .pill{ display:inline-block; padding:4px 10px; border-radius:999px; background:#f3f3f3; font-weight:600; }
  h2{ margin-top: 18px; font-size: 16px; }
  ul{ margin: 8px 0 0 18px; }
  pre{ background:#f7f7f7; padding:12px; border-radius:10px; white-space:pre-wrap; }
  table{ border-collapse: collapse; margin-top: 8px; width: 100%; }
  td{ padding:6px 8px; border-bottom:1px solid #eee; vertical-align: top; }
  td:first-child{ width: 180px; color:#444; }
  .small{ font-size:12px; color:#666; margin-top: 22px; }
  @media print{ body{ margin: 18mm; } }
</style>
</head>
<body>
  <h1>DealShield Report</h1>
  <p class="meta"><span class="pill">Risk: ${esc(result.level)} (${result.score}/100)</span></p>

  <h2>Deal Snapshot</h2>
  <table>
    <tr><td>Parties</td><td>${esc(snapshot.parties || "—")}</td></tr>
    <tr><td>Amount</td><td>${esc(snapshot.amount || "—")}</td></tr>
    <tr><td>Deadline</td><td>${esc(snapshot.deadline || "—")}</td></tr>
    <tr><td>Payment</td><td>${esc(snapshot.payment || "—")}</td></tr>
    <tr><td>Links</td><td>${esc((snapshot.links && snapshot.links.length) ? snapshot.links.join(", ") : "—")}</td></tr>
  </table>

  <h2>Reasons</h2>
  <ul>${reasons}</ul>

  <h2>Verification Plan</h2>
  <ul>${plan}</ul>

  <h2>Safe Reply</h2>
  <pre>${esc(reply)}</pre>

  <h2>Original input</h2>
  <pre>${esc(rawText)}</pre>

  <div class="small">Generated by DealShield (client-side).</div>
</body>
</html>`;
}


async function analyze(){
  const text = input.value.trim();
  if (!text){
    alert("Paste some text first.");
    return;
  }

  // Rule-based signals (fast)
  const result = scoreRisk(text);
  scoreEl.textContent = String(result.score);
  setPill(result.level);
  renderReasons(result.reasons);
  renderPlan(result.plan);

  // Snapshot (rules)
  const snapshot = {
    amount: extractAmount(text),
    deadline: extractDeadline(text),
    payment: detectPayment(text),
    links: result.links,
    parties: null,
  };

  // AI: parties via NER (best-effort)
  let triggers = [];
  for (const r of result.reasons) triggers = triggers.concat(r.triggerPhrases || []);
  triggers = triggers.concat(result.links.filter(Boolean));

  try{
    const classifier = await ensureNER();
    aiStatus.textContent = "AI: extracting…";
    const ents = await classifier(text);
    // Keep only top entities; group by label
    const keep = ents
      .filter(e => e.score >= 0.60 && ["PER","ORG","LOC","MISC"].includes(e.entity))
      .slice(0, 18);

    const pretty = [];
    const seen = new Set();
    for (const e of keep){
      const key = `${e.word}|${e.entity}`;
      if (seen.has(key)) continue;
      seen.add(key);
      pretty.push(`${e.word} (${e.entity})`);
    }
    snapshot.parties = pretty.length ? pretty.join(", ") : null;
    aiStatus.textContent = "AI: on-device ✓";
  } catch (e){
    console.warn(e);
    aiStatus.textContent = "AI: optional";
  }

  renderSnapshot(snapshot);

  // Safe reply
  const reply = buildSafeReply(result.level, snapshot);
  replyBox.textContent = reply;

  // Highlighted view
  const highlighted = highlightMatches(text, triggers);
  highlightedEl.innerHTML = highlighted;
}

demo1.addEventListener("click", () => { input.value = DEMOS.clean; });
demo2.addEventListener("click", () => { input.value = DEMOS.bank_change; });
demo3.addEventListener("click", () => { input.value = DEMOS.advance_fee; });

analyzeBtn.addEventListener("click", analyze);

copyReplyBtn.addEventListener("click", async () => {
  const t = replyBox.textContent.trim();
  if (!t || t === "—") return;
  await navigator.clipboard.writeText(t);
  copyReplyBtn.textContent = "Copied ✓";
  setTimeout(()=>copyReplyBtn.textContent="Copy safe reply", 900);
});

// --- Export (single button + format menu) ---
function getReportBundle() {
  const text = input.value.trim();
  if (!text) return null;

  const result = scoreRisk(text);

  const snapshot = {
    amount: extractAmount(text),
    deadline: extractDeadline(text),
    payment: detectPayment(text),
    links: result.links,
    parties: partiesEl.textContent === "—" ? null : partiesEl.textContent,
  };

  const reply =
    replyBox.textContent.trim() === "—"
      ? buildSafeReply(result.level, snapshot)
      : replyBox.textContent.trim();

  const md = buildMarkdownSummary(result, snapshot, reply, text);
  const txt = markdownToPlain(md);
  const html = buildHtmlReport(result, snapshot, reply, text);

  return { txt, html };
}

function downloadText(filename, content) {
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function exportAsTxt() {
  const bundle = getReportBundle();
  if (!bundle) return;
  downloadText("dealshield_report.txt", bundle.txt);
}

function exportAsPdf() {
  const bundle = getReportBundle();
  if (!bundle) return;

  const d = new Date();
  const ymd = d.toISOString().slice(0, 10);
  const filename = `dealshield_report_${ymd}.pdf`;

  // Preferred: direct download to the browser’s default Downloads folder (no print dialog).
  try {
    const jspdf = window.jspdf;
    if (jspdf && typeof jspdf.jsPDF === "function") {
      const { jsPDF } = jspdf;
      const doc = new jsPDF({ unit: "pt", format: "a4" });

      const margin = 48;
      const pageW = doc.internal.pageSize.getWidth();
      const pageH = doc.internal.pageSize.getHeight();
      const maxW = pageW - margin * 2;

      doc.setFont("courier", "normal");
      doc.setFontSize(11);

      const lineH = 14;
      const raw = (bundle.txt || "").replace(/\r/g, "");
      const paragraphs = raw.split("\n");

      let y = margin;
      for (const p of paragraphs) {
        if (p.trim() === "") {
          y += lineH;
          if (y > pageH - margin) {
            doc.addPage();
            y = margin;
          }
          continue;
        }
        const lines = doc.splitTextToSize(p, maxW);
        for (const line of lines) {
          doc.text(String(line), margin, y);
          y += lineH;
          if (y > pageH - margin) {
            doc.addPage();
            y = margin;
          }
        }
      }

      doc.save(filename);
      return;
    }
  } catch (err) {
    console.warn("Direct PDF export failed; falling back to print.", err);
  }

  // Fallback: browser print dialog (still produces a valid PDF via “Save as PDF”).
  const w = window.open("", "_blank");
  if (!w) return;
  w.document.open();
  w.document.write(bundle.html);
  w.document.close();
  w.focus();
  setTimeout(() => { w.print(); }, 250);
}

// Menu open/close
function openExportMenu() {
  if (!exportMenu || !exportBtn) return;
  exportMenu.hidden = false;
  exportBtn.setAttribute("aria-expanded", "true");
}
function closeExportMenu() {
  if (!exportMenu || !exportBtn) return;
  exportMenu.hidden = true;
  exportBtn.setAttribute("aria-expanded", "false");
}
function toggleExportMenu() {
  if (!exportMenu) return;
  exportMenu.hidden ? openExportMenu() : closeExportMenu();
}

if (exportBtn) {
  exportBtn.addEventListener("click", (e) => {
    e.preventDefault();
    toggleExportMenu();
  });
}
if (exportTxtBtn) {
  exportTxtBtn.addEventListener("click", (e) => {
    e.preventDefault();
    closeExportMenu();
    exportAsTxt();
  });
}
if (exportPdfBtn) {
  exportPdfBtn.addEventListener("click", (e) => {
    e.preventDefault();
    closeExportMenu();
    exportAsPdf();
  });
}

document.addEventListener("click", (e) => {
  if (!exportMenu || exportMenu.hidden) return;
  if (!e.target.closest("#exportWrap")) closeExportMenu();
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeExportMenu();
});


// Default demo text
input.value = DEMOS.clean;
highlightedEl.textContent = "Run analysis to see highlighted signals.";

// Warm up the on-device NER model in the background (non-blocking).
setTimeout(() => {
  ensureNER({ silent: true }).catch(() => {});
}, 300);
