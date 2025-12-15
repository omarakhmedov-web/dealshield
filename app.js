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
const exportMdBtn = $("exportMd");

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

async function ensureNER(){
  if (ner) return ner;

  aiStatus.textContent = "Mode: loading…";
  try {
    const pipeline = await _getPipelineFn();
    // Smaller NER model = faster cold-start.
    const modelId = "Xenova/distilbert-base-cased-finetuned-conll03-english";

    try {
      // Use quantized model for browsers when available.
      ner = await pipeline("token-classification", modelId, { quantized: true });
    } catch (e) {
      // Some environments don't support quantized weights; retry.
      ner = await pipeline("token-classification", modelId);
    }

    aiStatus.textContent = "Mode: AI + rules";
    return ner;
  } catch (e) {
    console.warn(e);
    aiStatus.textContent = "Mode: rules-only";
    throw e;
  }
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
    aiStatus.textContent = "Mode: extracting…";
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
    aiStatus.textContent = "Mode: AI + rules";
  } catch (e){
    console.warn(e);
    aiStatus.textContent = "Mode: rules-only";
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

exportMdBtn.addEventListener("click", async () => {
  const text = input.value.trim();
  if (!text) return;
  const result = scoreRisk(text);

  const snapshot = {
    amount: extractAmount(text),
    deadline: extractDeadline(text),
    payment: detectPayment(text),
    links: result.links,
    parties: partiesEl.textContent === "—" ? null : partiesEl.textContent,
  };
  const reply = replyBox.textContent.trim() === "—" ? buildSafeReply(result.level, snapshot) : replyBox.textContent.trim();

  const md = buildMarkdownSummary(result, snapshot, reply, text);
  const blob = new Blob([md], { type: "text/markdown" });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = "dealshield_summary.md";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
});

// Default demo text
input.value = DEMOS.clean;
highlightedEl.textContent = "Run analysis to see highlighted signals.";

// Warm up the on-device NER model in the background (non-blocking).
setTimeout(() => {
  ensureNER().catch(() => {});
}, 300);
