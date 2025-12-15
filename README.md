# DealShield — AI Deal Risk & Verification Copilot (VisaVerse AI Hackathon)

DealShield turns messy cross-border chats/emails/invoice text into:
- a **Deal Snapshot** (who / what / how much / when / how to pay),
- an **explainable risk score** with highlighted signals,
- a **step-by-step verification plan**,
- a **safe reply** you can copy/paste.

This prototype runs fully in the browser (no server required).
It uses optional **on-device AI** (Named Entity Recognition) and deterministic rules for risk signals.

## Live demo
https://<your-project>.pages.dev/

## How to run locally
Just open `index.html` in a modern browser (Chrome recommended).  
Note: the first AI run may take longer while the model downloads.

## Tech
- HTML/CSS/JavaScript
- Transformers.js (ONNX Runtime in the browser)
- Cloudflare Pages (deployment)

## Responsible use
DealShield is a risk-awareness and verification assistant. It does not replace legal advice or KYC.
Always verify payment details via a second channel before sending money.
