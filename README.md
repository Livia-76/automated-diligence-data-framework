# Automated Diligence Data Framework
> Hybrid AI and local automation framework developed to unify fragmented diligence data at Evercore PCA.

🧱 Excel → 🧮 Python (local) → 🤖 LLM → 📊 Unified Client IDs

**Context**  
During my internship at Evercore’s Private Capital Advisory (PCA) group, I encountered a core diligence challenge in a **$1B+ continuation vehicle**: understanding the target company’s future organic revenue growth by analyzing **competitive wins and losses across 3,000+ customers**.  
The underlying data was fragmented—some records had typos, others only included logos—making it nearly impossible to align client names across different datasets.

Instead of treating this as a manual data-cleaning task, I reframed it as a **design problem**: how could we create a single source of truth?

---

## 💡 Problem → Solution

| Stage | Approach | Limitation |
|-------|-----------|------------|
| 1️⃣ Excel lookup formulas | Initial attempt using `VLOOKUP`/`XLOOKUP` | Broke down on typos and inconsistent naming |
| 2️⃣ Copilot fuzzy matching *(early test, before integrated release)* | Excel Copilot previewed fuzzy match capability but lacked control and accuracy | Inconsistent and black-box |
| 3️⃣ Python automation (final) | Designed a hybrid automation system combining **LLM-based semantic reasoning** and **local logic-driven fuzzy matching** | Produced scalable, transparent results |

> *At the time (July 2025), while Microsoft Copilot offered enhanced Excel/AI productivity features, I found the enterprise-scale fuzzy-matching and data-cleanup control I needed was not yet fully mature—so I built my own Python solution for accuracy and transparency.*

---

## 🧩 Framework Overview

**Goal:** Create a unified mapping of all client entities across multiple diligence datasets.


**Outcome:**  
- Reduced manual processing time by ~90%.  
- Produced a reusable workflow for future revenue-growth and competitive-win analyses.  
- Improved accuracy and traceability of client matching for investor reporting.

---

## 🧠 Components

### `keyword_linker_local.py`
Local, rule-based fuzzy matcher built for privacy-safe execution behind corporate firewalls.  
- Uses `rapidfuzz` for string similarity scoring.  
- Handles normalization, abbreviation mapping, and deterministic rule layers.  
- Designed to run entirely offline for data-sensitive diligence workflows.

### `semantic_matcher_llm.py`
LLM-powered semantic matcher leveraging Anthropic’s Claude API (key removed for security).  
- Reads multiple Excel sheets and compares company metadata contextually.  
- Uses semantic embeddings to link entities missed by pure fuzzy matching.  
- Modular design for integration into broader diligence automation pipelines.

---

## ⚙️ Key Learnings

- **Design thinking in finance:** Ambiguity can be engineered away by layering complementary tools rather than seeking a single perfect one.  
- **From finance to product mindset:** Framing “data cleaning” as a **user-experience problem** led to reusable workflows and deeper insight generation.  
- **Technical growth:** From installing Python on a restricted corporate laptop to optimizing a local script under tight runtime constraints, this project stretched both my technical and adaptive skills.

---

## 📘 Reflection

What began as a bottleneck became my most valuable learning experience that summer.  
The project taught me that **product thinking isn’t limited to software teams**—it’s about bringing clarity and structure to complex, ambiguous systems. In investment banking, that means transforming unstructured information into insights investors can act on.

---

## 🔗 Repository Contents
| File | Description |
|------|--------------|
| `semantic_matcher_llm.py` | LLM-based semantic matcher for context-aware entity linkage. |
| `keyword_linker_local.py` | Offline fuzzy matcher for secure diligence workflows. |
| `README.md` | Project context, methodology, and reflection. |

---

**Note:**  
All data references are synthetic. No client or confidential information is included.  
API keys have been removed; use environment variables to authenticate locally.
