# Classificador AHU-Sul v3.1 — Redesign plan

Status: approved by the author on 2026-10-09, after one hour of hands-on use of the F3 build (commit df0afc2) on localhost.
This plan replaces the open parts of F3 and goes **before** F4 and the first deploy.
Keep it in `docs/plano_remodelacao_v3_1.md`. Each R-package is one commit, and `scripts/gate.py` must PASS before each commit.

---

## 0. What the author asked for

**What the author found, in his own words (summarised):**

1. **Browser tab:** remove the emoji from the tab icon.
2. **Labels:** remove careless parentheticals such as `Rei (King of Portugal)` and `Pessoas (remetentes)`. A king is just "Rei", unless he is not the king of Portugal, in which case it is "Rei (Espanha)".
3. **Selection is slow:** every click greys the page, flashes white and shows the loading animation, even after the models are warm.
4. **JSON export is useless:** it holds only `doc_uid`s plus provenance. The author needs **only a PDF table** of the selected documents, like the one built for the AHU e-mail. No JSON, no CSV, no `doc_uid` shown anywhere.
5. **The PDF did not work.** It must be minimal and functional, a list for photographic record at the AHU:
   - title "Lista para consulta no AHU";
   - columns year, cota, summary and DigitArq link;
   - no explanations.
6. **SRSP audit line:** replace the line "SRSP do DeepSeek v3, auditado no preprint §2.1: MAE 1,13 · r = 0,48 …" with a single "?" circle that opens a "saiba mais" explaining the inferences and the audit.
7. **Lenses:** remove the methodological lenses ("lentes") completely. They will not be used.
8. **Visual identity:** adopt the EINS visual identity (`relatorio_identidade_visual_eins.md`) creatively. Take only its 2–3 most recognisable traits: manuscript glyphs instead of icons, the fonts, and the bordô accent. The goal is a family resemblance across the author's tools, not a pixel copy.

**Carried over from the previous round, not yet done:**

- **Torch-free query path (ONNX).** The real peak is 948 MiB, too close to the ~1 GiB where Community Cloud kills apps.
- **The preprint's own wording for the scribe-mediation caveat.** It now lives in the "saiba mais" (see R5).

---

## 1. Decisions (new, numbered after D1–D6)

**D7 — Export is one thing: the AHU consultation PDF.**
- Remove from the UI:
  - CSV export;
  - JSON export of results;
  - JSON download/upload of the selection;
  - the PDF dossier (WP 3.3).
- Delete the code paths that only served them (`ahu/exportar.py`, `ahu/dossie.py`, their tests and their UI). One maintainer (R2) should not carry dead code. Git history keeps them if ever needed.
- Acceptance criteria A8 (dossier) and the CSV/JSON parts of §8.2 are **retired**. Record this in the spec.

**D8 — `scribe_mediation_likely` leaves the per-document UI and PDFs entirely.**
- D1 put it "only in the PDF". The PDF that carried it (the dossier) is gone, and the AHU list must stay minimal.
- The field stays in the data. It is explained, with the preprint's caveat, only in the "saiba mais" (R5). It is never shown as a per-document value.
- Update AGENTS.md rule 4 accordingly.

**D9 — Lenses removed.**
- Remove the `lente` field from the `Consulta` schema and bump `Consulta.versao` to 2.
- Old links that carry `lente` must still open. Ignore the field silently and keep all other fields. Add a test for this.

**D10 — No internal identifiers in the interface or in the PDF.**
- `doc_uid` and `pessoa_uid` are internal. The user sees cota, year, summary and names.
- The "Reportar erro" issue body may still carry the uid, since that goes to the maintainer.

**D11 — Kept on purpose.**
- The **Amostras** tab and its **EINS plan export** (`eins-plano-coleta/1`) stay, because D4 requires it. The author's "no JSON" request was about the selection export.
- Flag this in the report. If the author later wants Amostras gone too, it is a separate change.

---

## 2. Work packages

### R1 — Remove (S)
- Delete the lenses (D9): UI, the `Consulta` field, query-param handling, tests.
- Delete CSV, JSON and dossier exports, and the selection JSON download/upload (D7).
- Remove from all UI text every reference to `doc_uid`, `pessoa_uid` and the scribe field (D8, D10).
- Update `spec`, `plano`, AGENTS.md and the "Sobre" tab to match.
- **Done when:** gate PASS, the app runs, and a grep of the UI layer finds no `doc_uid` string rendered to the user.

### R2 — Fast selection (M) — highest priority
Root cause to confirm first. Every checkbox click triggers a full Streamlit rerun, which re-runs the hybrid search and re-renders the whole result list.

**Fix with:**
- **Cached search results.** Cache by the canonical `Consulta` (e.g. `st.cache_data`), so a rerun with an unchanged query never re-scores BM25 or embeddings.
- **Fragments.** Render the result list and the selection controls inside `st.fragment`, so ticking a box reruns only that fragment.
- **Session state.** Update the selection with `on_change` callbacks on `st.session_state`, not by re-reading every widget.
- **Selection counter.** Keep it inside the fragment, or update it without a full rerun.

**Acceptance:**
- Toggling a selection does **not** call the search engine. Prove it with a test using a call counter on the search function.
- On the author's machine, with models warm, toggling shows no greyed page and no loading animation.
- Server time for a toggle is under 300 ms. Report the measurement.

**Persistence (no server storage, R4).**
- Mirror the selection in the URL query params, compactly (a short hash prefix per document, checked for collisions against the corpus).
- A page reload or a bookmarked link then restores it without showing any id.
- If the URL would exceed ~1,800 characters, stop mirroring and show one muted line: "Seleção grande demais para o link; gere o PDF para guardá-la."

**Selection panel, minimal:**
- one line per document (⸿ · ano · cota · start of the summary), with ✕ to remove;
- "Limpar seleção";
- "Gerar PDF".
- No sorting options and no explanatory text.
- Keep "colar cotas" (pasting a list of cotas, with `#` groups and `|` notes) as a closed drawer (❧). It is how the author built his real AHU list.

### R3 — The AHU list PDF (S)
One PDF, built from the selection or from pasted cotas.

**Page:**
- A4 landscape.
- Title **"Lista para consulta no AHU"**, the generation date, and nothing else on top.

**Columns:**

| Nº | Ano | Cota | Resumo | DigitArq | ☐ |
|---|---|---|---|---|---|

- **Nº:** matches photos to rows.
- **Cota:** the AHU cota on one line, the `PT/AHU/CU/…` reference below it in small type.
- **Resumo:** full, wrapped.
- **DigitArq:** a clickable short label ("abrir"), not the raw URL.
- **☐:** to tick while photographing.

**Excluded:** no `doc_uid`, no provenance block, no tipologia, no sender, no SRSP, no scribe field.

**Format:**
- Groups from pasted `#` lines become group header rows.
- Font: Noto Serif (OFL, embedded), body 9 pt, line height 1.35.
- Footer, 8 pt: "Classificador AHU-Sul · página p/P".
- No logos.

**Done when:**
- The PDF opens correctly in Edge, Adobe Reader and a phone viewer.
- Accents, links and page breaks are right, and no row is split mid-summary.
- The test with all 7,051 documents still passes: 7,050 with a reference, the one known exception reported.
- **Attach a sample PDF of 15 documents to the report.**

### R4 — Labels and text (S)
Full sweep of every label and value shown to the user.

**Recipient and sender values:**
- Normalise DeepSeek's English `recipient_name` and `sender_name` values to Portuguese **at build time**, in `build/construir_app.py`, through a small reviewed mapping table (`build/rotulos_pt.json`).
- Examples:
  - `King of Portugal` → `Rei`;
  - `King of Spain` → `Rei (Espanha)`;
  - `Overseas Council` → `Conselho Ultramarino`.
- Values with no mapping keep the original, and are listed in a review CSV for the author.

**Labels:**
- Remove redundant parentheticals: `Pessoas (remetentes)` becomes `Remetentes`.
- Category, vector and typology labels from the DeepSeek layer (e.g. `Metropolitan Elite`, `Bottom-Up`) must appear in Portuguese everywhere they are shown.

**Tab icon:**
- Remove the emoji from `page_icon`.
- Use the Streamlit default, or a Material icon if one fits clearly (e.g. `:material/history_edu:`).

**Done when:** a test fails if any English category or vector label reaches the UI layer, and the review CSV of unmapped names is produced.

### R5 — "Saiba mais" (S)
- Replace the audit line with a small "?" control (e.g. `st.popover`) next to the SRSP / inference filters.
- **Contents**, in Portuguese and short:
  1. What the inferences are: DeepSeek v3, zero-shot, applied to the catalogue summary and not to the manuscript. They produce:
     - the sender category;
     - the communication vector;
     - the typology impact;
     - the SRSP score from 1 to 10.
  2. The audit numbers: MAE 1,13 · r = 0,48 · viés +0,59 · N = 705 · one evaluator. Differences of ±1 point are not distinguishable.
  3. Scribe mediation (D8): what it is, and the preprint's own caveat, quoted verbatim:

     > "Essa inferência, contudo, é metodologicamente delicada, por mais que seja uma informação fundamental para a linguística diacrônica (uma vez que a escolaridade variável desses notários coloniais atuava como um filtro permeável, ora favorecendo a hipercorreção, ora permitindo o escape indeliberado da sintaxe local), a sua classificação de forma puramente heurística é perigosa. Não é possível atestar tal mediação de forma categórica sem um extenso estudo paleográfico e diplomático do manuscrito original."

  4. A link to the preprint: https://doi.org/10.1590/SciELOPreprints.16461
- Biography confidence (ALTA/MÉDIA) gets the same "?" treatment if it currently carries a long explanatory line.

### R6 — Visual identity, EINS family (M)
Adopt **three** recognisable traits. Do not attempt full parity with a Qt app.

1. **Bordô as the single accent.**
   - Configure the theme in `.streamlit/config.toml`. Use separate light and dark themes if Streamlit 1.55 supports them; check the docs and do not guess.
   - Light ("papel envelhecido"):

     | Role | Colour |
     |---|---|
     | background | `#f5f2eb` |
     | secondary background | `#ece8dd` |
     | text | `#1c1b18` |
     | primary | `#800000` |

   - Dark:

     | Role | Colour |
     |---|---|
     | background | `#0d0d0d` |
     | secondary background | `#161616` |
     | text | `#e0e0e0` |
     | primary | `#a83030` |

     Pick whichever bordô passes 4.5:1 for text on the dark background.
   - The country palette of the charts stays as validated in WP 2.4. Vivid colour belongs to data, bordô to the interface, as in EINS.
2. **Manuscript glyphs instead of icons.**
   - **⸿ marks "selected".** Use it on selected result cards, in the selection panel and in the selection counter ("⸿ 5"). Selected cards also get the EINS 3-px bordô left bar, via the container `key` CSS class.
   - **❧ marks secondary, collapsible sections.** Put it at the start of expander and drawer labels, e.g. "❧ Colar cotas" and "❧ Filtros avançados".
   - Remove decorative emoji everywhere in the UI.
   - **Allowed symbols:** ❧ ❦ ⸿ ✕ ◀ ▶ ⚠ ?
3. **Type: "the screen is an instrument, the document is paper".**
   - Monospace for interface chrome and readouts: cotas, years, counts, labels.
   - Noto Serif for document text: the summaries in the result cards and the PDF.
   - Load the fonts through Streamlit's supported font configuration, at no runtime cost. If a web font cannot be configured cleanly, fall back to the system `monospace` / `serif` stacks rather than injecting fragile CSS.
   - Check that ⸿ (U+2E3F) renders with the chosen fonts on Windows and Android. If not, embed a font that has it.

**Rule:** minimal CSS, kept in one place (`ahu/estilo.py`), injected once. No styling scattered through the pages.

**Done when:** the author, looking at the app next to EINS Lite, recognises the family. Send three screenshots in the report: Busca light, Busca dark, and the selection panel with ⸿.

### R7 — Torch-free query path (M), before deploy
This is the item from the previous round, unchanged.

**Runtime vs build:**
- At runtime the app must not import `torch` or `sentence_transformers`.
- Encode queries with `onnxruntime` + `tokenizers` (or the lightest equivalent), using an ONNX export of `intfloat/multilingual-e5-small`.
- Keep the `"query: "` prefix, mean pooling and L2 normalisation identical to the build path.
- Split requirements:
  - `requirements.txt` — runtime, used by Streamlit Cloud;
  - `requirements-build.txt` — the author's machine, keeps sentence-transformers/torch for `build/construir_indice.py`.
- Stored document embeddings stay as they are. Do not re-index unless equivalence fails.

**Model files:**
- No model file in git may exceed 100 MB. Either download from the Hugging Face Hub at first use (cached), or commit a quantized model under the limit.
- Choose the option with the lower peak RSS that also passes equivalence, and report the trade-off.

**Equivalence test** (it must pass before anything else counts):
- Use about 30 realistic Portuguese queries: themes, names and places.
- Per query, cosine ≥ 0.99 against the sentence-transformers vector.
- Identical top-10 hybrid results for ≥ 90 % of the queries.
- Report the worst cases. If the quantized model fails, use fp32 ONNX.

**Memory:**
- Re-run `scripts/medir_memoria.py` with the cache warm and the app modules loaded.
- **Target: peak RSS < 600 MiB** for a semantic query.
- Also report the cold-start time of the first semantic query.

Update `docs/publicacao.md` and AGENTS.md to describe the two requirement files.

### R8 — Cleanup and handoff to deploy (S)
- Replace `use_container_width` with `width=` across the UI (Streamlit 1.55).
- Fast-forward `main` to `v3`.
- Re-check that `docs/publicacao.md` is exact for a brand-new repo `saulorrp/catalogo-ahu-sul-v3` and a brand-new Streamlit app on Python 3.11.
- **Final report:**
  - per R-package, what changed and which tests prove it;
  - peak RSS before and after;
  - the sample PDF;
  - the screenshots;
  - a short Portuguese **checklist for the author** (≤ 10 items) to verify on localhost before deploying.

---

## 3. Order and why

Run the packages in this order: **R1 → R2 → R3 → R4 → R5 → R6 → R7 → R8**.

- **R1 first:** removing code shrinks everything that follows.
- **R2 and R3 next:** they are the author's real workflow (select → PDF → photograph at the AHU), and the visit is in October.
- **R4 and R5:** text only, low risk.
- **R6:** touches every page, so it comes after the content has settled.
- **R7:** independent of the UI, but it blocks deploy.
- **After R8,** the author deploys. F4 (gold set and calibration on e5-small, CI, Zenodo) follows unchanged.

## 4. Constraints (unchanged)

- **R1:** zero running cost.
- **R2:** one maintainer.
- **R3:** builds on the author's Windows machine, CPU only.
- **R4:** nothing persisted server-side.
- **R5:** a single evaluator.
- **New dependencies:** must be CPU-only and fit the memory budget.
- **Language:** the UI is in Portuguese; code and docs are in English.
