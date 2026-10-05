"""Researcher-facing Explanation Graph Explorer.

Researcher view — not for blinded participant evaluation.

From `EMaES/rag`:

    streamlit run streamlit_explanation_graphs.py

From the repository root:

    streamlit run rag/streamlit_explanation_graphs.py

Expected input: exported `quiz_explanation_experiment_*.json`.

This app is a pure viewer. Graph semantics, relevance, and AXIS ranking are
decided only by `mas_langgraph.visualization.build_explanation_graph`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

_RAG_DIR = Path(__file__).resolve().parent
if str(_RAG_DIR) not in sys.path:
    sys.path.insert(0, str(_RAG_DIR))

import streamlit as st

from mas_langgraph.visualization import (
    VisualizationDataError,
    build_explanation_graph,
    render_dot,
    render_mermaid,
    render_tikz,
)
from mas_langgraph.visualization.models import ExplanationGraph
from mas_langgraph.visualization.case_inspect import (
    ARTIFACT_CONTENT_UNAVAILABLE,
    CONCEPT_CHANGE_NOTE,
    CONCEPT_CHANGE_UNASSESSED,
    display_user_question,
    match_concept_change_for_unit,
    replay_artifact_views,
)
from mas_langgraph.visualization.ui_helpers import (
    ARTIFACT_BROWSER_SUBTYPES,
    CONDITION_LABELS,
    CONDITION_VALUES,
    DETAIL_HELP,
    DETAIL_LABELS,
    DETAIL_VALUES,
    LEGEND_EDGES,
    LEGEND_NODES,
    LEGEND_SYMBOLS,
    PROVENANCE_NOTE,
    RELATION_NOTE,
    ExperimentPayloadError,
    artifact_rows,
    case_visualization_issue,
    condition_text,
    download_filename,
    edge_rows,
    experiment_overview,
    find_local_experiment_files,
    inspector_sections,
    list_case_options,
    node_display_label,
    parse_experiment_json_bytes,
    parse_experiment_json_text,
    question_view,
    question_change_summary,
    extract_replay_question_views,
    mcq_views_equal,
    replay_meaning_sentences,
    replay_selection_caption,
    replay_selection_explanation,
    resolve_replay_question_comparison,
    resolve_selected_key,
    resolve_selected_node_id,
    role_glossary,
    summarize_graph,
    validate_experiment_payload,
    visible_replay_units,
    CHECKPOINT_DEPENDENCE_NOTE,
    CONTENT_VS_ACCEPTANCE_FOOTNOTE,
    CONTENT_VS_ACCEPTANCE_NOTES,
    MISSING_QUESTION_COMPARISON,
    evidence_status_explanation,
    control_stability_explanation,
)


st.set_page_config(
    page_title="Explanation Graph Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def _parse_uploaded_json(data: bytes) -> Mapping[str, Any]:
    return parse_experiment_json_bytes(data)


@st.cache_data(show_spinner=False)
def _load_local_json(path: str, mtime: float) -> Mapping[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    return parse_experiment_json_text(text)


def _render_sections(sections) -> None:
    for section in sections:
        title = section.get("title")
        if title:
            st.markdown(f"**{title}**")
        text = section.get("text")
        if text not in (None, ""):
            st.write(text)
        for line in section.get("lines") or []:
            st.write(line)
        pairs = section.get("pairs") or []
        if pairs:
            for key, value in pairs:
                st.markdown(f"- **{key}:** {value}")
        st.write("")


def _render_legend() -> None:
    st.markdown("**Node meaning**")
    for name, shape in LEGEND_NODES:
        st.markdown(f"- {name} — {shape}")
    st.markdown("**Edge meaning**")
    for style, meaning in LEGEND_EDGES:
        st.markdown(f"- {style}: {meaning}")
    st.markdown("**Symbols**")
    for symbol, meaning in LEGEND_SYMBOLS:
        st.markdown(f"- `{symbol}` {meaning}")
    st.caption(PROVENANCE_NOTE)


def _render_question_card(case: Mapping[str, Any]) -> None:
    view = question_view(case)
    st.markdown("### Question")
    st.write(view["stem"] or "—")
    for opt in view["options"]:
        if not isinstance(opt, Mapping):
            continue
        oid = str(opt.get("option_id") or "")
        text = str(opt.get("text") or "")
        st.write(f"{oid}. {text}")
    correct = view.get("correct_option")
    decision = str(view.get("factual_decision") or "—").upper()
    st.markdown(f"**Correct answer:** {correct or '—'}")
    st.markdown(f"**Factual decision:** {decision}")
    st.markdown(f"**Candidate:** `{view.get('candidate_key')}`")


def _render_overview(payload: Mapping[str, Any]) -> None:
    overview = experiment_overview(payload)
    cols = st.columns(4)
    cols[0].metric("Run", str(overview.get("run_id") or "—"))
    cols[1].metric("Branch", str(overview.get("branch_id") or "—"))
    cols[2].metric("Questions", str(overview.get("question_count") or "—"))
    cols[3].metric("Seed", str(overview.get("experiment_seed") or "—"))
    models = overview.get("models") or {}
    present = {k: v for k, v in models.items() if v}
    if present:
        with st.expander("Models", expanded=False):
            for name, model in present.items():
                st.markdown(f"- **{name}:** `{model}`")
    warnings = overview.get("warnings") or []
    if warnings:
        with st.expander("Experiment warnings", expanded=False):
            for warning in warnings:
                st.warning(str(warning))


def _render_user_question(
    case: Mapping[str, Any], payload: Mapping[str, Any]
) -> None:
    st.divider()
    st.markdown("### User question")
    st.caption("AXIS explanation request for the selected case. This is not the generated MCQ stem.")
    st.text(display_user_question(case, payload))
    st.divider()


def _plain(text: Any) -> None:
    st.text("" if text in (None, "") else str(text))


def _render_artifact_view(view: Mapping[str, Any], *, expander_key: str = "") -> None:
    role = str(view.get("role") or "").strip()
    ref = str(view.get("ref") or "—")
    section = str(view.get("section_label") or "").strip()
    if not section:
        section = f"{role} artifact" if role else "Artifact"
    st.markdown(f"**{section}:** `{ref}`")
    atype = view.get("artifact_type")
    if atype:
        st.caption(str(atype))
    expander_kwargs: dict = {"expanded": False}
    if expander_key:
        expander_kwargs["key"] = expander_key
    with st.expander(f"View content — {ref}", **expander_kwargs):
        if not view.get("available"):
            st.caption(ARTIFACT_CONTENT_UNAVAILABLE)
            return
        body = str(view.get("body") or "")
        if body:
            _plain(body)
        else:
            st.caption("No primary text is stored on this artifact.")
        for key, value in view.get("fields") or []:
            st.markdown(f"- **{key}:**")
            _plain(value)
        for line in view.get("lines") or []:
            _plain(line)
        raw = view.get("raw")
        if isinstance(raw, Mapping) and raw and str(view.get("artifact_type") or "") not in {
            "EvidenceChunk",
            "KnowledgeAtom",
            "MCQBlueprint",
            "CandidateMCQ",
        }:
            st.json(dict(raw))


def _render_concept_change(assessment: Optional[Mapping[str, Any]], *, attach: bool) -> None:
    if not attach:
        return
    if not assessment:
        st.markdown(f"- **Tested-concept change:** {CONCEPT_CHANGE_UNASSESSED}")
        return
    label = assessment.get("label_display") or CONCEPT_CHANGE_UNASSESSED
    st.markdown(f"- **Tested-concept change:** {label}")
    reason = assessment.get("reason_display") or ""
    if reason:
        st.markdown("- **Reason:**")
        _plain(reason)
    noop_concept = assessment.get("noop_tested_concept")
    if noop_concept not in (None, ""):
        st.markdown("- **NOOP tested concept:**")
        _plain(noop_concept)
    cf_concept = assessment.get("counterfactual_tested_concept")
    if cf_concept not in (None, ""):
        st.markdown("- **Counterfactual tested concept:**")
        _plain(cf_concept)


def _build_graph(case: Mapping[str, Any], condition: str, detail_level: str):
    issue = case_visualization_issue(case)
    if issue:
        return None, issue
    try:
        graph = build_explanation_graph(
            case, condition=condition, detail_level=detail_level
        )
        return graph, None
    except VisualizationDataError as exc:
        return None, str(exc)


def _show_graphviz(dot: str) -> None:
    try:
        st.graphviz_chart(dot, use_container_width=True)
    except Exception as exc:  # pragma: no cover - environment-dependent
        st.warning(f"Graphviz chart could not be rendered ({exc}). Showing DOT source.")
        st.code(dot, language="dot")


def main() -> None:
    st.title("Explanation Graph Explorer")
    st.caption("Researcher view · Not for blinded participant evaluation")

    with st.sidebar:
        st.header("Data source")
        source = st.radio(
            "Load experiment JSON",
            ("Existing experiment file", "Upload"),
            key="data_source",
        )
        payload: Optional[Mapping[str, Any]] = None
        source_label = ""
        load_error: Optional[str] = None

        if source == "Upload":
            uploaded = st.file_uploader(
                "Experiment JSON", type=["json"], key="upload_json"
            )
            if uploaded is not None:
                source_label = uploaded.name
                try:
                    payload = _parse_uploaded_json(uploaded.getvalue())
                except ExperimentPayloadError as exc:
                    load_error = str(exc)
        else:
            files = find_local_experiment_files()
            if not files:
                st.info(
                    "No `quiz_explanation_experiment_*.json` files found in the "
                    "working directory or `rag/`."
                )
            else:
                labels = [p.name for p in files]
                choice = st.selectbox(
                    "Existing experiment file",
                    options=list(range(len(files))),
                    format_func=lambda i: labels[i],
                    key="local_file_index",
                )
                path = files[int(choice)]
                source_label = str(path)
                try:
                    payload = _load_local_json(str(path), path.stat().st_mtime)
                except ExperimentPayloadError as exc:
                    load_error = str(exc)
                except OSError as exc:
                    load_error = f"Could not read {path.name}: {exc}"

        condition_label = st.radio(
            "Condition",
            [CONDITION_LABELS[v] for v in CONDITION_VALUES],
            horizontal=True,
            key="condition_label",
        )
        detail_label = st.radio(
            "Detail",
            [DETAIL_LABELS[v] for v in DETAIL_VALUES],
            horizontal=True,
            key="detail_label",
        )
        condition = {v: k for k, v in CONDITION_LABELS.items()}[condition_label]
        detail_level = {v: k for k, v in DETAIL_LABELS.items()}[detail_label]
        st.caption(DETAIL_HELP[detail_level])

    if load_error:
        st.error(load_error)
        return
    if payload is None:
        st.info("Load an exported explanation-experiment JSON to begin.")
        return

    try:
        payload = validate_experiment_payload(payload)
    except ExperimentPayloadError as exc:
        st.error(str(exc))
        return

    fingerprint = f"{source}|{source_label}"
    if st.session_state.get("_payload_fp") != fingerprint:
        st.session_state["_payload_fp"] = fingerprint
        st.session_state.pop("case_key", None)
        st.session_state.pop("inspect_node_id", None)
        st.session_state.pop("inspect_edge_id", None)

    _render_overview(payload)
    if source_label:
        st.caption(f"Source: {source_label}")

    options = list_case_options(payload)
    if not options:
        st.info("This experiment has no cases.")
        return

    with st.sidebar:
        st.header("Question")
        keys = [row["key"] for row in options]
        current_key = resolve_selected_key(st.session_state.get("case_key"), keys)
        selected_key = st.selectbox(
            "MCQ",
            keys,
            index=keys.index(current_key) if current_key in keys else 0,
            format_func=lambda k: next(
                (row["label"] for row in options if row["key"] == k), k
            ),
            key="case_select",
        )
        st.session_state["case_key"] = selected_key

    case = next(
        (
            payload["cases"][row["index"]]
            for row in options
            if row["key"] == selected_key
        ),
        None,
    )
    if not isinstance(case, Mapping):
        st.error("The selected case could not be loaded.")
        return

    _render_user_question(case, payload)
    _render_question_card(case)
    issue = case_visualization_issue(case)
    if issue:
        st.warning(issue)

    graph, graph_error = _build_graph(case, condition, detail_level)
    if graph_error and graph is None:
        st.error(graph_error)

    tab_graph, tab_nodes, tab_rel, tab_axis, tab_cmp, tab_text, tab_src = st.tabs(
        [
            "Graph",
            "Node details / Artifacts",
            "Relations",
            "AXIS replays",
            "Condition comparison",
            "Text explanations",
            "Graph source",
        ]
    )

    with tab_graph:
        _render_graph_tab(graph, graph_error, selected_key, condition, detail_level)

    with tab_nodes:
        _render_nodes_tab(graph)

    with tab_rel:
        _render_relations_tab(graph)

    with tab_axis:
        _render_axis_tab(graph, condition, case)

    with tab_cmp:
        _render_comparison_tab(case, detail_level, issue)

    with tab_text:
        _render_text_tab(case)

    with tab_src:
        _render_source_tab(graph, selected_key, condition, detail_level)


def _render_graph_tab(
    graph: Optional[ExplanationGraph],
    graph_error: Optional[str],
    candidate_key: str,
    condition: str,
    detail_level: str,
) -> None:
    left, right = st.columns((3, 1))
    with right:
        st.markdown("#### Legend")
        _render_legend()
    with left:
        if graph is None:
            st.info("No graph is available for this case and condition.")
            if graph_error:
                st.error(graph_error)
            return
        stats = summarize_graph(graph)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Visible nodes", stats["visible_nodes"])
        m2.metric("Visible edges", stats["visible_edges"])
        m3.metric("Hidden nodes", stats["hidden_nodes"])
        m4.metric("Hidden edges", stats["hidden_edges"])
        m5.metric("Hidden replays", stats["hidden_replays"])
        if stats["selection_scope"]:
            st.caption(
                f"Selection scope: `{stats['selection_scope']}` · "
                f"target `{stats['target_ref']}`"
            )
        for warning in stats["warnings"]:
            st.warning(str(warning))
        if stats["visible_nodes"] == 0:
            st.info("The graph contains no visible nodes at this detail level.")
            return
        _show_graphviz(render_dot(graph))
        node_map = graph.node_map()
        visible_ids = list(graph.node_ids())
        fallback = f"artifact::{graph.target_ref}" if graph.target_ref else None
        chosen = resolve_selected_node_id(
            st.session_state.get("inspect_node_id"),
            visible_ids,
            fallback_id=fallback,
        )
        selected = st.selectbox(
            "Inspect graph node",
            visible_ids,
            index=visible_ids.index(chosen) if chosen in visible_ids else 0,
            format_func=lambda nid: node_display_label(node_map[nid]),
            key="inspect_node_select",
        )
        st.session_state["inspect_node_id"] = selected
        node = node_map.get(selected)
        if node is not None:
            st.markdown("#### Selected node")
            _render_sections(inspector_sections(node))
            with st.expander("Raw node details"):
                st.json(dict(node.details or {}))


def _render_nodes_tab(graph: Optional[ExplanationGraph]) -> None:
    if graph is None:
        st.info("No graph is available.")
        return
    node_map = graph.node_map()
    visible_ids = list(graph.node_ids())
    fallback = f"artifact::{graph.target_ref}" if graph.target_ref else None
    chosen = resolve_selected_node_id(
        st.session_state.get("inspect_node_id"),
        visible_ids,
        fallback_id=fallback,
    )
    selected = st.selectbox(
        "Inspect graph node",
        visible_ids,
        index=visible_ids.index(chosen) if chosen in visible_ids else 0,
        format_func=lambda nid: node_display_label(node_map[nid]),
        key="inspect_node_select_tab",
    )
    st.session_state["inspect_node_id"] = selected
    node = node_map.get(selected)
    if node is not None:
        _render_sections(inspector_sections(node))
        with st.expander("Raw node details"):
            st.json(dict(node.details or {}))

    st.markdown("### Artifacts")
    st.caption(
        "Browser over nodes already visible in the current graph. "
        "Filtering does not change the graph."
    )
    subtype = st.selectbox(
        "Filter by subtype",
        ["(all)"] + list(ARTIFACT_BROWSER_SUBTYPES),
        key="artifact_subtype_filter",
    )
    query = st.text_input("Search artifacts", key="artifact_search")
    rows = artifact_rows(graph)
    if subtype != "(all)":
        rows = [r for r in rows if r["type"] == subtype]
    if query.strip():
        q = query.strip().lower()
        rows = [
            r
            for r in rows
            if q in str(r["type"]).lower()
            or q in str(r["ref"]).lower()
            or q in str(r["label"]).lower()
            or q in str(r["preview"]).lower()
        ]
    st.dataframe(
        [
            {
                "type": r["type"],
                "exact ref": r["ref"],
                "label": r["label"],
                "preview": r["preview"],
            }
            for r in rows
        ],
        use_container_width=True,
        hide_index=True,
    )


def _render_relations_tab(graph: Optional[ExplanationGraph]) -> None:
    if graph is None:
        st.info("No graph is available.")
        return
    st.caption(RELATION_NOTE)
    st.caption(PROVENANCE_NOTE)
    rows = edge_rows(graph)
    st.dataframe(
        [
            {
                "source": r["source"],
                "target": r["target"],
                "kind": r["kind"],
                "relation": r["relation"],
                "label": r["label"],
                "source_id": r["source_id"],
                "target_id": r["target_id"],
            }
            for r in rows
        ],
        use_container_width=True,
        hide_index=True,
    )
    if not rows:
        st.info("No relations are visible in the current graph.")
        return
    ids = [r["edge_id"] for r in rows]
    current = resolve_selected_key(st.session_state.get("inspect_edge_id"), ids)
    chosen = st.selectbox(
        "Inspect relation",
        ids,
        index=ids.index(current) if current in ids else 0,
        format_func=lambda eid: next(
            (
                f"{r['kind']} · {r['source']} → {r['target']}"
                for r in rows
                if r["edge_id"] == eid
            ),
            eid,
        ),
        key="inspect_edge_select",
    )
    st.session_state["inspect_edge_id"] = chosen
    row = next((r for r in rows if r["edge_id"] == chosen), None)
    if row and row["details"]:
        st.markdown("**Edge details**")
        st.json(row["details"])
    elif row:
        st.caption("This relation has no extra details.")


def _render_mcq_question_block(title: str, view) -> None:
    st.markdown(f"**{title}**")
    if view is None:
        st.caption("Not available.")
        return
    st.write(view.get("stem") or "—")
    st.markdown(f"**Correct answer:** {view.get('correct_answer') or '—'}")
    st.markdown("**Distractors:**")
    distractors = list(view.get("distractors") or [])
    if not distractors:
        st.write("—")
    else:
        for item in distractors:
            st.write(f"- {item}")
    explanation = view.get("answer_explanation")
    if explanation:
        st.caption(f"Answer explanation: {explanation}")


def _render_axis_tab(
    graph: Optional[ExplanationGraph], condition: str, case: Optional[Mapping[str, Any]] = None
) -> None:
    if condition != "axis":
        st.info(
            "Switch the condition to AXIS to inspect replay units from the current graph."
        )
        return
    if graph is None:
        st.info("No graph is available.")
        return
    units = visible_replay_units(graph)
    caption = replay_selection_caption(graph)
    if caption:
        st.info(caption)
    st.caption(CHECKPOINT_DEPENDENCE_NOTE)
    st.caption(CONCEPT_CHANGE_NOTE)
    with st.expander("Question-content effect vs acceptance effect", expanded=False):
        for title, text in CONTENT_VS_ACCEPTANCE_NOTES:
            st.markdown(f"**{title}:** {text}")
        st.info(CONTENT_VS_ACCEPTANCE_FOOTNOTE)
    with st.expander("What do these effects mean?", expanded=False):
        for label, text in role_glossary():
            st.markdown(f"**{label}**")
            st.write(text)
    if not units:
        st.warning(
            "No AXIS replay units are visible in this graph. "
            "Raw replay comparisons may be absent from the exported case, "
            "or none were selected at this detail level."
        )
    for unit in units:
        title = (
            f"{unit.get('operation') or 'Replay'} · "
            f"{unit.get('target_ref') or unit.get('replay_id')}"
        )
        with st.expander(title, expanded=len(units) <= 3):
            st.markdown("##### Replay summary")
            st.markdown(f"- **evidence_status:** {unit.get('evidence_status') or '—'}")
            st.markdown(f"- **NOOP result:** {unit.get('noop_decision') or '—'}")
            st.markdown(
                f"- **Counterfactual result:** {unit.get('counterfactual_decision') or '—'}"
            )
            st.markdown(f"- **{unit.get('effect_line')}**")
            st.markdown(
                "- **causal/information role:** "
                f"{unit.get('display_role_label') or unit.get('information_role') or '—'}"
            )
            st.markdown(
                f"- **control stability:** {unit.get('decision_control_stable')}"
            )
            if unit.get("checkpoint_id") or unit.get("checkpoint_alias"):
                st.markdown(
                    f"- **checkpoint:** {unit.get('checkpoint_alias') or unit.get('checkpoint_id')}"
                )
            st.caption(f"Graph replay id: `{unit.get('replay_id')}`")
            assessment = match_concept_change_for_unit(case, unit)
            _render_concept_change(assessment, attach=True)

            st.markdown("##### Artifacts")
            artifact_views = replay_artifact_views(case, unit, graph=graph)
            if not artifact_views:
                st.caption("No artifact references are attached to this replay.")
            replay_id = str(unit.get("replay_id") or unit.get("comparison_index") or "replay")
            for view in artifact_views:
                role = str(view.get("role") or "artifact")
                ref = str(view.get("ref") or "")
                _render_artifact_view(
                    view,
                    expander_key=f"replay-artifact:{replay_id}:{role}:{ref}",
                )

            st.markdown("##### What this means")
            for sentence in replay_meaning_sentences(unit):
                st.write(sentence)
            status_help = evidence_status_explanation(unit.get("evidence_status"))
            if status_help:
                st.caption(f"evidence_status: {status_help}")
            st.caption(
                f"control_stability: {control_stability_explanation(unit.get('decision_control_stable'))}"
            )

            qcc = resolve_replay_question_comparison(unit, case)
            views = extract_replay_question_views(qcc)
            changes = question_change_summary(
                qcc,
                decision_changed=unit.get("decision_changed"),
                noop_decision=unit.get("noop_decision"),
                counterfactual_decision=unit.get("counterfactual_decision"),
            )
            st.markdown("##### Exact changes")
            for name, word in changes:
                st.markdown(f"- **{name}:** {word}")

            st.markdown("##### Question comparison")
            if views is None:
                st.caption(MISSING_QUESTION_COMPARISON)
            else:
                factual = views.get("factual")
                noop = views.get("noop")
                counterfactual = views.get("counterfactual")
                if factual is not None and noop is not None and not mcq_views_equal(factual, noop):
                    cols = st.columns(3)
                    with cols[0]:
                        _render_mcq_question_block("Factual question", factual)
                    with cols[1]:
                        _render_mcq_question_block("NOOP / control question", noop)
                    with cols[2]:
                        _render_mcq_question_block("Question after intervention", counterfactual)
                else:
                    left, right = st.columns(2)
                    original = noop if noop is not None else factual
                    original_title = "Original / control question"
                    if factual is not None and noop is not None:
                        original_title = "Original / NOOP question"
                    with left:
                        _render_mcq_question_block(original_title, original)
                    with right:
                        _render_mcq_question_block(
                            "Question after intervention", counterfactual
                        )
                    if factual is not None and noop is not None and mcq_views_equal(factual, noop):
                        st.caption("NOOP matched the factual question, so it is not duplicated.")

            with st.expander("Technical details"):
                st.caption(f"Graph replay id: `{unit.get('replay_id')}`")
                st.markdown(f"- **target_ref:** `{unit.get('target_ref') or '—'}`")
                st.markdown(
                    f"- **replacement_ref:** `{unit.get('replacement_ref') or '—'}`"
                )
                st.markdown(f"- **checkpoint_id:** `{unit.get('checkpoint_id') or '—'}`")
                st.markdown(
                    f"- **checkpoint_alias:** `{unit.get('checkpoint_alias') or '—'}`"
                )
                st.markdown(
                    f"- **comparison_index:** {unit.get('comparison_index')}"
                )
                st.markdown(
                    f"- **information_role:** `{unit.get('information_role') or '—'}`"
                )
                st.markdown(
                    f"- **causal_classification:** `{unit.get('causal_classification') or '—'}`"
                )
                st.markdown(
                    f"- **subject_content_control_stable:** "
                    f"{unit.get('subject_content_control_stable')}"
                )
                if qcc:
                    st.markdown("**question_content_comparison**")
                    st.json(qcc)
    ranking = list((graph.metadata or {}).get("replay_ranking") or [])
    with st.expander("Why were these AXIS replays selected?"):
        st.write(replay_selection_explanation(graph))
        st.caption(
            "This is the deterministic ranking already computed by the graph builder. "
            "It is not a causal claim."
        )
        if ranking:
            st.json(ranking)
        else:
            st.write("No ranking metadata is present on this graph.")
        st.json(
            {
                "selected_replay_indices": (graph.metadata or {}).get(
                    "selected_replay_indices"
                ),
                "hidden_replay_count": (graph.metadata or {}).get("hidden_replay_count"),
                "replay_count": (graph.metadata or {}).get("replay_count"),
            }
        )


def _render_comparison_tab(
    case: Mapping[str, Any], detail_level: str, issue: Optional[str]
) -> None:
    st.caption(
        "The same case and detail level, built three times by the canonical builder. "
        "This tab does not invent a fourth condition."
    )
    if issue:
        st.warning(issue)
        return
    graphs = {}
    errors = {}
    for cond in CONDITION_VALUES:
        g, err = _build_graph(case, cond, detail_level)
        graphs[cond] = g
        if err:
            errors[cond] = err
    cols = st.columns(3)
    for i, cond in enumerate(CONDITION_VALUES):
        g = graphs[cond]
        stats = summarize_graph(g) if g is not None else None
        with cols[i]:
            st.markdown(f"**{CONDITION_LABELS[cond]}**")
            if stats is None:
                st.write(errors.get(cond) or "unavailable")
            else:
                st.write(f"nodes: {stats['visible_nodes']}")
                st.write(f"edges: {stats['visible_edges']}")
                st.write(f"replay units: {stats['replay_units']}")
    sub = st.tabs([CONDITION_LABELS[c] for c in CONDITION_VALUES])
    for tab, cond in zip(sub, CONDITION_VALUES):
        with tab:
            g = graphs[cond]
            if g is None:
                st.error(errors.get(cond) or "Graph unavailable.")
                continue
            if len(g.nodes) == 0:
                st.info("Empty graph.")
                continue
            _show_graphviz(render_dot(g))


def _render_text_tab(case: Mapping[str, Any]) -> None:
    st.caption(
        "Participant-facing prose from the exported experiment. "
        "It is not an input to graph construction."
    )
    for cond in CONDITION_VALUES:
        block = condition_text(case, cond)
        st.markdown(f"### {block['label']}")
        st.markdown(f"Status: `{block.get('status') or '—'}`")
        for warning in block.get("warnings") or []:
            st.warning(str(warning))
        explanation = block.get("explanation")
        if explanation:
            st.write(explanation)
        else:
            st.info("No explanation text is present for this condition.")


def _render_source_tab(
    graph: Optional[ExplanationGraph],
    candidate_key: str,
    condition: str,
    detail_level: str,
) -> None:
    if graph is None:
        st.info("No graph is available.")
        return
    dot = render_dot(graph)
    mermaid = render_mermaid(graph)
    tikz = render_tikz(graph)
    payload = graph.to_dict()
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button(
        "Download DOT",
        data=dot.encode("utf-8"),
        file_name=download_filename(candidate_key, condition, detail_level, "dot"),
        mime="text/plain",
        key="dl_dot",
    )
    d2.download_button(
        "Download Mermaid",
        data=mermaid.encode("utf-8"),
        file_name=download_filename(candidate_key, condition, detail_level, "mmd"),
        mime="text/plain",
        key="dl_mmd",
    )
    d3.download_button(
        "Download TikZ",
        data=tikz.encode("utf-8"),
        file_name=download_filename(candidate_key, condition, detail_level, "tex"),
        mime="text/plain",
        key="dl_tex",
    )
    d4.download_button(
        "Download graph JSON",
        data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=download_filename(candidate_key, condition, detail_level, "json"),
        mime="application/json",
        key="dl_json",
    )
    src_dot, src_mmd, src_tex, src_json = st.tabs(
        ["DOT", "Mermaid", "TikZ", "Graph JSON"]
    )
    with src_dot:
        st.code(dot, language="dot")
    with src_mmd:
        st.code(mermaid, language="text")
    with src_tex:
        st.code(tikz, language="latex")
    with src_json:
        st.json(payload)


main()
