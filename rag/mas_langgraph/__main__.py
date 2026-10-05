"""CLI: python -m mas_langgraph --orchestrate --pdf-id 141 """

import argparse
import json
import sys


def _build_rag_config(args) -> "RAGConfig":
    from .input.rag_config import RAGConfig

    cfg = RAGConfig(
        pdf_id=args.pdf_id,
        vector_store_path=args.vector_store,
        llm_model=args.model,
        model_type=args.model_type,
        device=args.device,
    )
    if args.model_type == "huggingface":
        cfg.huggingface_api_key = args.api_key
    else:
        cfg.openai_api_key = args.api_key
    return cfg


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Cooperative multi-agent quiz pipeline. "
            "Quiz scope is derived from the indexed PDF material (outline sections, LOs, coverage) — no --topic."
        ),
    )
    parser.add_argument(
        "--orchestrator",
        choices=["classic", "langgraph"],
        default="classic",
        help="Orchestration engine: classic rule loop (default) or LangGraph skeleton.",
    )
    parser.add_argument(
        "--orchestrate",
        action="store_true",
        help="Run full rule-based orchestration loop.",
    )
    parser.add_argument(
        "--pdf-id",
        type=str,
        default=None,
        help="PDF collection id (vector store). Required for real RAG (unless --stub).",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Force stub tools (no PDF/LLM) even if --pdf-id is set.",
    )
    parser.add_argument("--vector-store", default="vector_store", help="Vector store directory.")
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini-2025-04-14",
        help="Workflow LLM model snapshot (default: gpt-4.1-mini-2025-04-14).",
    )
    parser.add_argument(
        "--model-type",
        choices=["openai", "huggingface"],
        default="openai",
        help="LLM provider (RAG mode).",
    )
    parser.add_argument("--api-key", default=None, help="OpenAI or HuggingFace API key.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Unused (API only).")
    parser.add_argument(
        "--agent",
        choices=[
            "structure_outline", "curriculum_lo", "retriever",
            "quality_duplicate_detector", "evidence_curator", "atomizer",
            "quiz_generator", "verifier", "coverage_controller", "packager_export",
        ],
        help="Run a single agent instead of full workflow.",
    )
    parser.add_argument(
        "--input",
        default="",
        help="Optional override input for single-agent run (default: derived from material).",
    )
    parser.add_argument("--json", action="store_true", help="Print result as JSON.")
    parser.add_argument(
        "--dump-pack",
        metavar="PATH",
        default=None,
        help="Write final QuizPack(s) to JSON file after run.",
    )
    args = parser.parse_args(argv)

    from .backbone import CooperativeBackbone
    from .export_utils import dump_quiz_packs
    from .orchestrator.material_context import MaterialContext

    rag_config = None
    if args.pdf_id and not args.stub:
        rag_config = _build_rag_config(args)
        print(f"RAG mode: pdf_id={args.pdf_id}, model={args.model_type}:{args.model}")
    elif not args.stub and (args.orchestrate or args.agent):
        print("Stub mode (no --pdf-id). Use --pdf-id 141 for real RAG input.")
        print("  Index first: python rag_pipeline.py --pdf-id 141")

    backbone = CooperativeBackbone(
        rag_config=rag_config,
        orchestrator_mode=args.orchestrator,
    )

    if args.orchestrate:
        if args.orchestrator == "langgraph":
            print("Orchestrator: LangGraph with MCQBlueprint planning")
        result = backbone.run_full_workflow()
    elif args.agent:
        inp = args.input
        if not inp:
            mat = MaterialContext(
                backbone.state,
                backbone.profile.coverage_plan,
                pdf_id=backbone.pdf_id,
            )
            if args.agent == "retriever":
                inp = mat.retrieval_query(0)
            elif args.agent == "quiz_generator":
                inp = mat.quiz_target()
            elif args.agent == "packager_export":
                inp = mat.pack_title()
            else:
                inp = mat.document_theme()
        result = backbone.run_agent(args.agent, inp)
    else:
        parser.print_help()
        return 1

    if args.dump_pack:
        packs = backbone.quiz_packs()
        path = dump_quiz_packs(packs, args.dump_pack)
        print(f"Wrote {len(packs)} quiz pack(s) to {path}")

    if args.json:
        def _serialize(obj):
            if hasattr(obj, "__dict__"):
                return obj.__dict__
            return str(obj)

        payload = dict(result) if isinstance(result, dict) else {"result": result}
        if args.orchestrate or args.agent:
            from .export_utils import packs_to_dicts
            payload["quiz_packs"] = packs_to_dicts(backbone.quiz_packs())
        print(json.dumps(payload, default=_serialize, indent=2))
    else:
        if args.orchestrate:
            theme = result.get("material_theme") or result.get("topic")
            print(f"Material theme: {theme}")
            if backbone.pdf_id:
                print(f"PDF id: {backbone.pdf_id}")
            print(f"Steps: {result.get('steps')}")
            print(f"Validated: {result.get('validated_questions')}")
            print(f"Packs exported: {result.get('packages_exported')}")
            print("Final state:", result.get("final_state"))
            packs = backbone.quiz_packs()
            if packs:
                print(f"\nQuiz pack: {packs[-1].pack_id} — {packs[-1].title} ({len(packs[-1].questions)} questions)")
        else:
            print(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
