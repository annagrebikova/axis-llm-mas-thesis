"""Skeleton for a new MAS domain — copy to domains/my_domain/ and wire in input/profile.py.

Steps:
  1. Copy this folder to domains/my_domain/
  2. Implement MyState(BaseCooperativeState), MyNeedsEvaluator(BaseNeedsEvaluator)
  3. Optional: my_coverage.py, format_my_workflow_result()
  4. Point MASProfile in input/profile.py to your classes
  5. Replace input/agents.py, orchestration_rules.py, tools/, etc.
"""

# Example stubs (not runnable until completed):

# from mas_langgraph.core.state_base import BaseCooperativeState
# from mas_langgraph.orchestrator.needs_base import BaseNeedsEvaluator
#
# class MyState(BaseCooperativeState):
#     def __init__(self):
#         self.items = {}
#
#     def snapshot_counts(self):
#         return {"items_len": len(self.items)}
#
# class MyNeedsEvaluator(BaseNeedsEvaluator):
#     def as_dict(self):
#         return self.register({
#             "needs_items": lambda **_: len(self.state.items) < 1,
#             "ready_to_finish": lambda **_: len(self.state.items) >= 1,
#         })
#
# def format_my_workflow_result(*, final_state, **kwargs):
#     return {"items": final_state.get("items_len", 0)}

DOMAIN_NAME = "template"
