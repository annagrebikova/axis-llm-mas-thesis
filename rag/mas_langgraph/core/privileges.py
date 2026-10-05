"""Privilege checks — loads matrix from input/privileges.py."""

from ..input.privileges import AGENT_PRIVILEGES, ENTITY_TYPES, Privilege

__all__ = ["Privilege", "ENTITY_TYPES", "AGENT_PRIVILEGES", "agent_can"]


def agent_can(agent_id: str, entity: str, privilege: str) -> bool:
    if agent_id not in AGENT_PRIVILEGES:
        return False
    return privilege in AGENT_PRIVILEGES[agent_id].get(entity, set())
