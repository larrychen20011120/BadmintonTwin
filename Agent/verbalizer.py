"""
BadmintonVerbalizer - Backward-compatible wrapper.

Delegates to method-specific verbalizers:
  - create_verbalizer() → NaiveRAGVerbalizer (for naive_rag / timerag compat)
  - verbalize_state() / verbalize_experiences() kept for legacy callers

New code should import directly from:
  - Agent.naive_rag_verbalizer
  - Agent.timerag_verbalizer
  - Agent.twin_verbalizer
"""

from Agent.naive_rag_verbalizer import NaiveRAGVerbalizer
from Agent.timerag_verbalizer import TimeRAGVerbalizer
from Agent.twin_verbalizer import TwinVerbalizer


class BadmintonVerbalizer(NaiveRAGVerbalizer):
    """Backward-compatible alias for NaiveRAGVerbalizer.
    Adds legacy method names for existing eval scripts.
    """
    def verbalize_state(self, state_dict):
        return self.narrative_state(state_dict)

    def verbalize_experience(self, exp, exp_idx=1):
        return self.narrative_case(exp, exp_idx, exp.get('similarity', 0))

    def verbalize_experiences(self, experiences):
        block, _ = self.build_cases_section(experiences)
        return block


def create_verbalizer():
    """Backward-compatible factory. Returns a NaiveRAGVerbalizer instance."""
    return NaiveRAGVerbalizer()


def verbalize_state(state_dict):
    verbalizer = NaiveRAGVerbalizer()
    return verbalizer.narrative_state(state_dict)


def verbalize_experiences(experiences):
    verbalizer = NaiveRAGVerbalizer()
    narrative, _, _ = verbalizer.build_cases_section(experiences)
    return narrative
