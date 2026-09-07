"""
Specialized Domain Sub-Agents for Visit Ethiopia RAG System.
Each agent provides domain expertise (Itinerary Planning, Cultural Heritage, Logistics & Visa)
by reusing core LLM and prompt generation infrastructure.
"""

from typing import Dict

class BaseSpecializedAgent:
    """Base class Serves as the parent class for all domain experts """

    def __init__(self, name: str, domain_focus: str):
        self.name = name
        self.domain_focus = domain_focus

    def format_domain_context(self, context_text: str) -> str:
        """Expert System Header to the retrieved document snippets."""
        return f"=== SPECIALIZED EXPERT: {self.name.upper()} ===\n[Domain Focus: {self.domain_focus}]\n\n{context_text}"


class ItineraryPlannerAgent(BaseSpecializedAgent):
    """Specialized agent for trip itineraries, routes, transport, and travel scheduling."""

    def __init__(self):
        super().__init__(
            name="Itinerary & Route Planning Expert",
            domain_focus="Multi-day trip itineraries, regional routes , transport options, and optimal travel days."
        )


class CultureHeritageAgent(BaseSpecializedAgent):
    """Specialized agent for UNESCO sites, history, festivals, cuisine, and local customs."""

    def __init__(self):
        super().__init__(
            name="Culture & Heritage Expert",
            domain_focus="UNESCO World Heritage sites, Ethiopian history, traditional festivals (Timkat, Meskel, Enkutatash), food (Injera, Coffee Ceremony), and local cultural etiquette."
        )


class LogisticsVisaAgent(BaseSpecializedAgent):
    """Specialized agent for e-Visa rules, currency conversion, flights, and safety."""

    def __init__(self):
        super().__init__(
            name="Logistics, Visa & Travel Safety Expert",
            domain_focus="e-Visa regulations, passport requirements, currency exchange (ETB), Ethiopian Airlines flights, health guidelines, and emergency contacts."
        )


class GeneralTourismAgent(BaseSpecializedAgent):
    """Default specialized agent for general tourism inquiries."""

    def __init__(self):
        super().__init__(
            name="General Tourism & Information Expert",
            domain_focus="General Ethiopian travel guidance, official government infrastructure, and regional tourism bureau information."
        )

# Agent Registry Dictionary for fast routing
AGENT_REGISTRY: Dict[str, BaseSpecializedAgent] = {
    "itinerary": ItineraryPlannerAgent(),
    "culture": CultureHeritageAgent(),
    "logistics": LogisticsVisaAgent(),
    "general": GeneralTourismAgent(),
}


def get_specialized_agent(domain_key: str) -> BaseSpecializedAgent:
    """Returns the matching specialized agent instance or falls back to GeneralTourismAgent."""
    return AGENT_REGISTRY.get(domain_key.lower().strip(), AGENT_REGISTRY["general"])
