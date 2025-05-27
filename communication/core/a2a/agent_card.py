"""
Agent Card implementation for A2A protocol.

Agent Cards are metadata documents that describe agent capabilities,
connection information, and authentication requirements.
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from urllib.parse import urljoin

import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentCapability:
    """Represents an agent capability."""
    name: str
    description: str
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    parameters: Optional[Dict[str, Any]] = None


@dataclass
class AgentEndpoint:
    """Represents an agent communication endpoint."""
    url: str
    protocol: str = "https"
    authentication: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None


@dataclass
class AgentCard:
    """
    Agent Card following Google A2A specification.
    
    Contains metadata describing agent capabilities, connection info,
    and authentication requirements.
    """
    
    # Required fields
    agent_id: str
    name: str
    description: str
    version: str
    
    # Optional fields
    capabilities: List[AgentCapability] = None
    endpoints: List[AgentEndpoint] = None
    authentication: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.capabilities is None:
            self.capabilities = []
        if self.endpoints is None:
            self.endpoints = []
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent card to dictionary."""
        data = asdict(self)
        
        # Convert datetime objects to ISO format
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.updated_at:
            data['updated_at'] = self.updated_at.isoformat()
            
        return data
    
    def to_json(self) -> str:
        """Convert agent card to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentCard':
        """Create agent card from dictionary."""
        # Convert datetime strings back to datetime objects
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        # Convert capability dictionaries to AgentCapability objects
        if 'capabilities' in data and data['capabilities']:
            data['capabilities'] = [
                AgentCapability(**cap) if isinstance(cap, dict) else cap
                for cap in data['capabilities']
            ]
        
        # Convert endpoint dictionaries to AgentEndpoint objects
        if 'endpoints' in data and data['endpoints']:
            data['endpoints'] = [
                AgentEndpoint(**ep) if isinstance(ep, dict) else ep
                for ep in data['endpoints']
            ]
        
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AgentCard':
        """Create agent card from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def add_capability(self, capability: AgentCapability) -> None:
        """Add a capability to the agent card."""
        self.capabilities.append(capability)
        self.updated_at = datetime.utcnow()
        logger.info(f"Added capability '{capability.name}' to agent {self.agent_id}")
    
    def add_endpoint(self, endpoint: AgentEndpoint) -> None:
        """Add an endpoint to the agent card."""
        self.endpoints.append(endpoint)
        self.updated_at = datetime.utcnow()
        logger.info(f"Added endpoint '{endpoint.url}' to agent {self.agent_id}")
    
    def get_primary_endpoint(self) -> Optional[AgentEndpoint]:
        """Get the primary communication endpoint."""
        if not self.endpoints:
            return None
        return self.endpoints[0]  # First endpoint is considered primary
    
    def supports_capability(self, capability_name: str) -> bool:
        """Check if agent supports a specific capability."""
        return any(cap.name == capability_name for cap in self.capabilities)
    
    def validate(self) -> List[str]:
        """Validate agent card and return list of errors."""
        errors = []
        
        if not self.agent_id:
            errors.append("agent_id is required")
        if not self.name:
            errors.append("name is required")
        if not self.description:
            errors.append("description is required")
        if not self.version:
            errors.append("version is required")
        
        # Validate endpoints
        for i, endpoint in enumerate(self.endpoints):
            if not endpoint.url:
                errors.append(f"endpoint[{i}].url is required")
            if endpoint.protocol not in ['http', 'https']:
                errors.append(f"endpoint[{i}].protocol must be 'http' or 'https'")
        
        return errors


class AgentCardGenerator:
    """Generator for creating agent cards with common patterns."""
    
    @staticmethod
    def create_basic_card(
        name: str,
        description: str,
        base_url: str,
        capabilities: Optional[List[str]] = None
    ) -> AgentCard:
        """Create a basic agent card with common defaults."""
        
        agent_id = str(uuid.uuid4())
        
        # Create basic capabilities
        agent_capabilities = []
        if capabilities:
            for cap_name in capabilities:
                agent_capabilities.append(
                    AgentCapability(
                        name=cap_name,
                        description=f"Agent capability: {cap_name}"
                    )
                )
        
        # Create default endpoints
        endpoints = [
            AgentEndpoint(
                url=urljoin(base_url, "/a2a/"),
                protocol="https" if base_url.startswith("https") else "http"
            )
        ]
        
        card = AgentCard(
            agent_id=agent_id,
            name=name,
            description=description,
            version="1.0.0",
            capabilities=agent_capabilities,
            endpoints=endpoints,
            metadata={
                "framework": "mkt-a2a",
                "created_by": "AgentCardGenerator"
            }
        )
        
        logger.info(f"Generated basic agent card for '{name}' with ID {agent_id}")
        return card
    
    @staticmethod
    def create_ai_agent_card(
        name: str,
        description: str,
        base_url: str,
        model_info: Dict[str, Any],
        supported_tasks: List[str]
    ) -> AgentCard:
        """Create an agent card specifically for AI agents."""
        
        # Create capabilities based on supported tasks
        capabilities = []
        for task in supported_tasks:
            capabilities.append(
                AgentCapability(
                    name=task,
                    description=f"AI task: {task}",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string"},
                            "parameters": {"type": "object"}
                        },
                        "required": ["prompt"]
                    },
                    output_schema={
                        "type": "object", 
                        "properties": {
                            "result": {"type": "string"},
                            "confidence": {"type": "number"}
                        }
                    }
                )
            )
        
        card = AgentCardGenerator.create_basic_card(
            name=name,
            description=description,
            base_url=base_url
        )
        
        card.capabilities = capabilities
        card.metadata.update({
            "agent_type": "ai_agent",
            "model_info": model_info,
            "supported_tasks": supported_tasks
        })
        
        logger.info(f"Generated AI agent card for '{name}' with {len(supported_tasks)} tasks")
        return card
    
    @staticmethod
    def create_service_agent_card(
        name: str,
        description: str,
        base_url: str,
        service_endpoints: Dict[str, str]
    ) -> AgentCard:
        """Create an agent card for service-based agents."""
        
        # Create capabilities based on service endpoints
        capabilities = []
        for service_name, endpoint in service_endpoints.items():
            capabilities.append(
                AgentCapability(
                    name=service_name,
                    description=f"Service endpoint: {service_name}",
                    parameters={"endpoint": endpoint}
                )
            )
        
        card = AgentCardGenerator.create_basic_card(
            name=name,
            description=description,
            base_url=base_url
        )
        
        card.capabilities = capabilities
        card.metadata.update({
            "agent_type": "service_agent",
            "service_endpoints": service_endpoints
        })
        
        logger.info(f"Generated service agent card for '{name}' with {len(service_endpoints)} services")
        return card


# Example usage and testing
if __name__ == "__main__":
    # Create a sample agent card
    generator = AgentCardGenerator()
    
    card = generator.create_ai_agent_card(
        name="Text Analysis Agent",
        description="An AI agent specialized in text analysis and processing",
        base_url="https://api.example.com",
        model_info={
            "model_name": "gpt-4",
            "provider": "openai",
            "version": "2024-01"
        },
        supported_tasks=["text_summarization", "sentiment_analysis", "entity_extraction"]
    )
    
    print("Generated Agent Card:")
    print(card.to_json())
    
    # Validate the card
    errors = card.validate()
    if errors:
        print("Validation errors:", errors)
    else:
        print("Agent card is valid!")
