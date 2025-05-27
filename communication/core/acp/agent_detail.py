# communication/core/acp/agent_detail.py
"""
Agent Detail implementation for ACP protocol.

Agent Details are model descriptions that describe agent capabilities 
without exposing implementation details, following the ACP specification
from the Linux Foundation AI & Data program / BeeAI project.
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from urllib.parse import urljoin
from enum import Enum

import logging

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Agent type enumeration."""
    AI_ASSISTANT = "ai_assistant"
    TASK_EXECUTOR = "task_executor"
    DATA_PROCESSOR = "data_processor"
    SERVICE_WRAPPER = "service_wrapper"
    WORKFLOW_ORCHESTRATOR = "workflow_orchestrator"
    CUSTOM = "custom"


class InputType(Enum):
    """Input type enumeration."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"
    JSON = "json"
    STRUCTURED_DATA = "structured_data"


class OutputType(Enum):
    """Output type enumeration."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"
    JSON = "json"
    STRUCTURED_DATA = "structured_data"


@dataclass
class AgentCapability:
    """Represents a specific agent capability."""
    
    name: str
    description: str
    input_types: List[InputType]
    output_types: List[OutputType]
    parameters: Optional[Dict[str, Any]] = None
    examples: Optional[List[Dict[str, Any]]] = None
    constraints: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.parameters is None:
            self.parameters = {}
        if self.examples is None:
            self.examples = []
        if self.constraints is None:
            self.constraints = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert capability to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "input_types": [t.value for t in self.input_types],
            "output_types": [t.value for t in self.output_types],
            "parameters": self.parameters,
            "examples": self.examples,
            "constraints": self.constraints
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentCapability':
        """Create capability from dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            input_types=[InputType(t) for t in data.get("input_types", [])],
            output_types=[OutputType(t) for t in data.get("output_types", [])],
            parameters=data.get("parameters", {}),
            examples=data.get("examples", []),
            constraints=data.get("constraints", {})
        )


@dataclass
class AgentEndpoint:
    """Represents an agent communication endpoint."""
    
    url: str
    protocol: str = "https"
    methods: List[str] = None
    authentication: Optional[Dict[str, Any]] = None
    rate_limits: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.methods is None:
            self.methods = ["POST"]
        if self.rate_limits is None:
            self.rate_limits = {"requests_per_minute": 60}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert endpoint to dictionary."""
        return {
            "url": self.url,
            "protocol": self.protocol,
            "methods": self.methods,
            "authentication": self.authentication,
            "rate_limits": self.rate_limits
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentEndpoint':
        """Create endpoint from dictionary."""
        return cls(
            url=data["url"],
            protocol=data.get("protocol", "https"),
            methods=data.get("methods", ["POST"]),
            authentication=data.get("authentication"),
            rate_limits=data.get("rate_limits", {"requests_per_minute": 60})
        )


@dataclass
class AgentMetrics:
    """Agent performance and usage metrics."""
    
    average_response_time: Optional[float] = None
    success_rate: Optional[float] = None
    total_runs: Optional[int] = None
    uptime_percentage: Optional[float] = None
    last_updated: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.last_updated is None:
            self.last_updated = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "average_response_time": self.average_response_time,
            "success_rate": self.success_rate,
            "total_runs": self.total_runs,
            "uptime_percentage": self.uptime_percentage,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMetrics':
        """Create metrics from dictionary."""
        last_updated = None
        if data.get("last_updated"):
            last_updated = datetime.fromisoformat(data["last_updated"])
        
        return cls(
            average_response_time=data.get("average_response_time"),
            success_rate=data.get("success_rate"),
            total_runs=data.get("total_runs"),
            uptime_percentage=data.get("uptime_percentage"),
            last_updated=last_updated
        )


@dataclass
class AgentDetail:
    """
    Agent Detail following ACP specification.
    
    Describes agent capabilities without exposing implementation details,
    enabling framework-agnostic agent communication.
    """
    
    # Required fields
    agent_id: str
    name: str
    description: str
    version: str
    agent_type: AgentType
    
    # Optional fields
    capabilities: List[AgentCapability] = None
    endpoints: List[AgentEndpoint] = None
    supported_languages: List[str] = None
    tags: List[str] = None
    metadata: Optional[Dict[str, Any]] = None
    metrics: Optional[AgentMetrics] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    # Deployment information
    deployment_info: Optional[Dict[str, Any]] = None
    scaling_info: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.capabilities is None:
            self.capabilities = []
        if self.endpoints is None:
            self.endpoints = []
        if self.supported_languages is None:
            self.supported_languages = ["en"]
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent detail to dictionary."""
        data = {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "agent_type": self.agent_type.value,
            "capabilities": [cap.to_dict() for cap in self.capabilities],
            "endpoints": [ep.to_dict() for ep in self.endpoints],
            "supported_languages": self.supported_languages,
            "tags": self.tags,
            "metadata": self.metadata,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "deployment_info": self.deployment_info,
            "scaling_info": self.scaling_info
        }
        
        # Remove None values
        return {k: v for k, v in data.items() if v is not None}
    
    def to_json(self) -> str:
        """Convert agent detail to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentDetail':
        """Create agent detail from dictionary."""
        # Convert datetime strings
        created_at = None
        if data.get('created_at'):
            created_at = datetime.fromisoformat(data['created_at'])
        
        updated_at = None
        if data.get('updated_at'):
            updated_at = datetime.fromisoformat(data['updated_at'])
        
        # Convert capabilities
        capabilities = []
        if data.get('capabilities'):
            capabilities = [
                AgentCapability.from_dict(cap) if isinstance(cap, dict) else cap
                for cap in data['capabilities']
            ]
        
        # Convert endpoints
        endpoints = []
        if data.get('endpoints'):
            endpoints = [
                AgentEndpoint.from_dict(ep) if isinstance(ep, dict) else ep
                for ep in data['endpoints']
            ]
        
        # Convert metrics
        metrics = None
        if data.get('metrics'):
            metrics = AgentMetrics.from_dict(data['metrics'])
        
        return cls(
            agent_id=data['agent_id'],
            name=data['name'],
            description=data['description'],
            version=data['version'],
            agent_type=AgentType(data['agent_type']),
            capabilities=capabilities,
            endpoints=endpoints,
            supported_languages=data.get('supported_languages', ["en"]),
            tags=data.get('tags', []),
            metadata=data.get('metadata', {}),
            metrics=metrics,
            created_at=created_at,
            updated_at=updated_at,
            deployment_info=data.get('deployment_info'),
            scaling_info=data.get('scaling_info')
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AgentDetail':
        """Create agent detail from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def add_capability(self, capability: AgentCapability) -> None:
        """Add a capability to the agent detail."""
        self.capabilities.append(capability)
        self.updated_at = datetime.utcnow()
        logger.info(f"Added capability '{capability.name}' to agent {self.agent_id}")
    
    def add_endpoint(self, endpoint: AgentEndpoint) -> None:
        """Add an endpoint to the agent detail."""
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
    
    def supports_input_type(self, input_type: InputType) -> bool:
        """Check if agent supports a specific input type."""
        for capability in self.capabilities:
            if input_type in capability.input_types:
                return True
        return False
    
    def supports_output_type(self, output_type: OutputType) -> bool:
        """Check if agent supports a specific output type."""
        for capability in self.capabilities:
            if output_type in capability.output_types:
                return True
        return False
    
    def get_capabilities_by_input_type(self, input_type: InputType) -> List[AgentCapability]:
        """Get all capabilities that support a specific input type."""
        return [cap for cap in self.capabilities if input_type in cap.input_types]
    
    def get_capabilities_by_output_type(self, output_type: OutputType) -> List[AgentCapability]:
        """Get all capabilities that support a specific output type."""
        return [cap for cap in self.capabilities if output_type in cap.output_types]
    
    def update_metrics(self, metrics: AgentMetrics) -> None:
        """Update agent metrics."""
        self.metrics = metrics
        self.updated_at = datetime.utcnow()
        logger.info(f"Updated metrics for agent {self.agent_id}")
    
    def validate(self) -> List[str]:
        """Validate agent detail and return list of errors."""
        errors = []
        
        # Required field validation
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
        
        # Validate capabilities
        for i, capability in enumerate(self.capabilities):
            if not capability.name:
                errors.append(f"capability[{i}].name is required")
            if not capability.input_types:
                errors.append(f"capability[{i}].input_types cannot be empty")
            if not capability.output_types:
                errors.append(f"capability[{i}].output_types cannot be empty")
        
        return errors


class AgentDetailGenerator:
    """Generator for creating agent details with common patterns."""
    
    @staticmethod
    def create_basic_detail(
        name: str,
        description: str,
        agent_type: AgentType,
        base_url: str,
        capabilities: Optional[List[str]] = None
    ) -> AgentDetail:
        """Create a basic agent detail with common defaults."""
        
        agent_id = str(uuid.uuid4())
        
        # Create basic capabilities
        agent_capabilities = []
        if capabilities:
            for cap_name in capabilities:
                agent_capabilities.append(
                    AgentCapability(
                        name=cap_name,
                        description=f"Agent capability: {cap_name}",
                        input_types=[InputType.TEXT],
                        output_types=[OutputType.TEXT]
                    )
                )
        
        # Create default endpoint
        endpoints = [
            AgentEndpoint(
                url=urljoin(base_url, "/acp/"),
                protocol="https" if base_url.startswith("https") else "http",
                methods=["POST"],
                rate_limits={"requests_per_minute": 60}
            )
        ]
        
        detail = AgentDetail(
            agent_id=agent_id,
            name=name,
            description=description,
            version="1.0.0",
            agent_type=agent_type,
            capabilities=agent_capabilities,
            endpoints=endpoints,
            metadata={
                "framework": "mkt-acp",
                "created_by": "AgentDetailGenerator"
            }
        )
        
        logger.info(f"Generated basic agent detail for '{name}' with ID {agent_id}")
        return detail
    
    @staticmethod
    def create_ai_assistant_detail(
        name: str,
        description: str,
        base_url: str,
        model_info: Dict[str, Any],
        supported_tasks: List[str],
        input_types: List[InputType] = None,
        output_types: List[OutputType] = None
    ) -> AgentDetail:
        """Create an agent detail specifically for AI assistants."""
        
        if input_types is None:
            input_types = [InputType.TEXT, InputType.JSON]
        if output_types is None:
            output_types = [OutputType.TEXT, OutputType.JSON]
        
        # Create capabilities based on supported tasks
        capabilities = []
        for task in supported_tasks:
            capabilities.append(
                AgentCapability(
                    name=task,
                    description=f"AI task: {task}",
                    input_types=input_types,
                    output_types=output_types,
                    parameters={
                        "max_tokens": 4000,
                        "temperature": 0.7,
                        "timeout": 30
                    },
                    examples=[
                        {
                            "input": f"Example input for {task}",
                            "output": f"Example output for {task}"
                        }
                    ],
                    constraints={
                        "max_input_length": 10000,
                        "supported_formats": ["text", "json"]
                    }
                )
            )
        
        detail = AgentDetailGenerator.create_basic_detail(
            name=name,
            description=description,
            agent_type=AgentType.AI_ASSISTANT,
            base_url=base_url
        )
        
        detail.capabilities = capabilities
        detail.metadata.update({
            "agent_type": "ai_assistant",
            "model_info": model_info,
            "supported_tasks": supported_tasks
        })
        
        # Add performance metrics
        detail.metrics = AgentMetrics(
            average_response_time=2.5,
            success_rate=0.95,
            total_runs=0,
            uptime_percentage=99.5
        )
        
        logger.info(f"Generated AI assistant detail for '{name}' with {len(supported_tasks)} tasks")
        return detail
    
    @staticmethod
    def create_data_processor_detail(
        name: str,
        description: str,
        base_url: str,
        processing_types: List[str],
        supported_formats: List[str]
    ) -> AgentDetail:
        """Create an agent detail for data processing agents."""
        
        # Map formats to input/output types
        format_type_mapping = {
            "text": [InputType.TEXT, OutputType.TEXT],
            "json": [InputType.JSON, OutputType.JSON],
            "csv": [InputType.STRUCTURED_DATA, OutputType.STRUCTURED_DATA],
            "image": [InputType.IMAGE, OutputType.IMAGE],
            "audio": [InputType.AUDIO, OutputType.AUDIO],
            "video": [InputType.VIDEO, OutputType.VIDEO]
        }
        
        # Determine input/output types from supported formats
        input_types = set()
        output_types = set()
        for fmt in supported_formats:
            if fmt in format_type_mapping:
                types = format_type_mapping[fmt]
                input_types.update(types[0] if isinstance(types[0], list) else [types[0]])
                output_types.update(types[1] if isinstance(types[1], list) else [types[1]])
        
        # Create capabilities based on processing types
        capabilities = []
        for proc_type in processing_types:
            capabilities.append(
                AgentCapability(
                    name=proc_type,
                    description=f"Data processing: {proc_type}",
                    input_types=list(input_types),
                    output_types=list(output_types),
                    parameters={
                        "batch_size": 100,
                        "timeout": 60,
                        "parallel_processing": True
                    },
                    constraints={
                        "max_file_size": "100MB",
                        "supported_formats": supported_formats
                    }
                )
            )
        
        detail = AgentDetailGenerator.create_basic_detail(
            name=name,
            description=description,
            agent_type=AgentType.DATA_PROCESSOR,
            base_url=base_url
        )
        
        detail.capabilities = capabilities
        detail.metadata.update({
            "agent_type": "data_processor",
            "processing_types": processing_types,
            "supported_formats": supported_formats
        })
        
        # Add scaling information
        detail.scaling_info = {
            "min_instances": 1,
            "max_instances": 10,
            "scale_metric": "queue_length",
            "scale_threshold": 50
        }
        
        logger.info(f"Generated data processor detail for '{name}' with {len(processing_types)} processing types")
        return detail
    
    @staticmethod
    def create_service_wrapper_detail(
        name: str,
        description: str,
        base_url: str,
        wrapped_service: Dict[str, Any],
        service_endpoints: Dict[str, str]
    ) -> AgentDetail:
        """Create an agent detail for service wrapper agents."""
        
        # Create capabilities based on service endpoints
        capabilities = []
        for service_name, endpoint in service_endpoints.items():
            capabilities.append(
                AgentCapability(
                    name=service_name,
                    description=f"Service wrapper for: {service_name}",
                    input_types=[InputType.JSON, InputType.TEXT],
                    output_types=[OutputType.JSON, OutputType.TEXT],
                    parameters={
                        "service_endpoint": endpoint,
                        "timeout": 30,
                        "retry_count": 3
                    },
                    constraints={
                        "rate_limit": "100/minute",
                        "auth_required": True
                    }
                )
            )
        
        detail = AgentDetailGenerator.create_basic_detail(
            name=name,
            description=description,
            agent_type=AgentType.SERVICE_WRAPPER,
            base_url=base_url
        )
        
        detail.capabilities = capabilities
        detail.metadata.update({
            "agent_type": "service_wrapper",
            "wrapped_service": wrapped_service,
            "service_endpoints": service_endpoints
        })
        
        # Add deployment information
        detail.deployment_info = {
            "deployment_type": "containerized",
            "resource_requirements": {
                "cpu": "0.5",
                "memory": "512Mi"
            },
            "health_check": {
                "endpoint": "/health",
                "interval": 30
            }
        }
        
        logger.info(f"Generated service wrapper detail for '{name}' with {len(service_endpoints)} services")
        return detail
    
    @staticmethod
    def create_workflow_orchestrator_detail(
        name: str,
        description: str,
        base_url: str,
        workflow_types: List[str],
        supported_agents: List[str]
    ) -> AgentDetail:
        """Create an agent detail for workflow orchestrator agents."""
        
        # Create capabilities for workflow orchestration
        capabilities = []
        for workflow_type in workflow_types:
            capabilities.append(
                AgentCapability(
                    name=f"orchestrate_{workflow_type}",
                    description=f"Orchestrate {workflow_type} workflows",
                    input_types=[InputType.JSON, InputType.STRUCTURED_DATA],
                    output_types=[InputType.JSON, InputType.STRUCTURED_DATA],
                    parameters={
                        "max_parallel_tasks": 10,
                        "timeout": 300,
                        "retry_policy": "exponential_backoff"
                    },
                    constraints={
                        "max_workflow_depth": 5,
                        "supported_agents": supported_agents
                    }
                )
            )
        
        detail = AgentDetailGenerator.create_basic_detail(
            name=name,
            description=description,
            agent_type=AgentType.WORKFLOW_ORCHESTRATOR,
            base_url=base_url
        )
        
        detail.capabilities = capabilities
        detail.metadata.update({
            "agent_type": "workflow_orchestrator",
            "workflow_types": workflow_types,
            "supported_agents": supported_agents
        })
        
        # Add scaling and deployment info
        detail.scaling_info = {
            "min_instances": 2,
            "max_instances": 20,
            "scale_metric": "active_workflows",
            "scale_threshold": 80
        }
        
        detail.deployment_info = {
            "deployment_type": "kubernetes",
            "resource_requirements": {
                "cpu": "1",
                "memory": "1Gi"
            },
            "persistent_storage": True
        }
        
        logger.info(f"Generated workflow orchestrator detail for '{name}' with {len(workflow_types)} workflow types")
        return detail


class AgentDetailRegistry:
    """Registry for managing agent details."""
    
    def __init__(self):
        """Initialize the registry."""
        self._agents: Dict[str, AgentDetail] = {}
        self._agents_by_type: Dict[AgentType, List[str]] = {
            agent_type: [] for agent_type in AgentType
        }
        self._agents_by_capability: Dict[str, List[str]] = {}
    
    def register(self, agent_detail: AgentDetail) -> None:
        """Register an agent detail."""
        agent_id = agent_detail.agent_id
        
        # Store agent
        self._agents[agent_id] = agent_detail
        
        # Index by type
        if agent_id not in self._agents_by_type[agent_detail.agent_type]:
            self._agents_by_type[agent_detail.agent_type].append(agent_id)
        
        # Index by capabilities
        for capability in agent_detail.capabilities:
            if capability.name not in self._agents_by_capability:
                self._agents_by_capability[capability.name] = []
            if agent_id not in self._agents_by_capability[capability.name]:
                self._agents_by_capability[capability.name].append(agent_id)
        
        logger.info(f"Registered agent {agent_id} ({agent_detail.name})")
    
    def unregister(self, agent_id: str) -> bool:
        """Unregister an agent detail."""
        if agent_id not in self._agents:
            return False
        
        agent_detail = self._agents[agent_id]
        
        # Remove from type index
        if agent_id in self._agents_by_type[agent_detail.agent_type]:
            self._agents_by_type[agent_detail.agent_type].remove(agent_id)
        
        # Remove from capability index
        for capability in agent_detail.capabilities:
            if capability.name in self._agents_by_capability:
                if agent_id in self._agents_by_capability[capability.name]:
                    self._agents_by_capability[capability.name].remove(agent_id)
                # Clean up empty capability lists
                if not self._agents_by_capability[capability.name]:
                    del self._agents_by_capability[capability.name]
        
        # Remove agent
        del self._agents[agent_id]
        
        logger.info(f"Unregistered agent {agent_id}")
        return True
    
    def get(self, agent_id: str) -> Optional[AgentDetail]:
        """Get an agent detail by ID."""
        return self._agents.get(agent_id)
    
    def get_by_type(self, agent_type: AgentType) -> List[AgentDetail]:
        """Get all agents of a specific type."""
        agent_ids = self._agents_by_type.get(agent_type, [])
        return [self._agents[agent_id] for agent_id in agent_ids]
    
    def get_by_capability(self, capability_name: str) -> List[AgentDetail]:
        """Get all agents that support a specific capability."""
        agent_ids = self._agents_by_capability.get(capability_name, [])
        return [self._agents[agent_id] for agent_id in agent_ids]
    
    def search(
        self,
        agent_type: Optional[AgentType] = None,
        capability: Optional[str] = None,
        input_type: Optional[InputType] = None,
        output_type: Optional[OutputType] = None,
        tags: Optional[List[str]] = None
    ) -> List[AgentDetail]:
        """Search for agents based on criteria."""
        results = list(self._agents.values())
        
        # Filter by type
        if agent_type:
            results = [agent for agent in results if agent.agent_type == agent_type]
        
        # Filter by capability
        if capability:
            results = [agent for agent in results if agent.supports_capability(capability)]
        
        # Filter by input type
        if input_type:
            results = [agent for agent in results if agent.supports_input_type(input_type)]
        
        # Filter by output type
        if output_type:
            results = [agent for agent in results if agent.supports_output_type(output_type)]
        
        # Filter by tags
        if tags:
            results = [
                agent for agent in results
                if any(tag in agent.tags for tag in tags)
            ]
        
        return results
    
    def list_all(self) -> List[AgentDetail]:
        """List all registered agents."""
        return list(self._agents.values())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        total_agents = len(self._agents)
        agents_by_type = {
            agent_type.value: len(agent_ids)
            for agent_type, agent_ids in self._agents_by_type.items()
        }
        total_capabilities = len(self._agents_by_capability)
        
        return {
            "total_agents": total_agents,
            "agents_by_type": agents_by_type,
            "total_capabilities": total_capabilities,
            "capabilities": list(self._agents_by_capability.keys())
        }


# Example usage and testing
if __name__ == "__main__":
    # Create a sample agent detail
    generator = AgentDetailGenerator()
    
    # Create an AI assistant
    ai_detail = generator.create_ai_assistant_detail(
        name="Text Analysis Assistant",
        description="An AI assistant specialized in text analysis and processing",
        base_url="https://api.example.com",
        model_info={
            "model_name": "gpt-4",
            "provider": "openai",
            "version": "2024-01"
        },
        supported_tasks=["text_summarization", "sentiment_analysis", "entity_extraction", "translation"],
        input_types=[InputType.TEXT, InputType.JSON],
        output_types=[OutputType.TEXT, OutputType.JSON]
    )
    
    print("Generated AI Assistant Agent Detail:")
    print(ai_detail.to_json())
    print("\n" + "="*50 + "\n")
    
    # Create a data processor
    data_detail = generator.create_data_processor_detail(
        name="CSV Data Processor",
        description="Processes and transforms CSV data files",
        base_url="https://data.example.com",
        processing_types=["data_cleaning", "data_transformation", "data_validation"],
        supported_formats=["csv", "json", "text"]
    )
    
    print("Generated Data Processor Agent Detail:")
    print(data_detail.to_json())
    print("\n" + "="*50 + "\n")
    
    # Create a service wrapper
    service_detail = generator.create_service_wrapper_detail(
        name="Weather Service Wrapper",
        description="Wrapper for external weather API services",
        base_url="https://weather.example.com",
        wrapped_service={
            "name": "OpenWeatherMap",
            "version": "2.5",
            "documentation": "https://openweathermap.org/api"
        },
        service_endpoints={
            "current_weather": "/weather/current",
            "forecast": "/weather/forecast",
            "historical": "/weather/history"
        }
    )
    
    print("Generated Service Wrapper Agent Detail:")
    print(service_detail.to_json())
    print("\n" + "="*50 + "\n")
    
    # Create a workflow orchestrator
    workflow_detail = generator.create_workflow_orchestrator_detail(
        name="AI Workflow Orchestrator",
        description="Orchestrates complex AI workflows with multiple agents",
        base_url="https://orchestrator.example.com",
        workflow_types=["sequential", "parallel", "conditional", "loop"],
        supported_agents=["text_processor", "image_analyzer", "data_validator"]
    )
    
    print("Generated Workflow Orchestrator Agent Detail:")
    print(workflow_detail.to_json())
    print("\n" + "="*50 + "\n")
    
    # Test registry functionality
    registry = AgentDetailRegistry()
    
    # Register all agents
    registry.register(ai_detail)
    registry.register(data_detail)
    registry.register(service_detail)
    registry.register(workflow_detail)
    
    print("Registry Statistics:")
    stats = registry.get_statistics()
    print(json.dumps(stats, indent=2))
    print("\n" + "="*50 + "\n")
    
    # Test search functionality
    print("Search Results:")
    
    # Search by type
    ai_agents = registry.get_by_type(AgentType.AI_ASSISTANT)
    print(f"AI Assistants: {len(ai_agents)}")
    for agent in ai_agents:
        print(f"  - {agent.name}")
    
    # Search by capability
    text_agents = registry.get_by_capability("text_summarization")
    print(f"Agents with text_summarization: {len(text_agents)}")
    for agent in text_agents:
        print(f"  - {agent.name}")
    
    # Search by input type
    json_agents = registry.search(input_type=InputType.JSON)
    print(f"Agents supporting JSON input: {len(json_agents)}")
    for agent in json_agents:
        print(f"  - {agent.name}")
    
    # Complex search
    complex_results = registry.search(
        agent_type=AgentType.AI_ASSISTANT,
        input_type=InputType.TEXT,
        output_type=OutputType.JSON
    )
    print(f"AI Assistants with TEXT->JSON: {len(complex_results)}")
    for agent in complex_results:
        print(f"  - {agent.name}")
    
    print("\n" + "="*50 + "\n")
    
    # Validate all agent details
    print("Validation Results:")
    all_agents = [ai_detail, data_detail, service_detail, workflow_detail]
    for agent in all_agents:
        errors = agent.validate()
        if errors:
            print(f"{agent.name}: INVALID")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"{agent.name}: VALID ✓")
    
    print("\n" + "="*50 + "\n")
    
    # Test serialization/deserialization
    print("Serialization Test:")
    original = ai_detail
    json_str = original.to_json()
    reconstructed = AgentDetail.from_json(json_str)
    
    print(f"Original agent ID: {original.agent_id}")
    print(f"Reconstructed agent ID: {reconstructed.agent_id}")
    print(f"Names match: {original.name == reconstructed.name}")
    print(f"Capabilities count match: {len(original.capabilities) == len(reconstructed.capabilities)}")
    print(f"Agent types match: {original.agent_type == reconstructed.agent_type}")
    
    # Test capability methods
    print("\n" + "="*50 + "\n")
    print("Capability Tests:")
    
    print(f"AI agent supports text_summarization: {ai_detail.supports_capability('text_summarization')}")
    print(f"AI agent supports TEXT input: {ai_detail.supports_input_type(InputType.TEXT)}")
    print(f"AI agent supports IMAGE output: {ai_detail.supports_output_type(OutputType.IMAGE)}")
    
    text_caps = ai_detail.get_capabilities_by_input_type(InputType.TEXT)
    print(f"AI agent TEXT input capabilities: {[cap.name for cap in text_caps]}")
    
    json_caps = ai_detail.get_capabilities_by_output_type(OutputType.JSON)
    print(f"AI agent JSON output capabilities: {[cap.name for cap in json_caps]}")
    
    print("\n" + "="*50 + "\n")
    print("Agent Detail Implementation Complete! ✓")
    print("Features implemented:")
    print("✓ Complete ACP Agent Detail structure")
    print("✓ Multiple agent type generators")
    print("✓ Agent registry with search capabilities")
    print("✓ Input/Output type support")
    print("✓ Capability management")
    print("✓ Metrics and deployment info")
    print("✓ Validation and error handling")
    print("✓ JSON serialization/deserialization")
    print("✓ Comprehensive examples and testing")
