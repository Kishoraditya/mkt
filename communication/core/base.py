"""
Base classes and interfaces for the communication module.

This module provides abstract base classes and common interfaces that all
communication protocols (A2A, ACP, ANP) should implement for consistency
and interoperability.
"""

import uuid
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


# Enums and Constants

class ProtocolType(Enum):
    """Supported communication protocols."""
    A2A = "a2a"  # Agent-to-Agent Protocol (Google)
    ACP = "acp"  # Agent Communication Protocol (Linux Foundation/BeeAI)
    ANP = "anp"  # Agent Network Protocol (ANP Community)


class MessageStatus(Enum):
    """Message processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class ConnectionStatus(Enum):
    """Connection status for agents."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    ERROR = "error"


class AgentCapabilityType(Enum):
    """Types of agent capabilities."""
    TEXT_PROCESSING = "text_processing"
    DATA_ANALYSIS = "data_analysis"
    FILE_PROCESSING = "file_processing"
    API_INTEGRATION = "api_integration"
    WORKFLOW_EXECUTION = "workflow_execution"
    CUSTOM = "custom"


# Base Data Classes

@dataclass
class BaseMessage:
    """Base message structure for all protocols."""
    
    message_id: str
    protocol: ProtocolType
    sender_id: str
    recipient_id: Optional[str]
    timestamp: datetime
    payload: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "message_id": self.message_id,
            "protocol": self.protocol.value,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """Convert message to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseMessage':
        """Create message from dictionary."""
        return cls(
            message_id=data["message_id"],
            protocol=ProtocolType(data["protocol"]),
            sender_id=data["sender_id"],
            recipient_id=data.get("recipient_id"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            payload=data["payload"],
            metadata=data.get("metadata", {})
        )


@dataclass
class AgentCapability:
    """Represents an agent capability."""
    
    name: str
    capability_type: AgentCapabilityType
    description: str
    version: str = "1.0.0"
    parameters: Optional[Dict[str, Any]] = None
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert capability to dictionary."""
        return {
            "name": self.name,
            "capability_type": self.capability_type.value,
            "description": self.description,
            "version": self.version,
            "parameters": self.parameters,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentCapability':
        """Create capability from dictionary."""
        return cls(
            name=data["name"],
            capability_type=AgentCapabilityType(data["capability_type"]),
            description=data["description"],
            version=data.get("version", "1.0.0"),
            parameters=data.get("parameters"),
            input_schema=data.get("input_schema"),
            output_schema=data.get("output_schema"),
            metadata=data.get("metadata", {})
        )


@dataclass
class AgentEndpoint:
    """Represents an agent communication endpoint."""
    
    url: str
    protocol: ProtocolType
    transport: str = "https"  # https, http, ws, wss
    authentication: Optional[Dict[str, Any]] = None
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert endpoint to dictionary."""
        return {
            "url": self.url,
            "protocol": self.protocol.value,
            "transport": self.transport,
            "authentication": self.authentication,
            "capabilities": self.capabilities,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentEndpoint':
        """Create endpoint from dictionary."""
        return cls(
            url=data["url"],
            protocol=ProtocolType(data["protocol"]),
            transport=data.get("transport", "https"),
            authentication=data.get("authentication"),
            capabilities=data.get("capabilities", []),
            metadata=data.get("metadata", {})
        )


@dataclass
class ConnectionInfo:
    """Information about a connection between agents."""
    
    connection_id: str
    local_agent_id: str
    remote_agent_id: str
    protocol: ProtocolType
    status: ConnectionStatus
    established_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_active(self, timeout_seconds: int = 300) -> bool:
        """Check if connection is active within timeout."""
        if not self.last_activity:
            return False
        
        timeout = timedelta(seconds=timeout_seconds)
        return (datetime.utcnow() - self.last_activity) < timeout
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()


# Abstract Base Classes

class BaseAgent(ABC):
    """Abstract base class for all communication agents."""
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        capabilities: Optional[List[AgentCapability]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize base agent."""
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.capabilities = capabilities or []
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        
        # Connection management
        self.connections: Dict[str, ConnectionInfo] = {}
        self.message_handlers: Dict[str, Callable] = {}
        
        logger.info(f"Base agent initialized: {self.agent_id}")
    
    @abstractmethod
    async def start(self) -> None:
        """Start the agent."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the agent."""
        pass
    
    @abstractmethod
    async def send_message(
        self,
        recipient_id: str,
        payload: Dict[str, Any],
        message_type: Optional[str] = None
    ) -> str:
        """Send a message to another agent."""
        pass
    
    @abstractmethod
    async def receive_message(self, message: BaseMessage) -> Optional[BaseMessage]:
        """Receive and process a message."""
        pass
    
    def add_capability(self, capability: AgentCapability) -> None:
        """Add a capability to the agent."""
        self.capabilities.append(capability)
        logger.debug(f"Added capability '{capability.name}' to agent {self.agent_id}")
    
    def get_capability(self, name: str) -> Optional[AgentCapability]:
        """Get a capability by name."""
        return next((cap for cap in self.capabilities if cap.name == name), None)
    
    def has_capability(self, name: str) -> bool:
        """Check if agent has a specific capability."""
        return self.get_capability(name) is not None
    
    def register_message_handler(
        self,
        message_type: str,
        handler: Callable[[BaseMessage], Optional[BaseMessage]]
    ) -> None:
        """Register a message handler for a specific message type."""
        self.message_handlers[message_type] = handler
        logger.debug(f"Registered handler for message type '{message_type}' on agent {self.agent_id}")
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "capabilities": [cap.to_dict() for cap in self.capabilities],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "active_connections": len([c for c in self.connections.values() if c.is_active()])
        }


class BaseClient(ABC):
    """Abstract base class for protocol clients."""
    
    def __init__(
        self,
        agent_id: str,
        protocol: ProtocolType,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize base client."""
        self.agent_id = agent_id
        self.protocol = protocol
        self.config = config or {}
        self.connections: Dict[str, ConnectionInfo] = {}
        self.is_running = False
        
        logger.info(f"Base client initialized: {agent_id} ({protocol.value})")
    
    @abstractmethod
    async def connect(
        self,
        remote_agent_id: str,
        endpoint: AgentEndpoint
    ) -> ConnectionInfo:
        """Connect to a remote agent."""
        pass
    
    @abstractmethod
    async def disconnect(self, connection_id: str) -> bool:
        """Disconnect from a remote agent."""
        pass
    
    @abstractmethod
    async def send_message(
        self,
        connection_id: str,
        message: BaseMessage
    ) -> bool:
        """Send a message through a connection."""
        pass
    
    @abstractmethod
    async def discover_agents(
        self,
        criteria: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Discover available agents."""
        pass
    
    def get_connection(self, connection_id: str) -> Optional[ConnectionInfo]:
        """Get connection information."""
        return self.connections.get(connection_id)
    
    def list_connections(self) -> List[ConnectionInfo]:
        """List all connections."""
        return list(self.connections.values())
    
    def get_active_connections(self) -> List[ConnectionInfo]:
        """Get active connections."""
        return [conn for conn in self.connections.values() if conn.is_active()]


class BaseServer(ABC):
    """Abstract base class for protocol servers."""
    
    def __init__(
        self,
        agent_id: str,
        protocol: ProtocolType,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize base server."""
        self.agent_id = agent_id
        self.protocol = protocol
        self.config = config or {}
        self.is_running = False
        self.sessions: Dict[str, Any] = {}
        self.message_handlers: Dict[str, Callable] = {}
        
        logger.info(f"Base server initialized: {agent_id} ({protocol.value})")
    
    @abstractmethod
    async def start(self) -> None:
        """Start the server."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the server."""
        pass
    
    @abstractmethod
    async def handle_message(
        self,
        message: BaseMessage,
        session_id: str
    ) -> Optional[BaseMessage]:
        """Handle an incoming message."""
        pass
    
    @abstractmethod
    async def authenticate_agent(
        self,
        agent_id: str,
        credentials: Dict[str, Any]
    ) -> bool:
        """Authenticate an agent."""
        pass
    
    def register_message_handler(
        self,
        message_type: str,
        handler: Callable[[BaseMessage, str], Optional[BaseMessage]]
    ) -> None:
        """Register a message handler."""
        self.message_handlers[message_type] = handler
        logger.debug(f"Registered handler for message type '{message_type}' on server {self.agent_id}")
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            "agent_id": self.agent_id,
            "protocol": self.protocol.value,
            "is_running": self.is_running,
            "active_sessions": len(self.sessions),
            "message_handlers": list(self.message_handlers.keys())
        }


class BaseMessageProcessor(ABC):
    """Abstract base class for message processors."""
    
    def __init__(self, processor_id: str):
        """Initialize message processor."""
        self.processor_id = processor_id
        self.processed_count = 0
        self.error_count = 0
        
    @abstractmethod
    async def process_message(
        self,
        message: BaseMessage,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[BaseMessage]:
        """Process a message."""
        pass
    
    @abstractmethod
    def can_process(self, message: BaseMessage) -> bool:
        """Check if this processor can handle the message."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        return {
            "processor_id": self.processor_id,
            "processed_count": self.processed_count,
            "error_count": self.error_count,
            "success_rate": self.processed_count / max(self.processed_count + self.error_count, 1)
        }


class BaseDiscoveryService(ABC):
    """Abstract base class for agent discovery services."""
    
    def __init__(self, service_id: str):
        """Initialize discovery service."""
        self.service_id = service_id
        self.registered_agents: Dict[str, Dict[str, Any]] = {}
        
    @abstractmethod
    async def register_agent(
        self,
        agent_id: str,
        agent_info: Dict[str, Any]
    ) -> bool:
        """Register an agent with the discovery service."""
        pass
    
    @abstractmethod
    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the discovery service."""
        pass
    
    @abstractmethod
    async def discover_agents(
        self,
        criteria: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Discover agents based on criteria."""
        pass
    
    @abstractmethod
    async def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific agent."""
        pass
    
    def get_registered_count(self) -> int:
        """Get number of registered agents."""
        return len(self.registered_agents)
    
    def list_agent_ids(self) -> List[str]:
        """List all registered agent IDs."""
        return list(self.registered_agents.keys())


class BaseAuthenticationProvider(ABC):
    """Abstract base class for authentication providers."""
    
    def __init__(self, provider_id: str):
        """Initialize authentication provider."""
        self.provider_id = provider_id
        
    @abstractmethod
    async def authenticate(
        self,
        agent_id: str,
        credentials: Dict[str, Any]
    ) -> bool:
        """Authenticate an agent."""
        pass
    
    @abstractmethod
    async def generate_token(
        self,
        agent_id: str,
        permissions: Optional[List[str]] = None
    ) -> str:
        """Generate an authentication token."""
        pass
    
    @abstractmethod
    async def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate an authentication token."""
        pass
    
    @abstractmethod
    async def revoke_token(self, token: str) -> bool:
        """Revoke an authentication token."""
        pass


# Protocol Factory and Registry

class ProtocolFactory:
    """Factory for creating protocol-specific instances."""
    
    _client_classes: Dict[ProtocolType, type] = {}
    _server_classes: Dict[ProtocolType, type] = {}
    _agent_classes: Dict[ProtocolType, type] = {}
    
    @classmethod
    def register_client(cls, protocol: ProtocolType, client_class: type):
        """Register a client class for a protocol."""
        cls._client_classes[protocol] = client_class
        logger.info(f"Registered client class for protocol {protocol.value}")
    
    @classmethod
    def register_server(cls, protocol: ProtocolType, server_class: type):
        """Register a server class for a protocol."""
        cls._server_classes[protocol] = server_class
        logger.info(f"Registered server class for protocol {protocol.value}")
    
    @classmethod
    def register_agent(cls, protocol: ProtocolType, agent_class: type):
        """Register an agent class for a protocol."""
        cls._agent_classes[protocol] = agent_class
        logger.info(f"Registered agent class for protocol {protocol.value}")
    
    @classmethod
    def create_client(
        cls,
        protocol: ProtocolType,
        agent_id: str,
        config: Optional[Dict[str, Any]] = None
    ) -> BaseClient:
        """Create a client instance for the specified protocol."""
        client_class = cls._client_classes.get(protocol)
        if not client_class:
            raise ValueError(f"No client class registered for protocol {protocol.value}")
        
        return client_class(agent_id=agent_id, config=config)
    
    @classmethod
    def create_server(
        cls,
        protocol: ProtocolType,
        agent_id: str,
        config: Optional[Dict[str, Any]] = None
    ) -> BaseServer:
        """Create a server instance for the specified protocol."""
        server_class = cls._server_classes.get(protocol)
        if not server_class:
            raise ValueError(f"No server class registered for protocol {protocol.value}")
        
        return server_class(agent_id=agent_id, config=config)
    
    @classmethod
    def create_agent(
        cls,
        protocol: ProtocolType,
        agent_id: str,
        name: str,
        description: str,
        config: Optional[Dict[str, Any]] = None
    ) -> BaseAgent:
        """Create an agent instance for the specified protocol."""
        agent_class = cls._agent_classes.get(protocol)
        if not agent_class:
            raise ValueError(f"No agent class registered for protocol {protocol.value}")
        
        return agent_class(
            agent_id=agent_id,
            name=name,
            description=description,
            config=config
        )
    
    @classmethod
    def get_supported_protocols(cls) -> List[ProtocolType]:
        """Get list of supported protocols."""
        protocols = set()
        protocols.update(cls._client_classes.keys())
        protocols.update(cls._server_classes.keys())
        protocols.update(cls._agent_classes.keys())
        return list(protocols)


# Message Processing Pipeline

class MessagePipeline:
    """Pipeline for processing messages through multiple processors."""
    
    def __init__(self, pipeline_id: str):
        """Initialize message pipeline."""
        self.pipeline_id = pipeline_id
        self.processors: List[BaseMessageProcessor] = []
        self.middleware: List[Callable] = []
        
    def add_processor(self, processor: BaseMessageProcessor) -> None:
        """Add a processor to the pipeline."""
        self.processors.append(processor)
        logger.debug(f"Added processor {processor.processor_id} to pipeline {self.pipeline_id}")
    
    def add_middleware(self, middleware: Callable) -> None:
        """Add middleware to the pipeline."""
        self.middleware.append(middleware)
        logger.debug(f"Added middleware to pipeline {self.pipeline_id}")
    
    async def process_message(
        self,
        message: BaseMessage,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[BaseMessage]:
        """Process a message through the pipeline."""
        current_message = message
        processing_context = context or {}
        
        try:
            # Apply middleware
            for middleware in self.middleware:
                current_message = await middleware(current_message, processing_context)
                if current_message is None:
                    return None
            
            # Process through processors
            for processor in self.processors:
                if processor.can_process(current_message):
                    try:
                        result = await processor.process_message(current_message, processing_context)
                        if result is not None:
                            current_message = result
                        processor.processed_count += 1
                    except Exception as e:
                        processor.error_count += 1
                        logger.error(f"Error in processor {processor.processor_id}: {e}")
                        raise
            
            return current_message
            
        except Exception as e:
            logger.error(f"Error processing message in pipeline {self.pipeline_id}: {e}")
            raise
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "pipeline_id": self.pipeline_id,
            "processor_count": len(self.processors),
            "middleware_count": len(self.middleware),
            "processors": [proc.get_stats() for proc in self.processors]
        }


# Connection Management

class ConnectionManager:
    """Manages connections across different protocols."""
    
    def __init__(self):
        """Initialize connection manager."""
        self.connections: Dict[str, ConnectionInfo] = {}
        self.connection_pools: Dict[ProtocolType, List[ConnectionInfo]] = {
            protocol: [] for protocol in ProtocolType
        }
        
    def add_connection(self, connection: ConnectionInfo) -> None:
        """Add a connection to the manager."""
        self.connections[connection.connection_id] = connection
        self.connection_pools[connection.protocol].append(connection)
        logger.debug(f"Added connection {connection.connection_id} ({connection.protocol.value})")
    
    def remove_connection(self, connection_id: str) -> bool:
        """Remove a connection from the manager."""
        connection = self.connections.pop(connection_id, None)
        if connection:
            pool = self.connection_pools[connection.protocol]
            if connection in pool:
                pool.remove(connection)
            logger.debug(f"Removed connection {connection_id}")
            return True
        return False
    
    def get_connection(self, connection_id: str) -> Optional[ConnectionInfo]:
        """Get a connection by ID."""
        return self.connections.get(connection_id)
    
    def get_connections_by_protocol(self, protocol: ProtocolType) -> List[ConnectionInfo]:
        """Get all connections for a specific protocol."""
        return self.connection_pools[protocol].copy()
    
    def get_active_connections(self, protocol: Optional[ProtocolType] = None) -> List[ConnectionInfo]:
        """Get active connections, optionally filtered by protocol."""
        if protocol:
            return [conn for conn in self.connection_pools[protocol] if conn.is_active()]
        else:
            return [conn for conn in self.connections.values() if conn.is_active()]
    
    def cleanup_inactive_connections(self, timeout_seconds: int = 300) -> int:
        """Clean up inactive connections and return count of removed connections."""
        inactive_connections = [
            conn_id for conn_id, conn in self.connections.items()
            if not conn.is_active(timeout_seconds)
        ]
        
        for conn_id in inactive_connections:
            self.remove_connection(conn_id)
        
        logger.info(f"Cleaned up {len(inactive_connections)} inactive connections")
        return len(inactive_connections)
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get connection manager statistics."""
        stats = {
            "total_connections": len(self.connections),
            "active_connections": len(self.get_active_connections()),
            "by_protocol": {}
        }
        
        for protocol in ProtocolType:
            protocol_connections = self.connection_pools[protocol]
            active_protocol_connections = [conn for conn in protocol_connections if conn.is_active()]
            stats["by_protocol"][protocol.value] = {
                "total": len(protocol_connections),
                "active": len(active_protocol_connections)
            }
        
        return stats


# Event System

class CommunicationEvent:
    """Represents a communication event."""
    
    def __init__(
        self,
        event_type: str,
        agent_id: str,
        protocol: ProtocolType,
        data: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ):
        """Initialize communication event."""
        self.event_id = str(uuid.uuid4())
        self.event_type = event_type
        self.agent_id = agent_id
        self.protocol = protocol
        self.data = data
        self.timestamp = timestamp or datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "agent_id": self.agent_id,
            "protocol": self.protocol.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat()
        }


class EventBus:
    """Event bus for communication events."""
    
    def __init__(self):
        """Initialize event bus."""
        self.subscribers: Dict[str, List[Callable]] = {}
        self.event_history: List[CommunicationEvent] = []
        self.max_history = 1000
    
    def subscribe(self, event_type: str, callback: Callable[[CommunicationEvent], None]) -> None:
        """Subscribe to events of a specific type."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        logger.debug(f"Subscribed to event type: {event_type}")
    
    def unsubscribe(self, event_type: str, callback: Callable) -> bool:
        """Unsubscribe from events."""
        if event_type in self.subscribers:
            try:
                self.subscribers[event_type].remove(callback)
                return True
            except ValueError:
                pass
        return False
    
    async def publish(self, event: CommunicationEvent) -> None:
        """Publish an event to subscribers."""
        # Add to history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history.pop(0)
        
        # Notify subscribers
        subscribers = self.subscribers.get(event.event_type, [])
        for callback in subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Error in event subscriber: {e}")
    
    def get_recent_events(self, count: int = 100) -> List[CommunicationEvent]:
        """Get recent events."""
        return self.event_history[-count:]
    
    def get_events_by_type(self, event_type: str, count: int = 100) -> List[CommunicationEvent]:
        """Get recent events of a specific type."""
        filtered_events = [e for e in self.event_history if e.event_type == event_type]
        return filtered_events[-count:]


# Configuration Management

@dataclass
class CommunicationConfig:
    """Configuration for communication module."""
    
    # General settings
    default_protocol: ProtocolType = ProtocolType.A2A
    max_connections_per_protocol: int = 100
    connection_timeout: int = 300
    message_timeout: int = 30
    
    # Discovery settings
    enable_discovery: bool = True
    discovery_interval: int = 60
    
    # Authentication settings
    require_authentication: bool = True
    token_expiry: int = 3600
    
    # Monitoring settings
    enable_monitoring: bool = True
    metrics_interval: int = 30
    
    # Protocol-specific settings
    a2a_config: Dict[str, Any] = field(default_factory=dict)
    acp_config: Dict[str, Any] = field(default_factory=dict)
    anp_config: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CommunicationConfig':
        """Create config from dictionary."""
        # Handle protocol enum conversion
        if 'default_protocol' in data:
            data['default_protocol'] = ProtocolType(data['default_protocol'])
        
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        data = {}
        for field_name, field_def in self.__dataclass_fields__.items():
            value = getattr(self, field_name)
            if isinstance(value, ProtocolType):
                data[field_name] = value.value
            else:
                data[field_name] = value
        return data


# Utility Functions

def generate_message_id() -> str:
    """Generate a unique message ID."""
    return str(uuid.uuid4())


def generate_connection_id(local_agent_id: str, remote_agent_id: str, protocol: ProtocolType) -> str:
    """Generate a unique connection ID."""
    timestamp = int(datetime.utcnow().timestamp())
    return f"{protocol.value}_{local_agent_id}_{remote_agent_id}_{timestamp}"


def validate_agent_id(agent_id: str) -> bool:
    """Validate agent ID format."""
    if not agent_id or not isinstance(agent_id, str):
        return False
    
    # Basic validation - alphanumeric, hyphens, underscores
    import re
    pattern = r'^[a-zA-Z0-9_-]+$'
    return bool(re.match(pattern, agent_id)) and len(agent_id) <= 255


def validate_message_payload(payload: Dict[str, Any]) -> List[str]:
    """Validate message payload and return list of errors."""
    errors = []
    
    if not isinstance(payload, dict):
        errors.append("Payload must be a dictionary")
        return errors
    
    # Check for required fields based on common patterns
    if 'type' not in payload:
        errors.append("Payload must include 'type' field")
    
    # Validate payload size (example: 10MB limit)
    try:
        payload_size = len(json.dumps(payload).encode('utf-8'))
        if payload_size > 10 * 1024 * 1024:  # 10MB
            errors.append("Payload size exceeds 10MB limit")
    except Exception as e:
        errors.append(f"Error serializing payload: {e}")
    
    return errors


def create_error_response(
    original_message: BaseMessage,
    error_code: str,
    error_message: str,
    error_details: Optional[Dict[str, Any]] = None
) -> BaseMessage:
    """Create an error response message."""
    error_payload = {
        "type": "error",
        "error_code": error_code,
        "error_message": error_message,
        "original_message_id": original_message.message_id
    }
    
    if error_details:
        error_payload["error_details"] = error_details
    
    return BaseMessage(
        message_id=generate_message_id(),
        protocol=original_message.protocol,
        sender_id=original_message.recipient_id or "system",
        recipient_id=original_message.sender_id,
        timestamp=datetime.utcnow(),
        payload=error_payload,
        metadata={"in_response_to": original_message.message_id}
    )


def create_success_response(
    original_message: BaseMessage,
    response_data: Dict[str, Any]
) -> BaseMessage:
    """Create a success response message."""
    success_payload = {
        "type": "response",
        "status": "success",
        "data": response_data,
        "original_message_id": original_message.message_id
    }
    
    return BaseMessage(
        message_id=generate_message_id(),
        protocol=original_message.protocol,
        sender_id=original_message.recipient_id or "system",
        recipient_id=original_message.sender_id,
        timestamp=datetime.utcnow(),
        payload=success_payload,
        metadata={"in_response_to": original_message.message_id}
    )


async def timeout_wrapper(
    coro: Callable,
    timeout_seconds: int,
    timeout_message: str = "Operation timed out"
) -> Any:
    """Wrap a coroutine with timeout handling."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning(f"Timeout after {timeout_seconds}s: {timeout_message}")
        raise TimeoutError(timeout_message)


def serialize_datetime(dt: datetime) -> str:
    """Serialize datetime to ISO format string."""
    return dt.isoformat()


def deserialize_datetime(dt_str: str) -> datetime:
    """Deserialize datetime from ISO format string."""
    return datetime.fromisoformat(dt_str)


# Exception Classes

class CommunicationError(Exception):
    """Base exception for communication module."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Initialize communication error."""
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}


class ProtocolError(CommunicationError):
    """Exception for protocol-specific errors."""
    
    def __init__(self, protocol: ProtocolType, message: str, error_code: Optional[str] = None):
        """Initialize protocol error."""
        super().__init__(f"[{protocol.value}] {message}", error_code)
        self.protocol = protocol


class AuthenticationError(CommunicationError):
    """Exception for authentication errors."""
    pass


class ConnectionError(CommunicationError):
    """Exception for connection errors."""
    pass


class MessageError(CommunicationError):
    """Exception for message processing errors."""
    pass


class DiscoveryError(CommunicationError):
    """Exception for discovery service errors."""
    pass


class ConfigurationError(CommunicationError):
    """Exception for configuration errors."""
    pass


# Monitoring and Metrics

class CommunicationMetrics:
    """Metrics collector for communication module."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.message_counts: Dict[str, int] = {}
        self.connection_counts: Dict[str, int] = {}
        self.error_counts: Dict[str, int] = {}
        self.response_times: Dict[str, List[float]] = {}
        self.start_time = datetime.utcnow()
    
    def record_message(self, protocol: ProtocolType, message_type: str) -> None:
        """Record a message."""
        key = f"{protocol.value}_{message_type}"
        self.message_counts[key] = self.message_counts.get(key, 0) + 1
    
    def record_connection(self, protocol: ProtocolType, action: str) -> None:
        """Record a connection event."""
        key = f"{protocol.value}_{action}"
        self.connection_counts[key] = self.connection_counts.get(key, 0) + 1
    
    def record_error(self, protocol: ProtocolType, error_type: str) -> None:
        """Record an error."""
        key = f"{protocol.value}_{error_type}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
    
    def record_response_time(self, protocol: ProtocolType, operation: str, time_ms: float) -> None:
        """Record response time."""
        key = f"{protocol.value}_{operation}"
        if key not in self.response_times:
            self.response_times[key] = []
        self.response_times[key].append(time_ms)
        
        # Keep only last 1000 measurements
        if len(self.response_times[key]) > 1000:
            self.response_times[key] = self.response_times[key][-1000:]
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        # Calculate average response times
        avg_response_times = {}
        for key, times in self.response_times.items():
            if times:
                avg_response_times[key] = sum(times) / len(times)
        
        return {
            "uptime_seconds": uptime,
            "message_counts": self.message_counts.copy(),
            "connection_counts": self.connection_counts.copy(),
            "error_counts": self.error_counts.copy(),
            "average_response_times_ms": avg_response_times,
            "total_messages": sum(self.message_counts.values()),
            "total_connections": sum(self.connection_counts.values()),
            "total_errors": sum(self.error_counts.values())
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.message_counts.clear()
        self.connection_counts.clear()
        self.error_counts.clear()
        self.response_times.clear()
        self.start_time = datetime.utcnow()


# Global instances and registry

# Global metrics instance
_global_metrics = CommunicationMetrics()

# Global event bus
_global_event_bus = EventBus()

# Global connection manager
_global_connection_manager = ConnectionManager()

# Global protocol factory
_global_protocol_factory = ProtocolFactory()


def get_global_metrics() -> CommunicationMetrics:
    """Get global metrics instance."""
    return _global_metrics


def get_global_event_bus() -> EventBus:
    """Get global event bus instance."""
    return _global_event_bus


def get_global_connection_manager() -> ConnectionManager:
    """Get global connection manager instance."""
    return _global_connection_manager


def get_global_protocol_factory() -> ProtocolFactory:
    """Get global protocol factory instance."""
    return _global_protocol_factory


# Module initialization

def initialize_communication_module(config: Optional[CommunicationConfig] = None) -> None:
    """Initialize the communication module with configuration."""
    if config is None:
        config = CommunicationConfig()
    
    # Store config globally
    global _module_config
    _module_config = config
    
    logger.info(f"Communication module initialized with default protocol: {config.default_protocol.value}")
    
    # Publish initialization event
    init_event = CommunicationEvent(
        event_type="module_initialized",
        agent_id="system",
        protocol=config.default_protocol,
        data={"config": config.to_dict()}
    )
    
    asyncio.create_task(_global_event_bus.publish(init_event))


def get_module_config() -> CommunicationConfig:
    """Get module configuration."""
    global _module_config
    if '_module_config' not in globals():
        _module_config = CommunicationConfig()
    return _module_config


# Cleanup function

async def cleanup_communication_module() -> None:
    """Clean up communication module resources."""
    logger.info("Cleaning up communication module")
    
    # Clean up connections
    _global_connection_manager.cleanup_inactive_connections(0)  # Clean all
    
    # Reset metrics
    _global_metrics.reset_metrics()
    
    # Clear event history
    _global_event_bus.event_history.clear()
    
    # Publish cleanup event
    cleanup_event = CommunicationEvent(
        event_type="module_cleanup",
        agent_id="system",
        protocol=get_module_config().default_protocol,
        data={"timestamp": datetime.utcnow().isoformat()}
    )
    
    await _global_event_bus.publish(cleanup_event)
    
    logger.info("Communication module cleanup complete")


# Module exports
__all__ = [
    # Enums
    'ProtocolType',
    'MessageStatus',
    'ConnectionStatus',
    'AgentCapabilityType',
    
    # Data classes
    'BaseMessage',
    'AgentCapability',
    'AgentEndpoint',
    'ConnectionInfo',
    'CommunicationConfig',
    
    # Abstract base classes
    'BaseAgent',
    'BaseClient',
    'BaseServer',
    'BaseMessageProcessor',
    'BaseDiscoveryService',
    'BaseAuthenticationProvider',
    
    # Core classes
    'ProtocolFactory',
    'MessagePipeline',
    'ConnectionManager',
    'CommunicationEvent',
    'EventBus',
    'CommunicationMetrics',
    
    # Exceptions
    'CommunicationError',
    'ProtocolError',
    'AuthenticationError',
    'ConnectionError',
    'MessageError',
    'DiscoveryError',
    'ConfigurationError',
    
    # Utility functions
    'generate_message_id',
    'generate_connection_id',
    'validate_agent_id',
    'validate_message_payload',
    'create_error_response',
    'create_success_response',
    'timeout_wrapper',
    'serialize_datetime',
    'deserialize_datetime',
    
    # Global accessors
    'get_global_metrics',
    'get_global_event_bus',
    'get_global_connection_manager',
    'get_global_protocol_factory',
    'get_module_config',
    
    # Module functions
    'initialize_communication_module',
    'cleanup_communication_module'
]

# Module metadata
__version__ = "1.0.0"
__author__ = "MKT Communication Team"
__license__ = "MIT"

# Initialize with default config if not already initialized
if '_module_config' not in globals():
    initialize_communication_module()

logger.info(f"Communication base module loaded (version {__version__})")
