"""
Agent-to-Agent (A2A) Protocol Implementation

This module implements Google's A2A specification for agent communication.
Supports JSON-RPC 2.0 over HTTP(S) with Agent Cards, Tasks, and Messages.

Key Components:
- AgentCard: Agent metadata and capabilities
- Task: Unit of work with lifecycle management
- Message: Communication turns between agents
- Client/Server: Communication endpoints
"""

from .agent_card import AgentCard, AgentCardGenerator
from .client import A2AClient
from .server import A2AServer
from .message import A2AMessage, MessagePart, TextPart, FilePart, DataPart
from .task import Task, TaskManager
from .streaming import StreamingHandler
"""
A2A Protocol Implementation.

This module provides a complete implementation of the Agent-to-Agent (A2A)
communication protocol, including client, server, streaming, and message
handling capabilities.
"""

from .agent_card import AgentCard, Endpoint, Capability
from .message import A2AMessage, MessageBuilder, MessagePart
from .task import Task, TaskManager, TaskStatus, TaskPriority, TaskResult
from .client import A2AClient, A2AClientError
from .server import A2AServer
from .streaming import StreamingHandler

__version__ = "1.0.0"
__author__ = "MKT Communication Team"
__email__ = "communication@mkt.dev"

# Protocol constants
PROTOCOL_NAME = "a2a"
PROTOCOL_VERSION = "1.0"
DEFAULT_TIMEOUT = 30  # seconds
MAX_MESSAGE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_TASK_DURATION = 3600  # 1 hour

# Well-known endpoints
AGENT_CARD_ENDPOINT = "/.well-known/agent-card"
TASKS_ENDPOINT = "/tasks"
MESSAGES_ENDPOINT = "/messages"
STREAM_ENDPOINT = "/stream"
WEBSOCKET_ENDPOINT = "/ws"

# Content types
CONTENT_TYPE_JSON = "application/json"
CONTENT_TYPE_SSE = "text/event-stream"

__all__ = [
    # Core classes
    "AgentCard",
    "Endpoint", 
    "Capability",
    "A2AMessage",
    "MessageBuilder",
    "MessagePart",
    "Task",
    "TaskManager",
    "TaskStatus",
    "TaskPriority", 
    "TaskResult",
    "A2AClient",
    "A2AClientError",
    "A2AServer",
    "StreamingHandler",
    
    # Constants
    "PROTOCOL_NAME",
    "PROTOCOL_VERSION",
    "DEFAULT_TIMEOUT",
    "MAX_MESSAGE_SIZE",
    "MAX_TASK_DURATION",
    "AGENT_CARD_ENDPOINT",
    "TASKS_ENDPOINT",
    "MESSAGES_ENDPOINT",
    "STREAM_ENDPOINT",
    "WEBSOCKET_ENDPOINT",
    "CONTENT_TYPE_JSON",
    "CONTENT_TYPE_SSE",
]


__version__ = "1.0.0"
__all__ = [
    "AgentCard",
    "AgentCardGenerator", 
    "A2AClient",
    "A2AServer",
    "A2AMessage",
    "MessagePart",
    "TextPart",
    "FilePart", 
    "DataPart",
    "Task",
    "TaskManager",
    "StreamingHandler"
]
