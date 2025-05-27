"""
A2A Message implementation.

Handles message structure, parts, and serialization for A2A protocol communication.
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import base64
import mimetypes

import logging

logger = logging.getLogger(__name__)


class MessagePart(ABC):
    """Abstract base class for message parts."""
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert part to dictionary."""
        pass
    
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MessagePart':
        """Create part from dictionary."""
        pass


@dataclass
class TextPart(MessagePart):
    """Text content part of a message."""
    
    text: str
    format: str = "plain"  # plain, markdown, html
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": "text",
            "text": self.text,
            "format": self.format
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TextPart':
        """Create from dictionary."""
        return cls(
            text=data["text"],
            format=data.get("format", "plain")
        )


@dataclass
class FilePart(MessagePart):
    """File content part of a message."""
    
    filename: str
    content: bytes
    mime_type: Optional[str] = None
    size: Optional[int] = None
    
    def __post_init__(self):
        """Initialize computed fields."""
        if self.mime_type is None:
            self.mime_type, _ = mimetypes.guess_type(self.filename)
        if self.size is None:
            self.size = len(self.content)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": "file",
            "filename": self.filename,
            "content": base64.b64encode(self.content).decode('utf-8'),
            "mime_type": self.mime_type,
            "size": self.size
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FilePart':
        """Create from dictionary."""
        content = base64.b64decode(data["content"])
        return cls(
            filename=data["filename"],
            content=content,
            mime_type=data.get("mime_type"),
            size=data.get("size")
        )


@dataclass
class DataPart(MessagePart):
    """Structured data part of a message."""
    
    data: Dict[str, Any]
    schema: Optional[Dict[str, Any]] = None
    format: str = "json"  # json, xml, yaml
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": "data",
            "data": self.data,
            "schema": self.schema,
            "format": self.format
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataPart':
        """Create from dictionary."""
        return cls(
            data=data["data"],
            schema=data.get("schema"),
            format=data.get("format", "json")
        )


@dataclass
class A2AMessage:
    """
    A2A Message following Google A2A specification.
    
    Represents a communication turn between agents with support for
    multimodal content through message parts.
    """
    
    # Required fields
    message_id: str
    role: str  # "user" or "agent"
    parts: List[MessagePart]
    
    # Optional fields
    task_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    parent_message_id: Optional[str] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        data = {
            "message_id": self.message_id,
            "role": self.role,
            "parts": [part.to_dict() for part in self.parts],
            "task_id": self.task_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata,
            "parent_message_id": self.parent_message_id
        }
        return {k: v for k, v in data.items() if v is not None}
    
    def to_json(self) -> str:
        """Convert message to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'A2AMessage':
        """Create message from dictionary."""
        # Convert timestamp string back to datetime
        timestamp = None
        if data.get('timestamp'):
            timestamp = datetime.fromisoformat(data['timestamp'])
        
        # Convert part dictionaries to MessagePart objects
        parts = []
        for part_data in data.get('parts', []):
            part_type = part_data.get('type')
            if part_type == 'text':
                parts.append(TextPart.from_dict(part_data))
            elif part_type == 'file':
                parts.append(FilePart.from_dict(part_data))
            elif part_type == 'data':
                parts.append(DataPart.from_dict(part_data))
            else:
                logger.warning(f"Unknown part type: {part_type}")
        
        return cls(
            message_id=data['message_id'],
            role=data['role'],
            parts=parts,
            task_id=data.get('task_id'),
            timestamp=timestamp,
            metadata=data.get('metadata', {}),
            parent_message_id=data.get('parent_message_id')
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'A2AMessage':
        """Create message from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def add_text_part(self, text: str, format: str = "plain") -> None:
        """Add a text part to the message."""
        part = TextPart(text=text, format=format)
        self.parts.append(part)
        logger.debug(f"Added text part to message {self.message_id}")
    
    def add_file_part(self, filename: str, content: bytes, mime_type: Optional[str] = None) -> None:
        """Add a file part to the message."""
        part = FilePart(filename=filename, content=content, mime_type=mime_type)
        self.parts.append(part)
        logger.debug(f"Added file part '{filename}' to message {self.message_id}")
    
    def add_data_part(self, data: Dict[str, Any], schema: Optional[Dict[str, Any]] = None, format: str = "json") -> None:
        """Add a data part to the message."""
        part = DataPart(data=data, schema=schema, format=format)
        self.parts.append(part)
        logger.debug(f"Added data part to message {self.message_id}")
    
    def get_text_content(self) -> str:
        """Get all text content from the message."""
        text_parts = [part for part in self.parts if isinstance(part, TextPart)]
        return "\n".join(part.text for part in text_parts)
    
    def get_file_parts(self) -> List[FilePart]:
        """Get all file parts from the message."""
        return [part for part in self.parts if isinstance(part, FilePart)]
    
    def get_data_parts(self) -> List[DataPart]:
        """Get all data parts from the message."""
        return [part for part in self.parts if isinstance(part, DataPart)]
    
    def validate(self) -> List[str]:
        """Validate message and return list of errors."""
        errors = []
        
        if not self.message_id:
            errors.append("message_id is required")
        if self.role not in ['user', 'agent']:
            errors.append("role must be 'user' or 'agent'")
        if not self.parts:
            errors.append("at least one message part is required")
        
        # Validate each part
        for i, part in enumerate(self.parts):
            if not isinstance(part, MessagePart):
                errors.append(f"part[{i}] must be a MessagePart instance")
        
        return errors
    
    def get_size(self) -> int:
        """Get total size of message in bytes."""
        total_size = 0
        for part in self.parts:
            if isinstance(part, TextPart):
                total_size += len(part.text.encode('utf-8'))
            elif isinstance(part, FilePart):
                total_size += part.size or 0
            elif isinstance(part, DataPart):
                total_size += len(json.dumps(part.data).encode('utf-8'))
        return total_size


class MessageBuilder:
    """Builder class for creating A2A messages."""
    
    def __init__(self, role: str, task_id: Optional[str] = None):
        """Initialize message builder."""
        self.message_id = str(uuid.uuid4())
        self.role = role
        self.task_id = task_id
        self.parts = []
        self.metadata = {}
        self.parent_message_id = None
    
    def add_text(self, text: str, format: str = "plain") -> 'MessageBuilder':
        """Add text part to message."""
        self.parts.append(TextPart(text=text, format=format))
        return self
    
    def add_file(self, filename: str, content: bytes, mime_type: Optional[str] = None) -> 'MessageBuilder':
        """Add file part to message."""
        self.parts.append(FilePart(filename=filename, content=content, mime_type=mime_type))
        return self
    
    def add_data(self, data: Dict[str, Any], schema: Optional[Dict[str, Any]] = None, format: str = "json") -> 'MessageBuilder':
        """Add data part to message."""
        self.parts.append(DataPart(data=data, schema=schema, format=format))
        return self
    
    def set_metadata(self, metadata: Dict[str, Any]) -> 'MessageBuilder':
        """Set message metadata."""
        self.metadata = metadata
        return self
    
    def set_parent(self, parent_message_id: str) -> 'MessageBuilder':
        """Set parent message ID."""
        self.parent_message_id = parent_message_id
        return self
    
    def build(self) -> A2AMessage:
        """Build the A2A message."""
        message = A2AMessage(
            message_id=self.message_id,
            role=self.role,
            parts=self.parts,
            task_id=self.task_id,
            metadata=self.metadata,
            parent_message_id=self.parent_message_id
        )
        
        logger.info(f"Built A2A message {self.message_id} with {len(self.parts)} parts")
        return message


# Example usage and testing
if __name__ == "__main__":
    # Create a message using the builder
    builder = MessageBuilder(role="user", task_id="task-123")
    
    message = (builder
               .add_text("Hello, can you analyze this data?")
               .add_data({"values": [1, 2, 3, 4, 5], "type": "numeric"})
               .set_metadata({"priority": "high", "source": "api"})
               .build())
    
    print("Created A2A Message:")
    print(message.to_json())
    
    # Validate the message
    errors = message.validate()
    if errors:
        print("Validation errors:", errors)
    else:
        print("Message is valid!")
    
    # Test serialization/deserialization
    json_str = message.to_json()
    reconstructed = A2AMessage.from_json(json_str)
    print(f"Original message ID: {message.message_id}")
    print(f"Reconstructed message ID: {reconstructed.message_id}")
    print(f"Text content: {reconstructed.get_text_content()}")
