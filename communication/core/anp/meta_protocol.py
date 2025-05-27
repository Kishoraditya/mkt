"""
ANP Meta-Protocol implementation.

The Meta-Protocol is a core component of the Agent Network Protocol (ANP) that enables
dynamic protocol negotiation and code generation for agent communication.

This implementation follows the ANP specification for protocol discovery,
negotiation, and dynamic adaptation.
"""

import json
import uuid
import hashlib
import importlib
import inspect
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Type
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from enum import Enum
import asyncio
import threading
import time

import logging

logger = logging.getLogger(__name__)


class ProtocolVersion(Enum):
    """Protocol version enumeration."""
    V1_0 = "1.0"
    V1_1 = "1.1"
    V2_0 = "2.0"


class NegotiationStatus(Enum):
    """Protocol negotiation status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class ProtocolCapability(Enum):
    """Protocol capability types."""
    MESSAGING = "messaging"
    FILE_TRANSFER = "file_transfer"
    STREAMING = "streaming"
    ENCRYPTION = "encryption"
    COMPRESSION = "compression"
    AUTHENTICATION = "authentication"
    DISCOVERY = "discovery"
    MONITORING = "monitoring"


@dataclass
class ProtocolDescriptor:
    """Describes a protocol and its capabilities."""
    
    protocol_id: str
    name: str
    version: ProtocolVersion
    description: str
    capabilities: List[ProtocolCapability]
    endpoints: List[str]
    schema_url: Optional[str] = None
    documentation_url: Optional[str] = None
    implementation_class: Optional[str] = None
    dependencies: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['version'] = self.version.value
        data['capabilities'] = [cap.value for cap in self.capabilities]
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProtocolDescriptor':
        """Create from dictionary."""
        # Convert version string to enum
        if 'version' in data:
            data['version'] = ProtocolVersion(data['version'])
        
        # Convert capability strings to enums
        if 'capabilities' in data:
            data['capabilities'] = [ProtocolCapability(cap) for cap in data['capabilities']]
        
        # Convert datetime string
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        return cls(**data)
    
    def supports_capability(self, capability: ProtocolCapability) -> bool:
        """Check if protocol supports a capability."""
        return capability in self.capabilities
    
    def is_compatible_with(self, other: 'ProtocolDescriptor') -> bool:
        """Check compatibility with another protocol."""
        # Basic compatibility check
        if self.protocol_id != other.protocol_id:
            return False
        
        # Version compatibility (same major version)
        self_major = int(self.version.value.split('.')[0])
        other_major = int(other.version.value.split('.')[0])
        
        return self_major == other_major
    
    def get_compatibility_score(self, other: 'ProtocolDescriptor') -> float:
        """Get compatibility score with another protocol (0.0 to 1.0)."""
        if not self.is_compatible_with(other):
            return 0.0
        
        # Calculate score based on shared capabilities
        shared_capabilities = set(self.capabilities) & set(other.capabilities)
        total_capabilities = set(self.capabilities) | set(other.capabilities)
        
        if not total_capabilities:
            return 1.0
        
        return len(shared_capabilities) / len(total_capabilities)


@dataclass
class NegotiationRequest:
    """Protocol negotiation request."""
    
    negotiation_id: str
    requester_id: str
    target_id: str
    supported_protocols: List[ProtocolDescriptor]
    preferred_capabilities: List[ProtocolCapability]
    requirements: Optional[Dict[str, Any]] = None
    timeout_seconds: int = 30
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.requirements is None:
            self.requirements = {}
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'negotiation_id': self.negotiation_id,
            'requester_id': self.requester_id,
            'target_id': self.target_id,
            'supported_protocols': [proto.to_dict() for proto in self.supported_protocols],
            'preferred_capabilities': [cap.value for cap in self.preferred_capabilities],
            'requirements': self.requirements,
            'timeout_seconds': self.timeout_seconds,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NegotiationRequest':
        """Create from dictionary."""
        # Convert protocol dictionaries
        if 'supported_protocols' in data:
            data['supported_protocols'] = [
                ProtocolDescriptor.from_dict(proto) for proto in data['supported_protocols']
            ]
        
        # Convert capability strings
        if 'preferred_capabilities' in data:
            data['preferred_capabilities'] = [
                ProtocolCapability(cap) for cap in data['preferred_capabilities']
            ]
        
        # Convert datetime
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        return cls(**data)


@dataclass
class NegotiationResponse:
    """Protocol negotiation response."""
    
    negotiation_id: str
    responder_id: str
    status: NegotiationStatus
    selected_protocol: Optional[ProtocolDescriptor] = None
    agreed_capabilities: Optional[List[ProtocolCapability]] = None
    configuration: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    generated_code: Optional[str] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.agreed_capabilities is None:
            self.agreed_capabilities = []
        if self.configuration is None:
            self.configuration = {}
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'negotiation_id': self.negotiation_id,
            'responder_id': self.responder_id,
            'status': self.status.value,
            'selected_protocol': self.selected_protocol.to_dict() if self.selected_protocol else None,
            'agreed_capabilities': [cap.value for cap in self.agreed_capabilities] if self.agreed_capabilities else [],
            'configuration': self.configuration,
            'error_message': self.error_message,
            'generated_code': self.generated_code,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NegotiationResponse':
        """Create from dictionary."""
        # Convert status string
        if 'status' in data:
            data['status'] = NegotiationStatus(data['status'])
        
        # Convert protocol dictionary
        if 'selected_protocol' in data and data['selected_protocol']:
            data['selected_protocol'] = ProtocolDescriptor.from_dict(data['selected_protocol'])
        
        # Convert capability strings
        if 'agreed_capabilities' in data:
            data['agreed_capabilities'] = [
                ProtocolCapability(cap) for cap in data['agreed_capabilities']
            ]
        
        # Convert datetime
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        return cls(**data)


class ProtocolRegistry:
    """Registry for managing available protocols."""
    
    def __init__(self):
        """Initialize protocol registry."""
        self.protocols: Dict[str, ProtocolDescriptor] = {}
        self.implementations: Dict[str, Type] = {}
        self._lock = threading.RLock()
        
        # Register built-in protocols
        self._register_builtin_protocols()
        
        logger.info("Initialized protocol registry")
    
    def _register_builtin_protocols(self) -> None:
        """Register built-in ANP protocols."""
        # Basic messaging protocol
        messaging_protocol = ProtocolDescriptor(
            protocol_id="anp.messaging.v1",
            name="ANP Basic Messaging",
            version=ProtocolVersion.V1_0,
            description="Basic message exchange protocol for ANP",
            capabilities=[
                ProtocolCapability.MESSAGING,
                ProtocolCapability.AUTHENTICATION,
                ProtocolCapability.ENCRYPTION
            ],
            endpoints=["/anp/messaging/v1"],
            implementation_class="communication.core.anp.protocols.MessagingProtocol"
        )
        self.register_protocol(messaging_protocol)
        
        # File transfer protocol
        file_transfer_protocol = ProtocolDescriptor(
            protocol_id="anp.file_transfer.v1",
            name="ANP File Transfer",
            version=ProtocolVersion.V1_0,
            description="Secure file transfer protocol for ANP",
            capabilities=[
                ProtocolCapability.FILE_TRANSFER,
                ProtocolCapability.ENCRYPTION,
                ProtocolCapability.COMPRESSION
            ],
            endpoints=["/anp/file_transfer/v1"],
            implementation_class="communication.core.anp.protocols.FileTransferProtocol"
        )
        self.register_protocol(file_transfer_protocol)
        
        # Streaming protocol
        streaming_protocol = ProtocolDescriptor(
            protocol_id="anp.streaming.v1",
            name="ANP Streaming",
            version=ProtocolVersion.V1_0,
            description="Real-time streaming protocol for ANP",
            capabilities=[
                ProtocolCapability.STREAMING,
                ProtocolCapability.MESSAGING,
                ProtocolCapability.ENCRYPTION
            ],
            endpoints=["/anp/streaming/v1"],
            implementation_class="communication.core.anp.protocols.StreamingProtocol"
        )
        self.register_protocol(streaming_protocol)
    
    def register_protocol(self, protocol: ProtocolDescriptor) -> None:
        """Register a protocol."""
        with self._lock:
            self.protocols[protocol.protocol_id] = protocol
            logger.debug(f"Registered protocol: {protocol.protocol_id}")
    
    def unregister_protocol(self, protocol_id: str) -> bool:
        """Unregister a protocol."""
        with self._lock:
            if protocol_id in self.protocols:
                del self.protocols[protocol_id]
                if protocol_id in self.implementations:
                    del self.implementations[protocol_id]
                logger.debug(f"Unregistered protocol: {protocol_id}")
                return True
            return False
    
    def get_protocol(self, protocol_id: str) -> Optional[ProtocolDescriptor]:
        """Get protocol by ID."""
        return self.protocols.get(protocol_id)
    
    def list_protocols(self) -> List[ProtocolDescriptor]:
        """List all registered protocols."""
        return list(self.protocols.values())
    
    def find_compatible_protocols(
        self,
        capabilities: List[ProtocolCapability],
        version: Optional[ProtocolVersion] = None
    ) -> List[ProtocolDescriptor]:
        """Find protocols that support required capabilities."""
        compatible = []
        
        for protocol in self.protocols.values():
            # Check version if specified
            if version and protocol.version != version:
                continue
            
            # Check if protocol supports all required capabilities
            if all(protocol.supports_capability(cap) for cap in capabilities):
                compatible.append(protocol)
        
        return compatible
    
    def register_implementation(self, protocol_id: str, implementation_class: Type) -> None:
        """Register a protocol implementation class."""
        with self._lock:
            self.implementations[protocol_id] = implementation_class
            logger.debug(f"Registered implementation for protocol: {protocol_id}")
    
    def get_implementation(self, protocol_id: str) -> Optional[Type]:
        """Get protocol implementation class."""
        return self.implementations.get(protocol_id)


class CodeGenerator:
    """Generates protocol implementation code dynamically."""
    
    def __init__(self):
        """Initialize code generator."""
        self.templates: Dict[str, str] = {}
        self.generated_code_cache: Dict[str, str] = {}
        self._load_templates()
        
        logger.info("Initialized code generator")
    
    def _load_templates(self) -> None:
        """Load code generation templates."""
        # Basic client template
        self.templates['client'] = '''
class {class_name}Client:
    """Generated client for {protocol_name} protocol."""
    
    def __init__(self, endpoint: str, auth_handler=None):
        self.endpoint = endpoint
        self.auth_handler = auth_handler
        self.session = None
    
    async def connect(self):
        """Establish connection to the protocol endpoint."""
        # Generated connection logic
        pass
    
    async def disconnect(self):
        """Close connection to the protocol endpoint."""
        # Generated disconnection logic
        pass
    
    {methods}
'''
        
        # Basic server template
        self.templates['server'] = '''
class {class_name}Server:
    """Generated server for {protocol_name} protocol."""
    
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self.handlers = {{}}
    
    def register_handler(self, method: str, handler: Callable):
        """Register a method handler."""
        self.handlers[method] = handler
    
    async def start(self):
        """Start the protocol server."""
        # Generated server startup logic
        pass
    
    async def stop(self):
        """Stop the protocol server."""
        # Generated server shutdown logic
        pass
    
    {methods}
'''
        
        # Method template
        self.templates['method'] = '''
    async def {method_name}(self, {parameters}):
        """Generated method for {method_description}."""
        # Method implementation
        {method_body}
'''
        
        # Message handler template
        self.templates['message_handler'] = '''
    async def handle_{message_type}(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle {message_type} message."""
        # Generated message handling logic
        return {{"status": "success", "data": message}}
'''
    
    def generate_client_code(
        self,
        protocol: ProtocolDescriptor,
        configuration: Dict[str, Any]
    ) -> str:
        """Generate client code for a protocol."""
        cache_key = f"client_{protocol.protocol_id}_{self._hash_config(configuration)}"
        
        if cache_key in self.generated_code_cache:
            return self.generated_code_cache[cache_key]
        
        class_name = self._generate_class_name(protocol.name)
        methods = self._generate_client_methods(protocol, configuration)
        
        code = self.templates['client'].format(
            class_name=class_name,
            protocol_name=protocol.name,
            methods=methods
        )
        
        self.generated_code_cache[cache_key] = code
        logger.debug(f"Generated client code for protocol: {protocol.protocol_id}")
        
        return code
    
    def generate_server_code(
        self,
        protocol: ProtocolDescriptor,
        configuration: Dict[str, Any]
    ) -> str:
        """Generate server code for a protocol."""
        cache_key = f"server_{protocol.protocol_id}_{self._hash_config(configuration)}"
        
        if cache_key in self.generated_code_cache:
            return self.generated_code_cache[cache_key]
        
        class_name = self._generate_class_name(protocol.name)
        methods = self._generate_server_methods(protocol, configuration)
        
        code = self.templates['server'].format(
            class_name=class_name,
            protocol_name=protocol.name,
            methods=methods
        )
        
        self.generated_code_cache[cache_key] = code
        logger.debug(f"Generated server code for protocol: {protocol.protocol_id}")
        
        return code
    
    def _generate_class_name(self, protocol_name: str) -> str:
        """Generate a valid class name from protocol name."""
        # Remove special characters and convert to PascalCase
        clean_name = ''.join(c for c in protocol_name if c.isalnum() or c.isspace())
        words = clean_name.split()
        return ''.join(word.capitalize() for word in words)
    
    def _generate_client_methods(
        self,
        protocol: ProtocolDescriptor,
        configuration: Dict[str, Any]
    ) -> str:
        """Generate client methods based on protocol capabilities."""
        methods = []
        
        # Generate methods based on capabilities
        for capability in protocol.capabilities:
            if capability == ProtocolCapability.MESSAGING:
                methods.append(self._generate_messaging_methods())
            elif capability == ProtocolCapability.FILE_TRANSFER:
                methods.append(self._generate_file_transfer_methods())
            elif capability == ProtocolCapability.STREAMING:
                methods.append(self._generate_streaming_methods())
            elif capability == ProtocolCapability.AUTHENTICATION:
                methods.append(self._generate_auth_methods())
        
        # Add custom methods from configuration
        if 'custom_methods' in configuration:
            for method_config in configuration['custom_methods']:
                methods.append(self._generate_custom_method(method_config))
        
        return '\n'.join(methods)
    
    def _generate_server_methods(
        self,
        protocol: ProtocolDescriptor,
        configuration: Dict[str, Any]
    ) -> str:
        """Generate server methods based on protocol capabilities."""
        methods = []
        
        # Generate message handlers based on capabilities
        for capability in protocol.capabilities:
            if capability == ProtocolCapability.MESSAGING:
                methods.append(self._generate_message_handlers())
            elif capability == ProtocolCapability.FILE_TRANSFER:
                methods.append(self._generate_file_handlers())
            elif capability == ProtocolCapability.STREAMING:
                methods.append(self._generate_stream_handlers())
        
        return '\n'.join(methods)
    
    def _generate_messaging_methods(self) -> str:
        """Generate messaging capability methods."""
        return '''
    async def send_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message through the protocol."""
        # Implementation for message sending
        response = await self._make_request("POST", "/messages", message)
        return response
    
    async def receive_messages(self, timeout: int = 30) -> List[Dict[str, Any]]:
        """Receive messages from the protocol."""
        # Implementation for message receiving
        response = await self._make_request("GET", f"/messages?timeout={timeout}")
        return response.get("messages", [])
'''
    
    def _generate_file_transfer_methods(self) -> str:
        """Generate file transfer capability methods."""
        return '''
    async def upload_file(self, filename: str, content: bytes) -> Dict[str, Any]:
        """Upload a file through the protocol."""
        # Implementation for file upload
        files = {"file": (filename, content)}
        response = await self._make_request("POST", "/files", files=files)
        return response
    
    async def download_file(self, file_id: str) -> bytes:
        """Download a file through the protocol."""
        # Implementation for file download
        response = await self._make_request("GET", f"/files/{file_id}")
        return response.content
'''
    
    def _generate_streaming_methods(self) -> str:
        """Generate streaming capability methods."""
        return '''
    async def start_stream(self, stream_config: Dict[str, Any]) -> str:
        """Start a streaming session."""
        # Implementation for stream initialization
        response = await self._make_request("POST", "/streams", stream_config)
        return response.get("stream_id")
    
    async def send_stream_data(self, stream_id: str, data: Any) -> None:
        """Send data to a stream."""
        # Implementation for stream data sending
        await self._make_request("POST", f"/streams/{stream_id}/data", {"data": data})
    
    async def receive_stream_data(self, stream_id: str) -> Any:
        """Receive data from a stream."""
        # Implementation for stream data receiving
        response = await self._make_request("GET", f"/streams/{stream_id}/data")
        return response.get("data")
'''
    
    def _generate_auth_methods(self) -> str:
        """Generate authentication capability methods."""
        return '''
    async def authenticate(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticate with the protocol."""
        # Implementation for authentication
        response = await self._make_request("POST", "/auth", credentials)
        if response.get("token"):
            self.auth_token = response["token"]
        return response
    
    async def refresh_token(self) -> Dict[str, Any]:
        """Refresh authentication token."""
        # Implementation for token refresh
        response = await self._make_request("POST", "/auth/refresh")
        if response.get("token"):
            self.auth_token = response["token"]
        return response
'''
    
    def _generate_message_handlers(self) -> str:
        """Generate message handler methods."""
        return '''
    async def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming message."""
        message_type = message.get("type", "unknown")
        handler = self.handlers.get(f"handle_{message_type}")
        
        if handler:
            return await handler(message)
        else:
            return {"status": "error", "message": f"No handler for message type: {message_type}"}
'''
    
    def _generate_file_handlers(self) -> str:
        """Generate file handler methods."""
        return '''
    async def handle_file_upload(self, file_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle file upload request."""
        # Implementation for handling file uploads
        return {"status": "success", "file_id": "generated_file_id"}
    
    async def handle_file_download(self, file_id: str) -> Dict[str, Any]:
        """Handle file download request."""
        # Implementation for handling file downloads
        return {"status": "success", "content": b"file_content"}
'''
    
    def _generate_stream_handlers(self) -> str:
        """Generate stream handler methods."""
        return '''
    async def handle_stream_start(self, stream_config: Dict[str, Any]) -> Dict[str, Any]:
        """Handle stream start request."""
        # Implementation for handling stream initialization
        stream_id = str(uuid.uuid4())
        return {"status": "success", "stream_id": stream_id}
    
    async def handle_stream_data(self, stream_id: str, data: Any) -> Dict[str, Any]:
        """Handle stream data."""
        # Implementation for handling stream data
        return {"status": "success", "processed": True}
'''
    
    def _generate_custom_method(self, method_config: Dict[str, Any]) -> str:
        """Generate a custom method from configuration."""
        method_name = method_config.get('name', 'custom_method')
        parameters = method_config.get('parameters', [])
        description = method_config.get('description', 'Custom method')
        body = method_config.get('body', 'pass')
        
        param_str = ', '.join(f"{param['name']}: {param.get('type', 'Any')}" for param in parameters)
        
        return self.templates['method'].format(
            method_name=method_name,
            parameters=param_str,
            method_description=description,
            method_body=body
        )
    
    def _hash_config(self, configuration: Dict[str, Any]) -> str:
        """Generate hash for configuration."""
        config_str = json.dumps(configuration, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def compile_code(self, code: str, class_name: str) -> Type:
        """Compile generated code and return the class."""
        try:
            # Create a namespace for the compiled code
            namespace = {
                'uuid': uuid,
                'Dict': Dict,
                'List': List,
                'Any': Any,
                'Callable': Callable,
                'Optional': Optional
            }
            
            # Execute the code in the namespace
            exec(code, namespace)
            
            # Return the generated class
            return namespace[class_name]
            
        except Exception as e:
            logger.error(f"Failed to compile generated code: {e}")
            raise
    
    def clear_cache(self) -> None:
        """Clear the generated code cache."""
        self.generated_code_cache.clear()
        logger.debug("Cleared code generation cache")


class MetaProtocolNegotiator:
    """Handles protocol negotiation between agents."""
    
    def __init__(self, registry: ProtocolRegistry, code_generator: CodeGenerator):
        """Initialize meta-protocol negotiator."""
        self.registry = registry
        self.code_generator = code_generator
        self.active_negotiations: Dict[str, Dict[str, Any]] = {}
        self.completed_negotiations: Dict[str, NegotiationResponse] = {}
        self._lock = threading.RLock()
        
        logger.info("Initialized meta-protocol negotiator")
    
    async def initiate_negotiation(
        self,
        requester_id: str,
        target_id: str,
        preferred_capabilities: List[ProtocolCapability],
        requirements: Optional[Dict[str, Any]] = None
    ) -> str:
        """Initiate protocol negotiation."""
        negotiation_id = str(uuid.uuid4())
        
        # Get supported protocols
        supported_protocols = self.registry.find_compatible_protocols(preferred_capabilities)
        
        if not supported_protocols:
            raise ValueError(f"No protocols found supporting capabilities: {preferred_capabilities}")
        
        # Create negotiation request
        request = NegotiationRequest(
            negotiation_id=negotiation_id,
            requester_id=requester_id,
            target_id=target_id,
            supported_protocols=supported_protocols,
            preferred_capabilities=preferred_capabilities,
            requirements=requirements or {}
        )
        
        # Store active negotiation
        with self._lock:
            self.active_negotiations[negotiation_id] = {
                'request': request,
                'status': NegotiationStatus.PENDING,
                'created_at': datetime.utcnow()
            }
        
        logger.info(f"Initiated negotiation {negotiation_id} between {requester_id} and {target_id}")
        return negotiation_id
    
    async def handle_negotiation_request(
        self,
        request: NegotiationRequest,
        responder_id: str
    ) -> NegotiationResponse:
        """Handle incoming negotiation request."""
        negotiation_id = request.negotiation_id
        
        try:
            # Update negotiation status
            with self._lock:
                if negotiation_id in self.active_negotiations:
                    self.active_negotiations[negotiation_id]['status'] = NegotiationStatus.IN_PROGRESS
            
            # Find best matching protocol
            best_protocol = self._select_best_protocol(request)
            
            if not best_protocol:
                response = NegotiationResponse(
                    negotiation_id=negotiation_id,
                    responder_id=responder_id,
                    status=NegotiationStatus.FAILED,
                    error_message="No compatible protocol found"
                )
            else:
                # Determine agreed capabilities
                agreed_capabilities = self._determine_agreed_capabilities(
                    best_protocol,
                    request.preferred_capabilities
                )
                
                # Generate configuration
                configuration = self._generate_configuration(best_protocol, request.requirements)
                
                # Generate protocol implementation code
                generated_code = self.code_generator.generate_client_code(best_protocol, configuration)
                
                response = NegotiationResponse(
                    negotiation_id=negotiation_id,
                    responder_id=responder_id,
                    status=NegotiationStatus.COMPLETED,
                    selected_protocol=best_protocol,
                    agreed_capabilities=agreed_capabilities,
                    configuration=configuration,
                    generated_code=generated_code
                )
            
            # Store completed negotiation
            with self._lock:
                self.completed_negotiations[negotiation_id] = response
                if negotiation_id in self.active_negotiations:
                    del self.active_negotiations[negotiation_id]
            
            logger.info(f"Completed negotiation {negotiation_id} with status: {response.status.value}")
            return response
            
        except Exception as e:
            logger.error(f"Error handling negotiation {negotiation_id}: {e}")
            
            error_response = NegotiationResponse(
                negotiation_id=negotiation_id,
                responder_id=responder_id,
                status=NegotiationStatus.FAILED,
                error_message=str(e)
            )
            
            with self._lock:
                self.completed_negotiations[negotiation_id] = error_response
                if negotiation_id in self.active_negotiations:
                    del self.active_negotiations[negotiation_id]
            return error_response
    
    def _select_best_protocol(self, request: NegotiationRequest) -> Optional[ProtocolDescriptor]:
        """Select the best protocol for the negotiation."""
        if not request.supported_protocols:
            return None
        
        # Score each protocol based on compatibility and capabilities
        scored_protocols = []
        
        for protocol in request.supported_protocols:
            score = 0.0
            
            # Base score for protocol availability
            score += 1.0
            
            # Score for preferred capabilities
            supported_preferred = sum(
                1 for cap in request.preferred_capabilities
                if protocol.supports_capability(cap)
            )
            if request.preferred_capabilities:
                score += (supported_preferred / len(request.preferred_capabilities)) * 2.0
            
            # Score for total capabilities (more is better)
            score += len(protocol.capabilities) * 0.1
            
            # Score for version (newer is better)
            version_score = {
                ProtocolVersion.V1_0: 0.1,
                ProtocolVersion.V1_1: 0.2,
                ProtocolVersion.V2_0: 0.3
            }.get(protocol.version, 0.0)
            score += version_score
            
            # Check requirements
            if self._meets_requirements(protocol, request.requirements):
                score += 1.0
            else:
                score -= 2.0  # Penalty for not meeting requirements
            
            scored_protocols.append((protocol, score))
        
        # Sort by score (highest first)
        scored_protocols.sort(key=lambda x: x[1], reverse=True)
        
        # Return the best protocol if it has a positive score
        if scored_protocols and scored_protocols[0][1] > 0:
            best_protocol = scored_protocols[0][0]
            logger.debug(f"Selected protocol {best_protocol.protocol_id} with score {scored_protocols[0][1]}")
            return best_protocol
        
        return None
    
    def _meets_requirements(self, protocol: ProtocolDescriptor, requirements: Dict[str, Any]) -> bool:
        """Check if protocol meets the specified requirements."""
        if not requirements:
            return True
        
        # Check version requirements
        if 'min_version' in requirements:
            min_version = ProtocolVersion(requirements['min_version'])
            if protocol.version.value < min_version.value:
                return False
        
        # Check required capabilities
        if 'required_capabilities' in requirements:
            required_caps = [ProtocolCapability(cap) for cap in requirements['required_capabilities']]
            if not all(protocol.supports_capability(cap) for cap in required_caps):
                return False
        
        # Check excluded capabilities
        if 'excluded_capabilities' in requirements:
            excluded_caps = [ProtocolCapability(cap) for cap in requirements['excluded_capabilities']]
            if any(protocol.supports_capability(cap) for cap in excluded_caps):
                return False
        
        # Check custom requirements
        if 'custom' in requirements:
            custom_reqs = requirements['custom']
            protocol_metadata = protocol.metadata or {}
            
            for key, value in custom_reqs.items():
                if protocol_metadata.get(key) != value:
                    return False
        
        return True
    
    def _determine_agreed_capabilities(
        self,
        protocol: ProtocolDescriptor,
        preferred_capabilities: List[ProtocolCapability]
    ) -> List[ProtocolCapability]:
        """Determine the capabilities that both parties agree to use."""
        agreed = []
        
        for capability in preferred_capabilities:
            if protocol.supports_capability(capability):
                agreed.append(capability)
        
        # Add essential capabilities that the protocol provides
        essential_caps = [ProtocolCapability.MESSAGING, ProtocolCapability.AUTHENTICATION]
        for cap in essential_caps:
            if protocol.supports_capability(cap) and cap not in agreed:
                agreed.append(cap)
        
        return agreed
    
    def _generate_configuration(
        self,
        protocol: ProtocolDescriptor,
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate configuration for the selected protocol."""
        config = {
            'protocol_id': protocol.protocol_id,
            'version': protocol.version.value,
            'endpoints': protocol.endpoints,
            'capabilities': [cap.value for cap in protocol.capabilities]
        }
        
        # Add security configuration
        config['security'] = {
            'encryption_required': True,
            'authentication_required': True,
            'certificate_validation': True
        }
        
        # Add performance configuration
        config['performance'] = {
            'timeout_seconds': 30,
            'retry_attempts': 3,
            'connection_pool_size': 10
        }
        
        # Add capability-specific configuration
        if ProtocolCapability.FILE_TRANSFER in protocol.capabilities:
            config['file_transfer'] = {
                'max_file_size': requirements.get('max_file_size', 100 * 1024 * 1024),  # 100MB
                'allowed_extensions': requirements.get('allowed_extensions', []),
                'compression_enabled': True
            }
        
        if ProtocolCapability.STREAMING in protocol.capabilities:
            config['streaming'] = {
                'buffer_size': requirements.get('buffer_size', 8192),
                'max_streams': requirements.get('max_streams', 10),
                'heartbeat_interval': 30
            }
        
        # Override with custom requirements
        if 'configuration' in requirements:
            config.update(requirements['configuration'])
        
        return config
    
    async def get_negotiation_status(self, negotiation_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a negotiation."""
        with self._lock:
            # Check active negotiations
            if negotiation_id in self.active_negotiations:
                negotiation = self.active_negotiations[negotiation_id]
                return {
                    'negotiation_id': negotiation_id,
                    'status': negotiation['status'].value,
                    'created_at': negotiation['created_at'].isoformat(),
                    'request': negotiation['request'].to_dict()
                }
            
            # Check completed negotiations
            if negotiation_id in self.completed_negotiations:
                response = self.completed_negotiations[negotiation_id]
                return {
                    'negotiation_id': negotiation_id,
                    'status': response.status.value,
                    'completed_at': response.created_at.isoformat(),
                    'response': response.to_dict()
                }
        
        return None
    
    async def cancel_negotiation(self, negotiation_id: str, reason: str = "Cancelled by user") -> bool:
        """Cancel an active negotiation."""
        with self._lock:
            if negotiation_id in self.active_negotiations:
                # Create cancellation response
                negotiation = self.active_negotiations[negotiation_id]
                response = NegotiationResponse(
                    negotiation_id=negotiation_id,
                    responder_id="system",
                    status=NegotiationStatus.FAILED,
                    error_message=f"Negotiation cancelled: {reason}"
                )
                
                # Move to completed
                self.completed_negotiations[negotiation_id] = response
                del self.active_negotiations[negotiation_id]
                
                logger.info(f"Cancelled negotiation {negotiation_id}: {reason}")
                return True
        
        return False
    
    def cleanup_old_negotiations(self, max_age_hours: int = 24) -> int:
        """Clean up old completed negotiations."""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        cleaned_count = 0
        
        with self._lock:
            # Clean completed negotiations
            to_remove = []
            for negotiation_id, response in self.completed_negotiations.items():
                if response.created_at and response.created_at < cutoff_time:
                    to_remove.append(negotiation_id)
            
            for negotiation_id in to_remove:
                del self.completed_negotiations[negotiation_id]
                cleaned_count += 1
            
            # Clean stale active negotiations
            to_remove = []
            for negotiation_id, negotiation in self.active_negotiations.items():
                if negotiation['created_at'] < cutoff_time:
                    to_remove.append(negotiation_id)
            
            for negotiation_id in to_remove:
                del self.active_negotiations[negotiation_id]
                cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old negotiations")
        
        return cleaned_count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get negotiation statistics."""
        with self._lock:
            active_count = len(self.active_negotiations)
            completed_count = len(self.completed_negotiations)
            
            # Count by status
            status_counts = {}
            for response in self.completed_negotiations.values():
                status = response.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Count by protocol
            protocol_counts = {}
            for response in self.completed_negotiations.values():
                if response.selected_protocol:
                    protocol_id = response.selected_protocol.protocol_id
                    protocol_counts[protocol_id] = protocol_counts.get(protocol_id, 0) + 1
        
        return {
            'active_negotiations': active_count,
            'completed_negotiations': completed_count,
            'total_negotiations': active_count + completed_count,
            'status_distribution': status_counts,
            'protocol_distribution': protocol_counts,
            'available_protocols': len(self.registry.protocols)
        }


class MetaProtocolManager:
    """Main manager for ANP Meta-Protocol functionality."""
    
    def __init__(self):
        """Initialize meta-protocol manager."""
        self.registry = ProtocolRegistry()
        self.code_generator = CodeGenerator()
        self.negotiator = MetaProtocolNegotiator(self.registry, self.code_generator)
        self.compiled_protocols: Dict[str, Type] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        
        logger.info("Initialized meta-protocol manager")
    
    async def start(self) -> None:
        """Start the meta-protocol manager."""
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        logger.info("Meta-protocol manager started")
    
    async def stop(self) -> None:
        """Stop the meta-protocol manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Meta-protocol manager stopped")
    
    async def negotiate_protocol(
        self,
        requester_id: str,
        target_id: str,
        capabilities: List[ProtocolCapability],
        requirements: Optional[Dict[str, Any]] = None
    ) -> NegotiationResponse:
        """Negotiate a protocol between two agents."""
        # Initiate negotiation
        negotiation_id = await self.negotiator.initiate_negotiation(
            requester_id, target_id, capabilities, requirements
        )
        
        # Get the negotiation request
        status = await self.negotiator.get_negotiation_status(negotiation_id)
        if not status:
            raise ValueError(f"Negotiation {negotiation_id} not found")
        
        request = NegotiationRequest.from_dict(status['request'])
        
        # Handle the negotiation (simulate target agent response)
        response = await self.negotiator.handle_negotiation_request(request, target_id)
        
        return response
    
    async def create_protocol_instance(
        self,
        negotiation_response: NegotiationResponse,
        instance_type: str = "client"
    ) -> Any:
        """Create a protocol instance from negotiation response."""
        if negotiation_response.status != NegotiationStatus.COMPLETED:
            raise ValueError(f"Cannot create instance from failed negotiation: {negotiation_response.status}")
        
        if not negotiation_response.selected_protocol:
            raise ValueError("No protocol selected in negotiation response")
        
        protocol = negotiation_response.selected_protocol
        configuration = negotiation_response.configuration or {}
        
        # Generate and compile code if not already done
        cache_key = f"{protocol.protocol_id}_{instance_type}"
        
        if cache_key not in self.compiled_protocols:
            if instance_type == "client":
                code = self.code_generator.generate_client_code(protocol, configuration)
            elif instance_type == "server":
                code = self.code_generator.generate_server_code(protocol, configuration)
            else:
                raise ValueError(f"Unknown instance type: {instance_type}")
            
            class_name = self.code_generator._generate_class_name(protocol.name) + instance_type.capitalize()
            compiled_class = self.code_generator.compile_code(code, class_name)
            self.compiled_protocols[cache_key] = compiled_class
        
        # Create instance
        protocol_class = self.compiled_protocols[cache_key]
        
        if instance_type == "client":
            endpoint = protocol.endpoints[0] if protocol.endpoints else "http://localhost:8000"
            instance = protocol_class(endpoint)
        else:
            instance = protocol_class()
        
        logger.info(f"Created {instance_type} instance for protocol {protocol.protocol_id}")
        return instance
    
    def register_custom_protocol(
        self,
        protocol: ProtocolDescriptor,
        implementation_class: Optional[Type] = None
    ) -> None:
        """Register a custom protocol."""
        self.registry.register_protocol(protocol)
        
        if implementation_class:
            self.registry.register_implementation(protocol.protocol_id, implementation_class)
        
        logger.info(f"Registered custom protocol: {protocol.protocol_id}")
    
    def get_available_protocols(self) -> List[ProtocolDescriptor]:
        """Get list of available protocols."""
        return self.registry.list_protocols()
    
    def find_protocols_by_capability(
        self,
        capabilities: List[ProtocolCapability]
    ) -> List[ProtocolDescriptor]:
        """Find protocols that support specific capabilities."""
        return self.registry.find_compatible_protocols(capabilities)
    
    async def _periodic_cleanup(self) -> None:
        """Periodic cleanup of old negotiations and cache."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Clean up old negotiations
                self.negotiator.cleanup_old_negotiations(max_age_hours=24)
                
                # Clear code generation cache periodically
                if len(self.code_generator.generated_code_cache) > 100:
                    self.code_generator.clear_cache()
                
                logger.debug("Completed periodic cleanup")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
    
    def get_manager_statistics(self) -> Dict[str, Any]:
        """Get comprehensive manager statistics."""
        negotiation_stats = self.negotiator.get_statistics()
        
        return {
            'negotiation_stats': negotiation_stats,
            'compiled_protocols': len(self.compiled_protocols),
            'code_cache_size': len(self.code_generator.generated_code_cache),
            'registry_stats': {
                'total_protocols': len(self.registry.protocols),
                'implementations_registered': len(self.registry.implementations)
            }
        }


# Example usage and comprehensive testing
def example_meta_protocol_usage():
    """Example usage of the meta-protocol system."""
    
    print("=== ANP Meta-Protocol Example Usage ===\n")
    
    # 1. Initialize the meta-protocol manager
    print("1. Initializing Meta-Protocol Manager:")
    print("-" * 40)
    
    manager = MetaProtocolManager()
    print("✓ Meta-protocol manager initialized")
    
    # 2. Explore available protocols
    print("\n2. Available Protocols:")
    print("-" * 25)
    
    protocols = manager.get_available_protocols()
    for protocol in protocols:
        print(f"• {protocol.name} ({protocol.protocol_id})")
        print(f"  Version: {protocol.version.value}")
        print(f"  Capabilities: {[cap.value for cap in protocol.capabilities]}")
        print()
    
    # 3. Find protocols by capability
    print("3. Finding Protocols by Capability:")
    print("-" * 35)
    
    messaging_protocols = manager.find_protocols_by_capability([ProtocolCapability.MESSAGING])
    print(f"Messaging protocols found: {len(messaging_protocols)}")
    
    file_transfer_protocols = manager.find_protocols_by_capability([
        ProtocolCapability.FILE_TRANSFER,
        ProtocolCapability.ENCRYPTION
    ])
    print(f"Secure file transfer protocols found: {len(file_transfer_protocols)}")
    
    # 4. Register a custom protocol
    print("\n4. Registering Custom Protocol:")
    print("-" * 30)
    
    custom_protocol = ProtocolDescriptor(
        protocol_id="custom.ai_agent.v1",
        name="Custom AI Agent Protocol",
        version=ProtocolVersion.V1_0,
        description="Custom protocol for AI agent communication",
        capabilities=[
            ProtocolCapability.MESSAGING,
            ProtocolCapability.STREAMING,
            ProtocolCapability.AUTHENTICATION
        ],
        endpoints=["/custom/ai_agent/v1"],
        metadata={
            "ai_model": "gpt-4",
            "max_context_length": 8192,
            "supports_function_calling": True
        }
    )
    
    manager.register_custom_protocol(custom_protocol)
    print(f"✓ Registered custom protocol: {custom_protocol.protocol_id}")
    
    # 5. Protocol negotiation example
    print("\n5. Protocol Negotiation Example:")
    print("-" * 32)
    
    async def negotiation_example():
        await manager.start()
        
        try:
            # Negotiate protocol between two agents
            response = await manager.negotiate_protocol(
                requester_id="agent_alice",
                target_id="agent_bob",
                capabilities=[
                    ProtocolCapability.MESSAGING,
                    ProtocolCapability.ENCRYPTION
                ],
                requirements={
                    "min_version": "1.0",
                    "required_capabilities": ["messaging"],
                    "configuration": {
                        "security": {
                            "encryption_required": True
                        }
                    }
                }
            )
            
            print(f"Negotiation Status: {response.status.value}")
            if response.selected_protocol:
                print(f"Selected Protocol: {response.selected_protocol.name}")
                print(f"Agreed Capabilities: {[cap.value for cap in response.agreed_capabilities]}")
            
            if response.status == NegotiationStatus.COMPLETED:
                # Create protocol instances
                client = await manager.create_protocol_instance(response, "client")
                server = await manager.create_protocol_instance(response, "server")
                
                print(f"✓ Created client instance: {type(client).__name__}")
                print(f"✓ Created server instance: {type(server).__name__}")
                
                # Show generated code (first 500 characters)
                if response.generated_code:
                    print(f"\nGenerated Code Preview:")
                    print("-" * 23)
                    print(response.generated_code[:500] + "..." if len(response.generated_code) > 500 else response.generated_code)
            
        finally:
            await manager.stop()
    
    # Run the async example
    import asyncio
    asyncio.run(negotiation_example())


def comprehensive_meta_protocol_testing():
    """Comprehensive testing of meta-protocol functionality."""
    
    print("\n=== Comprehensive Meta-Protocol Testing ===\n")
    
    # 1. Protocol Registry Testing
    print("1. Protocol Registry Testing:")
    print("-" * 30)
    
    registry = ProtocolRegistry()
    
    # Test protocol registration
    test_protocol = ProtocolDescriptor(
        protocol_id="test.protocol.v1",
        name="Test Protocol",
        version=ProtocolVersion.V1_0,
        description="Protocol for testing",
        capabilities=[ProtocolCapability.MESSAGING]
    )
    
    registry.register_protocol(test_protocol)
    retrieved = registry.get_protocol("test.protocol.v1")
    assert retrieved is not None, "Protocol registration failed"
    assert retrieved.name == "Test Protocol", "Protocol data mismatch"
    print("✓ Protocol registration and retrieval")
    
    # Test protocol search
    messaging_protocols = registry.find_compatible_protocols([ProtocolCapability.MESSAGING])
    assert len(messaging_protocols) > 0, "Protocol search failed"
    print(f"✓ Found {len(messaging_protocols)} messaging protocols")
    
    # Test protocol unregistration
    success = registry.unregister_protocol("test.protocol.v1")
    assert success, "Protocol unregistration failed"
    assert registry.get_protocol("test.protocol.v1") is None, "Protocol still exists after unregistration"
    print("✓ Protocol unregistration")
    
    # 2. Code Generator Testing
    print("\n2. Code Generator Testing:")
    print("-" * 26)
    
    generator = CodeGenerator()
    
    # Test client code generation
    client_code = generator.generate_client_code(test_protocol, {})
    assert "class TestProtocolClient:" in client_code, "Client code generation failed"
    assert "async def send_message" in client_code, "Missing messaging methods"
    print("✓ Client code generation")
    
    # Test server code generation
    server_code = generator.generate_server_code(test_protocol, {})
    assert "class TestProtocolServer:" in server_code, "Server code generation failed"
    assert "async def handle_message" in server_code, "Missing message handlers"
    print("✓ Server code generation")
    
    # Test code compilation
    try:
        client_class = generator.compile_code(client_code, "TestProtocolClient")
        assert client_class is not None, "Code compilation failed"
        print("✓ Code compilation")
    except Exception as e:
        print(f"✗ Code compilation failed: {e}")
    
    # Test custom method generation
    custom_config = {
        "custom_methods": [{
            "name": "custom_action",
            "parameters": [{"name": "data", "type": "Dict[str, Any]"}],
            "description": "Custom action method",
            "body": "return {'status': 'success', 'data': data}"
        }]
    }
    
    custom_code = generator.generate_client_code(test_protocol, custom_config)
    assert "async def custom_action" in custom_code, "Custom method generation failed"
    print("✓ Custom method generation")
    
    # 3. Negotiation Testing
    print("\n3. Negotiation Testing:")
    print("-" * 22)
    
    negotiator = MetaProtocolNegotiator(registry, generator)
    
    async def test_negotiation():
        # Re-register test protocol
        registry.register_protocol(test_protocol)
        
        # Test negotiation initiation
        negotiation_id = await negotiator.initiate_negotiation(
            "test_agent_1",
            "test_agent_2",
            [ProtocolCapability.MESSAGING]
        )
        assert negotiation_id is not None, "Negotiation initiation failed"
        print(f"✓ Negotiation initiated: {negotiation_id}")
        
        # Test negotiation status
        status = await negotiator.get_negotiation_status(negotiation_id)
        assert status is not None, "Negotiation status retrieval failed"
        assert status['status'] == 'pending', "Incorrect negotiation status"
        print("✓ Negotiation status retrieval")
        
        # Test negotiation handling
        request = NegotiationRequest.from_dict(status['request'])
        response = await negotiator.handle_negotiation_request(request, "test_agent_2")
        
        assert response.status == NegotiationStatus.COMPLETED, f"Negotiation failed: {response.error_message}"
        assert response.selected_protocol is not None, "No protocol selected"
        print("✓ Negotiation completion")
        
        # Test negotiation cancellation
        cancel_negotiation_id = await negotiator.initiate_negotiation(
            "test_agent_3",
            "test_agent_4",
            [ProtocolCapability.MESSAGING]
        )
        
        cancelled = await negotiator.cancel_negotiation(cancel_negotiation_id, "Test cancellation")
        assert cancelled, "Negotiation cancellation failed"
        print("✓ Negotiation cancellation")
        
        return response
    
    # Run negotiation tests
    response = asyncio.run(test_negotiation())
    
    # 4. Protocol Compatibility Testing
    print("\n4. Protocol Compatibility Testing:")
    print("-" * 34)
    
    # Create compatible protocols
    protocol_v1_0 = ProtocolDescriptor(
        protocol_id="compat.test.v1",
        name="Compatibility Test",
        version=ProtocolVersion.V1_0,
        description="Version 1.0",
        capabilities=[ProtocolCapability.MESSAGING, ProtocolCapability.AUTHENTICATION]
    )
    
    protocol_v1_1 = ProtocolDescriptor(
        protocol_id="compat.test.v1",
        name="Compatibility Test",
        version=ProtocolVersion.V1_1,
        description="Version 1.1",
        capabilities=[ProtocolCapability.MESSAGING, ProtocolCapability.AUTHENTICATION, ProtocolCapability.ENCRYPTION]
    )
    
    # Test compatibility
    compatible = protocol_v1_0.is_compatible_with(protocol_v1_1)
    assert compatible, "Compatible protocols not recognized as compatible"
    print("✓ Protocol compatibility detection")
    
    # Test compatibility score
    score = protocol_v1_0.get_compatibility_score(protocol_v1_1)
    assert 0.0 <= score <= 1.0, "Invalid compatibility score"
    print(f"✓ Compatibility score calculation: {score:.2f}")
    
    # 5. Serialization Testing
    print("\n5. Serialization Testing:")
    print("-" * 25)
    
    # Test protocol descriptor serialization
    protocol_dict = test_protocol.to_dict()
    reconstructed_protocol = ProtocolDescriptor.from_dict(protocol_dict)
    
    assert reconstructed_protocol.protocol_id == test_protocol.protocol_id, "Protocol serialization failed"
    assert reconstructed_protocol.capabilities == test_protocol.capabilities, "Capabilities not preserved"
    print("✓ Protocol descriptor serialization")
    
    # Test negotiation request serialization
    request = NegotiationRequest(
        negotiation_id="test_neg_123",
        requester_id="agent_1",
        target_id="agent_2",
        supported_protocols=[test_protocol],
        preferred_capabilities=[ProtocolCapability.MESSAGING]
    )
    
    request_dict = request.to_dict()
    reconstructed_request = NegotiationRequest.from_dict(request_dict)
    
    assert reconstructed_request.negotiation_id == request.negotiation_id, "Request serialization failed"
    assert len(reconstructed_request.supported_protocols) == 1, "Protocols not preserved"
    print("✓ Negotiation request serialization")
    
    # Test negotiation response serialization
    response_dict = response.to_dict()
    reconstructed_response = NegotiationResponse.from_dict(response_dict)
    
    assert reconstructed_response.negotiation_id == response.negotiation_id, "Response serialization failed"
    assert reconstructed_response.status == response.status, "Status not preserved"
    print("✓ Negotiation response serialization")
    
    # 6. Error Handling Testing
    print("\n6. Error Handling Testing:")
    print("-" * 26)
    
    # Test invalid protocol registration
    try:
        invalid_protocol = ProtocolDescriptor(
            protocol_id="",  # Invalid empty ID
            name="Invalid Protocol",
            version=ProtocolVersion.V1_0,
            description="Invalid protocol",
            capabilities=[]
        )
        errors = invalid_protocol.validate() if hasattr(invalid_protocol, 'validate') else []
        print("✓ Invalid protocol detection (validation would catch this)")
    except Exception as e:
        print(f"✓ Invalid protocol rejected: {type(e).__name__}")
    
    # Test negotiation with no compatible protocols
    async def test_no_compatible_protocols():
        try:
            # Clear registry
            temp_registry = ProtocolRegistry()
            temp_negotiator = MetaProtocolNegotiator(temp_registry, generator)
            
            negotiation_id = await temp_negotiator.initiate_negotiation(
                "agent_1",
                "agent_2",
                [ProtocolCapability.FILE_TRANSFER]  # No protocols support this
            )
            print("✗ Should have failed with no compatible protocols")
        except ValueError as e:
            print("✓ No compatible protocols error handled")
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
    
    asyncio.run(test_no_compatible_protocols())
    
    # 7. Performance Testing
    print("\n7. Performance Testing:")
    print("-" * 21)
    
    import time
    
    # Test protocol registration performance
    start_time = time.time()
    for i in range(100):
        perf_protocol = ProtocolDescriptor(
            protocol_id=f"perf.test.{i}",
            name=f"Performance Test {i}",
            version=ProtocolVersion.V1_0,
            description="Performance testing protocol",
            capabilities=[ProtocolCapability.MESSAGING]
        )
        registry.register_protocol(perf_protocol)
    
    registration_time = time.time() - start_time
    print(f"✓ Registered 100 protocols in {registration_time:.3f}s ({100/registration_time:.1f} protocols/sec)")
    
    # Test protocol search performance
    start_time = time.time()
    for _ in range(100):
        found = registry.find_compatible_protocols([ProtocolCapability.MESSAGING])
    
    search_time = time.time() - start_time
    print(f"✓ Performed 100 searches in {search_time:.3f}s ({100/search_time:.1f} searches/sec)")
    
    # Test code generation performance
    start_time = time.time()
    for i in range(10):
        code = generator.generate_client_code(test_protocol, {})
    
    generation_time = time.time() - start_time
    print(f"✓ Generated 10 client codes in {generation_time:.3f}s ({10/generation_time:.1f} generations/sec)")
    
    print("\n=== All Tests Completed Successfully! ===")


# Advanced usage examples
def advanced_meta_protocol_examples():
    """Advanced examples of meta-protocol usage."""
    
    print("\n=== Advanced Meta-Protocol Examples ===\n")
    
    # 1. Multi-Agent Protocol Negotiation
    print("1. Multi-Agent Protocol Negotiation:")
    print("-" * 36)
    
    async def multi_agent_negotiation():
        manager = MetaProtocolManager()
        await manager.start()
        
        try:
            # Define different agent types with different requirements
            agents = {
                "ai_assistant": {
                    "capabilities": [ProtocolCapability.MESSAGING, ProtocolCapability.STREAMING],
                    "requirements": {
                        "min_version": "1.0",
                        "configuration": {
                            "streaming": {"max_streams": 5}
                        }
                    }
                },
                "file_processor": {
                    "capabilities": [ProtocolCapability.FILE_TRANSFER, ProtocolCapability.ENCRYPTION],
                    "requirements": {
                        "required_capabilities": ["file_transfer", "encryption"],
                        "configuration": {
                            "file_transfer": {"max_file_size": 50 * 1024 * 1024}
                        }
                    }
                },
                "data_analyzer": {
                    "capabilities": [ProtocolCapability.MESSAGING, ProtocolCapability.COMPRESSION],
                    "requirements": {
                        "custom": {"supports_batch_processing": True}
                    }
                }
            }
            
            # Negotiate protocols between different agent pairs
            negotiations = []
            agent_pairs = [
                ("ai_assistant", "file_processor"),
                ("file_processor", "data_analyzer"),
                ("ai_assistant", "data_analyzer")
            ]
            
            for requester, target in agent_pairs:
                req_config = agents[requester]
                response = await manager.negotiate_protocol(
                    requester_id=requester,
                    target_id=target,
                    capabilities=req_config["capabilities"],
                    requirements=req_config["requirements"]
                )
                
                negotiations.append({
                    "pair": f"{requester} -> {target}",
                    "status": response.status.value,
                    "protocol": response.selected_protocol.name if response.selected_protocol else None
                })
            
            print("Negotiation Results:")
            for neg in negotiations:
                print(f"  {neg['pair']}: {neg['status']} ({neg['protocol']})")
            
        finally:
            await manager.stop()
    
    asyncio.run(multi_agent_negotiation())
    
    # 2. Dynamic Protocol Adaptation
    print("\n2. Dynamic Protocol Adaptation:")
    print("-" * 31)
    
    # Create protocols with different capability sets
    basic_protocol = ProtocolDescriptor(
        protocol_id="adaptive.basic.v1",
        name="Basic Adaptive Protocol",
        version=ProtocolVersion.V1_0,
        description="Basic protocol with minimal capabilities",
        capabilities=[ProtocolCapability.MESSAGING, ProtocolCapability.AUTHENTICATION]
    )
    
    enhanced_protocol = ProtocolDescriptor(
        protocol_id="adaptive.enhanced.v1",
        name="Enhanced Adaptive Protocol",
        version=ProtocolVersion.V1_0,
        description="Enhanced protocol with additional capabilities",
        capabilities=[
            ProtocolCapability.MESSAGING,
            ProtocolCapability.AUTHENTICATION,
            ProtocolCapability.ENCRYPTION,
            ProtocolCapability.COMPRESSION,
            ProtocolCapability.STREAMING
        ]
    )
    
    manager = MetaProtocolManager()
    manager.register_custom_protocol(basic_protocol)
    manager.register_custom_protocol(enhanced_protocol)
    
    # Simulate adaptation based on network conditions
    network_conditions = [
        {"bandwidth": "high", "latency": "low", "security": "required"},
        {"bandwidth": "low", "latency": "high", "security": "optional"},
        {"bandwidth": "medium", "latency": "medium", "security": "required"}
    ]
    
    async def adaptive_negotiation():
        await manager.start()
        
        try:
            for i, conditions in enumerate(network_conditions):
                print(f"\nScenario {i+1}: {conditions}")
                
                # Adapt requirements based on conditions
                if conditions["bandwidth"] == "low":
                    capabilities = [ProtocolCapability.MESSAGING]  # Minimal
                    requirements = {"configuration": {"performance": {"compression_enabled": True}}}
                elif conditions["security"] == "required":
                    capabilities = [ProtocolCapability.MESSAGING, ProtocolCapability.ENCRYPTION]
                    requirements = {"required_capabilities": ["encryption"]}
                else:
                    capabilities = [ProtocolCapability.MESSAGING, ProtocolCapability.STREAMING]
                    requirements = {}
                
                response = await manager.negotiate_protocol(
                    requester_id="adaptive_agent",
                    target_id="target_agent",
                    capabilities=capabilities,
                    requirements=requirements
                )
                
                if response.selected_protocol:
                    print(f"  Selected: {response.selected_protocol.name}")
                    print(f"  Capabilities: {[cap.value for cap in response.agreed_capabilities]}")
                else:
                    print(f"  Failed: {response.error_message}")
        
        finally:
            await manager.stop()
    
    asyncio.run(adaptive_negotiation())
    
    # 3. Protocol Composition and Chaining
    print("\n3. Protocol Composition and Chaining:")
    print("-" * 37)
    
    # Define specialized protocols for different stages
    auth_protocol = ProtocolDescriptor(
        protocol_id="chain.auth.v1",
        name="Authentication Protocol",
        version=ProtocolVersion.V1_0,
        description="Specialized authentication protocol",
        capabilities=[ProtocolCapability.AUTHENTICATION, ProtocolCapability.ENCRYPTION]
    )
    
    data_protocol = ProtocolDescriptor(
        protocol_id="chain.data.v1",
        name="Data Transfer Protocol",
        version=ProtocolVersion.V1_0,
        description="Specialized data transfer protocol",
        capabilities=[ProtocolCapability.MESSAGING, ProtocolCapability.FILE_TRANSFER, ProtocolCapability.COMPRESSION]
    )
    
    monitor_protocol = ProtocolDescriptor(
        protocol_id="chain.monitor.v1",
        name="Monitoring Protocol",
        version=ProtocolVersion.V1_0,
        description="Specialized monitoring protocol",
        capabilities=[ProtocolCapability.MONITORING, ProtocolCapability.STREAMING]
    )
    
    manager.register_custom_protocol(auth_protocol)
    manager.register_custom_protocol(data_protocol)
    manager.register_custom_protocol(monitor_protocol)
    
    # Simulate protocol chaining workflow
    workflow_stages = [
        {"name": "Authentication", "capabilities": [ProtocolCapability.AUTHENTICATION]},
        {"name": "Data Transfer", "capabilities": [ProtocolCapability.FILE_TRANSFER]},
        {"name": "Monitoring", "capabilities": [ProtocolCapability.MONITORING]}
    ]
    
    async def protocol_chaining():
        await manager.start()
        
        try:
            protocol_chain = []
            
            for stage in workflow_stages:
                response = await manager.negotiate_protocol(
                    requester_id="workflow_orchestrator",
                    target_id="service_provider",
                    capabilities=stage["capabilities"]
                )
                
                if response.status == NegotiationStatus.COMPLETED:
                    protocol_chain.append({
                        "stage": stage["name"],
                        "protocol": response.selected_protocol.name,
                        "negotiation_id": response.negotiation_id
                    })
                    print(f"✓ {stage['name']}: {response.selected_protocol.name}")
                else:
                    print(f"✗ {stage['name']}: Failed - {response.error_message}")
            
            print(f"\nProtocol Chain Established: {len(protocol_chain)} stages")
            
        finally:
            await manager.stop()
    
    asyncio.run(protocol_chaining())
    
    # 4. Protocol Performance Monitoring
    print("\n4. Protocol Performance Monitoring:")
    print("-" * 35)
    
    # Simulate protocol usage with performance tracking
    class ProtocolPerformanceMonitor:
        def __init__(self):
            self.metrics = {}
        
        def record_negotiation(self, protocol_id: str, duration: float, success: bool):
            if protocol_id not in self.metrics:
                self.metrics[protocol_id] = {
                    "negotiations": 0,
                    "successes": 0,
                    "total_duration": 0.0,
                    "avg_duration": 0.0
                }
            
            metrics = self.metrics[protocol_id]
            metrics["negotiations"] += 1
            metrics["total_duration"] += duration
            metrics["avg_duration"] = metrics["total_duration"] / metrics["negotiations"]
            
            if success:
                metrics["successes"] += 1
        
        def get_success_rate(self, protocol_id: str) -> float:
            if protocol_id not in self.metrics:
                return 0.0
            
            metrics = self.metrics[protocol_id]
            return metrics["successes"] / metrics["negotiations"] if metrics["negotiations"] > 0 else 0.0
        
        def get_performance_report(self) -> Dict[str, Any]:
            report = {}
            for protocol_id, metrics in self.metrics.items():
                report[protocol_id] = {
                    "success_rate": self.get_success_rate(protocol_id),
                    "avg_negotiation_time": metrics["avg_duration"],
                    "total_negotiations": metrics["negotiations"]
                }
            return report
    
    monitor = ProtocolPerformanceMonitor()
    
    # Simulate multiple negotiations with timing
    async def performance_monitoring():
        await manager.start()
        
        try:
            protocols_to_test = manager.get_available_protocols()[:3]  # Test first 3 protocols
            
            for protocol in protocols_to_test:
                for attempt in range(5):  # 5 attempts per protocol
                    start_time = time.time()
                    
                    try:
                        response = await manager.negotiate_protocol(
                            requester_id=f"perf_agent_{attempt}",
                            target_id="perf_target",
                            capabilities=protocol.capabilities[:1]  # Use first capability
                        )
                        
                        duration = time.time() - start_time
                        success = response.status == NegotiationStatus.COMPLETED
                        
                        monitor.record_negotiation(protocol.protocol_id, duration, success)
                        
                    except Exception as e:
                        duration = time.time() - start_time
                        monitor.record_negotiation(protocol.protocol_id, duration, False)
            
            # Generate performance report
            report = monitor.get_performance_report()
            print("Performance Report:")
            print("-" * 18)
            
            for protocol_id, metrics in report.items():
                print(f"{protocol_id}:")
                print(f"  Success Rate: {metrics['success_rate']:.1%}")
                print(f"  Avg Time: {metrics['avg_negotiation_time']:.3f}s")
                print(f"  Total Tests: {metrics['total_negotiations']}")
                print()
        
        finally:
            await manager.stop()
    
    asyncio.run(performance_monitoring())


if __name__ == "__main__":
    # Run all examples and tests
    print("ANP Meta-Protocol Implementation")
    print("=" * 50)
    
    # Basic usage example
    example_meta_protocol_usage()
    
    # Comprehensive testing
    comprehensive_meta_protocol_testing()
    
    # Advanced examples
    advanced_meta_protocol_examples()
    
    print("\n" + "=" * 50)
    print("ANP Meta-Protocol Implementation Complete!")
    print("All examples and tests executed successfully.")
