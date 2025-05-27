"""
ANP Client implementation.

This module provides the client-side implementation for the Agent Network Protocol (ANP),
handling outbound communication, DID-based authentication, protocol negotiation,
and secure message exchange with other ANP agents.
"""

import json
import uuid
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, urlparse
import logging
from enum import Enum

from .did import DIDManager, DIDDocument, DIDKeyPair
from .meta_protocol import MetaProtocolNegotiator, ProtocolSpec, NegotiationResult
from .encryption import ANPEncryption, EncryptedMessage, KeyExchange
from .discovery import DiscoveryService, DiscoverableAgent, AgentStatus

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection state enumeration."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    AUTHENTICATING = "authenticating"
    NEGOTIATING = "negotiating"
    CONNECTED = "connected"
    ERROR = "error"


class MessageType(Enum):
    """ANP message types."""
    HANDSHAKE = "handshake"
    AUTH_REQUEST = "auth_request"
    AUTH_RESPONSE = "auth_response"
    PROTOCOL_NEGOTIATION = "protocol_negotiation"
    PROTOCOL_RESPONSE = "protocol_response"
    DATA = "data"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    DISCONNECT = "disconnect"


@dataclass
class ANPMessage:
    """ANP protocol message structure."""
    
    message_id: str
    message_type: MessageType
    sender_did: str
    recipient_did: str
    timestamp: datetime
    payload: Dict[str, Any]
    signature: Optional[str] = None
    encryption_info: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "sender_did": self.sender_did,
            "recipient_did": self.recipient_did,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
            "signature": self.signature,
            "encryption_info": self.encryption_info
        }
    
    def to_json(self) -> str:
        """Convert message to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ANPMessage':
        """Create message from dictionary."""
        return cls(
            message_id=data["message_id"],
            message_type=MessageType(data["message_type"]),
            sender_did=data["sender_did"],
            recipient_did=data["recipient_did"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            payload=data["payload"],
            signature=data.get("signature"),
            encryption_info=data.get("encryption_info")
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ANPMessage':
        """Create message from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class ConnectionInfo:
    """Information about an ANP connection."""
    
    connection_id: str
    remote_agent_id: str
    remote_did: str
    remote_endpoint: str
    state: ConnectionState
    established_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    negotiated_protocols: Optional[List[ProtocolSpec]] = None
    encryption_session: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()
    
    def is_active(self, timeout_seconds: int = 300) -> bool:
        """Check if connection is active within timeout."""
        if not self.last_activity:
            return False
        
        timeout = timedelta(seconds=timeout_seconds)
        return (datetime.utcnow() - self.last_activity) < timeout


class ANPClientError(Exception):
    """Base exception for ANP client errors."""
    pass


class ConnectionError(ANPClientError):
    """Raised when connection operations fail."""
    pass


class AuthenticationError(ANPClientError):
    """Raised when authentication fails."""
    pass


class ProtocolNegotiationError(ANPClientError):
    """Raised when protocol negotiation fails."""
    pass


class MessageError(ANPClientError):
    """Raised when message operations fail."""
    pass


class ANPClient:
    """
    ANP Client for outbound communication.
    
    Handles DID-based authentication, protocol negotiation, encryption,
    and secure message exchange with other ANP agents.
    """
    
    def __init__(
        self,
        did_manager: DIDManager,
        discovery_service: Optional[DiscoveryService] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize ANP client."""
        self.did_manager = did_manager
        self.discovery_service = discovery_service
        self.config = config or {}
        
        # Core components
        self.meta_protocol = MetaProtocolNegotiator()
        self.encryption = ANPEncryption()
        
        # Connection management
        self.connections: Dict[str, ConnectionInfo] = {}
        self.active_sessions: Dict[str, aiohttp.ClientSession] = {}
        
        # Message handling
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.pending_responses: Dict[str, asyncio.Future] = {}
        
        # Configuration
        self.connection_timeout = self.config.get('connection_timeout', 30)
        self.message_timeout = self.config.get('message_timeout', 10)
        self.heartbeat_interval = self.config.get('heartbeat_interval', 60)
        self.max_connections = self.config.get('max_connections', 100)
        
        # Background tasks
        self._background_tasks: set = set()
        self._shutdown_event = asyncio.Event()
        
        # Setup default message handlers
        self._setup_default_handlers()
        
        logger.info(f"ANP Client initialized for DID: {self.did_manager.get_local_did()}")
    
    def _setup_default_handlers(self):
        """Setup default message handlers."""
        self.message_handlers[MessageType.AUTH_RESPONSE] = self._handle_auth_response
        self.message_handlers[MessageType.PROTOCOL_RESPONSE] = self._handle_protocol_response
        self.message_handlers[MessageType.HEARTBEAT] = self._handle_heartbeat
        self.message_handlers[MessageType.ERROR] = self._handle_error
        self.message_handlers[MessageType.DISCONNECT] = self._handle_disconnect
    
    async def start(self):
        """Start the ANP client."""
        try:
            # Start background tasks
            heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            self._background_tasks.add(heartbeat_task)
            self._background_tasks.add(cleanup_task)
            
            # Add done callbacks to remove completed tasks
            heartbeat_task.add_done_callback(self._background_tasks.discard)
            cleanup_task.add_done_callback(self._background_tasks.discard)
            
            logger.info("ANP Client started successfully")
            
        except Exception as e:
            logger.error(f"Error starting ANP client: {e}")
            raise
    
    async def stop(self):
        """Stop the ANP client."""
        try:
            # Signal shutdown
            self._shutdown_event.set()
            
            # Close all connections
            await self._close_all_connections()
            
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
            
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            # Close HTTP sessions
            for session in self.active_sessions.values():
                await session.close()
            self.active_sessions.clear()
            
            logger.info("ANP Client stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping ANP client: {e}")
    
    async def connect_to_agent(
        self,
        target_agent: Union[str, DiscoverableAgent],
        protocols: Optional[List[str]] = None
    ) -> str:
        """
        Connect to a remote agent.
        
        Args:
            target_agent: Agent ID, DID, or DiscoverableAgent object
            protocols: List of preferred protocols for negotiation
            
        Returns:
            Connection ID
        """
        try:
            # Resolve agent information
            agent_info = await self._resolve_agent(target_agent)
            if not agent_info:
                raise ConnectionError(f"Could not resolve agent: {target_agent}")
            
            # Check for existing connection
            existing_conn = self._find_existing_connection(agent_info.did)
            if existing_conn and existing_conn.state == ConnectionState.CONNECTED:
                logger.info(f"Reusing existing connection to {agent_info.did}")
                return existing_conn.connection_id
            
            # Create new connection
            connection_id = str(uuid.uuid4())
            endpoint = agent_info.endpoints[0].url if agent_info.endpoints else None
            
            if not endpoint:
                raise ConnectionError(f"No endpoint available for agent {agent_info.agent_id}")
            
            connection = ConnectionInfo(
                connection_id=connection_id,
                remote_agent_id=agent_info.agent_id,
                remote_did=agent_info.did,
                remote_endpoint=endpoint,
                state=ConnectionState.CONNECTING,
                metadata={"protocols_requested": protocols or []}
            )
            
            self.connections[connection_id] = connection
            
            # Perform connection handshake
            await self._perform_handshake(connection)
            
            # Authenticate with remote agent
            await self._authenticate(connection)
            
            # Negotiate protocols
            if protocols:
                await self._negotiate_protocols(connection, protocols)
            
            # Mark connection as established
            connection.state = ConnectionState.CONNECTED
            connection.established_at = datetime.utcnow()
            connection.update_activity()
            
            logger.info(f"Successfully connected to agent {agent_info.agent_id} (connection: {connection_id})")
            return connection_id
            
        except Exception as e:
            logger.error(f"Error connecting to agent: {e}")
            # Cleanup failed connection
            if connection_id in self.connections:
                del self.connections[connection_id]
            raise ConnectionError(f"Failed to connect to agent: {e}")
    
    async def disconnect_from_agent(self, connection_id: str) -> bool:
        """
        Disconnect from a remote agent.
        
        Args:
            connection_id: Connection ID to disconnect
            
        Returns:
            True if disconnected successfully
        """
        try:
            connection = self.connections.get(connection_id)
            if not connection:
                logger.warning(f"Connection {connection_id} not found")
                return False
            
            # Send disconnect message
            if connection.state == ConnectionState.CONNECTED:
                disconnect_msg = ANPMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.DISCONNECT,
                    sender_did=self.did_manager.get_local_did(),
                    recipient_did=connection.remote_did,
                    timestamp=datetime.utcnow(),
                    payload={"reason": "client_disconnect"}
                )
                
                try:
                    await self._send_message(connection, disconnect_msg)
                except Exception as e:
                    logger.warning(f"Error sending disconnect message: {e}")
            
            # Close connection
            await self._close_connection(connection_id)
            
            logger.info(f"Disconnected from agent {connection.remote_agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from agent: {e}")
            return False
    
    async def send_message(
        self,
        connection_id: str,
        payload: Dict[str, Any],
        message_type: MessageType = MessageType.DATA,
        wait_for_response: bool = False,
        timeout: Optional[int] = None
    ) -> Optional[ANPMessage]:
        """
        Send a message to a connected agent.
        
        Args:
            connection_id: Connection ID
            payload: Message payload
            message_type: Type of message
            wait_for_response: Whether to wait for a response
            timeout: Response timeout in seconds
            
        Returns:
            Response message if wait_for_response is True
        """
        try:
            connection = self.connections.get(connection_id)
            if not connection:
                raise MessageError(f"Connection {connection_id} not found")
            
            if connection.state != ConnectionState.CONNECTED:
                raise MessageError(f"Connection {connection_id} not in connected state")
            
            # Create message
            message = ANPMessage(
                message_id=str(uuid.uuid4()),
                message_type=message_type,
                sender_did=self.did_manager.get_local_did(),
                recipient_did=connection.remote_did,
                timestamp=datetime.utcnow(),
                payload=payload
            )
            
            # Setup response waiting if needed
            response_future = None
            if wait_for_response:
                response_future = asyncio.Future()
                self.pending_responses[message.message_id] = response_future
            
            # Send message
            await self._send_message(connection, message)
            
            # Wait for response if requested
            if wait_for_response and response_future:
                try:
                    timeout_val = timeout or self.message_timeout
                    response = await asyncio.wait_for(response_future, timeout=timeout_val)
                    return response
                except asyncio.TimeoutError:
                    raise MessageError(f"Response timeout for message {message.message_id}")
                finally:
                    self.pending_responses.pop(message.message_id, None)
            
            return None
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            raise MessageError(f"Failed to send message: {e}")
    
    async def send_data(
        self,
        connection_id: str,
        data: Any,
        data_type: str = "json",
        wait_for_response: bool = False
    ) -> Optional[Any]:
        """
        Send data to a connected agent with automatic serialization.
        
        Args:
            connection_id: Connection ID
            data: Data to send
            data_type: Data type (json, text, binary)
            wait_for_response: Whether to wait for a response
            
        Returns:
            Response data if wait_for_response is True
        """
        try:
            # Serialize data based on type
            if data_type == "json":
                payload = {"data": data, "type": "json"}
            elif data_type == "text":
                payload = {"data": str(data), "type": "text"}
            elif data_type == "binary":
                import base64
                encoded_data = base64.b64encode(data).decode('utf-8')
                payload = {"data": encoded_data, "type": "binary"}
            else:
                raise MessageError(f"Unsupported data type: {data_type}")
            
            # Send message
            response_msg = await self.send_message(
                connection_id=connection_id,
                payload=payload,
                message_type=MessageType.DATA,
                wait_for_response=wait_for_response
            )
            
            # Extract and deserialize response data if available
            if response_msg and wait_for_response:
                response_payload = response_msg.payload
                response_data = response_payload.get("data")
                response_type = response_payload.get("type", "json")
                
                if response_type == "json":
                    return response_data
                elif response_type == "text":
                    return str(response_data)
                elif response_type == "binary":
                    import base64
                    return base64.b64decode(response_data)
                else:
                    return response_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error sending data: {e}")
            raise MessageError(f"Failed to send data: {e}")
    
    async def broadcast_message(
        self,
        payload: Dict[str, Any],
        agent_filter: Optional[Callable[[DiscoverableAgent], bool]] = None,
        max_recipients: int = 10
    ) -> Dict[str, bool]:
        """
        Broadcast a message to multiple agents.
        
        Args:
            payload: Message payload
            agent_filter: Function to filter target agents
            max_recipients: Maximum number of recipients
            
        Returns:
            Dictionary mapping connection_id to success status
        """
        results = {}
        
        try:
            # Discover target agents
            if not self.discovery_service:
                raise MessageError("Discovery service not available for broadcasting")
            
            agents = await self.discovery_service.find_agents(max_results=max_recipients)
            
            # Apply filter if provided
            if agent_filter:
                agents = [agent for agent in agents if agent_filter(agent)]
            
            # Connect and send to each agent
            for agent in agents[:max_recipients]:
                try:
                    connection_id = await self.connect_to_agent(agent)
                    await self.send_message(connection_id, payload)
                    results[connection_id] = True
                    logger.debug(f"Broadcast message sent to {agent.agent_id}")
                    
                except Exception as e:
                    logger.warning(f"Failed to broadcast to {agent.agent_id}: {e}")
                    results[agent.agent_id] = False
            
            logger.info(f"Broadcast completed: {sum(results.values())}/{len(results)} successful")
            return results
            
        except Exception as e:
            logger.error(f"Error broadcasting message: {e}")
            raise MessageError(f"Failed to broadcast message: {e}")
    
    def get_connection_info(self, connection_id: str) -> Optional[ConnectionInfo]:
        """Get information about a connection."""
        return self.connections.get(connection_id)
    
    def get_active_connections(self) -> List[ConnectionInfo]:
        """Get list of active connections."""
        return [
            conn for conn in self.connections.values()
            if conn.state == ConnectionState.CONNECTED and conn.is_active()
        ]
    
    def register_message_handler(
        self,
        message_type: MessageType,
        handler: Callable[[ANPMessage, ConnectionInfo], None]
    ):
        """Register a custom message handler."""
        self.message_handlers[message_type] = handler
        logger.info(f"Registered handler for message type: {message_type}")
    
    # Private methods
    
    async def _resolve_agent(self, target: Union[str, DiscoverableAgent]) -> Optional[DiscoverableAgent]:
        """Resolve agent information from various input types."""
        if isinstance(target, DiscoverableAgent):
            return target
        
        if isinstance(target, str):
            # Try to find by agent ID or DID
            if self.discovery_service:
                # First try by agent ID
                agent = await self.discovery_service.find_agent_by_id(target)
                if agent:
                    return agent
                
                # Then try by DID
                agents = await self.discovery_service.find_agents(max_results=1)
                for agent in agents:
                    if agent.did == target:
                        return agent
        
        return None
    
    def _find_existing_connection(self, remote_did: str) -> Optional[ConnectionInfo]:
        """Find existing connection to a remote DID."""
        for connection in self.connections.values():
            if connection.remote_did == remote_did:
                return connection
        return None
    
    async def _perform_handshake(self, connection: ConnectionInfo):
        """Perform initial handshake with remote agent."""
        try:
            connection.state = ConnectionState.CONNECTING
            
            # Create handshake message
            handshake_msg = ANPMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.HANDSHAKE,
                sender_did=self.did_manager.get_local_did(),
                recipient_did=connection.remote_did,
                timestamp=datetime.utcnow(),
                payload={
                    "client_version": "1.0.0",
                    "supported_features": ["encryption", "protocol_negotiation"],
                    "connection_id": connection.connection_id
                }
            )
            
            # Send handshake
            await self._send_message(connection, handshake_msg)
            
            logger.debug(f"Handshake sent to {connection.remote_did}")
            
        except Exception as e:
            connection.state = ConnectionState.ERROR
            raise ConnectionError(f"Handshake failed: {e}")
    
    async def _authenticate(self, connection: ConnectionInfo):
        """Authenticate with remote agent using DID."""
        try:
            connection.state = ConnectionState.AUTHENTICATING
            
            # Create authentication challenge
            challenge = str(uuid.uuid4())
            
            # Sign challenge with our DID
            signature = await self.did_manager.sign_data(challenge.encode())
            
            # Create auth request
            auth_msg = ANPMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.AUTH_REQUEST,
                sender_did=self.did_manager.get_local_did(),
                recipient_did=connection.remote_did,
                timestamp=datetime.utcnow(),
                payload={
                    "challenge": challenge,
                    "signature": signature,
                    "did_document": self.did_manager.get_did_document().to_dict()
                }
            )
            
            # Send auth request and wait for response
            response = await self._send_message_and_wait(connection, auth_msg)
            
            if not response or response.message_type != MessageType.AUTH_RESPONSE:
                raise AuthenticationError("Invalid authentication response")
            
            # Verify response
            if not response.payload.get("authenticated"):
                raise AuthenticationError("Authentication rejected by remote agent")
            
            logger.debug(f"Authentication successful with {connection.remote_did}")
            
        except Exception as e:
            connection.state = ConnectionState.ERROR
            raise AuthenticationError(f"Authentication failed: {e}")
    
    async def _negotiate_protocols(self, connection: ConnectionInfo, protocols: List[str]):
        """Negotiate protocols with remote agent."""
        try:
            connection.state = ConnectionState.NEGOTIATING
            
            # Create protocol negotiation request
            negotiation_msg = ANPMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.PROTOCOL_NEGOTIATION,
                sender_did=self.did_manager.get_local_did(),
                recipient_did=connection.remote_did,
                timestamp=datetime.utcnow(),
                payload={
                    "requested_protocols": protocols,
                    "client_capabilities": self.meta_protocol.get_supported_protocols()
                }
            )
            
            # Send negotiation request and wait for response
            response = await self._send_message_and_wait(connection, negotiation_msg)
            
            if not response or response.message_type != MessageType.PROTOCOL_RESPONSE:
                raise ProtocolNegotiationError("Invalid protocol negotiation response")
            
            # Process negotiation result
            negotiated_protocols = response.payload.get("negotiated_protocols", [])
            connection.negotiated_protocols = [
                ProtocolSpec.from_dict(proto) for proto in negotiated_protocols
            ]
            
            logger.debug(f"Protocol negotiation successful: {len(negotiated_protocols)} protocols")
            
        except Exception as e:
            connection.state = ConnectionState.ERROR
            raise ProtocolNegotiationError(f"Protocol negotiation failed: {e}")
    
    async def _send_message(self, connection: ConnectionInfo, message: ANPMessage):
        """Send a message to a remote agent."""
        try:
            # Sign message
            message_data = message.to_json()
            signature = await self.did_manager.sign_data(message_data.encode())
            message.signature = signature
            
            # Encrypt message if encryption is enabled
            if connection.encryption_session:
                encrypted_msg = await self.encryption.encrypt_message(
                    message_data,
                    connection.encryption_session
                )
                message.encryption_info = encrypted_msg.to_dict()
            
            # Get or create HTTP session
            session = await self._get_http_session(connection.remote_endpoint)
            
            # Send HTTP request
            endpoint_url = urljoin(connection.remote_endpoint, "/anp/message")
            
            async with session.post(
                endpoint_url,
                json=message.to_dict(),
                timeout=aiohttp.ClientTimeout(total=self.message_timeout)
            ) as response:
                if response.status != 200:
                    raise MessageError(f"HTTP error {response.status}: {await response.text()}")
                
                response_data = await response.json()
                
                # Handle immediate response if present
                if "response_message" in response_data:
                    response_msg = ANPMessage.from_dict(response_data["response_message"])
                    await self._handle_received_message(response_msg, connection)
            
            connection.update_activity()
            logger.debug(f"Message sent successfully: {message.message_id}")
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            raise MessageError(f"Failed to send message: {e}")
    
    async def _send_message_and_wait(
        self,
        connection: ConnectionInfo,
        message: ANPMessage,
        timeout: Optional[int] = None
    ) -> Optional[ANPMessage]:
        """Send a message and wait for response."""
        # Setup response future
        response_future = asyncio.Future()
        self.pending_responses[message.message_id] = response_future
        
        try:
            # Send message
            await self._send_message(connection, message)
            
            # Wait for response
            timeout_val = timeout or self.message_timeout
            response = await asyncio.wait_for(response_future, timeout=timeout_val)
            return response
            
        except asyncio.TimeoutError:
            raise MessageError(f"Response timeout for message {message.message_id}")
        finally:
            self.pending_responses.pop(message.message_id, None)
    
    async def _handle_received_message(self, message: ANPMessage, connection: ConnectionInfo):
        """Handle a received message."""
        try:
            # Verify message signature
            if message.signature:
                # TODO: Implement signature verification
                pass
            
            # Decrypt message if encrypted
            if message.encryption_info:
                # TODO: Implement message decryption
                pass
            
            # Check for pending response
            if message.payload.get("response_to"):
                response_to_id = message.payload["response_to"]
                if response_to_id in self.pending_responses:
                    future = self.pending_responses.pop(response_to_id)
                    if not future.done():
                        future.set_result(message)
                    return
            
            # Handle message based on type
            handler = self.message_handlers.get(message.message_type)
            if handler:
                await handler(message, connection)
            else:
                logger.warning(f"No handler for message type: {message.message_type}")
            
            connection.update_activity()
            
        except Exception as e:
            logger.error(f"Error handling received message: {e}")
    
    async def _get_http_session(self, endpoint: str) -> aiohttp.ClientSession:
        """Get or create HTTP session for endpoint."""
        if endpoint not in self.active_sessions:
            connector = aiohttp.TCPConnector(
                limit=10,
                limit_per_host=5,
                ttl_dns_cache=300,
                use_dns_cache=True
            )
            
            self.active_sessions[endpoint] = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=self.connection_timeout)
            )
        
        return self.active_sessions[endpoint]
    
    async def _close_connection(self, connection_id: str):
        """Close a specific connection."""
        connection = self.connections.pop(connection_id, None)
        if connection:
            # Close HTTP session if no other connections use it
            endpoint_in_use = any(
                conn.remote_endpoint == connection.remote_endpoint
                for conn in self.connections.values()
            )
            
            if not endpoint_in_use and connection.remote_endpoint in self.active_sessions:
                session = self.active_sessions.pop(connection.remote_endpoint)
                await session.close()
            
            logger.debug(f"Connection {connection_id} closed")
    
    async def _close_all_connections(self):
        """Close all active connections."""
        connection_ids = list(self.connections.keys())
        
        for connection_id in connection_ids:
            try:
                await self.disconnect_from_agent(connection_id)
            except Exception as e:
                logger.warning(f"Error closing connection {connection_id}: {e}")
    
    # Default message handlers
    
    async def _handle_auth_response(self, message: ANPMessage, connection: ConnectionInfo):
        """Handle authentication response."""
        logger.debug(f"Received auth response from {message.sender_did}")
        # Response handling is done in _send_message_and_wait
    
    async def _handle_protocol_response(self, message: ANPMessage, connection: ConnectionInfo):
        """Handle protocol negotiation response."""
        logger.debug(f"Received protocol response from {message.sender_did}")
        # Response handling is done in _send_message_and_wait
    
    async def _handle_heartbeat(self, message: ANPMessage, connection: ConnectionInfo):
        """Handle heartbeat message."""
        logger.debug(f"Received heartbeat from {message.sender_did}")
        
        # Send heartbeat response
        response = ANPMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.HEARTBEAT,
            sender_did=self.did_manager.get_local_did(),
            recipient_did=message.sender_did,
            timestamp=datetime.utcnow(),
            payload={
                "response_to": message.message_id,
                "status": "alive"
            }
        )

        await self._send_message(connection, response)
    
    async def _handle_error(self, message: ANPMessage, connection: ConnectionInfo):
        """Handle error message."""
        error_info = message.payload
        logger.error(f"Received error from {message.sender_did}: {error_info}")
        
        # Update connection state if it's a critical error
        error_type = error_info.get("error_type")
        if error_type in ["authentication_failed", "protocol_error", "connection_error"]:
            connection.state = ConnectionState.ERROR
    
    async def _handle_disconnect(self, message: ANPMessage, connection: ConnectionInfo):
        """Handle disconnect message."""
        reason = message.payload.get("reason", "unknown")
        logger.info(f"Received disconnect from {message.sender_did}: {reason}")
        
        # Close the connection
        await self._close_connection(connection.connection_id)
    
    # Background tasks
    
    async def _heartbeat_loop(self):
        """Background task to send heartbeats to connected agents."""
        while not self._shutdown_event.is_set():
            try:
                current_time = datetime.utcnow()
                
                for connection in list(self.connections.values()):
                    if connection.state != ConnectionState.CONNECTED:
                        continue
                    
                    # Check if heartbeat is needed
                    if connection.last_activity:
                        time_since_activity = current_time - connection.last_activity
                        if time_since_activity.total_seconds() < self.heartbeat_interval:
                            continue
                    
                    try:
                        # Send heartbeat
                        heartbeat_msg = ANPMessage(
                            message_id=str(uuid.uuid4()),
                            message_type=MessageType.HEARTBEAT,
                            sender_did=self.did_manager.get_local_did(),
                            recipient_did=connection.remote_did,
                            timestamp=current_time,
                            payload={"timestamp": current_time.isoformat()}
                        )
                        
                        await self._send_message(connection, heartbeat_msg)
                        logger.debug(f"Heartbeat sent to {connection.remote_agent_id}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to send heartbeat to {connection.remote_agent_id}: {e}")
                        # Mark connection as potentially problematic
                        connection.state = ConnectionState.ERROR
                
                # Wait before next heartbeat cycle
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(10)  # Wait before retrying
    
    async def _cleanup_loop(self):
        """Background task to clean up inactive connections."""
        cleanup_interval = 300  # 5 minutes
        
        while not self._shutdown_event.is_set():
            try:
                current_time = datetime.utcnow()
                inactive_connections = []
                
                for connection_id, connection in self.connections.items():
                    # Check for inactive connections
                    if not connection.is_active(timeout_seconds=600):  # 10 minutes
                        inactive_connections.append(connection_id)
                        continue
                    
                    # Check for error state connections
                    if connection.state == ConnectionState.ERROR:
                        error_duration = current_time - (connection.last_activity or current_time)
                        if error_duration.total_seconds() > 300:  # 5 minutes in error state
                            inactive_connections.append(connection_id)
                
                # Clean up inactive connections
                for connection_id in inactive_connections:
                    try:
                        await self._close_connection(connection_id)
                        logger.info(f"Cleaned up inactive connection: {connection_id}")
                    except Exception as e:
                        logger.warning(f"Error cleaning up connection {connection_id}: {e}")
                
                # Clean up expired pending responses
                expired_responses = []
                for msg_id, future in self.pending_responses.items():
                    if future.done():
                        expired_responses.append(msg_id)
                
                for msg_id in expired_responses:
                    self.pending_responses.pop(msg_id, None)
                
                await asyncio.sleep(cleanup_interval)
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying


class ANPClientManager:
    """Manager for multiple ANP clients with different DIDs."""
    
    def __init__(self, discovery_service: Optional[DiscoveryService] = None):
        """Initialize client manager."""
        self.discovery_service = discovery_service
        self.clients: Dict[str, ANPClient] = {}
        self.default_client: Optional[ANPClient] = None
    
    async def create_client(
        self,
        did_manager: DIDManager,
        config: Optional[Dict[str, Any]] = None,
        set_as_default: bool = False
    ) -> ANPClient:
        """Create a new ANP client."""
        client = ANPClient(
            did_manager=did_manager,
            discovery_service=self.discovery_service,
            config=config
        )
        
        await client.start()
        
        client_id = did_manager.get_local_did()
        self.clients[client_id] = client
        
        if set_as_default or not self.default_client:
            self.default_client = client
        
        logger.info(f"Created ANP client for DID: {client_id}")
        return client
    
    async def remove_client(self, did: str) -> bool:
        """Remove an ANP client."""
        client = self.clients.pop(did, None)
        if client:
            await client.stop()
            
            if self.default_client == client:
                self.default_client = next(iter(self.clients.values()), None)
            
            logger.info(f"Removed ANP client for DID: {did}")
            return True
        
        return False
    
    def get_client(self, did: Optional[str] = None) -> Optional[ANPClient]:
        """Get an ANP client by DID or return default."""
        if did:
            return self.clients.get(did)
        return self.default_client
    
    def list_clients(self) -> List[str]:
        """List all client DIDs."""
        return list(self.clients.keys())
    
    async def shutdown_all(self):
        """Shutdown all clients."""
        for client in list(self.clients.values()):
            await client.stop()
        
        self.clients.clear()
        self.default_client = None
        logger.info("All ANP clients shut down")


# Utility functions and helpers

async def create_anp_client(
    private_key: Optional[str] = None,
    did: Optional[str] = None,
    discovery_service: Optional[DiscoveryService] = None,
    config: Optional[Dict[str, Any]] = None
) -> ANPClient:
    """
    Convenience function to create and start an ANP client.
    
    Args:
        private_key: Private key for DID (generates new if None)
        did: Existing DID (generates new if None)
        discovery_service: Discovery service instance
        config: Client configuration
        
    Returns:
        Started ANP client
    """
    # Create DID manager
    if private_key and did:
        did_manager = DIDManager.from_existing(private_key, did)
    else:
        did_manager = await DIDManager.create_new()
    
    # Create and start client
    client = ANPClient(
        did_manager=did_manager,
        discovery_service=discovery_service,
        config=config
    )
    
    await client.start()
    return client


class ANPClientPool:
    """Pool of ANP clients for load balancing and redundancy."""
    
    def __init__(self, max_clients: int = 5):
        """Initialize client pool."""
        self.max_clients = max_clients
        self.clients: List[ANPClient] = []
        self.current_index = 0
        self._lock = asyncio.Lock()
    
    async def add_client(self, client: ANPClient):
        """Add a client to the pool."""
        async with self._lock:
            if len(self.clients) < self.max_clients:
                self.clients.append(client)
                logger.info(f"Added client to pool (total: {len(self.clients)})")
            else:
                raise ValueError("Client pool is full")
    
    async def get_client(self) -> Optional[ANPClient]:
        """Get next available client using round-robin."""
        async with self._lock:
            if not self.clients:
                return None
            
            # Find next healthy client
            start_index = self.current_index
            while True:
                client = self.clients[self.current_index]
                self.current_index = (self.current_index + 1) % len(self.clients)
                
                # Check if client is healthy
                active_connections = client.get_active_connections()
                if len(active_connections) < client.max_connections:
                    return client
                
                # If we've checked all clients, return the first one anyway
                if self.current_index == start_index:
                    return client
    
    async def remove_client(self, client: ANPClient):
        """Remove a client from the pool."""
        async with self._lock:
            if client in self.clients:
                self.clients.remove(client)
                await client.stop()
                logger.info(f"Removed client from pool (total: {len(self.clients)})")
    
    async def shutdown_all(self):
        """Shutdown all clients in the pool."""
        async with self._lock:
            for client in self.clients:
                await client.stop()
            self.clients.clear()
            logger.info("Client pool shut down")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        total_connections = sum(
            len(client.get_active_connections()) for client in self.clients
        )
        
        return {
            "total_clients": len(self.clients),
            "max_clients": self.max_clients,
            "total_connections": total_connections,
            "average_connections_per_client": total_connections / max(len(self.clients), 1)
        }


# Message builder utilities

class ANPMessageBuilder:
    """Builder for creating ANP messages."""
    
    def __init__(self, sender_did: str, recipient_did: str):
        """Initialize message builder."""
        self.sender_did = sender_did
        self.recipient_did = recipient_did
        self.message_id = str(uuid.uuid4())
        self.message_type = MessageType.DATA
        self.payload = {}
        self.timestamp = datetime.utcnow()
    
    def set_type(self, message_type: MessageType) -> 'ANPMessageBuilder':
        """Set message type."""
        self.message_type = message_type
        return self
    
    def set_payload(self, payload: Dict[str, Any]) -> 'ANPMessageBuilder':
        """Set message payload."""
        self.payload = payload
        return self
    
    def add_data(self, key: str, value: Any) -> 'ANPMessageBuilder':
        """Add data to payload."""
        self.payload[key] = value
        return self
    
    def set_response_to(self, message_id: str) -> 'ANPMessageBuilder':
        """Mark as response to another message."""
        self.payload["response_to"] = message_id
        return self
    
    def build(self) -> ANPMessage:
        """Build the ANP message."""
        return ANPMessage(
            message_id=self.message_id,
            message_type=self.message_type,
            sender_did=self.sender_did,
            recipient_did=self.recipient_did,
            timestamp=self.timestamp,
            payload=self.payload
        )


# Configuration helpers

DEFAULT_CLIENT_CONFIG = {
    "connection_timeout": 30,
    "message_timeout": 10,
    "heartbeat_interval": 60,
    "max_connections": 100,
    "retry_attempts": 3,
    "retry_delay": 5,
    "enable_encryption": True,
    "enable_compression": False,
    "log_level": "INFO"
}


def create_client_config(**overrides) -> Dict[str, Any]:
    """Create client configuration with overrides."""
    config = DEFAULT_CLIENT_CONFIG.copy()
    config.update(overrides)
    return config


# Example usage and testing

async def example_client_usage():
    """Example of how to use the ANP client."""
    
    # Create client
    client = await create_anp_client()
    
    try:
        # Connect to an agent
        target_agent_id = "example-agent-123"
        connection_id = await client.connect_to_agent(target_agent_id)
        
        # Send a simple message
        response = await client.send_data(
            connection_id=connection_id,
            data={"query": "Hello, how are you?"},
            wait_for_response=True
        )
        
        print(f"Response: {response}")
        
        # Broadcast a message
        broadcast_results = await client.broadcast_message(
            payload={"announcement": "System maintenance in 1 hour"},
            max_recipients=5
        )
        
        print(f"Broadcast results: {broadcast_results}")
        
    finally:
        # Clean up
        await client.stop()


if __name__ == "__main__":
    # Run example
    asyncio.run(example_client_usage())


# Module exports
__all__ = [
    # Core classes
    'ANPClient',
    'ANPClientManager',
    'ANPClientPool',
    
    # Message classes
    'ANPMessage',
    'ANPMessageBuilder',
    'MessageType',
    
    # Connection classes
    'ConnectionInfo',
    'ConnectionState',
    
    # Exceptions
    'ANPClientError',
    'ConnectionError',
    'AuthenticationError',
    'ProtocolNegotiationError',
    'MessageError',
    
    # Utility functions
    'create_anp_client',
    'create_client_config',
    
    # Configuration
    'DEFAULT_CLIENT_CONFIG'
]

# Module metadata
__version__ = "1.0.0"
__author__ = "MKT Communication Team"
__license__ = "MIT"

logger.info(f"ANP Client module loaded (version {__version__})")
