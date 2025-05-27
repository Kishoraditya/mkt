"""
ANP Server implementation.

This module provides the server-side implementation for the Agent Network Protocol (ANP),
handling inbound communication, DID-based authentication, protocol negotiation,
and secure message processing from other ANP agents.
"""

import json
import uuid
import asyncio
import aiohttp
from aiohttp import web, WSMsgType
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Set
from dataclasses import dataclass, asdict
from urllib.parse import urlparse
import logging
from enum import Enum
import ssl
import weakref

from .did import DIDManager, DIDDocument, DIDKeyPair
from .meta_protocol import MetaProtocolNegotiator, ProtocolSpec, NegotiationResult
from .encryption import ANPEncryption, EncryptedMessage, KeyExchange
from .discovery import DiscoveryService, DiscoverableAgent, AgentStatus
from .client import ANPMessage, MessageType, ConnectionState, ConnectionInfo

logger = logging.getLogger(__name__)


class ServerState(Enum):
    """Server state enumeration."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class SessionState(Enum):
    """Session state enumeration."""
    HANDSHAKE = "handshake"
    AUTHENTICATING = "authenticating"
    NEGOTIATING = "negotiating"
    AUTHENTICATED = "authenticated"
    ACTIVE = "active"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass
class ClientSession:
    """Information about a connected client session."""
    
    session_id: str
    remote_did: Optional[str]
    remote_agent_id: Optional[str]
    remote_address: str
    state: SessionState
    created_at: datetime
    last_activity: datetime
    authenticated_at: Optional[datetime] = None
    negotiated_protocols: Optional[List[ProtocolSpec]] = None
    encryption_session: Optional[Dict[str, Any]] = None
    websocket: Optional[web.WebSocketResponse] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()
    
    def is_active(self, timeout_seconds: int = 300) -> bool:
        """Check if session is active within timeout."""
        timeout = timedelta(seconds=timeout_seconds)
        return (datetime.utcnow() - self.last_activity) < timeout
    
    def is_authenticated(self) -> bool:
        """Check if session is authenticated."""
        return self.state in [SessionState.AUTHENTICATED, SessionState.ACTIVE]


class ANPServerError(Exception):
    """Base exception for ANP server errors."""
    pass


class ServerConfigurationError(ANPServerError):
    """Raised when server configuration is invalid."""
    pass


class SessionError(ANPServerError):
    """Raised when session operations fail."""
    pass


class AuthenticationError(ANPServerError):
    """Raised when authentication fails."""
    pass


class ProtocolError(ANPServerError):
    """Raised when protocol operations fail."""
    pass


class ANPServer:
    """
    ANP Server for inbound communication.
    
    Handles DID-based authentication, protocol negotiation, encryption,
    and secure message processing from other ANP agents.
    """
    
    def __init__(
        self,
        did_manager: DIDManager,
        discovery_service: Optional[DiscoveryService] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize ANP server."""
        self.did_manager = did_manager
        self.discovery_service = discovery_service
        self.config = config or {}
        
        # Core components
        self.meta_protocol = MetaProtocolNegotiator()
        self.encryption = ANPEncryption()
        
        # Server state
        self.state = ServerState.STOPPED
        self.app: Optional[web.Application] = None
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        
        # Session management
        self.sessions: Dict[str, ClientSession] = {}
        self.websocket_sessions: Dict[str, ClientSession] = {}
        
        # Message handling
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.middleware_handlers: List[Callable] = []
        
        # Configuration
        self.host = self.config.get('host', '0.0.0.0')
        self.port = self.config.get('port', 8080)
        self.ssl_context = self._create_ssl_context()
        self.max_sessions = self.config.get('max_sessions', 1000)
        self.session_timeout = self.config.get('session_timeout', 3600)
        self.enable_websockets = self.config.get('enable_websockets', True)
        self.enable_cors = self.config.get('enable_cors', True)
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        
        # Setup default handlers
        self._setup_default_handlers()
        
        logger.info(f"ANP Server initialized for DID: {self.did_manager.get_local_did()}")
    
    def _create_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Create SSL context if certificates are configured."""
        ssl_config = self.config.get('ssl', {})
        
        if not ssl_config.get('enabled', False):
            return None
        
        cert_file = ssl_config.get('cert_file')
        key_file = ssl_config.get('key_file')
        
        if not cert_file or not key_file:
            logger.warning("SSL enabled but cert_file or key_file not specified")
            return None
        
        try:
            context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            context.load_cert_chain(cert_file, key_file)
            logger.info("SSL context created successfully")
            return context
        except Exception as e:
            logger.error(f"Failed to create SSL context: {e}")
            return None
    
    def _setup_default_handlers(self):
        """Setup default message handlers."""
        self.message_handlers[MessageType.HANDSHAKE] = self._handle_handshake
        self.message_handlers[MessageType.AUTH_REQUEST] = self._handle_auth_request
        self.message_handlers[MessageType.PROTOCOL_NEGOTIATION] = self._handle_protocol_negotiation
        self.message_handlers[MessageType.DATA] = self._handle_data_message
        self.message_handlers[MessageType.HEARTBEAT] = self._handle_heartbeat
        self.message_handlers[MessageType.DISCONNECT] = self._handle_disconnect
    
    async def start(self):
        """Start the ANP server."""
        try:
            if self.state != ServerState.STOPPED:
                raise ANPServerError(f"Server is not in stopped state: {self.state}")
            
            self.state = ServerState.STARTING
            logger.info(f"Starting ANP server on {self.host}:{self.port}")
            
            # Create web application
            self.app = web.Application(
                middlewares=self._create_middlewares()
            )
            
            # Setup routes
            self._setup_routes()
            
            # Create and start runner
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            
            # Create site
            self.site = web.TCPSite(
                self.runner,
                self.host,
                self.port,
                ssl_context=self.ssl_context
            )
            await self.site.start()
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Register with discovery service
            if self.discovery_service:
                await self._register_with_discovery()
            
            self.state = ServerState.RUNNING
            logger.info(f"ANP Server started successfully on {self.host}:{self.port}")
            
        except Exception as e:
            self.state = ServerState.ERROR
            logger.error(f"Failed to start ANP server: {e}")
            raise ANPServerError(f"Failed to start server: {e}")
    
    async def stop(self):
        """Stop the ANP server."""
        try:
            if self.state == ServerState.STOPPED:
                return
            
            self.state = ServerState.STOPPING
            logger.info("Stopping ANP server")
            
            # Signal shutdown
            self._shutdown_event.set()
            
            # Close all sessions
            await self._close_all_sessions()
            
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
            
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
            
            # Unregister from discovery service
            if self.discovery_service:
                await self._unregister_from_discovery()
            
            # Stop web server
            if self.site:
                await self.site.stop()
                self.site = None
            
            if self.runner:
                await self.runner.cleanup()
                self.runner = None
            
            self.app = None
            self.state = ServerState.STOPPED
            logger.info("ANP Server stopped successfully")
            
        except Exception as e:
            self.state = ServerState.ERROR
            logger.error(f"Error stopping ANP server: {e}")
            raise ANPServerError(f"Failed to stop server: {e}")
    
    def _create_middlewares(self) -> List[Callable]:
        """Create middleware stack."""
        middlewares = []
        
        # CORS middleware
        if self.enable_cors:
            middlewares.append(self._cors_middleware)
        
        # Authentication middleware
        middlewares.append(self._auth_middleware)
        
        # Session middleware
        middlewares.append(self._session_middleware)
        
        # Error handling middleware
        middlewares.append(self._error_middleware)
        
        # Custom middleware
        middlewares.extend(self.middleware_handlers)
        
        return middlewares
    
    def _setup_routes(self):
        """Setup HTTP routes."""
        # ANP protocol endpoints
        self.app.router.add_post('/anp/message', self._handle_http_message)
        self.app.router.add_get('/anp/agent-card', self._handle_agent_card_request)
        self.app.router.add_get('/anp/status', self._handle_status_request)
        
        # WebSocket endpoint
        if self.enable_websockets:
            self.app.router.add_get('/anp/ws', self._handle_websocket)
        
        # Discovery endpoints
        self.app.router.add_get('/.well-known/anp-agent', self._handle_well_known_agent)
        
        # Health check
        self.app.router.add_get('/health', self._handle_health_check)
        
        logger.debug("ANP server routes configured")
    
    async def _start_background_tasks(self):
        """Start background maintenance tasks."""
        # Session cleanup task
        cleanup_task = asyncio.create_task(self._session_cleanup_loop())
        self._background_tasks.add(cleanup_task)
        cleanup_task.add_done_callback(self._background_tasks.discard)
        
        # Heartbeat task
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._background_tasks.add(heartbeat_task)
        heartbeat_task.add_done_callback(self._background_tasks.discard)
        
        # Discovery update task
        if self.discovery_service:
            discovery_task = asyncio.create_task(self._discovery_update_loop())
            self._background_tasks.add(discovery_task)
            discovery_task.add_done_callback(self._background_tasks.discard)
        
        logger.debug("Background tasks started")
    
    # HTTP Route Handlers
    
    async def _handle_http_message(self, request: web.Request) -> web.Response:
        """Handle HTTP message endpoint."""
        try:
            # Get session
            session = request.get('anp_session')
            if not session:
                return web.json_response(
                    {"error": "No valid session"},
                    status=401
                )
            
            # Parse message
            try:
                data = await request.json()
                message = ANPMessage.from_dict(data)
            except Exception as e:
                return web.json_response(
                    {"error": f"Invalid message format: {e}"},
                    status=400
                )
            
            # Verify message
            if not await self._verify_message(message, session):
                return web.json_response(
                    {"error": "Message verification failed"},
                    status=403
                )
            
            # Process message
            response_message = await self._process_message(message, session)
            
            # Return response
            response_data = {"status": "processed"}
            if response_message:
                response_data["response_message"] = response_message.to_dict()
            
            return web.json_response(response_data)
            
        except Exception as e:
            logger.error(f"Error handling HTTP message: {e}")
            return web.json_response(
                {"error": "Internal server error"},
                status=500
            )
    
    async def _handle_agent_card_request(self, request: web.Request) -> web.Response:
        """Handle agent card request."""
        try:
            # Create agent card from DID document
            did_doc = self.did_manager.get_did_document()
            
            agent_card = {
                "agent_id": self.did_manager.get_local_did(),
                "did": self.did_manager.get_local_did(),
                "did_document": did_doc.to_dict(),
                "endpoints": [
                    {
                        "url": f"{'https' if self.ssl_context else 'http'}://{self.host}:{self.port}/anp/",
                        "protocol": "anp",
                        "version": "1.0.0"
                    }
                ],
                "capabilities": self.meta_protocol.get_supported_protocols(),
                "metadata": {
                    "server_version": "1.0.0",
                    "features": ["encryption", "websockets", "protocol_negotiation"]
                }
            }
            
            return web.json_response(agent_card)
            
        except Exception as e:
            logger.error(f"Error handling agent card request: {e}")
            return web.json_response(
                {"error": "Failed to generate agent card"},
                status=500
            )
    
    async def _handle_status_request(self, request: web.Request) -> web.Response:
        """Handle server status request."""
        try:
            status = {
                "server_state": self.state.value,
                "did": self.did_manager.get_local_did(),
                "active_sessions": len([s for s in self.sessions.values() if s.is_active()]),
                "total_sessions": len(self.sessions),
                "websocket_sessions": len(self.websocket_sessions),
                "uptime": (datetime.utcnow() - datetime.utcnow()).total_seconds(),  # TODO: Track actual uptime
                "features": {
                    "websockets": self.enable_websockets,
                    "ssl": self.ssl_context is not None,
                    "cors": self.enable_cors,
                    "discovery": self.discovery_service is not None
                }
            }
            
            return web.json_response(status)
            
        except Exception as e:
            logger.error(f"Error handling status request: {e}")
            return web.json_response(
                {"error": "Failed to get status"},
                status=500
            )
    
    async def _handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Handle WebSocket connection."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        session_id = str(uuid.uuid4())
        remote_address = request.remote
        
        # Create session
        session = ClientSession(
            session_id=session_id,
            remote_did=None,
            remote_agent_id=None,
            remote_address=remote_address,
            state=SessionState.HANDSHAKE,
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            websocket=ws
        )
        
        self.websocket_sessions[session_id] = session
        logger.info(f"WebSocket connection established: {session_id}")
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        message = ANPMessage.from_dict(data)
                        
                        # Process message
                        response = await self._process_message(message, session)
                        
                        # Send response if available
                        if response:
                            await ws.send_text(response.to_json())
                        
                        session.update_activity()
                        
                    except Exception as e:
                        logger.error(f"Error processing WebSocket message: {e}")
                        error_response = {
                            "error": "Message processing failed",
                            "details": str(e)
                        }
                        await ws.send_text(json.dumps(error_response))
                
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")
                    break
                
                elif msg.type == WSMsgType.CLOSE:
                    logger.info(f"WebSocket closed: {session_id}")
                    break
        
        except Exception as e:
            logger.error(f"WebSocket handler error: {e}")
        
        finally:
            # Clean up session
            self.websocket_sessions.pop(session_id, None)
            session.state = SessionState.DISCONNECTED
            logger.info(f"WebSocket session closed: {session_id}")
        
        return ws
    
    async def _handle_well_known_agent(self, request: web.Request) -> web.Response:
        """Handle .well-known/anp-agent discovery endpoint."""
        try:
            discovery_info = {
                "agent_id": self.did_manager.get_local_did(),
                "did": self.did_manager.get_local_did(),
                "name": self.config.get('agent_name', 'ANP Agent'),
                "description": self.config.get('agent_description', 'Agent Network Protocol Agent'),
                "version": "1.0.0",
                "endpoints": [
                    {
                        "url": f"{'https' if self.ssl_context else 'http'}://{self.host}:{self.port}/anp/",
                        "protocol": "anp",
                        "transport": "http"
                    }
                ],
                "capabilities": self.meta_protocol.get_supported_protocols(),
                "status": "active",
                "last_seen": datetime.utcnow().isoformat()
            }
            
            return web.json_response(discovery_info)
            
        except Exception as e:
            logger.error(f"Error handling well-known agent request: {e}")
            return web.json_response(
                {"error": "Discovery information unavailable"},
                status=500
            )
    
    async def _handle_health_check(self, request: web.Request) -> web.Response:
        """Handle health check endpoint."""
        health_status = {
            "status": "healthy" if self.state == ServerState.RUNNING else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0"
        }
        
        status_code = 200 if self.state == ServerState.RUNNING else 503
        return web.json_response(health_status, status=status_code)
    
    # Middleware
    
    @web.middleware
    async def _cors_middleware(self, request: web.Request, handler):
        """CORS middleware."""
        response = await handler(request)
        
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        
        return response
    
    @web.middleware
    async def _auth_middleware(self, request: web.Request, handler):
        """Authentication middleware."""
        # Skip auth for certain endpoints
        skip_auth_paths = ['/health', '/.well-known/anp-agent', '/anp/status']
        if request.path in skip_auth_paths:
            return await handler(request)
        
        # For WebSocket, auth is handled in the WebSocket handler
        if request.path == '/anp/ws':
            return await handler(request)
        
        # Check for existing session or create new one
        session_id = request.headers.get('X-ANP-Session-ID')
        session = None
        
        if session_id:
            session = self.sessions.get(session_id)
        
        if not session:
            # Create new session for this request
            session_id = str(uuid.uuid4())
            session = ClientSession(
                session_id=session_id,
                remote_did=None,
                remote_agent_id=None,
                remote_address=request.remote,
                state=SessionState.HANDSHAKE,
                created_at=datetime.utcnow(),
                last_activity=datetime.utcnow()
            )
            self.sessions[session_id] = session
        
        request['anp_session'] = session
        session.update_activity()
        
        return await handler(request)
    
    @web.middleware
    async def _session_middleware(self, request: web.Request, handler):
        """Session management middleware."""
        # Check session limits
        if len(self.sessions) >= self.max_sessions:
            return web.json_response(
                {"error": "Server at capacity"},
                status=503
            )
        
        response = await handler(request)
        
        # Add session ID to response headers
        session = request.get('anp_session')
        if session:
            response.headers['X-ANP-Session-ID'] = session.session_id
        
        return response
    
    @web.middleware
    async def _error_middleware(self, request: web.Request, handler):
        """Error handling middleware."""
        try:
            return await handler(request)
        except web.HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unhandled error in request {request.path}: {e}")
            return web.json_response(
                {"error": "Internal server error"},
                status=500
            )
    
    # Message Processing
    
    async def _process_message(
        self,
        message: ANPMessage,
        session: ClientSession
    ) -> Optional[ANPMessage]:
        """Process an incoming message."""
        try:
            logger.debug(f"Processing message {message.message_id} of type {message.message_type}")
            
            # Update session with sender info
            if message.sender_did and not session.remote_did:
                session.remote_did = message.sender_did
            
            # Get handler for message type
            handler = self.message_handlers.get(message.message_type)
            if not handler:
                logger.warning(f"No handler for message type: {message.message_type}")
                return self._create_error_response(
                    message,
                    "unsupported_message_type",
                    f"No handler for message type: {message.message_type}"
                )
            
            # Call handler
            response = await handler(message, session)
            
            session.update_activity()
            return response
            
        except Exception as e:
            logger.error(f"Error processing message {message.message_id}: {e}")
            return self._create_error_response(
                message,
                "processing_error",
                f"Message processing failed: {e}"
            )
    
    async def _verify_message(self, message: ANPMessage, session: ClientSession) -> bool:
        """Verify message authenticity and integrity."""
        try:
            # Check message structure
            if not message.message_id or not message.sender_did:
                return False
            
            # Verify signature if present
            if message.signature:
                # TODO: Implement signature verification using sender's DID
                pass
            
            # Decrypt message if encrypted
            if message.encryption_info:
                # TODO: Implement message decryption
                pass
            
            # Check if sender is authenticated for this session
            if session.state in [SessionState.AUTHENTICATED, SessionState.ACTIVE]:
                if session.remote_did != message.sender_did:
                    logger.warning(f"Message sender DID mismatch: {message.sender_did} vs {session.remote_did}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error verifying message: {e}")
            return False
    
    def _create_error_response(
        self,
        original_message: ANPMessage,
        error_type: str,
        error_message: str
    ) -> ANPMessage:
        """Create an error response message."""
        return ANPMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.ERROR,
            sender_did=self.did_manager.get_local_did(),
            recipient_did=original_message.sender_did,
            timestamp=datetime.utcnow(),
            payload={
                "response_to": original_message.message_id,
                "error_type": error_type,
                "error_message": error_message
            }
        )
    
    def _create_response_message(
        self,
        original_message: ANPMessage,
        payload: Dict[str, Any],
        message_type: MessageType = MessageType.DATA
    ) -> ANPMessage:
        """Create a response message."""
        response_payload = payload.copy()
        response_payload["response_to"] = original_message.message_id
        
        return ANPMessage(
            message_id=str(uuid.uuid4()),
            message_type=message_type,
            sender_did=self.did_manager.get_local_did(),
            recipient_did=original_message.sender_did,
            timestamp=datetime.utcnow(),
            payload=response_payload
        )
    
    # Default Message Handlers
    
    async def _handle_handshake(
        self,
        message: ANPMessage,
        session: ClientSession
    ) -> Optional[ANPMessage]:
        """Handle handshake message."""
        try:
            logger.info(f"Handling handshake from {message.sender_did}")
            
            # Update session state
            session.state = SessionState.HANDSHAKE
            session.remote_did = message.sender_did
            
            # Extract client info
            client_version = message.payload.get("client_version")
            supported_features = message.payload.get("supported_features", [])
            
            # Create handshake response
            response_payload = {
                "server_version": "1.0.0",
                "supported_features": ["encryption", "protocol_negotiation", "websockets"],
                "server_capabilities": self.meta_protocol.get_supported_protocols(),
                "session_id": session.session_id,
                "next_step": "authentication"
            }
            
            return self._create_response_message(message, response_payload)
            
        except Exception as e:
            logger.error(f"Error handling handshake: {e}")
            return self._create_error_response(message, "handshake_error", str(e))
    
    async def _handle_auth_request(
        self,
        message: ANPMessage,
        session: ClientSession
    ) -> Optional[ANPMessage]:
        """Handle authentication request."""
        try:
            logger.info(f"Handling auth request from {message.sender_did}")
            
            session.state = SessionState.AUTHENTICATING
            
            # Extract auth data
            challenge = message.payload.get("challenge")
            signature = message.payload.get("signature")
            did_document_data = message.payload.get("did_document")
            
            if not all([challenge, signature, did_document_data]):
                return self._create_error_response(
                    message,
                    "invalid_auth_request",
                    "Missing required authentication data"
                )
            
            # Verify DID document
            try:
                remote_did_doc = DIDDocument.from_dict(did_document_data)
            except Exception as e:
                return self._create_error_response(
                    message,
                    "invalid_did_document",
                    f"Invalid DID document: {e}"
                )
            
            # Verify signature
            # TODO: Implement signature verification using remote DID document
            signature_valid = True  # Placeholder
            
            if not signature_valid:
                return self._create_error_response(
                    message,
                    "authentication_failed",
                    "Signature verification failed"
                )
            
            # Authentication successful
            session.state = SessionState.AUTHENTICATED
            session.authenticated_at = datetime.utcnow()
            session.remote_did = message.sender_did
            
            # Extract agent ID from DID document if available
            session.remote_agent_id = remote_did_doc.id
            
            response_payload = {
                "authenticated": True,
                "session_expires": (datetime.utcnow() + timedelta(seconds=self.session_timeout)).isoformat(),
                "server_did": self.did_manager.get_local_did(),
                "next_step": "protocol_negotiation"
            }
            
            logger.info(f"Authentication successful for {message.sender_did}")
            return self._create_response_message(message, response_payload, MessageType.AUTH_RESPONSE)
            
        except Exception as e:
            logger.error(f"Error handling auth request: {e}")
            return self._create_error_response(message, "auth_error", str(e))
    
    async def _handle_protocol_negotiation(
        self,
        message: ANPMessage,
        session: ClientSession
    ) -> Optional[ANPMessage]:
        """Handle protocol negotiation."""
        try:
            logger.info(f"Handling protocol negotiation from {message.sender_did}")
            
            if not session.is_authenticated():
                return self._create_error_response(
                    message,
                    "not_authenticated",
                    "Authentication required for protocol negotiation"
                )
            
            session.state = SessionState.NEGOTIATING
            
            # Extract requested protocols
            requested_protocols = message.payload.get("requested_protocols", [])
            client_capabilities = message.payload.get("client_capabilities", [])
            
            # Negotiate protocols
            negotiation_result = await self.meta_protocol.negotiate_protocols(
                requested_protocols,
                client_capabilities
            )
            
            if not negotiation_result.success:
                return self._create_error_response(
                    message,
                    "protocol_negotiation_failed",
                    negotiation_result.error_message or "Protocol negotiation failed"
                )
            
            # Store negotiated protocols in session
            session.negotiated_protocols = negotiation_result.negotiated_protocols
            session.state = SessionState.ACTIVE
            
            # Prepare response
            response_payload = {
                "negotiation_successful": True,
                "negotiated_protocols": [proto.to_dict() for proto in negotiation_result.negotiated_protocols],
                "session_ready": True,
                "supported_features": negotiation_result.supported_features
            }
            
            logger.info(f"Protocol negotiation successful for {message.sender_did}: {len(negotiation_result.negotiated_protocols)} protocols")
            return self._create_response_message(message, response_payload, MessageType.PROTOCOL_RESPONSE)
            
        except Exception as e:
            logger.error(f"Error handling protocol negotiation: {e}")
            return self._create_error_response(message, "negotiation_error", str(e))
    
    async def _handle_data_message(
        self,
        message: ANPMessage,
        session: ClientSession
    ) -> Optional[ANPMessage]:
        """Handle data message."""
        try:
            logger.debug(f"Handling data message from {message.sender_did}")
            
            if session.state != SessionState.ACTIVE:
                return self._create_error_response(
                    message,
                    "session_not_ready",
                    "Session must be in active state to process data messages"
                )
            
            # Extract data from message
            data = message.payload.get("data")
            data_type = message.payload.get("type", "json")
            
            # Process data based on type
            processed_data = await self._process_data(data, data_type, session)
            
            # Create response if processing was successful
            if processed_data is not None:
                response_payload = {
                    "status": "processed",
                    "data": processed_data,
                    "type": data_type
                }
                return self._create_response_message(message, response_payload)
            else:
                # No response needed for this data
                return None
            
        except Exception as e:
            logger.error(f"Error handling data message: {e}")
            return self._create_error_response(message, "data_processing_error", str(e))
    
    async def _handle_heartbeat(
        self,
        message: ANPMessage,
        session: ClientSession
    ) -> Optional[ANPMessage]:
        """Handle heartbeat message."""
        logger.debug(f"Handling heartbeat from {message.sender_did}")
        
        # Update session activity
        session.update_activity()
        
        # Send heartbeat response
        response_payload = {
            "status": "alive",
            "timestamp": datetime.utcnow().isoformat(),
            "session_active": True
        }
        
        return self._create_response_message(message, response_payload, MessageType.HEARTBEAT)
    
    async def _handle_disconnect(
        self,
        message: ANPMessage,
        session: ClientSession
    ) -> Optional[ANPMessage]:
        """Handle disconnect message."""
        reason = message.payload.get("reason", "client_requested")
        logger.info(f"Handling disconnect from {message.sender_did}: {reason}")
        
        # Mark session as disconnected
        session.state = SessionState.DISCONNECTED
        
        # Send disconnect acknowledgment
        response_payload = {
            "disconnect_acknowledged": True,
            "reason": "server_acknowledged"
        }
        
        # Schedule session cleanup
        asyncio.create_task(self._cleanup_session(session.session_id, delay=5))
        
        return self._create_response_message(message, response_payload, MessageType.DISCONNECT)
    
    # Data Processing
    
    async def _process_data(
        self,
        data: Any,
        data_type: str,
        session: ClientSession
    ) -> Optional[Any]:
        """Process incoming data based on type."""
        try:
            if data_type == "json":
                # Process JSON data
                return await self._process_json_data(data, session)
            elif data_type == "text":
                # Process text data
                return await self._process_text_data(str(data), session)
            elif data_type == "binary":
                # Process binary data
                import base64
                binary_data = base64.b64decode(data)
                return await self._process_binary_data(binary_data, session)
            else:
                logger.warning(f"Unknown data type: {data_type}")
                return {"error": f"Unsupported data type: {data_type}"}
                
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return {"error": f"Data processing failed: {e}"}
    
    async def _process_json_data(self, data: Dict[str, Any], session: ClientSession) -> Optional[Dict[str, Any]]:
        """Process JSON data - override in subclasses for custom logic."""
        # Default implementation - echo the data back
        return {
            "echo": data,
            "processed_at": datetime.utcnow().isoformat(),
            "session_id": session.session_id
        }
    
    async def _process_text_data(self, text: str, session: ClientSession) -> Optional[str]:
        """Process text data - override in subclasses for custom logic."""
        # Default implementation - echo the text back
        return f"Echo: {text} (processed at {datetime.utcnow().isoformat()})"
    
    async def _process_binary_data(self, data: bytes, session: ClientSession) -> Optional[str]:
        """Process binary data - override in subclasses for custom logic."""
        # Default implementation - return info about the data
        return f"Received {len(data)} bytes of binary data"
    
    # Session Management
    
    async def _cleanup_session(self, session_id: str, delay: int = 0):
        """Clean up a session after optional delay."""
        if delay > 0:
            await asyncio.sleep(delay)
        
        # Remove from sessions
        session = self.sessions.pop(session_id, None)
        self.websocket_sessions.pop(session_id, None)
        
        if session:
            # Close WebSocket if present
            if session.websocket and not session.websocket.closed:
                await session.websocket.close()
            
            logger.debug(f"Session cleaned up: {session_id}")
    
    async def _close_all_sessions(self):
        """Close all active sessions."""
        session_ids = list(self.sessions.keys())
        
        for session_id in session_ids:
            try:
                await self._cleanup_session(session_id)
            except Exception as e:
                logger.warning(f"Error closing session {session_id}: {e}")
        
        # Close WebSocket sessions
        ws_session_ids = list(self.websocket_sessions.keys())
        for session_id in ws_session_ids:
            try:
                session = self.websocket_sessions.get(session_id)
                if session and session.websocket and not session.websocket.closed:
                    await session.websocket.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket session {session_id}: {e}")
        
        self.sessions.clear()
        self.websocket_sessions.clear()
        logger.info("All sessions closed")
    
    # Discovery Service Integration
    
    async def _register_with_discovery(self):
        """Register this server with the discovery service."""
        try:
            if not self.discovery_service:
                return
            
            # Create discoverable agent info
            agent_info = DiscoverableAgent(
                agent_id=self.did_manager.get_local_did(),
                did=self.did_manager.get_local_did(),
                name=self.config.get('agent_name', 'ANP Server'),
                description=self.config.get('agent_description', 'Agent Network Protocol Server'),
                capabilities=self.meta_protocol.get_supported_protocols(),
                endpoints=[
                    f"{'https' if self.ssl_context else 'http'}://{self.host}:{self.port}/anp/"
                ],
                status=AgentStatus.ACTIVE,
                metadata={
                    "server_version": "1.0.0",
                    "features": ["encryption", "websockets", "protocol_negotiation"]
                }
            )
            
            await self.discovery_service.register_agent(agent_info)
            logger.info("Registered with discovery service")
            
        except Exception as e:
            logger.error(f"Failed to register with discovery service: {e}")
    
    async def _unregister_from_discovery(self):
        """Unregister this server from the discovery service."""
        try:
            if not self.discovery_service:
                return
            
            await self.discovery_service.unregister_agent(self.did_manager.get_local_did())
            logger.info("Unregistered from discovery service")
            
        except Exception as e:
            logger.error(f"Failed to unregister from discovery service: {e}")
    
    # Background Tasks
    
    async def _session_cleanup_loop(self):
        """Background task to clean up expired sessions."""
        cleanup_interval = 60  # 1 minute
        
        while not self._shutdown_event.is_set():
            try:
                current_time = datetime.utcnow()
                expired_sessions = []
                
                # Check HTTP sessions
                for session_id, session in self.sessions.items():
                    if not session.is_active(self.session_timeout):
                        expired_sessions.append(session_id)
                    elif session.state == SessionState.ERROR:
                        error_duration = current_time - session.last_activity
                        if error_duration.total_seconds() > 300:  # 5 minutes in error state
                            expired_sessions.append(session_id)
                
                # Check WebSocket sessions
                for session_id, session in self.websocket_sessions.items():
                    if session.websocket and session.websocket.closed:
                        expired_sessions.append(session_id)
                    elif not session.is_active(self.session_timeout):
                        expired_sessions.append(session_id)
                
                # Clean up expired sessions
                for session_id in expired_sessions:
                    try:
                        await self._cleanup_session(session_id)
                        logger.debug(f"Cleaned up expired session: {session_id}")
                    except Exception as e:
                        logger.warning(f"Error cleaning up session {session_id}: {e}")
                
                if expired_sessions:
                    logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
                
                await asyncio.sleep(cleanup_interval)
                
            except Exception as e:
                logger.error(f"Error in session cleanup loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _heartbeat_loop(self):
        """Background task to send heartbeats to active sessions."""
        heartbeat_interval = 120  # 2 minutes
        
        while not self._shutdown_event.is_set():
            try:
                current_time = datetime.utcnow()
                
                # Send heartbeats to WebSocket sessions
                for session in list(self.websocket_sessions.values()):
                    if session.state == SessionState.ACTIVE and session.websocket:
                        try:
                            # Check if heartbeat is needed
                            time_since_activity = current_time - session.last_activity
                            if time_since_activity.total_seconds() < heartbeat_interval:
                                continue
                            
                            # Send heartbeat
                            heartbeat_msg = ANPMessage(
                                message_id=str(uuid.uuid4()),
                                message_type=MessageType.HEARTBEAT,
                                sender_did=self.did_manager.get_local_did(),
                                recipient_did=session.remote_did,
                                timestamp=current_time,
                                payload={"timestamp": current_time.isoformat()}
                            )
                            
                            await session.websocket.send_text(heartbeat_msg.to_json())
                            session.update_activity()
                            logger.debug(f"Heartbeat sent to session {session.session_id}")
                            
                        except Exception as e:
                            logger.warning(f"Failed to send heartbeat to session {session.session_id}: {e}")
                            # Mark session for cleanup
                            session.state = SessionState.ERROR
                
                await asyncio.sleep(heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _discovery_update_loop(self):
        """Background task to update discovery service registration."""
        update_interval = 300  # 5 minutes
        
        while not self._shutdown_event.is_set():
            try:
                if self.discovery_service:
                    # Update agent status
                    await self.discovery_service.update_agent_status(
                        self.did_manager.get_local_did(),
                        AgentStatus.ACTIVE
                    )
                    logger.debug("Updated discovery service registration")
                
                await asyncio.sleep(update_interval)
                
            except Exception as e:
                logger.error(f"Error in discovery update loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    # Public API for custom handlers
    
    def register_message_handler(
        self,
        message_type: MessageType,
        handler: Callable[[ANPMessage, ClientSession], Optional[ANPMessage]]
    ):
        """Register a custom message handler."""
        self.message_handlers[message_type] = handler
        logger.info(f"Registered custom handler for message type: {message_type}")
    
    def register_middleware(self, middleware: Callable):
        """Register custom middleware."""
        self.middleware_handlers.append(middleware)
        logger.info("Registered custom middleware")
    
    def register_data_processor(
        self,
        data_type: str,
        processor: Callable[[Any, ClientSession], Any]
    ):
        """Register a custom data processor."""
        # Store in a registry for custom data processors
        if not hasattr(self, '_data_processors'):
            self._data_processors = {}
        
        self._data_processors[data_type] = processor
        logger.info(f"Registered custom data processor for type: {data_type}")
    
    # Utility methods
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a session."""
        session = self.sessions.get(session_id) or self.websocket_sessions.get(session_id)
        if not session:
            return None
        
        return {
            "session_id": session.session_id,
            "remote_did": session.remote_did,
            "remote_agent_id": session.remote_agent_id,
            "remote_address": session.remote_address,
            "state": session.state.value,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "authenticated_at": session.authenticated_at.isoformat() if session.authenticated_at else None,
            "is_websocket": session.session_id in self.websocket_sessions,
            "is_active": session.is_active(),
            "negotiated_protocols": [proto.to_dict() for proto in (session.negotiated_protocols or [])]
        }
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        active_sessions = [s for s in self.sessions.values() if s.is_active()]
        active_ws_sessions = [s for s in self.websocket_sessions.values() if s.is_active()]
        
        return {
            "server_state": self.state.value,
            "did": self.did_manager.get_local_did(),
            "host": self.host,
            "port": self.port,
            "ssl_enabled": self.ssl_context is not None,
            "sessions": {
                "total_http": len(self.sessions),
                "active_http": len(active_sessions),
                "total_websocket": len(self.websocket_sessions),
                "active_websocket": len(active_ws_sessions),
                "max_sessions": self.max_sessions
            },
            "features": {
                "websockets": self.enable_websockets,
                "cors": self.enable_cors,
                "discovery": self.discovery_service is not None
            },
            "protocols": self.meta_protocol.get_supported_protocols(),
            "background_tasks": len(self._background_tasks)
        }
    
    def list_active_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions."""
        sessions = []
        
        # HTTP sessions
        for session in self.sessions.values():
            if session.is_active():
                sessions.append(self.get_session_info(session.session_id))
        
        # WebSocket sessions
        for session in self.websocket_sessions.values():
            if session.is_active():
                sessions.append(self.get_session_info(session.session_id))
        
        return [s for s in sessions if s is not None]
    
    async def broadcast_message(
        self,
        message: ANPMessage,
        session_filter: Optional[Callable[[ClientSession], bool]] = None
    ) -> Dict[str, Any]:
        """Broadcast a message to multiple sessions."""
        results = {
            "sent": 0,
            "failed": 0,
            "errors": []
        }
        
        # Get target sessions
        target_sessions = []
        
        # HTTP sessions
        for session in self.sessions.values():
            if session.is_active() and session.state == SessionState.ACTIVE:
                if not session_filter or session_filter(session):
                    target_sessions.append(session)
        
        # WebSocket sessions
        for session in self.websocket_sessions.values():
            if session.is_active() and session.state == SessionState.ACTIVE:
                if not session_filter or session_filter(session):
                    target_sessions.append(session)
        
        # Send to each session
        for session in target_sessions:
            try:
                if session.websocket and not session.websocket.closed:
                    # Send via WebSocket
                    await session.websocket.send_text(message.to_json())
                    results["sent"] += 1
                else:
                    # For HTTP sessions, we can't push messages directly
                    # This would require a different mechanism like Server-Sent Events
                    logger.debug(f"Cannot push message to HTTP session {session.session_id}")
                
            except Exception as e:
                results["failed"] += 1
                results["errors"].append({
                    "session_id": session.session_id,
                    "error": str(e)
                })
                logger.warning(f"Failed to send broadcast message to session {session.session_id}: {e}")
        
        logger.info(f"Broadcast complete: {results['sent']} sent, {results['failed']} failed")
        return results
    
    async def send_message_to_session(
        self,
        session_id: str,
        message: ANPMessage
    ) -> bool:
        """Send a message to a specific session."""
        try:
            # Check WebSocket sessions first
            session = self.websocket_sessions.get(session_id)
            if session and session.websocket and not session.websocket.closed:
                await session.websocket.send_text(message.to_json())
                session.update_activity()
                return True
            
            # HTTP sessions can't receive pushed messages directly
            session = self.sessions.get(session_id)
            if session:
                logger.warning(f"Cannot push message to HTTP session {session_id}")
                return False
            
            logger.warning(f"Session not found: {session_id}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to send message to session {session_id}: {e}")
            return False


class ANPServerManager:
    """Manager for multiple ANP servers with different configurations."""
    
    def __init__(self):
        """Initialize server manager."""
        self.servers: Dict[str, ANPServer] = {}
        self.default_server: Optional[ANPServer] = None
    
    async def create_server(
        self,
        did_manager: DIDManager,
        discovery_service: Optional[DiscoveryService] = None,
        config: Optional[Dict[str, Any]] = None,
        server_id: Optional[str] = None,
        set_as_default: bool = False
    ) -> ANPServer:
        """Create a new ANP server."""
        if not server_id:
            server_id = did_manager.get_local_did()
        
        if server_id in self.servers:
            raise ServerConfigurationError(f"Server with ID {server_id} already exists")
        
        server = ANPServer(
            did_manager=did_manager,
            discovery_service=discovery_service,
            config=config
        )
        
        await server.start()
        
        self.servers[server_id] = server
        
        if set_as_default or not self.default_server:
            self.default_server = server
        
        logger.info(f"Created ANP server: {server_id}")
        return server
    
    async def remove_server(self, server_id: str) -> bool:
        """Remove an ANP server."""
        server = self.servers.pop(server_id, None)
        if server:
            await server.stop()
            
            if self.default_server == server:
                self.default_server = next(iter(self.servers.values()), None)
            
            logger.info(f"Removed ANP server: {server_id}")
            return True
        
        return False
    
    def get_server(self, server_id: Optional[str] = None) -> Optional[ANPServer]:
        """Get an ANP server by ID or return default."""
        if server_id:
            return self.servers.get(server_id)
        return self.default_server
    
    def list_servers(self) -> List[str]:
        """List all server IDs."""
        return list(self.servers.keys())
    
    async def shutdown_all(self):
        """Shutdown all servers."""
        for server in list(self.servers.values()):
            await server.stop()
        
        self.servers.clear()
        self.default_server = None
        logger.info("All ANP servers shut down")
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        total_sessions = sum(
            len(server.sessions) + len(server.websocket_sessions)
            for server in self.servers.values()
        )
        
        return {
            "total_servers": len(self.servers),
            "running_servers": len([s for s in self.servers.values() if s.state == ServerState.RUNNING]),
            "total_sessions": total_sessions,
            "servers": {
                server_id: server.get_server_stats()
                for server_id, server in self.servers.items()
            }
        }


# Utility functions and helpers

async def create_anp_server(
    private_key: Optional[str] = None,
    did: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 8080,
    discovery_service: Optional[DiscoveryService] = None,
    config: Optional[Dict[str, Any]] = None
) -> ANPServer:
    """
    Convenience function to create and start an ANP server.
    
    Args:
        private_key: Private key for DID (generates new if None)
        did: Existing DID (generates new if None)
        host: Server host address
        port: Server port
        discovery_service: Discovery service instance
        config: Server configuration
        
    Returns:
        Started ANP server
    """
    # Create DID manager
    if private_key and did:
        did_manager = DIDManager.from_existing(private_key, did)
    else:
        did_manager = await DIDManager.create_new()
    
    # Merge configuration
    server_config = {
        "host": host,
        "port": port
    }
    if config:
        server_config.update(config)
    
    # Create and start server
    server = ANPServer(
        did_manager=did_manager,
        discovery_service=discovery_service,
        config=server_config
    )
    
    await server.start()
    return server


class ANPServerCluster:
    """Cluster of ANP servers for high availability and load balancing."""
    
    def __init__(self, load_balancer_config: Optional[Dict[str, Any]] = None):
        """Initialize server cluster."""
        self.servers: List[ANPServer] = []
        self.load_balancer_config = load_balancer_config or {}
        self.current_index = 0
        self._lock = asyncio.Lock()
    
    async def add_server(self, server: ANPServer):
        """Add a server to the cluster."""
        async with self._lock:
            self.servers.append(server)
            logger.info(f"Added server to cluster (total: {len(self.servers)})")
    
    async def remove_server(self, server: ANPServer):
        """Remove a server from the cluster."""
        async with self._lock:
            if server in self.servers:
                self.servers.remove(server)
                await server.stop()
                logger.info(f"Removed server from cluster (total: {len(self.servers)})")
    
    def get_next_server(self) -> Optional[ANPServer]:
        """Get next available server using round-robin."""
        if not self.servers:
            return None
        
        # Simple round-robin for now
        server = self.servers[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.servers)
        
        # Check if server is healthy
        if server.state == ServerState.RUNNING:
            return server
        
        # Find next healthy server
        for _ in range(len(self.servers)):
            server = self.servers[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.servers)
            if server.state == ServerState.RUNNING:
                return server
        
        return None  # No healthy servers
    
    async def shutdown_all(self):
        """Shutdown all servers in the cluster."""
        async with self._lock:
            for server in self.servers:
                await server.stop()
            self.servers.clear()
            logger.info("Server cluster shut down")
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get cluster statistics."""
        running_servers = [s for s in self.servers if s.state == ServerState.RUNNING]
        total_sessions = sum(
            len(s.sessions) + len(s.websocket_sessions)
            for s in self.servers
        )
        
        return {
            "total_servers": len(self.servers),
            "running_servers": len(running_servers),
            "total_sessions": total_sessions,
            "average_sessions_per_server": total_sessions / max(len(self.servers), 1),
            "load_balancer": self.load_balancer_config
        }


# Configuration helpers

DEFAULT_SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": 8080,
    "max_sessions": 1000,
    "session_timeout": 3600,
    "enable_websockets": True,
    "enable_cors": True,
    "ssl": {
        "enabled": False,
        "cert_file": None,
        "key_file": None
    },
    "agent_name": "ANP Server",
    "agent_description": "Agent Network Protocol Server"
}


def create_server_config(**overrides) -> Dict[str, Any]:
    """Create server configuration with overrides."""
    config = DEFAULT_SERVER_CONFIG.copy()
    config.update(overrides)
    return config


# Example usage and testing

async def example_server_usage():
    """Example of how to use the ANP server."""
    
    # Create server
    server = await create_anp_server(
        host="localhost",
        port=8080,
        config={
            "agent_name": "Example ANP Server",
            "agent_description": "Example server for testing ANP protocol"
        }
    )
    
    # Register custom message handler
    async def custom_data_handler(message: ANPMessage, session: ClientSession) -> Optional[ANPMessage]:
        """Custom handler for data messages."""
        data = message.payload.get("data")
        
        # Process the data
        result = f"Processed: {data}"
        
        # Create response
        response_payload = {
            "result": result,
            "processed_at": datetime.utcnow().isoformat()
        }
        
        return server._create_response_message(message, response_payload)
    
    server.register_message_handler(MessageType.DATA, custom_data_handler)
    
    try:
        # Server is now running and handling requests
        logger.info("Server is running. Press Ctrl+C to stop.")
        
        # Keep server running
        while server.state == ServerState.RUNNING:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    finally:
        # Clean up
        await server.stop()


if __name__ == "__main__":
    # Run example
    asyncio.run(example_server_usage())


# Module exports
__all__ = [
    # Core classes
    'ANPServer',
    'ANPServerManager',
    'ANPServerCluster',
    
    # Session classes
    'ClientSession',
    'SessionState',
    
    # State enums
    'ServerState',
    
    # Exceptions
    'ANPServerError',
    'ServerConfigurationError',
    'SessionError',
    'AuthenticationError',
    'ProtocolError',
    
    # Utility functions
    'create_anp_server',
    'create_server_config',
    
    # Configuration
    'DEFAULT_SERVER_CONFIG'
]

# Module metadata
__version__ = "1.0.0"
__author__ = "MKT Communication Team"
__license__ = "MIT"

logger.info(f"ANP Server module loaded (version {__version__})")
