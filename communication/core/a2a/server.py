"""
A2A Server implementation.

Handles inbound communication from other A2A agents, including
agent card serving, task management, and message processing.
"""

import json
import uuid
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import asdict
import traceback

from aiohttp import web, WSMsgType
from aiohttp.web_request import Request
from aiohttp.web_response import Response, StreamResponse
import aiohttp_cors

from .agent_card import AgentCard
from .message import A2AMessage, MessageBuilder
from .task import Task, TaskManager, TaskStatus, TaskPriority, TaskResult
from .streaming import StreamingHandler

import logging

logger = logging.getLogger(__name__)


class A2AServerError(Exception):
    """Base exception for A2A server errors."""
    pass


class InvalidRequestError(A2AServerError):
    """Raised when request is invalid."""
    pass


class AuthorizationError(A2AServerError):
    """Raised when authorization fails."""
    pass


class A2AServer:
    """
    A2A Server for handling inbound agent communication.
    
    Implements Google A2A specification server-side functionality
    including JSON-RPC 2.0 endpoints and WebSocket streaming.
    """
    
    def __init__(
        self,
        agent_card: AgentCard,
        host: str = "0.0.0.0",
        port: int = 8080,
        enable_cors: bool = True,
        auth_handler: Optional[Callable] = None
    ):
        """Initialize A2A server."""
        self.agent_card = agent_card
        self.host = host
        self.port = port
        self.enable_cors = enable_cors
        self.auth_handler = auth_handler
        
        # Core components
        self.task_manager = TaskManager()
        self.streaming_handler = StreamingHandler()
        self.message_handlers: Dict[str, Callable] = {}
        
        # Web application
        self.app = web.Application()
        self.setup_routes()
        
        if enable_cors:
            self.setup_cors()
        
        logger.info(f"A2A Server initialized for agent {agent_card.agent_id}")
    
    def setup_routes(self) -> None:
        """Setup HTTP routes."""
        # Agent card endpoint
        self.app.router.add_get("/.well-known/agent-card", self.get_agent_card)
        
        # Task management endpoints
        self.app.router.add_post("/tasks", self.create_task_endpoint)
        self.app.router.add_get("/tasks/{task_id}", self.get_task_status_endpoint)
        self.app.router.add_post("/tasks/{task_id}/cancel", self.cancel_task_endpoint)
        
        # Message endpoints
        self.app.router.add_post("/messages", self.send_message_endpoint)
        
        # Streaming endpoints
        self.app.router.add_get("/stream/{task_id}", self.stream_task_updates)
        self.app.router.add_get("/ws", self.websocket_handler)
        
        # Health check
        self.app.router.add_get("/health", self.health_check)
        
        # API documentation
        self.app.router.add_get("/api/docs", self.api_documentation)
    
    def setup_cors(self) -> None:
        """Setup CORS configuration."""
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        # Add CORS to all routes
        for route in list(self.app.router.routes()):
            cors.add(route)
    
    async def authenticate_request(self, request: Request) -> Optional[str]:
        """Authenticate incoming request."""
        if not self.auth_handler:
            return None  # No authentication required
        
        try:
            auth_header = request.headers.get("Authorization")
            if not auth_header:
                raise AuthorizationError("Missing Authorization header")
            
            # Extract token from "Bearer <token>" format
            if not auth_header.startswith("Bearer "):
                raise AuthorizationError("Invalid Authorization header format")
            
            token = auth_header[7:]  # Remove "Bearer " prefix
            
            # Call custom auth handler
            agent_id = await self.auth_handler(token)
            if not agent_id:
                raise AuthorizationError("Invalid or expired token")
            
            return agent_id
            
        except AuthorizationError:
            raise
        except Exception as e:
            raise AuthorizationError(f"Authentication failed: {str(e)}")
    
    def create_json_rpc_response(self, result: Any, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Create JSON-RPC 2.0 response."""
        return {
            "jsonrpc": "2.0",
            "result": result,
            "id": request_id
        }
    
    def create_json_rpc_error(self, code: int, message: str, request_id: Optional[str] = None, data: Any = None) -> Dict[str, Any]:
        """Create JSON-RPC 2.0 error response."""
        error = {
            "code": code,
            "message": message
        }
        if data is not None:
            error["data"] = data
        
        return {
            "jsonrpc": "2.0",
            "error": error,
            "id": request_id
        }
    
    async def get_agent_card(self, request: Request) -> Response:
        """Serve agent card at well-known endpoint."""
        try:
            logger.debug("Serving agent card")
            return web.json_response(self.agent_card.to_dict())
        except Exception as e:
            logger.error(f"Error serving agent card: {str(e)}")
            return web.json_response(
                {"error": "Internal server error"},
                status=500
            )
    
    async def create_task_endpoint(self, request: Request) -> Response:
        """Handle task creation requests."""
        try:
            # Authenticate request
            requester_id = await self.authenticate_request(request)
            
            # Parse JSON-RPC request
            data = await request.json()
            
            if data.get("jsonrpc") != "2.0":
                raise InvalidRequestError("Invalid JSON-RPC version")
            
            if data.get("method") != "create_task":
                raise InvalidRequestError("Invalid method")
            
            params = data.get("params", {})
            request_id = data.get("id")
            
            # Extract task parameters
            task_type = params.get("task_type")
            if not task_type:
                raise InvalidRequestError("task_type is required")
            
            input_data = params.get("input_data")
            priority_str = params.get("priority", "normal")
            timeout_seconds = params.get("timeout")
            
            # Convert priority
            try:
                priority = TaskPriority(priority_str)
            except ValueError:
                raise InvalidRequestError(f"Invalid priority: {priority_str}")
            
            # Convert timeout
            timeout = None
            if timeout_seconds:
                from datetime import timedelta
                timeout = timedelta(seconds=timeout_seconds)
            
            # Create task
            task = self.task_manager.create_task(
                agent_id=self.agent_card.agent_id,
                task_type=task_type,
                input_data=input_data,
                priority=priority,
                timeout=timeout
            )
            
            # Start task execution asynchronously
            asyncio.create_task(self._execute_task_async(task.task_id))
            
            # Return task information
            response_data = self.create_json_rpc_response(task.to_dict(), request_id)
            
            logger.info(f"Created task {task.task_id} for requester {requester_id}")
            return web.json_response(response_data)
            
        except AuthorizationError as e:
            logger.warning(f"Authorization failed: {str(e)}")
            return web.json_response(
                self.create_json_rpc_error(-32001, str(e), data.get("id") if 'data' in locals() else None),
                status=401
            )
        except InvalidRequestError as e:
            logger.warning(f"Invalid request: {str(e)}")
            return web.json_response(
                self.create_json_rpc_error(-32602, str(e), data.get("id") if 'data' in locals() else None),
                status=400
            )
        except Exception as e:
            logger.error(f"Error creating task: {str(e)}\n{traceback.format_exc()}")
            return web.json_response(
                self.create_json_rpc_error(-32603, "Internal error", data.get("id") if 'data' in locals() else None),
                status=500
            )
    
    async def get_task_status_endpoint(self, request: Request) -> Response:
        """Handle task status requests."""
        try:
            # Authenticate request
            await self.authenticate_request(request)
            
            task_id = request.match_info["task_id"]
            task = self.task_manager.get_task(task_id)
            
            if not task:
                return web.json_response(
                    {"error": "Task not found"},
                    status=404
                )
            
            logger.debug(f"Retrieved status for task {task_id}")
            return web.json_response(task.to_dict())
            
        except AuthorizationError as e:
            return web.json_response({"error": str(e)}, status=401)
        except Exception as e:
            logger.error(f"Error getting task status: {str(e)}")
            return web.json_response(
                {"error": "Internal server error"},
                status=500
            )
    
    async def cancel_task_endpoint(self, request: Request) -> Response:
        """Handle task cancellation requests."""
        try:
            # Authenticate request
            requester_id = await self.authenticate_request(request)
            
            # Parse JSON-RPC request
            data = await request.json()
            
            if data.get("jsonrpc") != "2.0":
                raise InvalidRequestError("Invalid JSON-RPC version")
            
            if data.get("method") != "cancel_task":
                raise InvalidRequestError("Invalid method")
            
            params = data.get("params", {})
            request_id = data.get("id")
            task_id = request.match_info["task_id"]
            
            # Verify task exists
            task = self.task_manager.get_task(task_id)
            if not task:
                return web.json_response(
                    self.create_json_rpc_error(-32001, "Task not found", request_id),
                    status=404
                )
            
            # Cancel the task
            success = self.task_manager.cancel_task(task_id)
            
            # Notify streaming clients
            await self.streaming_handler.broadcast_task_update(task_id, {
                "status": "cancelled",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            response_data = self.create_json_rpc_response(
                {"cancelled": success, "task_id": task_id},
                request_id
            )
            
            logger.info(f"Task {task_id} cancellation {'successful' if success else 'failed'} by {requester_id}")
            return web.json_response(response_data)
            
        except AuthorizationError as e:
            return web.json_response(
                self.create_json_rpc_error(-32001, str(e), data.get("id") if 'data' in locals() else None),
                status=401
            )
        except InvalidRequestError as e:
            return web.json_response(
                self.create_json_rpc_error(-32602, str(e), data.get("id") if 'data' in locals() else None),
                status=400
            )
        except Exception as e:
            logger.error(f"Error cancelling task: {str(e)}")
            return web.json_response(
                self.create_json_rpc_error(-32603, "Internal error", data.get("id") if 'data' in locals() else None),
                status=500
            )
    
    async def send_message_endpoint(self, request: Request) -> Response:
        """Handle message sending requests."""
        try:
            # Authenticate request
            sender_id = await self.authenticate_request(request)
            
            # Parse JSON-RPC request
            data = await request.json()
            
            if data.get("jsonrpc") != "2.0":
                raise InvalidRequestError("Invalid JSON-RPC version")
            
            if data.get("method") != "send_message":
                raise InvalidRequestError("Invalid method")
            
            params = data.get("params", {})
            request_id = data.get("id")
            
            # Extract message data
            message_data = params.get("message")
            if not message_data:
                raise InvalidRequestError("message is required")
            
            task_id = params.get("task_id")
            
            # Parse message
            message = A2AMessage.from_dict(message_data)
            
            # Validate message
            errors = message.validate()
            if errors:
                raise InvalidRequestError(f"Invalid message: {', '.join(errors)}")
            
            # Process message
            response_message = await self._process_message(message, task_id, sender_id)
            
            # Return response
            response_data = self.create_json_rpc_response(response_message.to_dict(), request_id)
            
            logger.info(f"Processed message {message.message_id} from {sender_id}")
            return web.json_response(response_data)
            
        except AuthorizationError as e:
            return web.json_response(
                self.create_json_rpc_error(-32001, str(e), data.get("id") if 'data' in locals() else None),
                status=401
            )
        except InvalidRequestError as e:
            return web.json_response(
                self.create_json_rpc_error(-32602, str(e), data.get("id") if 'data' in locals() else None),
                status=400
            )
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return web.json_response(
                self.create_json_rpc_error(-32603, "Internal error", data.get("id") if 'data' in locals() else None),
                status=500
            )
    
    async def stream_task_updates(self, request: Request) -> StreamResponse:
        """Handle Server-Sent Events for task updates."""
        try:
            # Authenticate request
            await self.authenticate_request(request)
            
            task_id = request.match_info["task_id"]
            
            # Verify task exists
            task = self.task_manager.get_task(task_id)
            if not task:
                return web.Response(text="Task not found", status=404)
            
            # Create SSE response
            response = web.StreamResponse(
                status=200,
                reason='OK',
                headers={
                    'Content-Type': 'text/event-stream',
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'Access-Control-Allow-Origin': '*',
                }
            )
            
            await response.prepare(request)
            
            # Register client for updates
            client_id = str(uuid.uuid4())
            await self.streaming_handler.register_client(task_id, client_id, response)
            
            try:
                # Send initial task status
                await self.streaming_handler.send_event(
                    response,
                    "task_status",
                    task.to_dict()
                )
                
                # Keep connection alive until client disconnects
                while not response.transport.is_closing():
                    await asyncio.sleep(1)
                    
                    # Send heartbeat
                    await self.streaming_handler.send_event(
                        response,
                        "heartbeat",
                        {"timestamp": datetime.utcnow().isoformat()}
                    )
                    
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error in SSE stream: {str(e)}")
            finally:
                # Unregister client
                await self.streaming_handler.unregister_client(task_id, client_id)
            
            return response
            
        except AuthorizationError as e:
            return web.Response(text=str(e), status=401)
        except Exception as e:
            logger.error(f"Error setting up SSE stream: {str(e)}")
            return web.Response(text="Internal server error", status=500)
    
    async def websocket_handler(self, request: Request) -> web.WebSocketResponse:
        """Handle WebSocket connections for real-time communication."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        client_id = str(uuid.uuid4())
        logger.info(f"WebSocket client {client_id} connected")
        
        try:
            # Authenticate WebSocket connection
            auth_message = await ws.receive()
            if auth_message.type != WSMsgType.TEXT:
                await ws.close(code=4000, message=b"Authentication required")
                return ws
            
            auth_data = json.loads(auth_message.data)
            token = auth_data.get("token")
            
            if self.auth_handler:
                agent_id = await self.auth_handler(token)
                if not agent_id:
                    await ws.close(code=4001, message=b"Authentication failed")
                    return ws
            
            # Send authentication success
            await ws.send_str(json.dumps({
                "type": "auth_success",
                "client_id": client_id
            }))
            
            # Handle messages
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self._handle_websocket_message(ws, client_id, data)
                    except json.JSONDecodeError:
                        await ws.send_str(json.dumps({
                            "type": "error",
                            "message": "Invalid JSON"
                        }))
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")
                    break
                
        except Exception as e:
            logger.error(f"WebSocket error for client {client_id}: {str(e)}")
        finally:
            logger.info(f"WebSocket client {client_id} disconnected")
        
        return ws
    
    async def health_check(self, request: Request) -> Response:
        """Health check endpoint."""
        stats = self.task_manager.get_statistics()
        health_data = {
            "status": "healthy",
            "agent_id": self.agent_card.agent_id,
            "timestamp": datetime.utcnow().isoformat(),
            "tasks": stats,
            "streaming_clients": self.streaming_handler.get_client_count()
        }
        
        return web.json_response(health_data)
    
    async def api_documentation(self, request: Request) -> Response:
        """Serve API documentation."""
        docs = {
            "agent_id": self.agent_card.agent_id,
            "name": self.agent_card.name,
            "version": self.agent_card.version,
            "endpoints": {
                "agent_card": "GET /.well-known/agent-card",
                "create_task": "POST /tasks",
                "get_task": "GET /tasks/{task_id}",
                "cancel_task": "POST /tasks/{task_id}/cancel",
                "send_message": "POST /messages",
                "stream_updates": "GET /stream/{task_id}",
                "websocket": "GET /ws",
                "health": "GET /health"
            },
            "supported_task_types": list(self.task_manager.task_handlers.keys()),
            "message_handlers": list(self.message_handlers.keys())
        }
        
        return web.json_response(docs)
    
    async def _execute_task_async(self, task_id: str) -> None:
        """Execute a task asynchronously and broadcast updates."""
        try:
            # Get the task
            task = self.task_manager.get_task(task_id)
            if not task:
                logger.error(f"Task {task_id} not found for execution")
                return
            
            # Broadcast task started
            await self.streaming_handler.broadcast_task_update(task_id, {
                "status": "running",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Execute the task
            result = await self.task_manager.execute_task(task_id)
            
            # Broadcast completion
            await self.streaming_handler.broadcast_task_update(task_id, {
                "status": task.status.value,
                "result": result.to_dict() if result else None,
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error executing task {task_id}: {str(e)}")
            
            # Broadcast failure
            await self.streaming_handler.broadcast_task_update(task_id, {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
    
    async def _process_message(self, message: A2AMessage, task_id: Optional[str], sender_id: str) -> A2AMessage:
        """Process an incoming message and generate response."""
        try:
            # Check if we have a handler for this message type
            message_type = message.metadata.get("type", "default")
            
            if message_type in self.message_handlers:
                handler = self.message_handlers[message_type]
                response_data = await handler(message, task_id, sender_id)
            else:
                # Default response
                response_data = {
                    "received": True,
                    "message_id": message.message_id,
                    "processed_at": datetime.utcnow().isoformat()
                }
            
            # Create response message
            response_builder = MessageBuilder(role="agent", task_id=task_id)
            response_builder.add_data(response_data)
            response_builder.set_parent(message.message_id)
            
            return response_builder.build()
            
        except Exception as e:
            logger.error(f"Error processing message {message.message_id}: {str(e)}")
            
            # Create error response
            error_builder = MessageBuilder(role="agent", task_id=task_id)
            error_builder.add_text(f"Error processing message: {str(e)}")
            error_builder.set_parent(message.message_id)
            
            return error_builder.build()
    
    async def _handle_websocket_message(self, ws: web.WebSocketResponse, client_id: str, data: Dict[str, Any]) -> None:
        """Handle incoming WebSocket message."""
        try:
            message_type = data.get("type")
            
            if message_type == "subscribe_task":
                task_id = data.get("task_id")
                if task_id:
                    await self.streaming_handler.subscribe_client_to_task(client_id, task_id, ws)
                    await ws.send_str(json.dumps({
                        "type": "subscribed",
                        "task_id": task_id
                    }))
            
            elif message_type == "unsubscribe_task":
                task_id = data.get("task_id")
                if task_id:
                    await self.streaming_handler.unsubscribe_client_from_task(client_id, task_id)
                    await ws.send_str(json.dumps({
                        "type": "unsubscribed",
                        "task_id": task_id
                    }))
            
            elif message_type == "ping":
                await ws.send_str(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                }))
            
            else:
                await ws.send_str(json.dumps({
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                }))
                
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {str(e)}")
            await ws.send_str(json.dumps({
                "type": "error",
                "message": "Internal server error"
            }))
    
    def register_task_handler(self, task_type: str, handler: Callable) -> None:
        """Register a handler for a specific task type."""
        self.task_manager.register_handler(task_type, handler)
        logger.info(f"Registered task handler for type: {task_type}")
    
    def register_message_handler(self, message_type: str, handler: Callable) -> None:
        """Register a handler for a specific message type."""
        self.message_handlers[message_type] = handler
        logger.info(f"Registered message handler for type: {message_type}")
    
    async def start(self) -> None:
        """Start the A2A server."""
        try:
            runner = web.AppRunner(self.app)
            await runner.setup()
            
            site = web.TCPSite(runner, self.host, self.port)
            await site.start()
            
            logger.info(f"A2A Server started on {self.host}:{self.port}")
            logger.info(f"Agent card available at: http://{self.host}:{self.port}/.well-known/agent-card")
            
        except Exception as e:
            logger.error(f"Failed to start A2A server: {str(e)}")
            raise
    
    async def stop(self) -> None:
        """Stop the A2A server."""
        try:
            await self.app.cleanup()
            logger.info("A2A Server stopped")
        except Exception as e:
            logger.error(f"Error stopping A2A server: {str(e)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            "agent_id": self.agent_card.agent_id,
            "host": self.host,
            "port": self.port,
            "task_manager": self.task_manager.get_statistics(),
            "streaming": self.streaming_handler.get_statistics(),
            "registered_handlers": {
                "tasks": list(self.task_manager.task_handlers.keys()),
                "messages": list(self.message_handlers.keys())
            }
        }


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    from .agent_card import AgentCard, Endpoint, Capability
    
    async def sample_task_handler(input_data):
        """Sample task handler for testing."""
        await asyncio.sleep(2)  # Simulate work
        return {"processed": True, "input": input_data}
    
    async def sample_message_handler(message, task_id, sender_id):
        """Sample message handler for testing."""
        return {
            "echo": message.get_text_content(),
            "sender": sender_id,
            "task_id": task_id
        }
    
    async def test_server():
        """Test A2A server functionality."""
        # Create agent card
        agent_card = AgentCard(
            agent_id="test-server-agent",
            name="Test A2A Server",
            version="1.0.0",
            description="Test server for A2A protocol",
            endpoints=[
                Endpoint(
                    url="http://localhost:8080",
                    protocol="a2a",
                    methods=["POST", "GET"]
                )
            ],
            capabilities=[
                Capability(
                    name="text_processing",
                    description="Process text input",
                    input_schema={"type": "object"},
                    output_schema={"type": "object"}
                )
            ]
        )
        
        # Create and configure server
        server = A2AServer(agent_card, port=8080)
        server.register_task_handler("text_processing", sample_task_handler)
        server.register_message_handler("echo", sample_message_handler)
        
        try:
            # Start server
            await server.start()
            
            # Keep server running
            print("Server running... Press Ctrl+C to stop")
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            print("Stopping server...")
        finally:
            await server.stop()
    
    # Run the test server
    # asyncio.run(test_server())
