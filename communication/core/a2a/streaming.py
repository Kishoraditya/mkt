"""
A2A Streaming implementation.

Handles Server-Sent Events (SSE) and WebSocket streaming for real-time
communication and task updates in the A2A protocol.
"""

import json
import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict

from aiohttp import web, WSMsgType
from aiohttp.web_response import StreamResponse

import logging

logger = logging.getLogger(__name__)


@dataclass
class StreamClient:
    """Represents a streaming client connection."""
    client_id: str
    connection_type: str  # 'sse' or 'websocket'
    connection: Any  # StreamResponse or WebSocketResponse
    subscribed_tasks: Set[str]
    connected_at: datetime
    last_activity: datetime
    
    def __post_init__(self):
        if not self.subscribed_tasks:
            self.subscribed_tasks = set()


class StreamingError(Exception):
    """Base exception for streaming errors."""
    pass


class ClientNotFoundError(StreamingError):
    """Raised when a client is not found."""
    pass


class StreamingHandler:
    """
    Handles real-time streaming for A2A protocol.
    
    Supports both Server-Sent Events (SSE) and WebSocket connections
    for broadcasting task updates and real-time communication.
    """
    
    def __init__(self, heartbeat_interval: float = 30.0, cleanup_interval: float = 300.0):
        """Initialize streaming handler."""
        self.heartbeat_interval = heartbeat_interval
        self.cleanup_interval = cleanup_interval
        
        # Client management
        self.clients: Dict[str, StreamClient] = {}
        self.task_subscribers: Dict[str, Set[str]] = defaultdict(set)  # task_id -> client_ids
        
        # Background tasks
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.total_connections = 0
        self.total_messages_sent = 0
        self.total_errors = 0
        
        logger.info("StreamingHandler initialized")
    
    async def start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        if not self.heartbeat_task:
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        if not self.cleanup_task:
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Background streaming tasks started")
    
    async def stop_background_tasks(self) -> None:
        """Stop background maintenance tasks."""
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
            self.heartbeat_task = None
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
            self.cleanup_task = None
        
        logger.info("Background streaming tasks stopped")
    
    async def register_client(
        self,
        task_id: str,
        client_id: str,
        connection: Any,
        connection_type: str = "sse"
    ) -> None:
        """Register a new streaming client."""
        try:
            now = datetime.utcnow()
            
            client = StreamClient(
                client_id=client_id,
                connection_type=connection_type,
                connection=connection,
                subscribed_tasks={task_id},
                connected_at=now,
                last_activity=now
            )
            
            self.clients[client_id] = client
            self.task_subscribers[task_id].add(client_id)
            self.total_connections += 1
            
            logger.info(f"Registered {connection_type} client {client_id} for task {task_id}")
            
        except Exception as e:
            logger.error(f"Error registering client {client_id}: {str(e)}")
            raise StreamingError(f"Failed to register client: {str(e)}")
    
    async def unregister_client(self, task_id: str, client_id: str) -> None:
        """Unregister a streaming client."""
        try:
            if client_id in self.clients:
                client = self.clients[client_id]
                
                # Remove from task subscriptions
                for subscribed_task in client.subscribed_tasks:
                    self.task_subscribers[subscribed_task].discard(client_id)
                
                # Clean up empty task subscriber sets
                self.task_subscribers = {
                    task: clients for task, clients in self.task_subscribers.items()
                    if clients
                }
                
                # Remove client
                del self.clients[client_id]
                
                logger.info(f"Unregistered client {client_id}")
            
        except Exception as e:
            logger.error(f"Error unregistering client {client_id}: {str(e)}")
    
    async def subscribe_client_to_task(self, client_id: str, task_id: str, connection: Any = None) -> None:
        """Subscribe an existing client to a task."""
        try:
            if client_id not in self.clients:
                if connection:
                    # Register new WebSocket client
                    await self.register_client(task_id, client_id, connection, "websocket")
                else:
                    raise ClientNotFoundError(f"Client {client_id} not found")
            else:
                client = self.clients[client_id]
                client.subscribed_tasks.add(task_id)
                self.task_subscribers[task_id].add(client_id)
                client.last_activity = datetime.utcnow()
                
                logger.debug(f"Client {client_id} subscribed to task {task_id}")
            
        except Exception as e:
            logger.error(f"Error subscribing client {client_id} to task {task_id}: {str(e)}")
            raise StreamingError(f"Failed to subscribe client: {str(e)}")
    
    async def unsubscribe_client_from_task(self, client_id: str, task_id: str) -> None:
        """Unsubscribe a client from a task."""
        try:
            if client_id in self.clients:
                client = self.clients[client_id]
                client.subscribed_tasks.discard(task_id)
                self.task_subscribers[task_id].discard(client_id)
                client.last_activity = datetime.utcnow()
                
                logger.debug(f"Client {client_id} unsubscribed from task {task_id}")
            
        except Exception as e:
            logger.error(f"Error unsubscribing client {client_id} from task {task_id}: {str(e)}")
    
    async def send_event(self, response: StreamResponse, event_type: str, data: Any) -> None:
        """Send a Server-Sent Event."""
        try:
            event_data = json.dumps(data)
            
            # Format SSE message
            message = f"event: {event_type}\n"
            message += f"data: {event_data}\n"
            message += f"id: {uuid.uuid4()}\n\n"
            
            await response.write(message.encode('utf-8'))
            self.total_messages_sent += 1
            
        except Exception as e:
            logger.error(f"Error sending SSE event: {str(e)}")
            self.total_errors += 1
            raise StreamingError(f"Failed to send event: {str(e)}")
    
    async def send_websocket_message(self, ws: web.WebSocketResponse, message_type: str, data: Any) -> None:
        """Send a WebSocket message."""
        try:
            message = {
                "type": message_type,
                "data": data,
                "timestamp": datetime.utcnow().isoformat(),
                "id": str(uuid.uuid4())
            }
            
            await ws.send_str(json.dumps(message))
            self.total_messages_sent += 1
            
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {str(e)}")
            self.total_errors += 1
            raise StreamingError(f"Failed to send WebSocket message: {str(e)}")
    
    async def broadcast_task_update(self, task_id: str, update_data: Dict[str, Any]) -> None:
        """Broadcast a task update to all subscribed clients."""
        try:
            client_ids = self.task_subscribers.get(task_id, set())
            
            if not client_ids:
                logger.debug(f"No clients subscribed to task {task_id}")
                return
            
            # Prepare update message
            message_data = {
                "task_id": task_id,
                "update": update_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Send to all subscribed clients
            failed_clients = []
            
            for client_id in client_ids:
                if client_id not in self.clients:
                    failed_clients.append(client_id)
                    continue
                
                client = self.clients[client_id]
                
                try:
                    if client.connection_type == "sse":
                        await self.send_event(client.connection, "task_update", message_data)
                    elif client.connection_type == "websocket":
                        await self.send_websocket_message(client.connection, "task_update", message_data)
                    
                    client.last_activity = datetime.utcnow()
                    
                except Exception as e:
                    logger.warning(f"Failed to send update to client {client_id}: {str(e)}")
                    failed_clients.append(client_id)
            
            # Clean up failed clients
            for client_id in failed_clients:
                await self.unregister_client(task_id, client_id)
            
            successful_sends = len(client_ids) - len(failed_clients)
            logger.debug(f"Broadcast task {task_id} update to {successful_sends} clients")
            
        except Exception as e:
            logger.error(f"Error broadcasting task update: {str(e)}")
            self.total_errors += 1
    
    async def broadcast_global_message(self, message_type: str, data: Dict[str, Any]) -> None:
        """Broadcast a message to all connected clients."""
        try:
            if not self.clients:
                logger.debug("No clients connected for global broadcast")
                return
            
            message_data = {
                "type": message_type,
                "data": data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            failed_clients = []
            
            for client_id, client in self.clients.items():
                try:
                    if client.connection_type == "sse":
                        await self.send_event(client.connection, message_type, message_data)
                    elif client.connection_type == "websocket":
                        await self.send_websocket_message(client.connection, message_type, message_data)
                    
                    client.last_activity = datetime.utcnow()
                    
                except Exception as e:
                    logger.warning(f"Failed to send global message to client {client_id}: {str(e)}")
                    failed_clients.append(client_id)
            
            # Clean up failed clients
            for client_id in failed_clients:
                for task_id in list(self.task_subscribers.keys()):
                    await self.unregister_client(task_id, client_id)
            
            successful_sends = len(self.clients) - len(failed_clients)
            logger.debug(f"Global broadcast sent to {successful_sends} clients")
            
        except Exception as e:
            logger.error(f"Error broadcasting global message: {str(e)}")
            self.total_errors += 1
    
    async def _heartbeat_loop(self) -> None:
        """Background task to send heartbeat messages."""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                if self.clients:
                    await self.broadcast_global_message("heartbeat", {
                        "server_time": datetime.utcnow().isoformat(),
                        "connected_clients": len(self.clients)
                    })
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {str(e)}")
                await asyncio.sleep(5)  # Brief pause before retrying
    
    async def _cleanup_loop(self) -> None:
        """Background task to clean up stale connections."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_stale_connections()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {str(e)}")
                await asyncio.sleep(30)  # Brief pause before retrying
    
    async def _cleanup_stale_connections(self) -> None:
        """Clean up stale or disconnected clients."""
        try:
            now = datetime.utcnow()
            stale_clients = []
            
            for client_id, client in self.clients.items():
                # Check if connection is still alive
                try:
                    if client.connection_type == "sse":
                        # For SSE, check if transport is closing
                        if hasattr(client.connection, 'transport') and client.connection.transport.is_closing():
                            stale_clients.append(client_id)
                    elif client.connection_type == "websocket":
                        # For WebSocket, check if connection is closed
                        if client.connection.closed:
                            stale_clients.append(client_id)
                    
                    # Check for inactive connections (no activity for 10 minutes)
                    inactive_duration = (now - client.last_activity).total_seconds()
                    if inactive_duration > 600:  # 10 minutes
                        stale_clients.append(client_id)
                        
                except Exception as e:
                    logger.warning(f"Error checking client {client_id} status: {str(e)}")
                    stale_clients.append(client_id)
            
            # Remove stale clients
            for client_id in stale_clients:
                for task_id in list(self.task_subscribers.keys()):
                    await self.unregister_client(task_id, client_id)
            
            if stale_clients:
                logger.info(f"Cleaned up {len(stale_clients)} stale connections")
                
        except Exception as e:
            logger.error(f"Error during connection cleanup: {str(e)}")
    
    def get_client_count(self) -> int:
        """Get the number of connected clients."""
        return len(self.clients)
    
    def get_task_subscriber_count(self, task_id: str) -> int:
        """Get the number of clients subscribed to a specific task."""
        return len(self.task_subscribers.get(task_id, set()))
    
    def get_client_info(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific client."""
        if client_id not in self.clients:
            return None
        
        client = self.clients[client_id]
        return {
            "client_id": client.client_id,
            "connection_type": client.connection_type,
            "subscribed_tasks": list(client.subscribed_tasks),
            "connected_at": client.connected_at.isoformat(),
            "last_activity": client.last_activity.isoformat(),
            "connection_duration": (datetime.utcnow() - client.connected_at).total_seconds()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get streaming handler statistics."""
        now = datetime.utcnow()
        
        # Calculate connection durations
        connection_durations = []
        for client in self.clients.values():
            duration = (now - client.connected_at).total_seconds()
            connection_durations.append(duration)
        
        # Group clients by connection type
        connection_types = {}
        for client in self.clients.values():
            conn_type = client.connection_type
            connection_types[conn_type] = connection_types.get(conn_type, 0) + 1
        
        # Task subscription statistics
        task_stats = {}
        for task_id, subscribers in self.task_subscribers.items():
            task_stats[task_id] = len(subscribers)
        
        return {
            "total_connections_ever": self.total_connections,
            "current_connections": len(self.clients),
            "connection_types": connection_types,
            "total_messages_sent": self.total_messages_sent,
            "total_errors": self.total_errors,
            "task_subscriptions": task_stats,
            "average_connection_duration": sum(connection_durations) / len(connection_durations) if connection_durations else 0,
            "background_tasks_running": {
                "heartbeat": self.heartbeat_task is not None and not self.heartbeat_task.done(),
                "cleanup": self.cleanup_task is not None and not self.cleanup_task.done()
            }
        }
    
    async def send_custom_event(self, task_id: str, event_type: str, data: Dict[str, Any]) -> int:
        """Send a custom event to clients subscribed to a specific task."""
        try:
            client_ids = self.task_subscribers.get(task_id, set())
            
            if not client_ids:
                return 0
            
            successful_sends = 0
            failed_clients = []
            
            for client_id in client_ids:
                if client_id not in self.clients:
                    failed_clients.append(client_id)
                    continue
                
                client = self.clients[client_id]
                
                try:
                    if client.connection_type == "sse":
                        await self.send_event(client.connection, event_type, data)
                    elif client.connection_type == "websocket":
                        await self.send_websocket_message(client.connection, event_type, data)
                    
                    client.last_activity = datetime.utcnow()
                    successful_sends += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to send custom event to client {client_id}: {str(e)}")
                    failed_clients.append(client_id)
            
            # Clean up failed clients
            for client_id in failed_clients:
                await self.unregister_client(task_id, client_id)
            
            logger.debug(f"Sent custom event '{event_type}' to {successful_sends} clients for task {task_id}")
            return successful_sends
            
        except Exception as e:
            logger.error(f"Error sending custom event: {str(e)}")
            self.total_errors += 1
            return 0
    
    async def notify_task_progress(self, task_id: str, progress: float, message: str = "") -> None:
        """Send a progress update for a task."""
        progress_data = {
            "task_id": task_id,
            "progress": max(0.0, min(1.0, progress)),  # Clamp between 0 and 1
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.send_custom_event(task_id, "task_progress", progress_data)
    
    async def notify_task_log(self, task_id: str, level: str, message: str) -> None:
        """Send a log message for a task."""
        log_data = {
            "task_id": task_id,
            "level": level,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.send_custom_event(task_id, "task_log", log_data)
    
    async def close_all_connections(self) -> None:
        """Close all active connections gracefully."""
        try:
            logger.info(f"Closing {len(self.clients)} active connections")
            
            # Send goodbye message
            await self.broadcast_global_message("server_shutdown", {
                "message": "Server is shutting down",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Close all connections
            for client_id, client in list(self.clients.items()):
                try:
                    if client.connection_type == "websocket":
                        await client.connection.close()
                    # SSE connections will be closed when the response ends
                    
                except Exception as e:
                    logger.warning(f"Error closing connection for client {client_id}: {str(e)}")
            
            # Clear all data
            self.clients.clear()
            self.task_subscribers.clear()
            
            # Stop background tasks
            await self.stop_background_tasks()
            
            logger.info("All streaming connections closed")
            
        except Exception as e:
            logger.error(f"Error closing streaming connections: {str(e)}")


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_streaming():
        """Test streaming handler functionality."""
        handler = StreamingHandler()
        
        try:
            # Start background tasks
            await handler.start_background_tasks()
            
            # Simulate some operations
            print("Testing streaming handler...")
            
            # Test statistics
            stats = handler.get_statistics()
            print(f"Initial stats: {stats}")
            
            # Test task update broadcast (no clients)
            await handler.broadcast_task_update("test-task-1", {
                "status": "running",
                "progress": 0.5
            })
            
            # Test global message broadcast (no clients)
            await handler.broadcast_global_message("test_message", {
                "content": "Hello, world!"
            })
            
            print("Streaming handler test completed")
            
        finally:
            await handler.close_all_connections()
    
    # Run the test
    # asyncio.run(test_streaming())
