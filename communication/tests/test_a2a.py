"""
Basic workability tests for A2A protocol implementation.

These tests verify that the core A2A components work correctly
and can communicate with each other.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from communication.core.a2a.agent_card import AgentCard, Endpoint, Capability
from communication.core.a2a.message import A2AMessage, MessageBuilder, MessagePart
from communication.core.a2a.task import Task, TaskManager, TaskStatus, TaskPriority, TaskResult
from communication.core.a2a.client import A2AClient, A2AClientError
from communication.core.a2a.server import A2AServer
from communication.core.a2a.streaming import StreamingHandler


class TestAgentCard:
    """Test AgentCard functionality."""
    
    def test_agent_card_creation(self):
        """Test basic agent card creation."""
        agent_card = AgentCard(
            agent_id="test-agent",
            name="Test Agent",
            version="1.0.0",
            description="A test agent",
            endpoints=[
                Endpoint(
                    url="http://localhost:8080",
                    protocol="a2a",
                    methods=["POST", "GET"]
                )
            ],
            capabilities=[
                Capability(
                    name="test_capability",
                    description="Test capability",
                    input_schema={"type": "object"},
                    output_schema={"type": "object"}
                )
            ]
        )
        
        assert agent_card.agent_id == "test-agent"
        assert agent_card.name == "Test Agent"
        assert len(agent_card.endpoints) == 1
        assert len(agent_card.capabilities) == 1
    
    def test_agent_card_serialization(self):
        """Test agent card serialization and deserialization."""
        agent_card = AgentCard(
            agent_id="test-agent",
            name="Test Agent",
            version="1.0.0",
            description="A test agent",
            endpoints=[
                Endpoint(
                    url="http://localhost:8080",
                    protocol="a2a",
                    methods=["POST"]
                )
            ],
            capabilities=[
                Capability(
                    name="test_capability",
                    description="Test capability",
                    input_schema={"type": "string"},
                    output_schema={"type": "string"}
                )
            ]
        )
        
        # Test to_dict
        card_dict = agent_card.to_dict()
        assert isinstance(card_dict, dict)
        assert card_dict["agent_id"] == "test-agent"
        
        # Test from_dict
        restored_card = AgentCard.from_dict(card_dict)
        assert restored_card.agent_id == agent_card.agent_id
        assert restored_card.name == agent_card.name
        assert len(restored_card.endpoints) == len(agent_card.endpoints)
        assert len(restored_card.capabilities) == len(agent_card.capabilities)
    
    def test_agent_card_validation(self):
        """Test agent card validation."""
        # Valid agent card
        valid_card = AgentCard(
            agent_id="valid-agent",
            name="Valid Agent",
            version="1.0.0",
            description="Valid agent",
            endpoints=[
                Endpoint(
                    url="http://localhost:8080",
                    protocol="a2a",
                    methods=["POST"]
                )
            ],
            capabilities=[]
        )
        
        errors = valid_card.validate()
        assert len(errors) == 0
        
        # Invalid agent card (missing required fields)
        invalid_card = AgentCard(
            agent_id="",  # Empty agent_id
            name="",      # Empty name
            version="1.0.0",
            description="Invalid agent",
            endpoints=[],  # No endpoints
            capabilities=[]
        )
        
        errors = invalid_card.validate()
        assert len(errors) > 0
        assert any("agent_id" in error for error in errors)
        assert any("name" in error for error in errors)
        assert any("endpoints" in error for error in errors)


class TestMessage:
    """Test A2A message functionality."""
    
    def test_message_creation(self):
        """Test basic message creation."""
        message = A2AMessage(
            message_id="test-msg-1",
            role="user",
            parts=[
                MessagePart(type="text", content="Hello, world!")
            ],
            metadata={"type": "greeting"}
        )
        
        assert message.message_id == "test-msg-1"
        assert message.role == "user"
        assert len(message.parts) == 1
        assert message.parts[0].content == "Hello, world!"
    
    def test_message_builder(self):
        """Test message builder functionality."""
        builder = MessageBuilder(role="user", task_id="task-123")
        builder.add_text("Hello!")
        builder.add_data({"key": "value"})
        builder.set_metadata({"type": "test"})
        builder.set_parent("parent-msg-id")
        
        message = builder.build()
        
        assert message.role == "user"
        assert message.task_id == "task-123"
        assert len(message.parts) == 2
        assert message.metadata["type"] == "test"
        assert message.parent_message_id == "parent-msg-id"
    
    def test_message_content_extraction(self):
        """Test message content extraction methods."""
        builder = MessageBuilder(role="user")
        builder.add_text("Hello, world!")
        builder.add_data({"result": "success"})
        
        message = builder.build()
        
        # Test text content extraction
        text_content = message.get_text_content()
        assert "Hello, world!" in text_content
        
        # Test data content extraction
        data_content = message.get_data_content()
        assert data_content["result"] == "success"
    
    def test_message_validation(self):
        """Test message validation."""
        # Valid message
        valid_message = A2AMessage(
            message_id="valid-msg",
            role="user",
            parts=[MessagePart(type="text", content="Valid content")],
            metadata={}
        )
        
        errors = valid_message.validate()
        assert len(errors) == 0
        
        # Invalid message (invalid role)
        invalid_message = A2AMessage(
            message_id="invalid-msg",
            role="invalid_role",
            parts=[],
            metadata={}
        )
        
        errors = invalid_message.validate()
        assert len(errors) > 0
        assert any("role" in error for error in errors)
    
    def test_message_serialization(self):
        """Test message serialization and deserialization."""
        original_message = A2AMessage(
            message_id="serialize-test",
            role="agent",
            parts=[
                MessagePart(type="text", content="Test message"),
                MessagePart(type="data", content={"key": "value"})
            ],
            metadata={"type": "test"},
            task_id="task-123"
        )
        
        # Test to_dict
        message_dict = original_message.to_dict()
        assert isinstance(message_dict, dict)
        assert message_dict["message_id"] == "serialize-test"
        
        # Test from_dict
        restored_message = A2AMessage.from_dict(message_dict)
        assert restored_message.message_id == original_message.message_id
        assert restored_message.role == original_message.role
        assert len(restored_message.parts) == len(original_message.parts)
        assert restored_message.task_id == original_message.task_id


class TestTask:
    """Test A2A task functionality."""
    
    def test_task_creation(self):
        """Test basic task creation."""
        task = Task(
            task_id="test-task-1",
            agent_id="test-agent",
            task_type="test_operation",
            status=TaskStatus.CREATED,
            priority=TaskPriority.NORMAL,
            input_data={"input": "test"},
            created_at=datetime.utcnow()
        )
        
        assert task.task_id == "test-task-1"
        assert task.agent_id == "test-agent"
        assert task.task_type == "test_operation"
        assert task.status == TaskStatus.CREATED
        assert task.priority == TaskPriority.NORMAL
    
    def test_task_lifecycle(self):
        """Test task lifecycle management."""
        task = Task(
            task_id="lifecycle-test",
            agent_id="test-agent",
            task_type="test_operation",
            status=TaskStatus.CREATED,
            priority=TaskPriority.NORMAL,
            input_data={},
            created_at=datetime.utcnow()
        )
        
        # Test status transitions
        assert task.status == TaskStatus.CREATED
        
        task.status = TaskStatus.RUNNING
        assert task.status == TaskStatus.RUNNING
        
        task.status = TaskStatus.COMPLETED
        assert task.status == TaskStatus.COMPLETED
    
    def test_task_timeout(self):
        """Test task timeout functionality."""
        # Task with timeout
        task_with_timeout = Task(
            task_id="timeout-test",
            agent_id="test-agent",
            task_type="test_operation",
            status=TaskStatus.CREATED,
            priority=TaskPriority.NORMAL,
            input_data={},
            created_at=datetime.utcnow(),
            timeout=timedelta(seconds=1)
        )
        
        # Should not be expired immediately
        assert not task_with_timeout.is_expired()
        
        # Task without timeout
        task_no_timeout = Task(
            task_id="no-timeout-test",
            agent_id="test-agent",
            task_type="test_operation",
            status=TaskStatus.CREATED,
            priority=TaskPriority.NORMAL,
            input_data={},
            created_at=datetime.utcnow()
        )
        
        # Should never be expired
        assert not task_no_timeout.is_expired()
    
    def test_task_serialization(self):
        """Test task serialization."""
        task = Task(
            task_id="serialize-test",
            agent_id="test-agent",
            task_type="test_operation",
            status=TaskStatus.RUNNING,
            priority=TaskPriority.HIGH,
            input_data={"test": "data"},
            created_at=datetime.utcnow()
        )
        
        task_dict = task.to_dict()
        assert isinstance(task_dict, dict)
        assert task_dict["task_id"] == "serialize-test"
        assert task_dict["status"] == "running"
        assert task_dict["priority"] == "high"


class TestTaskManager:
    """Test TaskManager functionality."""
    
    def test_task_manager_creation(self):
        """Test task manager creation."""
        manager = TaskManager()
        assert len(manager.tasks) == 0
        assert len(manager.task_handlers) == 0
    
    def test_task_creation_and_retrieval(self):
        """Test task creation and retrieval."""
        manager = TaskManager()
        
        task = manager.create_task(
            agent_id="test-agent",
            task_type="test_operation",
            input_data={"test": "data"},
            priority=TaskPriority.NORMAL
        )
        
        assert task.task_id is not None
        assert task.agent_id == "test-agent"
        assert task.task_type == "test_operation"
        assert task.status == TaskStatus.CREATED
        
        # Test retrieval
        retrieved_task = manager.get_task(task.task_id)
        assert retrieved_task is not None
        assert retrieved_task.task_id == task.task_id
    
    def test_task_handler_registration(self):
        """Test task handler registration."""
        manager = TaskManager()
        
        async def test_handler(input_data):
            return {"result": "success"}
        
        manager.register_handler("test_operation", test_handler)
        assert "test_operation" in manager.task_handlers
        assert manager.task_handlers["test_operation"] == test_handler
    
    @pytest.mark.asyncio
    async def test_task_execution(self):
        """Test task execution."""
        manager = TaskManager()
        
        # Register a test handler
        async def test_handler(input_data):
            return {"result": "processed", "input": input_data}
        
        manager.register_handler("test_operation", test_handler)
        
        # Create and execute task
        task = manager.create_task(
            agent_id="test-agent",
            task_type="test_operation",
            input_data={"test": "data"}
        )
        
        result = await manager.execute_task(task.task_id)
        
        assert result is not None
        assert result.success is True
        assert result.data["result"] == "processed"
        assert result.data["input"]["test"] == "data"
        
        # Check task status
        updated_task = manager.get_task(task.task_id)
        assert updated_task.status == TaskStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_task_execution_failure(self):
        """Test task execution failure handling."""
        manager = TaskManager()
        
        # Register a failing handler
        async def failing_handler(input_data):
            raise ValueError("Test error")
        
        manager.register_handler("failing_operation", failing_handler)
        
        # Create and execute task
        task = manager.create_task(
            agent_id="test-agent",
            task_type="failing_operation",
            input_data={}
        )
        
        result = await manager.execute_task(task.task_id)
        
        assert result is not None
        assert result.success is False
        assert "Test error" in result.error_message
        
        # Check task status
        updated_task = manager.get_task(task.task_id)
        assert updated_task.status == TaskStatus.FAILED
    
    def test_task_cancellation(self):
        """Test task cancellation."""
        manager = TaskManager()
        
        task = manager.create_task(
            agent_id="test-agent",
            task_type="test_operation",
            input_data={}
        )
        
        # Cancel the task
        success = manager.cancel_task(task.task_id)
        assert success is True
        
        # Check task status
        updated_task = manager.get_task(task.task_id)
        assert updated_task.status == TaskStatus.CANCELLED
    
    def test_task_statistics(self):
        """Test task statistics."""
        manager = TaskManager()
        
        # Create some tasks
        task1 = manager.create_task("agent1", "type1", {})
        task2 = manager.create_task("agent2", "type2", {})
        manager.cancel_task(task2.task_id)
        
        stats = manager.get_statistics()
        
        assert stats["total_tasks"] == 2
        assert stats["tasks_by_status"]["created"] == 1
        assert stats["tasks_by_status"]["cancelled"] == 1
        assert "type1" in stats["tasks_by_type"]
        assert "type2" in stats["tasks_by_type"]


class TestStreamingHandler:
    """Test StreamingHandler functionality."""
    
    def test_streaming_handler_creation(self):
        """Test streaming handler creation."""
        handler = StreamingHandler()
        assert len(handler.clients) == 0
        assert len(handler.task_subscribers) == 0
        assert handler.total_connections == 0
    
    @pytest.mark.asyncio
    async def test_client_registration(self):
        """Test client registration and unregistration."""
        handler = StreamingHandler()
        
        # Mock connection
        mock_connection = MagicMock()
        
        # Register client
        await handler.register_client("task-1", "client-1", mock_connection, "sse")
        
        assert len(handler.clients) == 1
        assert "client-1" in handler.clients
        assert "task-1" in handler.task_subscribers
        assert "client-1" in handler.task_subscribers["task-1"]
        assert handler.total_connections == 1
        
        # Unregister client
        await handler.unregister_client("task-1", "client-1")
        
        assert len(handler.clients) == 0
        assert len(handler.task_subscribers) == 0
    
    @pytest.mark.asyncio
    async def test_task_subscription(self):
        """Test task subscription functionality."""
        handler = StreamingHandler()
        mock_connection = MagicMock()
        
        # Register client first
        await handler.register_client("task-1", "client-1", mock_connection)
        
        # Subscribe to additional task
        await handler.subscribe_client_to_task("client-1", "task-2", mock_connection)
        
        client = handler.clients["client-1"]
        assert "task-1" in client.subscribed_tasks
        assert "task-2" in client.subscribed_tasks
        assert "client-1" in handler.task_subscribers["task-2"]
        
        # Unsubscribe from task
        await handler.unsubscribe_client_from_task("client-1", "task-1")
        
        assert "task-1" not in client.subscribed_tasks
        assert "task-2" in client.subscribed_tasks
    
    @pytest.mark.asyncio
    async def test_task_update_broadcast(self):
        """Test task update broadcasting."""
        handler = StreamingHandler()
        
        # Mock SSE connection
        mock_sse_connection = AsyncMock()
        mock_sse_connection.write = AsyncMock()
        
        # Mock WebSocket connection
        mock_ws_connection = AsyncMock()
        mock_ws_connection.send_str = AsyncMock()
        
        # Register clients
        await handler.register_client("task-1", "client-sse", mock_sse_connection, "sse")
        await handler.register_client("task-1", "client-ws", mock_ws_connection, "websocket")
        
        # Broadcast update
        update_data = {"status": "running", "progress": 0.5}
        await handler.broadcast_task_update("task-1", update_data)
        
        # Verify SSE client received update
        mock_sse_connection.write.assert_called_once()
        
        # Verify WebSocket client received update
        mock_ws_connection.send_str.assert_called_once()
    
    def test_client_statistics(self):
        """Test client statistics."""
        handler = StreamingHandler()
        
        stats = handler.get_statistics()
        
        assert "total_connections_ever" in stats
        assert "current_connections" in stats
        assert "connection_types" in stats
        assert "total_messages_sent" in stats
        assert "total_errors" in stats
        assert stats["current_connections"] == 0
        assert stats["total_connections_ever"] == 0


class TestA2AClient:
    """Test A2A client functionality."""
    
    def test_client_creation(self):
        """Test client creation."""
        client = A2AClient("test-client-agent")
        assert client.agent_id == "test-client-agent"
        assert client.session is not None
    
    @pytest.mark.asyncio
    async def test_agent_discovery_mock(self):
        """Test agent discovery with mocked response."""
        client = A2AClient("test-client")
        
        # Mock the HTTP session
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "agent_id": "discovered-agent",
            "name": "Discovered Agent",
            "version": "1.0.0",
            "description": "Test agent",
            "endpoints": [
                {
                    "url": "http://localhost:8080",
                    "protocol": "a2a",
                    "methods": ["POST", "GET"]
                }
            ],
            "capabilities": []
        })
        
        client.session.get = AsyncMock(return_value=mock_response)
        
        # Test discovery
        agent_card = await client.discover_agent("http://localhost:8080")
        
        assert agent_card.agent_id == "discovered-agent"
        assert agent_card.name == "Discovered Agent"
        assert len(agent_card.endpoints) == 1
        
        # Verify the correct URL was called
        client.session.get.assert_called_once_with(
            "http://localhost:8080/.well-known/agent-card"
        )
    
    @pytest.mark.asyncio
    async def test_message_sending_mock(self):
        """Test message sending with mocked response."""
        client = A2AClient("test-client")
        
        # Create test message
        builder = MessageBuilder(role="user")
        builder.add_text("Test message")
        message = builder.build()
        
        # Mock the HTTP session
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "jsonrpc": "2.0",
            "result": {
                "message_id": "response-msg-id",
                "role": "agent",
                "parts": [
                    {
                        "type": "text",
                        "content": "Response message"
                    }
                ],
                "metadata": {},
                "timestamp": datetime.utcnow().isoformat()
            },
            "id": "test-request-id"
        })
        
        client.session.post = AsyncMock(return_value=mock_response)
        
        # Test message sending
        response = await client.send_message("target-agent", message)
        
        assert response.role == "agent"
        assert "Response message" in response.get_text_content()
        
        # Verify the correct endpoint was called
        client.session.post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_task_creation_mock(self):
        """Test task creation with mocked response."""
        client = A2AClient("test-client")
        
        # Mock the HTTP session
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "jsonrpc": "2.0",
            "result": {
                "task_id": "created-task-id",
                "agent_id": "target-agent",
                "task_type": "test_operation",
                "status": "created",
                "priority": "normal",
                "created_at": datetime.utcnow().isoformat()
            },
            "id": "test-request-id"
        })
        
        client.session.post = AsyncMock(return_value=mock_response)
        
        # Test task creation
        task = await client.create_task(
            target_agent_id="target-agent",
            task_type="test_operation",
            input_data={"test": "data"}
        )
        
        assert task.task_id == "created-task-id"
        assert task.agent_id == "target-agent"
        assert task.task_type == "test_operation"
        assert task.status == TaskStatus.CREATED
    
    @pytest.mark.asyncio
    async def test_client_cleanup(self):
        """Test client cleanup."""
        client = A2AClient("test-client")
        
        # Mock session close
        client.session.close = AsyncMock()
        
        await client.close()
        
        client.session.close.assert_called_once()


class TestA2AServer:
    """Test A2A server functionality."""
    
    def test_server_creation(self):
        """Test server creation."""
        agent_card = AgentCard(
            agent_id="test-server-agent",
            name="Test Server",
            version="1.0.0",
            description="Test server",
            endpoints=[
                Endpoint(
                    url="http://localhost:8080",
                    protocol="a2a",
                    methods=["POST", "GET"]
                )
            ],
            capabilities=[]
        )
        
        server = A2AServer(agent_card, port=8080)
        
        assert server.agent_card.agent_id == "test-server-agent"
        assert server.port == 8080
        assert server.host == "localhost"
    
    def test_handler_registration(self):
        """Test handler registration."""
        agent_card = AgentCard(
            agent_id="test-server",
            name="Test Server",
            version="1.0.0",
            description="Test server",
            endpoints=[],
            capabilities=[]
        )
        
        server = A2AServer(agent_card)
        
        # Register task handler
        async def test_task_handler(input_data):
            return {"result": "success"}
        
        server.register_task_handler("test_operation", test_task_handler)
        assert "test_operation" in server.task_manager.task_handlers
        
        # Register message handler
        async def test_message_handler(message, task_id, sender_id):
            return {"response": "received"}
        
        server.register_message_handler("test_message", test_message_handler)
        assert "test_message" in server.message_handlers
    
    def test_server_statistics(self):
        """Test server statistics."""
        agent_card = AgentCard(
            agent_id="test-server",
            name="Test Server",
            version="1.0.0",
            description="Test server",
            endpoints=[],
            capabilities=[]
        )
        
        server = A2AServer(agent_card)
        
        stats = server.get_statistics()
        
        assert "agent_id" in stats
        assert "host" in stats
        assert "port" in stats
        assert "task_manager" in stats
        assert "streaming" in stats
        assert "registered_handlers" in stats
        
        assert stats["agent_id"] == "test-server"


class TestA2AIntegration:
    """Integration tests for A2A components working together."""
    
    @pytest.mark.asyncio
    async def test_message_round_trip(self):
        """Test complete message round trip."""
        # Create agent card
        agent_card = AgentCard(
            agent_id="integration-test-agent",
            name="Integration Test Agent",
            version="1.0.0",
            description="Agent for integration testing",
            endpoints=[
                Endpoint(
                    url="http://localhost:8085",
                    protocol="a2a",
                    methods=["POST", "GET"]
                )
            ],
            capabilities=[
                Capability(
                    name="echo",
                    description="Echo messages back",
                    input_schema={"type": "object"},
                    output_schema={"type": "object"}
                )
            ]
        )
        
        # Create server
        server = A2AServer(agent_card, port=8085)
        
        # Register echo handler
        async def echo_handler(message, task_id, sender_id):
            return {
                "echo": message.get_text_content(),
                "sender": sender_id,
                "task_id": task_id
            }
        
        server.register_message_handler("echo", echo_handler)
        
        # Create client
        client = A2AClient("integration-test-client")
        
        try:
            # Start server
            await server.start()
            await asyncio.sleep(0.5)  # Give server time to start
            
            # Test agent discovery
            discovered_agent = await client.discover_agent("http://localhost:8085")
            assert discovered_agent.agent_id == "integration-test-agent"
            
            # Test message sending
            builder = MessageBuilder(role="user")
            builder.add_text("Hello, integration test!")
            builder.set_metadata({"type": "echo"})
            message = builder.build()
            
            response = await client.send_message(discovered_agent.agent_id, message)
            response_data = response.get_data_content()
            
            assert response_data["echo"] == "Hello, integration test!"
            assert response_data["sender"] == "integration-test-client"
            
        finally:
            await server.stop()
            await client.close()
    
    @pytest.mark.asyncio
    async def test_task_execution_integration(self):
        """Test complete task execution workflow."""
        # Create agent card
        agent_card = AgentCard(
            agent_id="task-test-agent",
            name="Task Test Agent",
            version="1.0.0",
            description="Agent for task testing",
            endpoints=[
                Endpoint(
                    url="http://localhost:8086",
                    protocol="a2a",
                    methods=["POST", "GET"]
                )
            ],
            capabilities=[
                Capability(
                    name="add_numbers",
                    description="Add two numbers",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"}
                        },
                        "required": ["a", "b"]
                    },
                    output_schema={
                        "type": "object",
                        "properties": {
                            "result": {"type": "number"}
                        }
                    }
                )
            ]
        )
        
        # Create server
        server = A2AServer(agent_card, port=8086)
        
        # Register task handler
        async def add_numbers_handler(input_data):
            a = input_data.get("a", 0)
            b = input_data.get("b", 0)
            return {"result": a + b}
        
        server.register_task_handler("add_numbers", add_numbers_handler)
        
        # Create client
        client = A2AClient("task-test-client")
        
        try:
            # Start server
            await server.start()
            await asyncio.sleep(0.5)  # Give server time to start
            
            # Test task execution
            result = await client.execute_task_and_wait(
                target_agent_id="task-test-agent",
                task_type="add_numbers",
                input_data={"a": 10, "b": 20},
                timeout=timedelta(seconds=30)
            )
            
            assert result.success is True
            assert result.data["result"] == 30
            
        finally:
            await server.stop()
            await client.close()


# Pytest configuration
@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
