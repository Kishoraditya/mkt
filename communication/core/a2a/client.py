"""
A2A Client implementation.

Handles outbound communication to other A2A agents, including
agent discovery, task creation, and message exchange.
"""

import json
import uuid
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from urllib.parse import urljoin
import ssl
import certifi

from .agent_card import AgentCard
from .message import A2AMessage, MessageBuilder
from .task import Task, TaskStatus, TaskPriority, TaskResult

import logging

logger = logging.getLogger(__name__)


class A2AClientError(Exception):
    """Base exception for A2A client errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class AgentNotFoundError(A2AClientError):
    """Raised when an agent cannot be found or reached."""
    
    def __init__(self, agent_id: str, endpoint: Optional[str] = None):
        message = f"Agent {agent_id} not found"
        if endpoint:
            message += f" at endpoint {endpoint}"
        super().__init__(message)
        self.agent_id = agent_id
        self.endpoint = endpoint


class TaskExecutionError(A2AClientError):
    """Raised when task execution fails."""
    
    def __init__(self, task_id: str, error_message: str, task_status: Optional[TaskStatus] = None):
        message = f"Task {task_id} execution failed: {error_message}"
        super().__init__(message)
        self.task_id = task_id
        self.error_message = error_message
        self.task_status = task_status


class AuthenticationError(A2AClientError):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message)


class NetworkError(A2AClientError):
    """Raised when network communication fails."""
    
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error


class A2AClient:
    """
    A2A Client for communicating with other agents.
    
    Implements Google A2A specification for agent-to-agent communication
    using JSON-RPC 2.0 over HTTP(S).
    """
    
    def __init__(
        self,
        agent_id: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        auth_token: Optional[str] = None,
        verify_ssl: bool = True
    ):
        """Initialize A2A client."""
        self.agent_id = agent_id
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.auth_token = auth_token
        self.verify_ssl = verify_ssl
        self.session: Optional[aiohttp.ClientSession] = None
        self.discovered_agents: Dict[str, AgentCard] = {}
        self.active_tasks: Dict[str, Task] = {}
        
        # Setup SSL context
        self.ssl_context = None
        if verify_ssl:
            self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        
        logger.info(f"A2A Client initialized for agent {agent_id}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_session()
    
    async def start_session(self) -> None:
        """Start HTTP session."""
        if self.session is None:
            connector = aiohttp.TCPConnector(ssl=self.ssl_context)
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': f'A2A-Client/{self.agent_id}'
            }
            
            if self.auth_token:
                headers['Authorization'] = f'Bearer {self.auth_token}'
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=headers
            )
            
            logger.debug("HTTP session started")
    
    async def close_session(self) -> None:
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
            logger.debug("HTTP session closed")
    
    async def _make_request(
        self,
        method: str,
        url: str,
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic."""
        if not self.session:
            await self.start_session()
        
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                async with self.session.request(
                    method=method,
                    url=url,
                    json=data,
                    headers=headers
                ) as response:
                    
                    response_text = await response.text()
                    
                    if response.status == 401:
                        raise AuthenticationError("Invalid or expired authentication token")
                    
                    if response.status == 404:
                        raise AgentNotFoundError("Agent not found", endpoint=url)
                    
                    if response.status >= 400:
                        try:
                            error_data = json.loads(response_text)
                        except json.JSONDecodeError:
                            error_data = {"error": response_text}
                        
                        raise A2AClientError(
                            f"HTTP {response.status}: {error_data.get('error', 'Unknown error')}",
                            status_code=response.status,
                            response_data=error_data
                        )
                    
                    try:
                        return json.loads(response_text)
                    except json.JSONDecodeError as e:
                        raise A2AClientError(f"Invalid JSON response: {str(e)}")
            
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = NetworkError(f"Network error on attempt {attempt + 1}: {str(e)}", e)
                
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Request failed, retrying in {wait_time}s: {str(e)}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Request failed after {self.max_retries + 1} attempts")
                    raise last_error
        
        raise last_error
    
    async def discover_agent(self, agent_endpoint: str) -> AgentCard:
        """Discover an agent by fetching its agent card."""
        try:
            logger.info(f"Discovering agent at {agent_endpoint}")
            
            # Fetch agent card from well-known endpoint
            card_url = urljoin(agent_endpoint, "/.well-known/agent-card")
            response_data = await self._make_request("GET", card_url)
            
            agent_card = AgentCard.from_dict(response_data)
            
            # Validate the agent card
            errors = agent_card.validate()
            if errors:
                raise A2AClientError(f"Invalid agent card: {', '.join(errors)}")
            
            # Cache the discovered agent
            self.discovered_agents[agent_card.agent_id] = agent_card
            
            logger.info(f"Successfully discovered agent {agent_card.agent_id}: {agent_card.name}")
            return agent_card
            
        except A2AClientError:
            raise
        except Exception as e:
            raise A2AClientError(f"Failed to discover agent: {str(e)}")
    
    async def create_task(
        self,
        target_agent_id: str,
        task_type: str,
        input_data: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[timedelta] = None
    ) -> Task:
        """Create a task on a target agent."""
        try:
            # Get agent card
            agent_card = self.discovered_agents.get(target_agent_id)
            if not agent_card:
                raise AgentNotFoundError(target_agent_id)
            
            # Get primary endpoint
            endpoint = agent_card.get_primary_endpoint()
            if not endpoint:
                raise A2AClientError(f"No endpoints available for agent {target_agent_id}")
            
            # Prepare task data
            task_data = {
                "jsonrpc": "2.0",
                "method": "create_task",
                "params": {
                    "agent_id": target_agent_id,
                    "task_type": task_type,
                    "input_data": input_data,
                    "priority": priority.value,
                    "timeout": timeout.total_seconds() if timeout else None,
                    "requester_id": self.agent_id
                },
                "id": str(uuid.uuid4())
            }
            
            # Make request
            task_url = urljoin(endpoint.url, "/tasks")
            response_data = await self._make_request("POST", task_url, task_data)
            
            # Handle JSON-RPC response
            if "error" in response_data:
                error = response_data["error"]
                raise TaskExecutionError(
                    task_id="unknown",
                    error_message=f"Task creation failed: {error.get('message', 'Unknown error')}"
                )
            
            # Create task object from response
            task_info = response_data.get("result", {})
            task = Task.from_dict(task_info)
            
            # Cache the task
            self.active_tasks[task.task_id] = task
            
            logger.info(f"Created task {task.task_id} on agent {target_agent_id}")
            return task
            
        except A2AClientError:
            raise
        except Exception as e:
            raise TaskExecutionError("unknown", f"Failed to create task: {str(e)}")
    
    async def send_message(
        self,
        target_agent_id: str,
        message: A2AMessage,
        task_id: Optional[str] = None
    ) -> A2AMessage:
        """Send a message to a target agent."""
        try:
            # Get agent card
            agent_card = self.discovered_agents.get(target_agent_id)
            if not agent_card:
                raise AgentNotFoundError(target_agent_id)
            
            # Get primary endpoint
            endpoint = agent_card.get_primary_endpoint()
            if not endpoint:
                raise A2AClientError(f"No endpoints available for agent {target_agent_id}")
            
            # Prepare message data
            message_data = {
                "jsonrpc": "2.0",
                "method": "send_message",
                "params": {
                    "message": message.to_dict(),
                    "task_id": task_id,
                    "sender_id": self.agent_id
                },
                "id": str(uuid.uuid4())
            }
            
            # Make request
            message_url = urljoin(endpoint.url, "/messages")
            response_data = await self._make_request("POST", message_url, message_data)
            
            # Handle JSON-RPC response
            if "error" in response_data:
                error = response_data["error"]
                raise A2AClientError(f"Message sending failed: {error.get('message', 'Unknown error')}")
            
            # Parse response message
            response_message_data = response_data.get("result", {})
            response_message = A2AMessage.from_dict(response_message_data)
            
            logger.info(f"Sent message {message.message_id} to agent {target_agent_id}")
            return response_message
            
        except A2AClientError:
            raise
        except Exception as e:
            raise A2AClientError(f"Failed to send message: {str(e)}")
    
    async def get_task_status(self, target_agent_id: str, task_id: str) -> Task:
        """Get the status of a task on a target agent."""
        try:
            # Get agent card
            agent_card = self.discovered_agents.get(target_agent_id)
            if not agent_card:
                raise AgentNotFoundError(target_agent_id)
            
            # Get primary endpoint
            endpoint = agent_card.get_primary_endpoint()
            if not endpoint:
                raise A2AClientError(f"No endpoints available for agent {target_agent_id}")
            
            # Make request
            status_url = urljoin(endpoint.url, f"/tasks/{task_id}")
            response_data = await self._make_request("GET", status_url)
            
            # Parse task data
            task = Task.from_dict(response_data)
            
            # Update cached task
            if task_id in self.active_tasks:
                self.active_tasks[task_id] = task
            
            logger.debug(f"Retrieved status for task {task_id}: {task.status.value}")
            return task
            
        except A2AClientError:
            raise
        except Exception as e:
            raise TaskExecutionError(task_id, f"Failed to get task status: {str(e)}")
    
    async def cancel_task(self, target_agent_id: str, task_id: str) -> bool:
        """Cancel a task on a target agent."""
        try:
            # Get agent card
            agent_card = self.discovered_agents.get(target_agent_id)
            if not agent_card:
                raise AgentNotFoundError(target_agent_id)
            
            # Get primary endpoint
            endpoint = agent_card.get_primary_endpoint()
            if not endpoint:
                raise A2AClientError(f"No endpoints available for agent {target_agent_id}")
            
            # Prepare cancellation data
            cancel_data = {
                "jsonrpc": "2.0",
                "method": "cancel_task",
                "params": {
                    "task_id": task_id,
                    "requester_id": self.agent_id
                },
                "id": str(uuid.uuid4())
            }
            
            # Make request
            cancel_url = urljoin(endpoint.url, f"/tasks/{task_id}/cancel")
            response_data = await self._make_request("POST", cancel_url, cancel_data)
            
            # Handle JSON-RPC response
            if "error" in response_data:
                error = response_data["error"]
                raise TaskExecutionError(task_id, f"Task cancellation failed: {error.get('message', 'Unknown error')}")
            
            success = response_data.get("result", {}).get("cancelled", False)
            
            # Update cached task
            if task_id in self.active_tasks:
                self.active_tasks[task_id].cancel()
            
            logger.info(f"Task {task_id} cancellation {'successful' if success else 'failed'}")
            return success
            
        except A2AClientError:
            raise
        except Exception as e:
            raise TaskExecutionError(task_id, f"Failed to cancel task: {str(e)}")
    
    async def wait_for_task_completion(
        self,
        target_agent_id: str,
        task_id: str,
        poll_interval: float = 2.0,
        max_wait_time: Optional[float] = None
    ) -> Task:
        """Wait for a task to complete, polling for status updates."""
        start_time = datetime.utcnow()
        
        while True:
            try:
                task = await self.get_task_status(target_agent_id, task_id)
                
                if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    logger.info(f"Task {task_id} completed with status: {task.status.value}")
                    return task
                
                # Check for timeout
                if max_wait_time:
                    elapsed = (datetime.utcnow() - start_time).total_seconds()
                    if elapsed >= max_wait_time:
                        raise TaskExecutionError(
                            task_id,
                            f"Task did not complete within {max_wait_time} seconds"
                        )
                
                # Check if task is expired
                if task.is_expired():
                    raise TaskExecutionError(
                        task_id,
                        "Task has exceeded its timeout"
                    )
                
                await asyncio.sleep(poll_interval)
                
            except A2AClientError:
                raise
            except Exception as e:
                raise TaskExecutionError(task_id, f"Error while waiting for task completion: {str(e)}")
    
    async def execute_task_and_wait(
        self,
        target_agent_id: str,
        task_type: str,
        input_data: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[timedelta] = None,
        poll_interval: float = 2.0
    ) -> TaskResult:
        """Create a task and wait for its completion."""
        try:
            # Create the task
            task = await self.create_task(
                target_agent_id=target_agent_id,
                task_type=task_type,
                input_data=input_data,
                priority=priority,
                timeout=timeout
            )
            
            # Wait for completion
            completed_task = await self.wait_for_task_completion(
                target_agent_id=target_agent_id,
                task_id=task.task_id,
                poll_interval=poll_interval,
                max_wait_time=timeout.total_seconds() if timeout else None
            )
            
            if completed_task.status == TaskStatus.FAILED:
                raise TaskExecutionError(
                    task.task_id,
                    completed_task.result.error_message if completed_task.result else "Task failed"
                )
            
            return completed_task.result
            
        except A2AClientError:
            raise
        except Exception as e:
            raise TaskExecutionError("unknown", f"Failed to execute task: {str(e)}")
    
    def get_cached_agent(self, agent_id: str) -> Optional[AgentCard]:
        """Get a cached agent card."""
        return self.discovered_agents.get(agent_id)
    
    def get_active_task(self, task_id: str) -> Optional[Task]:
        """Get an active task."""
        return self.active_tasks.get(task_id)
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self.discovered_agents.clear()
        self.active_tasks.clear()
        logger.info("Client cache cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "discovered_agents": len(self.discovered_agents),
            "active_tasks": len(self.active_tasks),
            "agent_ids": list(self.discovered_agents.keys()),
            "task_statuses": {
                status.value: len([t for t in self.active_tasks.values() if t.status == status])
                for status in TaskStatus
            }
        }


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_client():
        """Test A2A client functionality."""
        async with A2AClient("test-agent-123") as client:
            try:
                # Test agent discovery
                agent_card = await client.discover_agent("https://api.example.com")
                print(f"Discovered agent: {agent_card.name}")
                
                # Test task creation and execution
                result = await client.execute_task_and_wait(
                    target_agent_id=agent_card.agent_id,
                    task_type="text_analysis",
                    input_data={"text": "Hello, world!"},
                    timeout=timedelta(minutes=5)
                )
                
                print(f"Task result: {result.to_dict()}")
                
            except A2AClientError as e:
                print(f"A2A Error: {e}")
            except Exception as e:
                print(f"Unexpected error: {e}")
    
    # Run the test
    # asyncio.run(test_client())
