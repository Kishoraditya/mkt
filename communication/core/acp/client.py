# communication/core/acp/client.py
"""
ACP Client implementation.

Provides client functionality for the Agent Communication Protocol (ACP)
following the Linux Foundation AI & Data / BeeAI specification.
"""

import asyncio
import json
import uuid
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Callable, AsyncGenerator
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, urlparse
import aiohttp
import ssl
from enum import Enum

from .agent_detail import AgentDetail, InputType, OutputType
from .message import ACPMessage, MessagePart
from .run import ACPRun, RunStatus, RunResult

import logging

logger = logging.getLogger(__name__)


class ACPClientError(Exception):
    """Base exception for ACP client errors."""
    pass


class AgentNotFoundError(ACPClientError):
    """Raised when an agent is not found."""
    pass


class RunCreationError(ACPClientError):
    """Raised when run creation fails."""
    pass


class CommunicationError(ACPClientError):
    """Raised when communication with agent fails."""
    pass


class AuthenticationError(ACPClientError):
    """Raised when authentication fails."""
    pass


class TimeoutError(ACPClientError):
    """Raised when operation times out."""
    pass


@dataclass
class ClientConfig:
    """Configuration for ACP client."""
    
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    verify_ssl: bool = True
    user_agent: str = "MKT-ACP-Client/1.0"
    max_concurrent_runs: int = 10
    default_headers: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.default_headers is None:
            self.default_headers = {
                "Content-Type": "application/json",
                "Accept": "application/json"
            }


@dataclass
class AuthConfig:
    """Authentication configuration."""
    
    auth_type: str = "none"  # none, bearer, api_key, basic
    token: Optional[str] = None
    api_key: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    
    def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        headers = {}
        
        if self.auth_type == "bearer" and self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        elif self.auth_type == "api_key" and self.api_key:
            headers["X-API-Key"] = self.api_key
        elif self.auth_type == "basic" and self.username and self.password:
            import base64
            credentials = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
            headers["Authorization"] = f"Basic {credentials}"
        
        if self.headers:
            headers.update(self.headers)
        
        return headers


class ACPClient:
    """
    ACP (Agent Communication Protocol) Client.
    
    Provides functionality to discover agents, create runs, and communicate
    with agents following the ACP specification.
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        config: Optional[ClientConfig] = None,
        auth: Optional[AuthConfig] = None
    ):
        """
        Initialize ACP client.
        
        Args:
            base_url: Base URL for agent communication
            config: Client configuration
            auth: Authentication configuration
        """
        self.base_url = base_url
        self.config = config or ClientConfig()
        self.auth = auth or AuthConfig()
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._active_runs: Dict[str, ACPRun] = {}
        self._agent_cache: Dict[str, AgentDetail] = {}
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_runs": 0,
            "active_runs": 0,
            "cached_agents": 0
        }
        
        logger.info(f"Initialized ACP client with base URL: {base_url}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_session(self) -> None:
        """Ensure HTTP session is created."""
        if self._session is None or self._session.closed:
            # Configure SSL context
            ssl_context = None
            if not self.config.verify_ssl:
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
            
            # Create session with configuration
            connector = aiohttp.TCPConnector(
                ssl=ssl_context,
                limit=100,
                limit_per_host=30
            )
            
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            
            headers = self.config.default_headers.copy()
            headers["User-Agent"] = self.config.user_agent
            headers.update(self.auth.get_auth_headers())
            
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=headers
            )
            
            logger.debug("Created new HTTP session")
    
    async def close(self) -> None:
        """Close the client and cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.debug("Closed HTTP session")
        
        # Cancel any active runs
        for run in self._active_runs.values():
            if run.status in [RunStatus.CREATED, RunStatus.RUNNING]:
                try:
                    await self.cancel_run(run.run_id)
                except Exception as e:
                    logger.warning(f"Failed to cancel run {run.run_id}: {e}")
    
    async def _make_request(
        self,
        method: str,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic."""
        await self._ensure_session()
        
        # Prepare request
        request_headers = {}
        if headers:
            request_headers.update(headers)
        
        # Add authentication headers
        auth_headers = self.auth.get_auth_headers()
        request_headers.update(auth_headers)
        
        # Retry logic
        last_exception = None
        for attempt in range(self.config.max_retries + 1):
            try:
                self._stats["total_requests"] += 1
                
                async with self._session.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                    headers=request_headers
                ) as response:
                    
                    # Log request details
                    logger.debug(f"{method} {url} -> {response.status}")
                    
                    # Handle response
                    if response.status == 200:
                        self._stats["successful_requests"] += 1
                        try:
                            return await response.json()
                        except json.JSONDecodeError as e:
                            raise CommunicationError(f"Invalid JSON response: {e}")
                    
                    elif response.status == 401:
                        self._stats["failed_requests"] += 1
                        raise AuthenticationError("Authentication failed")
                    
                    elif response.status == 404:
                        self._stats["failed_requests"] += 1
                        raise AgentNotFoundError("Agent or resource not found")
                    
                    elif response.status == 408:
                        self._stats["failed_requests"] += 1
                        raise TimeoutError("Request timed out")
                    
                    else:
                        self._stats["failed_requests"] += 1
                        error_text = await response.text()
                        raise CommunicationError(f"HTTP {response.status}: {error_text}")
            
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exception = e
                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    self._stats["failed_requests"] += 1
                    logger.error(f"Request failed after {self.config.max_retries + 1} attempts: {e}")
        
        raise CommunicationError(f"Request failed after retries: {last_exception}")
    
    async def discover_agent(self, agent_url: str) -> AgentDetail:
        """
        Discover an agent by fetching its agent detail.
        
        Args:
            agent_url: URL of the agent
            
        Returns:
            AgentDetail object
        """
        # Check cache first
        if agent_url in self._agent_cache:
            logger.debug(f"Using cached agent detail for {agent_url}")
            return self._agent_cache[agent_url]
        
        # Construct agent detail URL
        detail_url = urljoin(agent_url, "/.well-known/agent-detail")
        
        try:
            response_data = await self._make_request("GET", detail_url)
            agent_detail = AgentDetail.from_dict(response_data)
            
            # Cache the agent detail
            self._agent_cache[agent_url] = agent_detail
            self._stats["cached_agents"] = len(self._agent_cache)
            
            logger.info(f"Discovered agent: {agent_detail.name} ({agent_detail.agent_id})")
            return agent_detail
            
        except Exception as e:
            logger.error(f"Failed to discover agent at {agent_url}: {e}")
            raise AgentNotFoundError(f"Could not discover agent at {agent_url}: {e}")
    
    async def create_run(
        self,
        agent_url: str,
        input_messages: List[ACPMessage],
        run_config: Optional[Dict[str, Any]] = None
    ) -> ACPRun:
        """
        Create a new run with an agent.
        
        Args:
            agent_url: URL of the agent
            input_messages: List of input messages
            run_config: Optional run configuration
            
        Returns:
            ACPRun object
        """
        # Discover agent first
        agent_detail = await self.discover_agent(agent_url)
        
        # Validate input messages
        for msg in input_messages:
            errors = msg.validate()
            if errors:
                raise RunCreationError(f"Invalid input message: {errors}")
        
        # Prepare run data
        run_data = {
            "input": [msg.to_dict() for msg in input_messages],
            "config": run_config or {}
        }
        
        # Create run endpoint
        run_url = urljoin(agent_url, "/acp/runs")
        
        try:
            response_data = await self._make_request("POST", run_url, data=run_data)
            
            # Create run object
            run = ACPRun.from_dict(response_data)
            run.agent_url = agent_url
            run.agent_detail = agent_detail
            
            # Track active run
            self._active_runs[run.run_id] = run
            self._stats["total_runs"] += 1
            self._stats["active_runs"] = len(self._active_runs)
            
            logger.info(f"Created run {run.run_id} with agent {agent_detail.name}")
            return run
            
        except Exception as e:
            logger.error(f"Failed to create run with agent {agent_url}: {e}")
            raise RunCreationError(f"Could not create run: {e}")
    
    async def get_run_status(self, agent_url: str, run_id: str) -> ACPRun:
        """
        Get the status of a run.
        
        Args:
            agent_url: URL of the agent
            run_id: ID of the run
            
        Returns:
            Updated ACPRun object
        """
        status_url = urljoin(agent_url, f"/acp/runs/{run_id}")
        
        try:
            response_data = await self._make_request("GET", status_url)
            run = ACPRun.from_dict(response_data)
            
            # Update cached run if it exists
            if run_id in self._active_runs:
                self._active_runs[run_id] = run
                
                # Remove from active runs if completed
                if run.status in [RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED]:
                    del self._active_runs[run_id]
                    self._stats["active_runs"] = len(self._active_runs)
            
            logger.debug(f"Retrieved status for run {run_id}: {run.status.value}")
            return run
            
        except Exception as e:
            logger.error(f"Failed to get run status {run_id}: {e}")
            raise CommunicationError(f"Could not get run status: {e}")
    
    async def cancel_run(self, run_id: str, agent_url: Optional[str] = None) -> bool:
        """
        Cancel a running run.
        
        Args:
            run_id: ID of the run to cancel
            agent_url: URL of the agent (optional if run is tracked)
            
        Returns:
            True if cancelled successfully
        """
        # Get agent URL from tracked run if not provided
        if not agent_url and run_id in self._active_runs:
            agent_url = self._active_runs[run_id].agent_url
        
        if not agent_url:
            raise ValueError("Agent URL must be provided or run must be tracked")
        
        cancel_url = urljoin(agent_url, f"/acp/runs/{run_id}/cancel")
        
        try:
            await self._make_request("POST", cancel_url)
            
            # Update tracked run
            if run_id in self._active_runs:
                self._active_runs[run_id].status = RunStatus.CANCELLED
                del self._active_runs[run_id]
                self._stats["active_runs"] = len(self._active_runs)
            
            logger.info(f"Cancelled run {run_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel run {run_id}: {e}")
            return False
    
    async def wait_for_completion(
        self,
        run: ACPRun,
        poll_interval: float = 1.0,
        timeout: Optional[float] = None
    ) -> ACPRun:
        """
        Wait for a run to complete.
        
        Args:
            run: The run to wait for
            poll_interval: How often to check status (seconds)
            timeout: Maximum time to wait (seconds)
            
        Returns:
            Completed ACPRun object
        """
        start_time = time.time()
        
        while True:
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Run {run.run_id} did not complete within {timeout} seconds")
            
            # Get current status
            current_run = await self.get_run_status(run.agent_url, run.run_id)
            
            # Check if completed
            if current_run.status in [RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED]:
                logger.info(f"Run {run.run_id} completed with status: {current_run.status.value}")
                return current_run
            
            # Wait before next poll
            await asyncio.sleep(poll_interval)
    
    async def send_message(
        self,
        agent_url: str,
        run_id: str,
        message: ACPMessage
    ) -> ACPMessage:
        """
        Send a message to a running run.
        
        Args:
            agent_url: URL of the agent
            run_id: ID of the run
            message: Message to send
            
        Returns:
            Response message from agent
        """
        # Validate message
        errors = message.validate()
        if errors:
            raise CommunicationError(f"Invalid message: {errors}")
        
        message_url = urljoin(agent_url, f"/acp/runs/{run_id}/messages")
        message_data = message.to_dict()
        
        try:
            response_data = await self._make_request("POST", message_url, data=message_data)
            response_message = ACPMessage.from_dict(response_data)
            
            logger.debug(f"Sent message to run {run_id}, received response")
            return response_message
            
        except Exception as e:
            logger.error(f"Failed to send message to run {run_id}: {e}")
            raise CommunicationError(f"Could not send message: {e}")
    
    async def stream_run(
        self,
        run: ACPRun,
        poll_interval: float = 0.5
    ) -> AsyncGenerator[ACPRun, None]:
        """
        Stream run updates as they happen.
        
        Args:
            run: The run to stream
            poll_interval: How often to check for updates
            
        Yields:
            Updated ACPRun objects
        """
        last_update = run.updated_at
        
        while True:
            try:
                current_run = await self.get_run_status(run.agent_url, run.run_id)
                
                # Yield if there's an update
                if current_run.updated_at != last_update:
                    last_update = current_run.updated_at
                    yield current_run
                
                # Stop if completed
                if current_run.status in [RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED]:
                    break
                
                await asyncio.sleep(poll_interval)
                
            except Exception as e:
                logger.error(f"Error streaming run {run.run_id}: {e}")
                break
    
    async def execute_simple_run(
        self,
        agent_url: str,
        input_text: str,
        timeout: Optional[float] = None
    ) -> str:
        """
        Execute a simple text-based run and return the result.
        
        Args:
            agent_url: URL of the agent
            input_text: Input text for the agent
            timeout: Maximum time to wait for completion
            
        Returns:
            Output text from the agent
        """
        # Create input message
        from .message import MessageBuilder
        message = (MessageBuilder(role="user")
                  .add_text(input_text)
                  .build())
        
        # Create and execute run
        run = await self.create_run(agent_url, [message])
        completed_run = await self.wait_for_completion(run, timeout=timeout)
        
        # Extract text output
        if completed_run.status == RunStatus.COMPLETED and completed_run.output:
            output_messages = completed_run.output
            if output_messages:
                return output_messages[0].get_text_content()
        
        raise CommunicationError(f"Run failed or produced no output: {completed_run.status.value}")
    
    async def batch_execute(
        self,
        requests: List[Dict[str, Any]],
        max_concurrent: Optional[int] = None
    ) -> List[ACPRun]:
        """
        Execute multiple runs concurrently.
        
        Args:
            requests: List of request dictionaries with 'agent_url' and 'messages'
            max_concurrent: Maximum concurrent runs (defaults to config value)
            
        Returns:
            List of completed ACPRun objects
        """
        if max_concurrent is None:
            max_concurrent = self.config.max_concurrent_runs
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_single(request: Dict[str, Any]) -> ACPRun:
            async with semaphore:
                agent_url = request["agent_url"]
                messages = request["messages"]
                config = request.get("config")
                
                run = await self.create_run(agent_url, messages, config)
                return await self.wait_for_completion(run)
        
        # Execute all requests concurrently
        tasks = [execute_single(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        completed_runs = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch request {i} failed: {result}")
                # Create a failed run object
                failed_run = ACPRun(
                    run_id=f"failed-{i}",
                    status=RunStatus.FAILED,
                    error=str(result)
                )
                completed_runs.append(failed_run)
            else:
                completed_runs.append(result)
        
        logger.info(f"Completed batch execution of {len(requests)} requests")
        return completed_runs
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            **self._stats,
            "success_rate": (
                self._stats["successful_requests"] / max(self._stats["total_requests"], 1)
            ),
            "active_runs_list": list(self._active_runs.keys()),
            "cached_agents_list": list(self._agent_cache.keys())
        }
    
    def clear_cache(self) -> None:
        """Clear the agent cache."""
        self._agent_cache.clear()
        self._stats["cached_agents"] = 0
        logger.info("Cleared agent cache")
    
    async def health_check(self, agent_url: str) -> Dict[str, Any]:
        """
        Perform a health check on an agent.
        
        Args:
            agent_url: URL of the agent
            
        Returns:
            Health check results
        """
        health_url = urljoin(agent_url, "/health")
        
        try:
            start_time = time.time()
            response_data = await self._make_request("GET", health_url)
            response_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "response_time": response_time,
                "agent_url": agent_url,
                "details": response_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "agent_url": agent_url,
                "timestamp": datetime.utcnow().isoformat()
            }


class ACPClientPool:
    """Pool of ACP clients for load balancing and failover."""
    
    def __init__(
        self,
        agent_urls: List[str],
        config: Optional[ClientConfig] = None,
        auth: Optional[AuthConfig] = None
    ):
        """
        Initialize client pool.
        
        Args:
            agent_urls: List of agent URLs
            config: Client configuration
            auth: Authentication configuration
        """
        self.agent_urls = agent_urls
        self.config = config or ClientConfig()
        self.auth = auth or AuthConfig()
        
        self._clients: Dict[str, ACPClient] = {}
        self._health_status: Dict[str, bool] = {}
        self._last_health_check: Dict[str, datetime] = {}
        self._round_robin_index = 0
        
        logger.info(f"Initialized ACP client pool with {len(agent_urls)} agents")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._initialize_clients()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _initialize_clients(self) -> None:
        """Initialize clients for all agent URLs."""
        for url in self.agent_urls:
            client = ACPClient(base_url=url, config=self.config, auth=self.auth)
            await client._ensure_session()
            self._clients[url] = client
            self._health_status[url] = True  # Assume healthy initially
    
    async def close(self) -> None:
        """Close all clients."""
        for client in self._clients.values():
            await client.close()
        logger.info("Closed all clients in pool")
    
    async def _check_health(self, agent_url: str) -> bool:
        """Check health of a specific agent."""
        try:
            client = self._clients[agent_url]
            health_result = await client.health_check(agent_url)
            is_healthy = health_result["status"] == "healthy"
            
            self._health_status[agent_url] = is_healthy
            self._last_health_check[agent_url] = datetime.utcnow()
            
            return is_healthy
            
        except Exception as e:
            logger.warning(f"Health check failed for {agent_url}: {e}")
            self._health_status[agent_url] = False
            self._last_health_check[agent_url] = datetime.utcnow()
            return False
    
    async def get_healthy_client(self) -> Optional[ACPClient]:
        """Get a healthy client using round-robin selection."""
        healthy_urls = [
            url for url, is_healthy in self._health_status.items()
            if is_healthy
        ]
        
        if not healthy_urls:
            # Try to find a healthy client by checking all
            for url in self.agent_urls:
                if await self._check_health(url):
                    healthy_urls.append(url)
        
        if not healthy_urls:
            logger.error("No healthy clients available")
            return None
        
        # Round-robin selection
        selected_url = healthy_urls[self._round_robin_index % len(healthy_urls)]
        self._round_robin_index += 1
        
        return self._clients[selected_url]
    
    async def execute_with_failover(
        self,
        operation: Callable[[ACPClient], Any],
        max_attempts: int = 3
    ) -> Any:
        """
        Execute an operation with automatic failover.
        
        Args:
            operation: Function that takes an ACPClient and returns a result
            max_attempts: Maximum number of attempts across different clients
            
        Returns:
            Result of the operation
        """
        last_exception = None
        
        for attempt in range(max_attempts):
            client = await self.get_healthy_client()
            if not client:
                raise CommunicationError("No healthy clients available")
            
            try:
                result = await operation(client)
                logger.debug(f"Operation succeeded on attempt {attempt + 1}")
                return result
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Operation failed on attempt {attempt + 1}: {e}")
                
                # Mark client as unhealthy
                for url, c in self._clients.items():
                    if c == client:
                        self._health_status[url] = False
                        break
        
        raise CommunicationError(f"Operation failed after {max_attempts} attempts: {last_exception}")
    
    async def broadcast_operation(
        self,
        operation: Callable[[ACPClient], Any]
    ) -> Dict[str, Any]:
        """
        Execute an operation on all healthy clients.
        
        Args:
            operation: Function that takes an ACPClient and returns a result
            
        Returns:
            Dictionary mapping agent URLs to results
        """
        results = {}
        
        for url, client in self._clients.items():
            if self._health_status.get(url, False):
                try:
                    result = await operation(client)
                    results[url] = {"success": True, "result": result}
                except Exception as e:
                    results[url] = {"success": False, "error": str(e)}
                    self._health_status[url] = False
        
        return results
    
    def get_pool_statistics(self) -> Dict[str, Any]:
        """Get statistics for the entire pool."""
        total_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_runs": 0,
            "active_runs": 0,
            "cached_agents": 0
        }
        
        client_stats = {}
        for url, client in self._clients.items():
            stats = client.get_statistics()
            client_stats[url] = {
                **stats,
                "healthy": self._health_status.get(url, False),
                "last_health_check": self._last_health_check.get(url)
            }
            
            # Aggregate totals
            for key in total_stats:
                total_stats[key] += stats.get(key, 0)
        
        return {
            "pool_totals": total_stats,
            "client_details": client_stats,
            "healthy_clients": sum(1 for h in self._health_status.values() if h),
            "total_clients": len(self._clients)
        }


# Example usage and testing
async def example_usage():
    """Example usage of ACP client."""
    
    # Basic client usage
    config = ClientConfig(timeout=30, max_retries=3)
    auth = AuthConfig(auth_type="bearer", token="your-token-here")
    
    async with ACPClient(config=config, auth=auth) as client:
        
        # Discover an agent
        agent_url = "https://api.example.com/agent"
        try:
            agent_detail = await client.discover_agent(agent_url)
            print(f"Discovered agent: {agent_detail.name}")
            print(f"Capabilities: {[cap.name for cap in agent_detail.capabilities]}")
        except AgentNotFoundError:
            print("Agent not found")
            return
        
        # Create a simple message
        from .message import MessageBuilder
        message = (MessageBuilder(role="user")
                  .add_text("Hello, can you help me with text analysis?")
                  .build())
        
        # Create and execute a run
        try:
            run = await client.create_run(agent_url, [message])
            print(f"Created run: {run.run_id}")
            
            # Wait for completion
            completed_run = await client.wait_for_completion(run, timeout=60)
            print(f"Run completed with status: {completed_run.status.value}")
            
            if completed_run.output:
                for output_msg in completed_run.output:
                    print(f"Output: {output_msg.get_text_content()}")
        
        except RunCreationError as e:
            print(f"Failed to create run: {e}")
        except TimeoutError as e:
            print(f"Run timed out: {e}")
        
        # Simple execution
        try:
            result = await client.execute_simple_run(
                agent_url,
                "Analyze the sentiment of this text: I love this product!",
                timeout=30
            )
            print(f"Simple run result: {result}")
        except Exception as e:
            print(f"Simple run failed: {e}")
        
        # Get statistics
        stats = client.get_statistics()
        print(f"Client statistics: {stats}")


async def example_pool_usage():
    """Example usage of ACP client pool."""
    
    agent_urls = [
        "https://agent1.example.com",
        "https://agent2.example.com",
        "https://agent3.example.com"
    ]
    
    config = ClientConfig(timeout=30)
    
    async with ACPClientPool(agent_urls, config=config) as pool:
        
        # Execute with automatic failover
        async def discover_operation(client: ACPClient):
            return await client.discover_agent(client.base_url)
        
        try:
            agent_detail = await pool.execute_with_failover(discover_operation)
            print(f"Discovered agent with failover: {agent_detail.name}")
        except CommunicationError as e:
            print(f"All agents failed: {e}")
        
        # Broadcast operation to all agents
        async def health_check_operation(client: ACPClient):
            return await client.health_check(client.base_url)
        
        health_results = await pool.broadcast_operation(health_check_operation)
        print("Health check results:")
        for url, result in health_results.items():
            status = "✓" if result["success"] else "✗"
            print(f"  {status} {url}")
        
        # Get pool statistics
        pool_stats = pool.get_pool_statistics()
        print(f"Pool statistics: {pool_stats}")


if __name__ == "__main__":
    # Run examples
    print("=== ACP Client Example ===")
    asyncio.run(example_usage())
    
    print("\n=== ACP Client Pool Example ===")
    asyncio.run(example_pool_usage())
    
    print("\n=== ACP Client Implementation Complete! ===")
    print("Features implemented:")
    print("✓ Complete ACP client with discovery, runs, and messaging")
    print("✓ Async/await support with proper session management")
    print("✓ Retry logic with exponential backoff")
    print("✓ Authentication support (Bearer, API Key, Basic)")
    print("✓ SSL/TLS configuration")
    print("✓ Run lifecycle management")
    print("✓ Streaming and polling support")
    print("✓ Batch execution with concurrency control")
    print("✓ Client pooling with failover and load balancing")
    print("✓ Health checking and monitoring")
    print("✓ Comprehensive error handling")
    print("✓ Statistics and metrics collection")
    print("✓ Caching for agent details")
    print("✓ Simple execution helpers")
