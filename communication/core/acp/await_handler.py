"""
ACP Await Handler implementation.

Provides await mechanism functionality for the Agent Communication Protocol (ACP)
allowing agents to pause execution and request information from clients.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Set
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from enum import Enum
import weakref
from concurrent.futures import ThreadPoolExecutor

import logging

logger = logging.getLogger(__name__)


class AwaitError(Exception):
    """Base exception for await-related errors."""
    pass


class AwaitTimeoutError(AwaitError):
    """Raised when await request times out."""
    pass


class AwaitCancelledError(AwaitError):
    """Raised when await request is cancelled."""
    pass


class AwaitValidationError(AwaitError):
    """Raised when await request validation fails."""
    pass


class AwaitNotFoundError(AwaitError):
    """Raised when await request is not found."""
    pass


class AwaitStatus(Enum):
    """Status of an await request."""
    PENDING = "pending"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class AwaitType(Enum):
    """Types of await requests."""
    USER_INPUT = "user_input"
    CONFIRMATION = "confirmation"
    CHOICE = "choice"
    FILE_UPLOAD = "file_upload"
    APPROVAL = "approval"
    CUSTOM = "custom"


@dataclass
class AwaitRequest:
    """
    Represents an await request from an agent.
    
    This allows agents to pause execution and request information
    from the client before continuing.
    """
    
    # Required fields
    await_id: str
    run_id: str
    await_type: AwaitType
    prompt: str
    
    # Optional fields
    options: Optional[List[str]] = None
    default_value: Optional[Any] = None
    validation_schema: Optional[Dict[str, Any]] = None
    timeout: Optional[int] = None  # seconds
    metadata: Optional[Dict[str, Any]] = None
    
    # Status fields
    status: AwaitStatus = AwaitStatus.PENDING
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Response fields
    response: Optional[Any] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        """Initialize computed fields."""
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        
        if self.timeout and self.expires_at is None:
            self.expires_at = self.created_at + timedelta(seconds=self.timeout)
        
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert await request to dictionary."""
        data = {
            "await_id": self.await_id,
            "run_id": self.run_id,
            "await_type": self.await_type.value,
            "prompt": self.prompt,
            "status": self.status.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "options": self.options,
            "default_value": self.default_value,
            "validation_schema": self.validation_schema,
            "timeout": self.timeout,
            "metadata": self.metadata,
            "response": self.response,
            "error": self.error
        }
        return {k: v for k, v in data.items() if v is not None}
    
    def to_json(self) -> str:
        """Convert await request to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AwaitRequest':
        """Create await request from dictionary."""
        # Convert datetime strings
        created_at = None
        if data.get('created_at'):
            created_at = datetime.fromisoformat(data['created_at'])
        
        expires_at = None
        if data.get('expires_at'):
            expires_at = datetime.fromisoformat(data['expires_at'])
        
        completed_at = None
        if data.get('completed_at'):
            completed_at = datetime.fromisoformat(data['completed_at'])
        
        return cls(
            await_id=data['await_id'],
            run_id=data['run_id'],
            await_type=AwaitType(data['await_type']),
            prompt=data['prompt'],
            options=data.get('options'),
            default_value=data.get('default_value'),
            validation_schema=data.get('validation_schema'),
            timeout=data.get('timeout'),
            metadata=data.get('metadata', {}),
            status=AwaitStatus(data.get('status', 'pending')),
            created_at=created_at,
            expires_at=expires_at,
            completed_at=completed_at,
            response=data.get('response'),
            error=data.get('error')
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AwaitRequest':
        """Create await request from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def is_expired(self) -> bool:
        """Check if the await request has expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at
    
    def validate_response(self, response: Any) -> List[str]:
        """
        Validate a response against the await request requirements.
        
        Args:
            response: The response to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check if response is required
        if response is None and self.default_value is None:
            if self.await_type in [AwaitType.USER_INPUT, AwaitType.CHOICE]:
                errors.append("Response is required")
        
        # Validate choice options
        if self.await_type == AwaitType.CHOICE and self.options:
            if response not in self.options:
                errors.append(f"Response must be one of: {self.options}")
        
        # Validate confirmation
        if self.await_type == AwaitType.CONFIRMATION:
            if not isinstance(response, bool):
                errors.append("Confirmation response must be boolean")
        
        # Validate against schema if provided
        if self.validation_schema and response is not None:
            try:
                import jsonschema
                jsonschema.validate(response, self.validation_schema)
            except ImportError:
                logger.warning("jsonschema not available for validation")
            except Exception as e:
                errors.append(f"Schema validation failed: {str(e)}")
        
        return errors


class AwaitHandler(ABC):
    """Abstract base class for await handlers."""
    
    @abstractmethod
    async def handle_await(self, await_request: AwaitRequest) -> Any:
        """
        Handle an await request.
        
        Args:
            await_request: The await request to handle
            
        Returns:
            The response to the await request
        """
        pass
    
    @abstractmethod
    async def cancel_await(self, await_id: str) -> bool:
        """
        Cancel an await request.
        
        Args:
            await_id: ID of the await request to cancel
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        pass


class InteractiveAwaitHandler(AwaitHandler):
    """Interactive await handler that prompts for user input."""
    
    def __init__(self):
        """Initialize interactive handler."""
        self._active_awaits: Dict[str, AwaitRequest] = {}
    
    async def handle_await(self, await_request: AwaitRequest) -> Any:
        """Handle await request interactively."""
        self._active_awaits[await_request.await_id] = await_request
        
        try:
            print(f"\n=== Await Request ===")
            print(f"Type: {await_request.await_type.value}")
            print(f"Prompt: {await_request.prompt}")
            
            if await_request.options:
                print(f"Options: {await_request.options}")
            
            if await_request.default_value is not None:
                print(f"Default: {await_request.default_value}")
            
            if await_request.timeout:
                print(f"Timeout: {await_request.timeout} seconds")
            
            # Get user input based on type
            if await_request.await_type == AwaitType.CONFIRMATION:
                response = await self._get_confirmation()
            elif await_request.await_type == AwaitType.CHOICE:
                response = await self._get_choice(await_request.options)
            elif await_request.await_type == AwaitType.USER_INPUT:
                response = await self._get_user_input()
            else:
                response = await self._get_custom_input(await_request)
            
            # Use default if no response provided
            if response is None and await_request.default_value is not None:
                response = await_request.default_value
            
            # Validate response
            errors = await_request.validate_response(response)
            if errors:
                raise AwaitValidationError(f"Validation errors: {errors}")
            
            return response
            
        finally:
            if await_request.await_id in self._active_awaits:
                del self._active_awaits[await_request.await_id]
    
    async def cancel_await(self, await_id: str) -> bool:
        """Cancel an await request."""
        if await_id in self._active_awaits:
            del self._active_awaits[await_id]
            print(f"Cancelled await request: {await_id}")
            return True
        return False
    
    async def _get_confirmation(self) -> bool:
        """Get confirmation from user."""
        while True:
            response = input("Confirm (y/n): ").strip().lower()
            if response in ['y', 'yes', 'true', '1']:
                return True
            elif response in ['n', 'no', 'false', '0']:
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no")
    
    async def _get_choice(self, options: List[str]) -> str:
        """Get choice from user."""
        while True:
            print("Available options:")
            for i, option in enumerate(options, 1):
                print(f"  {i}. {option}")
            
            response = input("Enter choice (number or text): ").strip()
            
            # Try to parse as number
            try:
                choice_num = int(response)
                if 1 <= choice_num <= len(options):
                    return options[choice_num - 1]
            except ValueError:
                pass
            
            # Try to match text
            if response in options:
                return response
            
            print("Invalid choice. Please try again.")
    
    async def _get_user_input(self) -> str:
        """Get text input from user."""
        return input("Enter response: ").strip()
    
    async def _get_custom_input(self, await_request: AwaitRequest) -> Any:
        """Get custom input based on await request."""
        response = input("Enter response: ").strip()
        
        # Try to parse as JSON if it looks like structured data
        if response.startswith('{') or response.startswith('['):
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                pass
        
        return response


class AutoAwaitHandler(AwaitHandler):
    """Automatic await handler that provides default responses."""
    
    def __init__(self, default_responses: Optional[Dict[str, Any]] = None):
        """
        Initialize auto handler.
        
        Args:
            default_responses: Default responses for different await types
        """
        self.default_responses = default_responses or {
            AwaitType.CONFIRMATION: True,
            AwaitType.USER_INPUT: "Auto-generated response",
            AwaitType.CHOICE: None,  # Will use first option
            AwaitType.APPROVAL: True,
            AwaitType.CUSTOM: "Auto response"
        }
    
    async def handle_await(self, await_request: AwaitRequest) -> Any:
        """Handle await request automatically."""
        logger.info(f"Auto-handling await request: {await_request.await_type.value}")
        
        # Use default value if provided
        if await_request.default_value is not None:
            return await_request.default_value
        
        # Use configured default response
        if await_request.await_type in self.default_responses:
            response = self.default_responses[await_request.await_type]
            
            # Special handling for choices
            if await_request.await_type == AwaitType.CHOICE and response is None:
                if await_request.options:
                    response = await_request.options[0]
            
            return response
        
        # Fallback
        return "Auto-generated response"
    
    async def cancel_await(self, await_id: str) -> bool:
        """Cancel await request (always succeeds for auto handler)."""
        logger.info(f"Auto-cancelled await request: {await_id}")
        return True


class CallbackAwaitHandler(AwaitHandler):
    """Callback-based await handler for custom processing."""
    
    def __init__(self):
        """Initialize callback handler."""
        self._handlers: Dict[AwaitType, Callable] = {}
        self._active_awaits: Dict[str, AwaitRequest] = {}
    
    def register_handler(self, await_type: AwaitType, handler: Callable) -> None:
        """
        Register a handler for a specific await type.
        
        Args:
            await_type: Type of await request
            handler: Callable that handles the await request
        """
        self._handlers[await_type] = handler
        logger.info(f"Registered await handler for type: {await_type.value}")
    
    async def handle_await(self, await_request: AwaitRequest) -> Any:
        """Handle await request using registered handlers."""
        self._active_awaits[await_request.await_id] = await_request
        
        try:
            if await_request.await_type in self._handlers:
                handler = self._handlers[await_request.await_type]
                
                if asyncio.iscoroutinefunction(handler):
                    response = await handler(await_request)
                else:
                    # Run sync handler in thread pool
                    loop = asyncio.get_event_loop()
                    with ThreadPoolExecutor() as executor:
                        response = await loop.run_in_executor(executor, handler, await_request)
                
                return response
            else:
                raise AwaitError(f"No handler registered for await type: {await_request.await_type.value}")
        
        finally:
            if await_request.await_id in self._active_awaits:
                del self._active_awaits[await_request.await_id]
    
    async def cancel_await(self, await_id: str) -> bool:
        """Cancel an await request."""
        if await_id in self._active_awaits:
            del self._active_awaits[await_id]
            logger.info(f"Cancelled await request: {await_id}")
            return True
        return False


class AwaitManager:
    """
    Manages await requests and their lifecycle.
    
    Provides centralized management of await requests including
    timeout handling, status tracking, and response coordination.
    """
    
    def __init__(self, default_timeout: int = 300):
        """
        Initialize await manager.
        
        Args:
            default_timeout: Default timeout for await requests in seconds
        """
        self.default_timeout = default_timeout
        self._awaits: Dict[str, AwaitRequest] = {}
        self._handlers: Dict[str, AwaitHandler] = {}
        self._futures: Dict[str, asyncio.Future] = {}
        self._timeout_tasks: Dict[str, asyncio.Task] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Statistics
        self._stats = {
            "total_awaits": 0,
            "active_awaits": 0,
            "completed_awaits": 0,
            "failed_awaits": 0,
            "cancelled_awaits": 0,
            "timeout_awaits": 0
        }
    
    async def start(self) -> None:
        """Start the await manager."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Await manager started")
    
    async def stop(self) -> None:
        """Stop the await manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all active awaits
        for await_id in list(self._awaits.keys()):
            await self.cancel_await(await_id)
        
        logger.info("Await manager stopped")
    
    def register_handler(self, name: str, handler: AwaitHandler) -> None:
        """
        Register an await handler.
        
        Args:
            name: Name of the handler
            handler: The await handler instance
        """
        self._handlers[name] = handler
        logger.info(f"Registered await handler: {name}")
    
    async def create_await(
        self,
        run_id: str,
        await_type: AwaitType,
        prompt: str,
        options: Optional[List[str]] = None,
        default_value: Optional[Any] = None,
        validation_schema: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        handler_name: Optional[str] = None
    ) -> AwaitRequest:
        """
        Create a new await request.
        
        Args:
            run_id: ID of the run making the request
            await_type: Type of await request
            prompt: Prompt to display to user
            options: Available options for choice requests
            default_value: Default value if no response provided
            validation_schema: JSON schema for response validation
            timeout: Timeout in seconds
            metadata: Additional metadata
            handler_name: Name of handler to use
            
        Returns:
            The created await request
        """
        await_id = str(uuid.uuid4())
        
        await_request = AwaitRequest(
            await_id=await_id,
            run_id=run_id,
            await_type=await_type,
            prompt=prompt,
            options=options,
            default_value=default_value,
            validation_schema=validation_schema,
            timeout=timeout or self.default_timeout,
            metadata=metadata or {}
        )
        
        # Store the request
        self._awaits[await_id] = await_request
        
        # Create future for response
        self._futures[await_id] = asyncio.Future()
        
        # Set up timeout
        if await_request.timeout:
            timeout_task = asyncio.create_task(
                self._handle_timeout(await_id, await_request.timeout)
            )
            self._timeout_tasks[await_id] = timeout_task
        
        # Update statistics
        self._stats["total_awaits"] += 1
        self._stats["active_awaits"] = len([a for a in self._awaits.values() 
                                          if a.status in [AwaitStatus.PENDING, AwaitStatus.WAITING]])
        
        logger.info(f"Created await request {await_id} for run {run_id}")
        return await_request
    
    async def process_await(
        self,
        await_id: str,
        handler_name: Optional[str] = None
    ) -> Any:
        """
        Process an await request.
        
        Args:
            await_id: ID of the await request
            handler_name: Name of handler to use (optional)
            
        Returns:
            The response from the await request
        """
        if await_id not in self._awaits:
            raise AwaitNotFoundError(f"Await request {await_id} not found")
        
        await_request = self._awaits[await_id]
        
        if await_request.status != AwaitStatus.PENDING:
            raise AwaitError(f"Await request {await_id} is not pending")
        
        # Check if expired
        if await_request.is_expired():
            await_request.status = AwaitStatus.TIMEOUT
            await_request.error = "Request expired"
            await_request.completed_at = datetime.utcnow()
            self._stats["timeout_awaits"] += 1
            raise AwaitTimeoutError(f"Await request {await_id} has expired")
        
        # Update status
        await_request.status = AwaitStatus.WAITING
        
        try:
            # Get handler
            handler = self._get_handler(handler_name)
            
            # Process the await request
            response = await handler.handle_await(await_request)
            
            # Validate response
            errors = await_request.validate_response(response)
            if errors:
                raise AwaitValidationError(f"Response validation failed: {errors}")
            
            # Update await request
            await_request.status = AwaitStatus.COMPLETED
            await_request.response = response
            await_request.completed_at = datetime.utcnow()
            
            # Complete the future
            if await_id in self._futures and not self._futures[await_id].done():
                self._futures[await_id].set_result(response)
            
            self._stats["completed_awaits"] += 1
            logger.info(f"Completed await request {await_id}")
            
            return response
            
        except Exception as e:
            # Update await request with error
            await_request.status = AwaitStatus.FAILED
            await_request.error = str(e)
            await_request.completed_at = datetime.utcnow()
            
            # Complete the future with exception
            if await_id in self._futures and not self._futures[await_id].done():
                self._futures[await_id].set_exception(e)
            
            self._stats["failed_awaits"] += 1
            logger.error(f"Failed await request {await_id}: {e}")
            
            raise
        
        finally:
            # Cancel timeout task
            if await_id in self._timeout_tasks:
                self._timeout_tasks[await_id].cancel()
                del self._timeout_tasks[await_id]
            
            # Update active count
            self._stats["active_awaits"] = len([a for a in self._awaits.values() 
                                              if a.status in [AwaitStatus.PENDING, AwaitStatus.WAITING]])
    
    async def respond_to_await(
        self,
        await_id: str,
        response: Any
    ) -> None:
        """
        Respond to an await request directly.
        
        Args:
            await_id: ID of the await request
            response: Response to provide
        """
        if await_id not in self._awaits:
            raise AwaitNotFoundError(f"Await request {await_id} not found")
        
        await_request = self._awaits[await_id]
        
        if await_request.status not in [AwaitStatus.PENDING, AwaitStatus.WAITING]:
            raise AwaitError(f"Await request {await_id} is not waiting for response")
        
        # Validate response
        errors = await_request.validate_response(response)
        if errors:
            raise AwaitValidationError(f"Response validation failed: {errors}")
        
        # Update await request
        await_request.status = AwaitStatus.COMPLETED
        await_request.response = response
        await_request.completed_at = datetime.utcnow()
        
        # Complete the future
        if await_id in self._futures and not self._futures[await_id].done():
            self._futures[await_id].set_result(response)
        
        # Cancel timeout task
        if await_id in self._timeout_tasks:
            self._timeout_tasks[await_id].cancel()
            del self._timeout_tasks[await_id]
        
        self._stats["completed_awaits"] += 1
        logger.info(f"Responded to await request {await_id}")
    
    async def cancel_await(self, await_id: str) -> bool:
        """
        Cancel an await request.
        
        Args:
            await_id: ID of the await request to cancel
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        if await_id not in self._awaits:
            return False
        
        await_request = self._awaits[await_id]
        
        if await_request.status in [AwaitStatus.COMPLETED, AwaitStatus.FAILED, AwaitStatus.CANCELLED]:
            return False
        
        # Update status
        await_request.status = AwaitStatus.CANCELLED
        await_request.completed_at = datetime.utcnow()
        
        # Cancel future
        if await_id in self._futures and not self._futures[await_id].done():
            self._futures[await_id].cancel()
        
        # Cancel timeout task
        if await_id in self._timeout_tasks:
            self._timeout_tasks[await_id].cancel()
            del self._timeout_tasks[await_id]
        
        # Notify handlers
        for handler in self._handlers.values():
            try:
                await handler.cancel_await(await_id)
            except Exception as e:
                logger.warning(f"Handler cancel failed for {await_id}: {e}")
        
        self._stats["cancelled_awaits"] += 1
        logger.info(f"Cancelled await request {await_id}")
        
        return True
    
    async def wait_for_await(self, await_id: str, timeout: Optional[float] = None) -> Any:
        """
        Wait for an await request to complete.
        
        Args:
            await_id: ID of the await request
            timeout: Timeout in seconds
            
        Returns:
            The response from the await request
        """
        if await_id not in self._futures:
            raise AwaitNotFoundError(f"Await request {await_id} not found")
        
        try:
            return await asyncio.wait_for(self._futures[await_id], timeout=timeout)
        except asyncio.TimeoutError:
            raise AwaitTimeoutError(f"Timeout waiting for await request {await_id}")
        except asyncio.CancelledError:
            raise AwaitCancelledError(f"Await request {await_id} was cancelled")
    
    def get_await(self, await_id: str) -> Optional[AwaitRequest]:
        """Get an await request by ID."""
        return self._awaits.get(await_id)
    
    def list_awaits(
        self,
        run_id: Optional[str] = None,
        status: Optional[AwaitStatus] = None
    ) -> List[AwaitRequest]:
        """
        List await requests with optional filtering.
        
        Args:
            run_id: Filter by run ID
            status: Filter by status
            
        Returns:
            List of matching await requests
        """
        awaits = list(self._awaits.values())
        
        if run_id:
            awaits = [a for a in awaits if a.run_id == run_id]
        
        if status:
            awaits = [a for a in awaits if a.status == status]
        
        return awaits
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get await manager statistics."""
        return {
            **self._stats,
            "active_awaits": len([a for a in self._awaits.values() 
                                if a.status in [AwaitStatus.PENDING, AwaitStatus.WAITING]]),
            "total_stored_awaits": len(self._awaits)
        }
    
    def _get_handler(self, handler_name: Optional[str] = None) -> AwaitHandler:
        """Get an await handler by name or default."""
        if handler_name and handler_name in self._handlers:
            return self._handlers[handler_name]
        
        # Return first available handler or create default
        if self._handlers:
            return next(iter(self._handlers.values()))
        
        # Create default auto handler
        return AutoAwaitHandler()
    
    async def _handle_timeout(self, await_id: str, timeout: int) -> None:
        """Handle timeout for an await request."""
        try:
            await asyncio.sleep(timeout)
            
            if await_id in self._awaits:
                await_request = self._awaits[await_id]
                if await_request.status in [AwaitStatus.PENDING, AwaitStatus.WAITING]:
                    await_request.status = AwaitStatus.TIMEOUT
                    await_request.error = "Request timed out"
                    await_request.completed_at = datetime.utcnow()
                    
                    # Complete future with timeout error
                    if await_id in self._futures and not self._futures[await_id].done():
                        self._futures[await_id].set_exception(
                            AwaitTimeoutError(f"Await request {await_id} timed out")
                        )
                    
                    self._stats["timeout_awaits"] += 1
                    logger.warning(f"Await request {await_id} timed out")
        
        except asyncio.CancelledError:
            pass  # Timeout was cancelled
    
    async def _cleanup_loop(self) -> None:
        """Cleanup completed await requests periodically."""
        while True:
            try:
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                await self._cleanup_old_awaits()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Await cleanup loop error: {e}")
    
    async def _cleanup_old_awaits(self) -> None:
        """Remove old completed await requests."""
        cutoff_time = datetime.utcnow() - timedelta(hours=1)  # Keep for 1 hour
        
        awaits_to_remove = []
        for await_id, await_request in self._awaits.items():
            if (await_request.status in [AwaitStatus.COMPLETED, AwaitStatus.FAILED, 
                                       AwaitStatus.CANCELLED, AwaitStatus.TIMEOUT] and
                await_request.completed_at and 
                await_request.completed_at < cutoff_time):
                awaits_to_remove.append(await_id)
        
        for await_id in awaits_to_remove:
            # Clean up associated resources
            if await_id in self._awaits:
                del self._awaits[await_id]
            if await_id in self._futures:
                if not self._futures[await_id].done():
                    self._futures[await_id].cancel()
                del self._futures[await_id]
            if await_id in self._timeout_tasks:
                self._timeout_tasks[await_id].cancel()
                del self._timeout_tasks[await_id]
        
        if awaits_to_remove:
            logger.info(f"Cleaned up {len(awaits_to_remove)} old await requests")


class AwaitRequestBuilder:
    """Builder class for creating await requests."""
    
    def __init__(self, run_id: str, await_type: AwaitType, prompt: str):
        """Initialize builder with required fields."""
        self.run_id = run_id
        self.await_type = await_type
        self.prompt = prompt
        self.options: Optional[List[str]] = None
        self.default_value: Optional[Any] = None
        self.validation_schema: Optional[Dict[str, Any]] = None
        self.timeout: Optional[int] = None
        self.metadata: Optional[Dict[str, Any]] = None
    
    def with_options(self, options: List[str]) -> 'AwaitRequestBuilder':
        """Add options for choice requests."""
        self.options = options
        return self
    
    def with_default(self, default_value: Any) -> 'AwaitRequestBuilder':
        """Add default value."""
        self.default_value = default_value
        return self
    
    def with_validation(self, schema: Dict[str, Any]) -> 'AwaitRequestBuilder':
        """Add validation schema."""
        self.validation_schema = schema
        return self
    
    def with_timeout(self, timeout: int) -> 'AwaitRequestBuilder':
        """Add timeout in seconds."""
        self.timeout = timeout
        return self
    
    def with_metadata(self, metadata: Dict[str, Any]) -> 'AwaitRequestBuilder':
        """Add metadata."""
        self.metadata = metadata
        return self
    
    async def create(self, manager: AwaitManager) -> AwaitRequest:
        """Create the await request using the manager."""
        return await manager.create_await(
            run_id=self.run_id,
            await_type=self.await_type,
            prompt=self.prompt,
            options=self.options,
            default_value=self.default_value,
            validation_schema=self.validation_schema,
            timeout=self.timeout,
            metadata=self.metadata
        )


# Convenience functions for common await patterns
async def await_user_input(
    manager: AwaitManager,
    run_id: str,
    prompt: str,
    default: Optional[str] = None,
    timeout: Optional[int] = None
) -> str:
    """
    Await user text input.
    
    Args:
        manager: Await manager instance
        run_id: ID of the run
        prompt: Prompt to display
        default: Default value
        timeout: Timeout in seconds
        
    Returns:
        User input string
    """
    builder = AwaitRequestBuilder(run_id, AwaitType.USER_INPUT, prompt)
    if default:
        builder.with_default(default)
    if timeout:
        builder.with_timeout(timeout)
    
    await_request = await builder.create(manager)
    await manager.process_await(await_request.await_id)
    return await manager.wait_for_await(await_request.await_id)


async def await_confirmation(
    manager: AwaitManager,
    run_id: str,
    prompt: str,
    default: Optional[bool] = None,
    timeout: Optional[int] = None
) -> bool:
    """
    Await user confirmation.
    
    Args:
        manager: Await manager instance
        run_id: ID of the run
        prompt: Prompt to display
        default: Default value
        timeout: Timeout in seconds
        
    Returns:
        Boolean confirmation
    """
    builder = AwaitRequestBuilder(run_id, AwaitType.CONFIRMATION, prompt)
    if default is not None:
        builder.with_default(default)
    if timeout:
        builder.with_timeout(timeout)
    
    await_request = await builder.create(manager)
    await manager.process_await(await_request.await_id)
    return await manager.wait_for_await(await_request.await_id)


async def await_choice(
    manager: AwaitManager,
    run_id: str,
    prompt: str,
    options: List[str],
    default: Optional[str] = None,
    timeout: Optional[int] = None
) -> str:
    """
    Await user choice from options.
    
    Args:
        manager: Await manager instance
        run_id: ID of the run
        prompt: Prompt to display
        options: Available choices
        default: Default choice
        timeout: Timeout in seconds
        
    Returns:
        Selected choice
    """
    builder = (AwaitRequestBuilder(run_id, AwaitType.CHOICE, prompt)
               .with_options(options))
    if default:
        builder.with_default(default)
    if timeout:
        builder.with_timeout(timeout)
    
    await_request = await builder.create(manager)
    await manager.process_await(await_request.await_id)
    return await manager.wait_for_await(await_request.await_id)


async def await_approval(
    manager: AwaitManager,
    run_id: str,
    prompt: str,
    timeout: Optional[int] = None
) -> bool:
    """
    Await approval from user.
    
    Args:
        manager: Await manager instance
        run_id: ID of the run
        prompt: Prompt to display
        timeout: Timeout in seconds
        
    Returns:
        Boolean approval
    """
    builder = AwaitRequestBuilder(run_id, AwaitType.APPROVAL, prompt)
    if timeout:
        builder.with_timeout(timeout)
    
    await_request = await builder.create(manager)
    await manager.process_await(await_request.await_id)
    return await manager.wait_for_await(await_request.await_id)


# Example usage and testing
async def example_await_usage():
    """Example usage of await functionality."""
    
    # Create await manager
    manager = AwaitManager(default_timeout=60)
    await manager.start()
    
    try:
        # Register handlers
        interactive_handler = InteractiveAwaitHandler()
        auto_handler = AutoAwaitHandler()
        callback_handler = CallbackAwaitHandler()
        
        manager.register_handler("interactive", interactive_handler)
        manager.register_handler("auto", auto_handler)
        manager.register_handler("callback", callback_handler)
        
        # Register custom callback handler
        async def custom_approval_handler(await_request: AwaitRequest) -> bool:
            """Custom approval handler."""
            print(f"Custom approval: {await_request.prompt}")
            # Simulate approval logic
            return True
        
        callback_handler.register_handler(AwaitType.APPROVAL, custom_approval_handler)
        
        print("=== Await Handler Examples ===")
        
        # Example 1: User input with auto handler
        print("\n1. Auto user input:")
        user_input = await await_user_input(
            manager, "run-123", "What is your name?", 
            default="Anonymous", timeout=30
        )
        print(f"Result: {user_input}")
        
        # Example 2: Confirmation with auto handler
        print("\n2. Auto confirmation:")
        confirmed = await await_confirmation(
            manager, "run-123", "Do you want to continue?", 
            default=True, timeout=30
        )
        print(f"Result: {confirmed}")
        
        # Example 3: Choice with auto handler
        print("\n3. Auto choice:")
        choice = await await_choice(
            manager, "run-123", "Select an option:",
            ["Option A", "Option B", "Option C"],
            default="Option A", timeout=30
        )
        print(f"Result: {choice}")
        
        # Example 4: Custom await with callback handler
        print("\n4. Custom approval with callback:")
        approval = await await_approval(manager, "run-123", "Approve this action?", timeout=30)
        print(f"Result: {approval}")
        
        # Example 5: Manual await request creation
        print("\n5. Manual await request:")
        await_request = await manager.create_await(
            run_id="run-456",
            await_type=AwaitType.USER_INPUT,
            prompt="Enter your email:",
            validation_schema={
                "type": "string",
                "format": "email"
            },
            timeout=60
        )
        
        # Simulate response
        await manager.respond_to_await(await_request.await_id, "user@example.com")
        result = await manager.wait_for_await(await_request.await_id)
        print(f"Email result: {result}")
        
        # Show statistics
        print("\n=== Statistics ===")
        stats = manager.get_statistics()
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # List all awaits
        print("\n=== All Awaits ===")
        all_awaits = manager.list_awaits()
        for await_req in all_awaits:
            print(f"- {await_req.await_id}: {await_req.status.value} ({await_req.await_type.value})")
    
    finally:
        await manager.stop()


def create_test_scenarios():
    """Create test scenarios for await functionality."""
    
    async def test_timeout_scenario():
        """Test timeout handling."""
        manager = AwaitManager(default_timeout=2)  # 2 second timeout
        await manager.start()
        
        try:
            # Create await that will timeout
            await_request = await manager.create_await(
                run_id="timeout-test",
                await_type=AwaitType.USER_INPUT,
                prompt="This will timeout",
                timeout=1  # 1 second timeout
            )
            
            # Don't process it, let it timeout
            try:
                result = await manager.wait_for_await(await_request.await_id, timeout=2)
                print(f"Unexpected result: {result}")
            except AwaitTimeoutError:
                print("✓ Timeout handled correctly")
            
            # Check status
            final_request = manager.get_await(await_request.await_id)
            print(f"Final status: {final_request.status.value}")
            
        finally:
            await manager.stop()
    
    async def test_validation_scenario():
        """Test response validation."""
        manager = AwaitManager()
        await manager.start()
        
        try:
            # Create await with validation
            await_request = await manager.create_await(
                run_id="validation-test",
                await_type=AwaitType.USER_INPUT,
                prompt="Enter a number:",
                validation_schema={
                    "type": "number",
                    "minimum": 0,
                    "maximum": 100
                }
            )
            
            # Try invalid response
            try:
                await manager.respond_to_await(await_request.await_id, "not a number")
                print("✗ Validation should have failed")
            except AwaitValidationError:
                print("✓ Validation correctly rejected invalid input")
            
            # Try valid response
            await manager.respond_to_await(await_request.await_id, 42)
            result = await manager.wait_for_await(await_request.await_id)
            print(f"✓ Valid input accepted: {result}")
            
        finally:
            await manager.stop()
    
    return [test_timeout_scenario, test_validation_scenario]


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run test scenarios
        async def run_tests():
            test_scenarios = create_test_scenarios()
            for i, test in enumerate(test_scenarios, 1):
                print(f"\n=== Test Scenario {i}: {test.__name__} ===")
                await test()
        
        asyncio.run(run_tests())
    else:
        # Run example usage
        asyncio.run(example_await_usage())
    
    print("\n=== ACP Await Handler Implementation Complete! ===")
    print("Features implemented:")
    print("✓ Complete await request lifecycle management")
    print("✓ Multiple await handler types (Interactive, Auto, Callback)")
    print("✓ Timeout handling and automatic cleanup")
    print("✓ Response validation with JSON schema support")
    print("✓ Centralized await manager with statistics")
    print("✓ Builder pattern for easy await creation")
    print("✓ Convenience functions for common patterns")
    print("✓ Comprehensive error handling")
    print("✓ Async/await throughout for performance")
    print("✓ Cancellation support")
    print("✓ Status tracking and monitoring")
    print("✓ Flexible handler registration system")
    print("✓ Production-ready resource management")
