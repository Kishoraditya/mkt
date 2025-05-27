"""
ACP Run implementation.

Handles run lifecycle management for Agent Communication Protocol (ACP).
A run represents a single agent execution with specific inputs and manages
the complete lifecycle from creation to completion.
"""

import json
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, AsyncGenerator
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future

import logging

logger = logging.getLogger(__name__)


class RunStatus(Enum):
    """Status values for ACP runs."""
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    AWAITING = "awaiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class RunPriority(Enum):
    """Priority levels for run execution."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class RunError(Exception):
    """Base exception for run-related errors."""
    pass


class RunValidationError(RunError):
    """Raised when run validation fails."""
    pass


class RunExecutionError(RunError):
    """Raised when run execution fails."""
    pass


class RunTimeoutError(RunError):
    """Raised when run execution times out."""
    pass


class RunCancellationError(RunError):
    """Raised when run is cancelled."""
    pass


@dataclass
class RunInput:
    """Input data for a run."""
    
    messages: List[Dict[str, Any]]
    parameters: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    
    def validate(self) -> List[str]:
        """Validate run input."""
        errors = []
        
        if not self.messages:
            errors.append("At least one input message is required")
        
        for i, message in enumerate(self.messages):
            if not isinstance(message, dict):
                errors.append(f"Message {i} must be a dictionary")
            elif "role" not in message:
                errors.append(f"Message {i} must have a 'role' field")
            elif "parts" not in message:
                errors.append(f"Message {i} must have a 'parts' field")
        
        return errors


@dataclass
class RunOutput:
    """Output data from a run."""
    
    messages: List[Dict[str, Any]]
    artifacts: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.artifacts is None:
            self.artifacts = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RunMetrics:
    """Performance metrics for a run."""
    
    execution_time: Optional[float] = None
    queue_time: Optional[float] = None
    await_time: Optional[float] = None
    total_time: Optional[float] = None
    memory_usage: Optional[int] = None
    cpu_usage: Optional[float] = None
    message_count: Optional[int] = None
    token_count: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class RunEventType(Enum):
    """Types of run events."""
    CREATED = "created"
    QUEUED = "queued"
    STARTED = "started"
    MESSAGE_SENT = "message_sent"
    MESSAGE_RECEIVED = "message_received"
    AWAIT_REQUESTED = "await_requested"
    AWAIT_RESOLVED = "await_resolved"
    PROGRESS_UPDATE = "progress_update"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class RunEvent:
    """Event that occurs during run execution."""
    
    event_id: str
    run_id: str
    event_type: RunEventType
    timestamp: datetime
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.data is None:
            self.data = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "run_id": self.run_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "message": self.message
        }


class ACPRun:
    """
    ACP Run implementation following Agent Communication Protocol specification.
    
    Represents a single agent execution with complete lifecycle management,
    event tracking, and support for asynchronous operations.
    """
    
    def __init__(
        self,
        run_id: str,
        agent_id: str,
        input_data: RunInput,
        priority: RunPriority = RunPriority.NORMAL,
        timeout: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize ACP run."""
        self.run_id = run_id
        self.agent_id = agent_id
        self.input_data = input_data
        self.priority = priority
        self.timeout = timeout
        self.metadata = metadata or {}
        
        # Status and timing
        self.status = RunStatus.CREATED
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.queued_at: Optional[datetime] = None
        
        # Output and results
        self.output_data: Optional[RunOutput] = None
        self.error: Optional[str] = None
        self.error_details: Optional[Dict[str, Any]] = None
        
        # Metrics and monitoring
        self.metrics = RunMetrics()
        self.events: List[RunEvent] = []
        self.progress: float = 0.0
        self.progress_message: Optional[str] = None
        
        # Execution control
        self._lock = threading.RLock()
        self._cancelled = threading.Event()
        self._await_requests: Dict[str, Any] = {}
        self._event_handlers: List[Callable[[RunEvent], None]] = []
        
        # Add creation event
        self._add_event(RunEventType.CREATED, message="Run created")
        
        logger.info(f"Created ACP run {self.run_id} for agent {self.agent_id}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert run to dictionary representation."""
        data = {
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "progress": self.progress,
            "metadata": self.metadata,
            "input_data": {
                "messages": self.input_data.messages,
                "parameters": self.input_data.parameters,
                "context": self.input_data.context
            }
        }
        
        # Add optional fields
        if self.started_at:
            data["started_at"] = self.started_at.isoformat()
        if self.completed_at:
            data["completed_at"] = self.completed_at.isoformat()
        if self.queued_at:
            data["queued_at"] = self.queued_at.isoformat()
        if self.timeout:
            data["timeout"] = self.timeout
        if self.output_data:
            data["output_data"] = {
                "messages": self.output_data.messages,
                "artifacts": self.output_data.artifacts,
                "metadata": self.output_data.metadata
            }
        if self.error:
            data["error"] = self.error
        if self.error_details:
            data["error_details"] = self.error_details
        if self.progress_message:
            data["progress_message"] = self.progress_message
        
        # Add metrics
        data["metrics"] = self.metrics.to_dict()
        
        return data
    
    def to_json(self) -> str:
        """Convert run to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ACPRun':
        """Create run from dictionary."""
        # Create input data
        input_dict = data["input_data"]
        input_data = RunInput(
            messages=input_dict["messages"],
            parameters=input_dict.get("parameters"),
            context=input_dict.get("context")
        )
        
        # Create run
        run = cls(
            run_id=data["run_id"],
            agent_id=data["agent_id"],
            input_data=input_data,
            priority=RunPriority(data["priority"]),
            timeout=data.get("timeout"),
            metadata=data.get("metadata", {})
        )
        
        # Set status and timestamps
        run.status = RunStatus(data["status"])
        run.created_at = datetime.fromisoformat(data["created_at"])
        run.progress = data.get("progress", 0.0)
        
        if data.get("started_at"):
            run.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            run.completed_at = datetime.fromisoformat(data["completed_at"])
        if data.get("queued_at"):
            run.queued_at = datetime.fromisoformat(data["queued_at"])
        
        # Set output data
        if data.get("output_data"):
            output_dict = data["output_data"]
            run.output_data = RunOutput(
                messages=output_dict["messages"],
                artifacts=output_dict.get("artifacts"),
                metadata=output_dict.get("metadata")
            )
        
        # Set error information
        run.error = data.get("error")
        run.error_details = data.get("error_details")
        run.progress_message = data.get("progress_message")
        
        # Set metrics
        if data.get("metrics"):
            metrics_dict = data["metrics"]
            run.metrics = RunMetrics(**metrics_dict)
        
        return run
    
    def validate(self) -> List[str]:
        """Validate run configuration."""
        errors = []
        
        if not self.run_id:
            errors.append("run_id is required")
        if not self.agent_id:
            errors.append("agent_id is required")
        
        # Validate input data
        input_errors = self.input_data.validate()
        errors.extend(input_errors)
        
        # Validate timeout
        if self.timeout is not None and self.timeout <= 0:
            errors.append("timeout must be positive")
        
        return errors
    
    def add_event_handler(self, handler: Callable[[RunEvent], None]) -> None:
        """Add event handler for run events."""
        with self._lock:
            self._event_handlers.append(handler)
    
    def remove_event_handler(self, handler: Callable[[RunEvent], None]) -> None:
        """Remove event handler."""
        with self._lock:
            if handler in self._event_handlers:
                self._event_handlers.remove(handler)
    
    def _add_event(
        self, 
        event_type: RunEventType, 
        data: Optional[Dict[str, Any]] = None,
        message: Optional[str] = None
    ) -> RunEvent:
        """Add event to run history."""
        event = RunEvent(
            event_id=str(uuid.uuid4()),
            run_id=self.run_id,
            event_type=event_type,
            timestamp=datetime.utcnow(),
            data=data,
            message=message
        )
        
        with self._lock:
            self.events.append(event)
            
            # Notify event handlers
            for handler in self._event_handlers:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Error in event handler: {e}")
        
        logger.debug(f"Added event {event_type.value} to run {self.run_id}")
        return event
    
    def queue(self) -> None:
        """Mark run as queued."""
        with self._lock:
            if self.status != RunStatus.CREATED:
                raise RunError(f"Cannot queue run in status {self.status.value}")
            
            self.status = RunStatus.QUEUED
            self.queued_at = datetime.utcnow()
            self._add_event(RunEventType.QUEUED, message="Run queued for execution")
    
    def start(self) -> None:
        """Start run execution."""
        with self._lock:
            if self.status not in [RunStatus.CREATED, RunStatus.QUEUED]:
                raise RunError(f"Cannot start run in status {self.status.value}")
            
            self.status = RunStatus.RUNNING
            self.started_at = datetime.utcnow()
            
            # Calculate queue time
            if self.queued_at:
                self.metrics.queue_time = (self.started_at - self.queued_at).total_seconds()
            
            self._add_event(RunEventType.STARTED, message="Run execution started")
    
    def update_progress(self, progress: float, message: Optional[str] = None) -> None:
        """Update run progress."""
        with self._lock:
            self.progress = max(0.0, min(1.0, progress))
            self.progress_message = message
            
            self._add_event(
                RunEventType.PROGRESS_UPDATE,
                data={"progress": self.progress, "message": message},
                message=f"Progress: {self.progress:.1%}"
            )
    
    def request_await(
        self, 
        await_id: str, 
        await_type: str, 
        prompt: str,
        options: Optional[Dict[str, Any]] = None
    ) -> None:
        """Request await from client."""
        with self._lock:
            if self.status != RunStatus.RUNNING:
                raise RunError(f"Cannot request await in status {self.status.value}")
            
            self.status = RunStatus.AWAITING
            await_request = {
                "await_id": await_id,
                "await_type": await_type,
                "prompt": prompt,
                "options": options or {},
                "requested_at": datetime.utcnow().isoformat()
            }
            
            self._await_requests[await_id] = await_request
            
            self._add_event(
                RunEventType.AWAIT_REQUESTED,
                data=await_request,
                message=f"Await requested: {prompt}"
            )
    
    def resolve_await(self, await_id: str, response: Any) -> None:
        """Resolve await request with response."""
        with self._lock:
            if await_id not in self._await_requests:
                raise RunError(f"Await request {await_id} not found")
            
            await_request = self._await_requests[await_id]
            await_request["response"] = response
            await_request["resolved_at"] = datetime.utcnow().isoformat()
            
            # Calculate await time
            requested_at = datetime.fromisoformat(await_request["requested_at"])
            resolved_at = datetime.fromisoformat(await_request["resolved_at"])
            await_time = (resolved_at - requested_at).total_seconds()
            
            if self.metrics.await_time is None:
                self.metrics.await_time = await_time
            else:
                self.metrics.await_time += await_time
            
            self.status = RunStatus.RUNNING
            
            self._add_event(
                RunEventType.AWAIT_RESOLVED,
                data={"await_id": await_id, "response": response, "await_time": await_time},
                message=f"Await {await_id} resolved"
            )
    
    def add_message(self, message: Dict[str, Any], direction: str = "outbound") -> None:
        """Add message to run (sent or received)."""
        event_type = RunEventType.MESSAGE_SENT if direction == "outbound" else RunEventType.MESSAGE_RECEIVED
        
        self._add_event(
            event_type,
            data={"message": message, "direction": direction},
            message=f"Message {direction}: {message.get('message_id', 'unknown')}"
        )
        
        # Update message count
        if self.metrics.message_count is None:
            self.metrics.message_count = 1
        else:
            self.metrics.message_count += 1
    
    def complete(self, output_data: RunOutput) -> None:
        """Complete run with output data."""
        with self._lock:
            if self.status in [RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED]:
                raise RunError(f"Run already finished with status {self.status.value}")
            
            self.status = RunStatus.COMPLETED
            self.completed_at = datetime.utcnow()
            self.output_data = output_data
            self.progress = 1.0
            
            # Calculate execution time
            if self.started_at:
                self.metrics.execution_time = (self.completed_at - self.started_at).total_seconds()
            
            # Calculate total time
            self.metrics.total_time = (self.completed_at - self.created_at).total_seconds()
            
            self._add_event(
                RunEventType.COMPLETED,
                data={"output_messages": len(output_data.messages)},
                message="Run completed successfully"
            )
            
            logger.info(f"Run {self.run_id} completed successfully")
    
    def fail(self, error: str, error_details: Optional[Dict[str, Any]] = None) -> None:
        """Fail run with error information."""
        with self._lock:
            if self.status in [RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED]:
                raise RunError(f"Run already finished with status {self.status.value}")
            
            self.status = RunStatus.FAILED
            self.completed_at = datetime.utcnow()
            self.error = error
            self.error_details = error_details or {}
            
            # Calculate execution time if started
            if self.started_at:
                self.metrics.execution_time = (self.completed_at - self.started_at).total_seconds()
            
            # Calculate total time
            self.metrics.total_time = (self.completed_at - self.created_at).total_seconds()
            
            self._add_event(
                RunEventType.FAILED,
                data={"error": error, "error_details": error_details},
                message=f"Run failed: {error}"
            )
            
            logger.error(f"Run {self.run_id} failed: {error}")
    
    def cancel(self, reason: Optional[str] = None) -> None:
        """Cancel run execution."""
        with self._lock:
            if self.status in [RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED]:
                raise RunError(f"Cannot cancel run in status {self.status.value}")
            
            self.status = RunStatus.CANCELLED
            self.completed_at = datetime.utcnow()
            self.error = reason or "Run cancelled"
            
            # Set cancellation flag
            self._cancelled.set()
            
            # Calculate execution time if started
            if self.started_at:
                self.metrics.execution_time = (self.completed_at - self.started_at).total_seconds()
            
            # Calculate total time
            self.metrics.total_time = (self.completed_at - self.created_at).total_seconds()
            
            self._add_event(
                RunEventType.CANCELLED,
                data={"reason": reason},
                message=f"Run cancelled: {reason or 'No reason provided'}"
            )
            
            logger.info(f"Run {self.run_id} cancelled: {reason}")
    
    def timeout_run(self) -> None:
        """Mark run as timed out."""
        with self._lock:
            if self.status in [RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED]:
                return  # Already finished
            
            self.status = RunStatus.TIMEOUT
            self.completed_at = datetime.utcnow()
            self.error = f"Run timed out after {self.timeout} seconds"
            
            # Calculate execution time if started
            if self.started_at:
                self.metrics.execution_time = (self.completed_at - self.started_at).total_seconds()
            
            # Calculate total time
            self.metrics.total_time = (self.completed_at - self.created_at).total_seconds()
            
            self._add_event(
                RunEventType.TIMEOUT,
                data={"timeout": self.timeout},
                message=f"Run timed out after {self.timeout} seconds"
            )
            
            logger.warning(f"Run {self.run_id} timed out after {self.timeout} seconds")
    
    def is_finished(self) -> bool:
        """Check if run is in a finished state."""
        return self.status in [
            RunStatus.COMPLETED, 
            RunStatus.FAILED, 
            RunStatus.CANCELLED, 
            RunStatus.TIMEOUT
        ]
    
    def is_cancelled(self) -> bool:
        """Check if run is cancelled."""
        return self._cancelled.is_set()
    
    def get_duration(self) -> Optional[float]:
        """Get run duration in seconds."""
        if not self.started_at:
            return None
        
        end_time = self.completed_at or datetime.utcnow()
        return (end_time - self.started_at).total_seconds()
    
    def get_total_duration(self) -> float:
        """Get total duration from creation to completion."""
        end_time = self.completed_at or datetime.utcnow()
        return (end_time - self.created_at).total_seconds()
    
    def get_events_by_type(self, event_type: RunEventType) -> List[RunEvent]:
        """Get events of specific type."""
        return [event for event in self.events if event.event_type == event_type]
    
    def get_await_requests(self) -> Dict[str, Any]:
        """Get all await requests."""
        with self._lock:
            return self._await_requests.copy()
    
    def get_pending_awaits(self) -> Dict[str, Any]:
        """Get pending await requests."""
        with self._lock:
            return {
                await_id: request 
                for await_id, request in self._await_requests.items()
                if "resolved_at" not in request
            }
    
    def update_metrics(self, **kwargs) -> None:
        """Update run metrics."""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self.metrics, key):
                    setattr(self.metrics, key, value)


class RunManager:
    """Manager for handling multiple ACP runs."""
    
    def __init__(self, max_concurrent_runs: int = 10):
        """Initialize run manager."""
        self.max_concurrent_runs = max_concurrent_runs
        self.runs: Dict[str, ACPRun] = {}
        self.run_queue: List[str] = []
        self.active_runs: Dict[str, Future] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_runs)
        self._lock = threading.RLock()
        self._shutdown = False
        
        # Statistics
        self.stats = {
            "total_runs": 0,
            "completed_runs": 0,
            "failed_runs": 0,
            "cancelled_runs": 0,
            "timeout_runs": 0
        }
        
        logger.info(f"Initialized run manager with {max_concurrent_runs} max concurrent runs")
    
    def create_run(
        self,
        agent_id: str,
        input_data: RunInput,
        run_id: Optional[str] = None,
        priority: RunPriority = RunPriority.NORMAL,
        timeout: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ACPRun:
        """Create a new run."""
        if run_id is None:
            run_id = str(uuid.uuid4())
        
        # Validate input
        validation_errors = input_data.validate()
        if validation_errors:
            raise RunValidationError(f"Invalid input data: {validation_errors}")
        
        run = ACPRun(
            run_id=run_id,
            agent_id=agent_id,
            input_data=input_data,
            priority=priority,
            timeout=timeout,
            metadata=metadata
        )
        
        # Validate run
        run_errors = run.validate()
        if run_errors:
            raise RunValidationError(f"Invalid run configuration: {run_errors}")
        
        with self._lock:
            if run_id in self.runs:
                raise RunError(f"Run {run_id} already exists")
            
            self.runs[run_id] = run
            self.stats["total_runs"] += 1
        
        logger.info(f"Created run {run_id} for agent {agent_id}")
        return run
    
    def submit_run(self, run_id: str, executor_func: Callable[[ACPRun], None]) -> None:
        """Submit run for execution."""
        with self._lock:
            if self._shutdown:
                raise RunError("Run manager is shutting down")
            
            if run_id not in self.runs:
                raise RunError(f"Run {run_id} not found")
            
            run = self.runs[run_id]
            
            if run.status != RunStatus.CREATED:
                raise RunError(f"Run {run_id} already submitted or finished")
            
            # Queue run based on priority
            self._queue_run(run_id)
            
            # Try to start run immediately if capacity available
            self._try_start_queued_runs(executor_func)
    
    def _queue_run(self, run_id: str) -> None:
        """Queue run based on priority."""
        run = self.runs[run_id]
        run.queue()
        
        # Insert based on priority
        if run.priority == RunPriority.URGENT:
            self.run_queue.insert(0, run_id)
        elif run.priority == RunPriority.HIGH:
            # Insert after other urgent runs
            insert_pos = 0
            for i, queued_id in enumerate(self.run_queue):
                if self.runs[queued_id].priority != RunPriority.URGENT:
                    insert_pos = i
                    break
            else:
                insert_pos = len(self.run_queue)
            self.run_queue.insert(insert_pos, run_id)
        else:
            # Normal and low priority go to the end
            self.run_queue.append(run_id)
    
    def _try_start_queued_runs(self, executor_func: Callable[[ACPRun], None]) -> None:
        """Try to start queued runs if capacity available."""
        while (len(self.active_runs) < self.max_concurrent_runs and 
               self.run_queue and not self._shutdown):
            
            run_id = self.run_queue.pop(0)
            run = self.runs[run_id]
            
            if run.status != RunStatus.QUEUED:
                continue  # Skip if status changed
            
            # Start run
            run.start()
            
            # Submit to executor
            future = self.executor.submit(self._execute_run, run, executor_func)
            self.active_runs[run_id] = future
            
            logger.info(f"Started execution of run {run_id}")
    
    def _execute_run(self, run: ACPRun, executor_func: Callable[[ACPRun], None]) -> None:
        """Execute run with timeout and error handling."""
        try:
            # Set up timeout if specified
            if run.timeout:
                timeout_timer = threading.Timer(run.timeout, run.timeout_run)
                timeout_timer.start()
            else:
                timeout_timer = None
            
            try:
                # Execute the run
                executor_func(run)
                
                # If run is still running, it means executor didn't complete it
                if run.status == RunStatus.RUNNING:
                    run.fail("Executor function completed without setting run status")
                
            finally:
                # Cancel timeout timer
                if timeout_timer:
                    timeout_timer.cancel()
            
        except RunCancellationError:
            # Run was cancelled
            pass
        except RunTimeoutError:
            # Run timed out
            pass
        except Exception as e:
            # Execution error
            if not run.is_finished():
                run.fail(f"Execution error: {str(e)}", {"exception_type": type(e).__name__})
            logger.error(f"Error executing run {run.run_id}: {e}")
        
        finally:
            # Clean up
            with self._lock:
                if run.run_id in self.active_runs:
                    del self.active_runs[run.run_id]
                
                # Update statistics
                if run.status == RunStatus.COMPLETED:
                    self.stats["completed_runs"] += 1
                elif run.status == RunStatus.FAILED:
                    self.stats["failed_runs"] += 1
                elif run.status == RunStatus.CANCELLED:
                    self.stats["cancelled_runs"] += 1
                elif run.status == RunStatus.TIMEOUT:
                    self.stats["timeout_runs"] += 1
                
                # Try to start more queued runs
                self._try_start_queued_runs(executor_func)
    
    def get_run(self, run_id: str) -> Optional[ACPRun]:
        """Get run by ID."""
        return self.runs.get(run_id)
    
    def cancel_run(self, run_id: str, reason: Optional[str] = None) -> bool:
        """Cancel a run."""
        with self._lock:
            if run_id not in self.runs:
                return False
            
            run = self.runs[run_id]
            
            if run.is_finished():
                return False
            
            # Remove from queue if queued
            if run_id in self.run_queue:
                self.run_queue.remove(run_id)
            
            # Cancel active run
            if run_id in self.active_runs:
                future = self.active_runs[run_id]
                future.cancel()
            
            # Mark run as cancelled
            run.cancel(reason)
            
            logger.info(f"Cancelled run {run_id}: {reason}")
            return True
    
    def get_runs_by_status(self, status: RunStatus) -> List[ACPRun]:
        """Get runs by status."""
        return [run for run in self.runs.values() if run.status == status]
    
    def get_runs_by_agent(self, agent_id: str) -> List[ACPRun]:
        """Get runs by agent ID."""
        return [run for run in self.runs.values() if run.agent_id == agent_id]
    
    def get_active_runs(self) -> List[ACPRun]:
        """Get currently active runs."""
        with self._lock:
            return [self.runs[run_id] for run_id in self.active_runs.keys()]
    
    def get_queued_runs(self) -> List[ACPRun]:
        """Get queued runs in order."""
        return [self.runs[run_id] for run_id in self.run_queue]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get run manager statistics."""
        with self._lock:
            stats = self.stats.copy()
            stats.update({
                "active_runs": len(self.active_runs),
                "queued_runs": len(self.run_queue),
                "total_stored_runs": len(self.runs),
                "max_concurrent_runs": self.max_concurrent_runs
            })
            return stats
    
    def cleanup_finished_runs(self, max_age_hours: int = 24) -> int:
        """Clean up finished runs older than specified age."""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        cleaned_count = 0
        
        with self._lock:
            runs_to_remove = []
            
            for run_id, run in self.runs.items():
                if (run.is_finished() and 
                    run.completed_at and 
                    run.completed_at < cutoff_time):
                    runs_to_remove.append(run_id)
            
            for run_id in runs_to_remove:
                del self.runs[run_id]
                cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} finished runs older than {max_age_hours} hours")
        
        return cleaned_count
    
    def shutdown(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        """Shutdown run manager."""
        with self._lock:
            self._shutdown = True
            
            # Cancel all queued runs
            for run_id in self.run_queue.copy():
                self.cancel_run(run_id, "Run manager shutting down")
            
            # Cancel all active runs
            for run_id in list(self.active_runs.keys()):
                self.cancel_run(run_id, "Run manager shutting down")
        
        if wait:
            self.executor.shutdown(wait=True, timeout=timeout)
        
        logger.info("Run manager shutdown complete")


class RunBuilder:
    """Builder for creating ACP runs with fluent interface."""
    
    def __init__(self, agent_id: str):
        """Initialize run builder."""
        self.agent_id = agent_id
        self.run_id: Optional[str] = None
        self.messages: List[Dict[str, Any]] = []
        self.parameters: Dict[str, Any] = {}
        self.context: Dict[str, Any] = {}
        self.priority = RunPriority.NORMAL
        self.timeout: Optional[int] = None
        self.metadata: Dict[str, Any] = {}
    
    def set_run_id(self, run_id: str) -> 'RunBuilder':
        """Set run ID."""
        self.run_id = run_id
        return self
    
    def add_message(self, message: Dict[str, Any]) -> 'RunBuilder':
        """Add input message."""
        self.messages.append(message)
        return self
    
    def add_text_message(self, role: str, text: str) -> 'RunBuilder':
        """Add text message."""
        message = {
            "role": role,
            "parts": [{"type": "text", "text": text}]
        }
        return self.add_message(message)
    
    def set_parameters(self, parameters: Dict[str, Any]) -> 'RunBuilder':
        """Set run parameters."""
        self.parameters = parameters
        return self
    
    def add_parameter(self, key: str, value: Any) -> 'RunBuilder':
        """Add single parameter."""
        self.parameters[key] = value
        return self
    
    def set_context(self, context: Dict[str, Any]) -> 'RunBuilder':
        """Set run context."""
        self.context = context
        return self
    
    def add_context(self, key: str, value: Any) -> 'RunBuilder':
        """Add single context item."""
        self.context[key] = value
        return self
    
    def set_priority(self, priority: RunPriority) -> 'RunBuilder':
        """Set run priority."""
        self.priority = priority
        return self
    
    def set_timeout(self, timeout: int) -> 'RunBuilder':
        """Set run timeout in seconds."""
        self.timeout = timeout
        return self
    
    def set_metadata(self, metadata: Dict[str, Any]) -> 'RunBuilder':
        """Set run metadata."""
        self.metadata = metadata
        return self
    
    def add_metadata(self, key: str, value: Any) -> 'RunBuilder':
        """Add single metadata item."""
        self.metadata[key] = value
        return self
    
    def build(self) -> ACPRun:
        """Build the ACP run."""
        if not self.messages:
            raise RunValidationError("At least one input message is required")
        
        input_data = RunInput(
            messages=self.messages,
            parameters=self.parameters if self.parameters else None,
            context=self.context if self.context else None
        )
        
        run = ACPRun(
            run_id=self.run_id or str(uuid.uuid4()),
            agent_id=self.agent_id,
            input_data=input_data,
            priority=self.priority,
            timeout=self.timeout,
            metadata=self.metadata if self.metadata else None
        )
        
        logger.info(f"Built run {run.run_id} for agent {self.agent_id}")
        return run


class RunMonitor:
    """Monitor for tracking run performance and health."""
    
    def __init__(self):
        """Initialize run monitor."""
        self.metrics_history: List[Dict[str, Any]] = []
        self.alert_handlers: List[Callable[[str, Dict[str, Any]], None]] = []
        self.thresholds = {
            "max_execution_time": 300,  # 5 minutes
            "max_queue_time": 60,       # 1 minute
            "max_failure_rate": 0.1,    # 10%
            "max_timeout_rate": 0.05    # 5%
        }
    
    def add_alert_handler(self, handler: Callable[[str, Dict[str, Any]], None]) -> None:
        """Add alert handler."""
        self.alert_handlers.append(handler)
    
    def set_threshold(self, metric: str, value: float) -> None:
        """Set alert threshold."""
        self.thresholds[metric] = value
    
    def record_run_metrics(self, run: ACPRun) -> None:
        """Record metrics for a completed run."""
        if not run.is_finished():
            return
        
        metrics = {
            "run_id": run.run_id,
            "agent_id": run.agent_id,
            "status": run.status.value,
            "priority": run.priority.value,
            "execution_time": run.metrics.execution_time,
            "queue_time": run.metrics.queue_time,
            "total_time": run.metrics.total_time,
            "message_count": run.metrics.message_count,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.metrics_history.append(metrics)
        
        # Check for alerts
        self._check_alerts(run, metrics)
        
        # Limit history size
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
    
    def _check_alerts(self, run: ACPRun, metrics: Dict[str, Any]) -> None:
        """Check for alert conditions."""
        alerts = []
        
        # Check execution time
        if (run.metrics.execution_time and 
            run.metrics.execution_time > self.thresholds["max_execution_time"]):
            alerts.append({
                "type": "long_execution_time",
                "message": f"Run {run.run_id} took {run.metrics.execution_time:.1f}s to execute",
                "threshold": self.thresholds["max_execution_time"],
                "actual": run.metrics.execution_time
            })
        
        # Check queue time
        if (run.metrics.queue_time and 
            run.metrics.queue_time > self.thresholds["max_queue_time"]):
            alerts.append({
                "type": "long_queue_time",
                "message": f"Run {run.run_id} was queued for {run.metrics.queue_time:.1f}s",
                "threshold": self.thresholds["max_queue_time"],
                "actual": run.metrics.queue_time
            })
        
        # Check recent failure rate
        recent_runs = self._get_recent_runs(minutes=10)
        if len(recent_runs) >= 10:  # Only check if we have enough data
            failure_rate = len([r for r in recent_runs if r["status"] == "failed"]) / len(recent_runs)
            if failure_rate > self.thresholds["max_failure_rate"]:
                alerts.append({
                    "type": "high_failure_rate",
                    "message": f"Failure rate is {failure_rate:.1%} in last 10 minutes",
                    "threshold": self.thresholds["max_failure_rate"],
                    "actual": failure_rate
                })
        
        # Send alerts
        for alert in alerts:
            self._send_alert(alert["type"], alert)
    
    def _get_recent_runs(self, minutes: int) -> List[Dict[str, Any]]:
        """Get runs from recent time period."""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        return [
            metrics for metrics in self.metrics_history
            if datetime.fromisoformat(metrics["timestamp"]) > cutoff
        ]
    
    def _send_alert(self, alert_type: str, alert_data: Dict[str, Any]) -> None:
        """Send alert to handlers."""
        for handler in self.alert_handlers:
            try:
                handler(alert_type, alert_data)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for time period."""
        recent_runs = self._get_recent_runs(minutes=hours * 60)
        
        if not recent_runs:
            return {"message": "No runs in time period"}
        
        total_runs = len(recent_runs)
        completed_runs = len([r for r in recent_runs if r["status"] == "completed"])
        failed_runs = len([r for r in recent_runs if r["status"] == "failed"])
        cancelled_runs = len([r for r in recent_runs if r["status"] == "cancelled"])
        timeout_runs = len([r for r in recent_runs if r["status"] == "timeout"])
        
        execution_times = [r["execution_time"] for r in recent_runs if r["execution_time"]]
        queue_times = [r["queue_time"] for r in recent_runs if r["queue_time"]]
        
        return {
            "time_period_hours": hours,
            "total_runs": total_runs,
            "completed_runs": completed_runs,
            "failed_runs": failed_runs,
            "cancelled_runs": cancelled_runs,
            "timeout_runs": timeout_runs,
            "success_rate": completed_runs / total_runs if total_runs > 0 else 0,
            "failure_rate": failed_runs / total_runs if total_runs > 0 else 0,
            "avg_execution_time": sum(execution_times) / len(execution_times) if execution_times else 0,
            "max_execution_time": max(execution_times) if execution_times else 0,
            "avg_queue_time": sum(queue_times) / len(queue_times) if queue_times else 0,
            "max_queue_time": max(queue_times) if queue_times else 0
        }


# Utility functions for common run patterns
def create_simple_run(agent_id: str, text: str, role: str = "user") -> ACPRun:
    """Create a simple text-based run."""
    return (RunBuilder(agent_id)
            .add_text_message(role, text)
            .build())


def create_analysis_run(
    agent_id: str, 
    data: Dict[str, Any], 
    analysis_type: str,
    parameters: Optional[Dict[str, Any]] = None
) -> ACPRun:
    """Create a run for data analysis."""
    builder = (RunBuilder(agent_id)
               .add_text_message("user", f"Please perform {analysis_type} analysis on the provided data")
               .add_message({
                   "role": "user",
                   "parts": [{"type": "json", "data": data, "description": "Data to analyze"}]
               })
               .add_metadata("analysis_type", analysis_type))
    
    if parameters:
        builder.set_parameters(parameters)
    
    return builder.build()


def create_file_processing_run(
    agent_id: str,
    filename: str,
    file_data: bytes,
    processing_instructions: str
) -> ACPRun:
    """Create a run for file processing."""
    return (RunBuilder(agent_id)
            .add_text_message("user", processing_instructions)
            .add_message({
                "role": "user",
                "parts": [{
                    "type": "file",
                    "filename": filename,
                    "data": file_data,
                    "description": "File to process"
                }]
            })
            .add_metadata("task_type", "file_processing")
            .build())


# Example usage and testing
def example_run_usage():
    """Example usage of ACP run functionality."""
    
    print("=== ACP Run Examples ===")
    
    # Example 1: Simple run creation
    print("\n1. Simple run creation:")
    simple_run = create_simple_run("agent-123", "Hello, how are you?")
    print(f"Created run: {simple_run.run_id}")
    print(f"Status: {simple_run.status.value}")
    print(f"Input messages: {len(simple_run.input_data.messages)}")
    
    # Example 2: Complex run with builder
    print("\n2. Complex run with builder:")
    complex_run = (RunBuilder("agent-456")
                   .add_text_message("user", "Analyze this data")
                   .add_message({
                       "role": "user",
                       "parts": [{
                           "type": "json",
                           "data": {"values": [1, 2, 3, 4, 5]},
                           "description": "Sample data"
                       }]
                   })
                   .set_priority(RunPriority.HIGH)
                   .set_timeout(300)
                   .add_parameter("analysis_type", "statistical")
                   .add_metadata("client_id", "test-client")
                   .build())
    
    print(f"Complex run: {complex_run.run_id}")
    print(f"Priority: {complex_run.priority.value}")
    print(f"Timeout: {complex_run.timeout}")
    print(f"Parameters: {complex_run.input_data.parameters}")
    
    # Example 3: Run lifecycle simulation
    print("\n3. Run lifecycle simulation:")
    
    def simulate_agent_execution(run: ACPRun) -> None:
        """Simulate agent execution."""
        import time
        
        # Simulate processing
        run.update_progress(0.2, "Starting analysis")
        time.sleep(0.1)
        
        run.update_progress(0.5, "Processing data")
        time.sleep(0.1)
        
        # Simulate await request
        run.request_await(
            await_id="user-input-1",
            await_type="confirmation",
            prompt="Do you want to continue with advanced analysis?",
            options={"type": "boolean", "default": True}
        )
        
        # Simulate user response (in real scenario, this would come from client)
        time.sleep(0.1)
        run.resolve_await("user-input-1", True)
        
        run.update_progress(0.8, "Completing analysis")
        time.sleep(0.1)
        
        # Complete with results
        output = RunOutput(
            messages=[{
                "role": "agent",
                "parts": [{
                    "type": "text",
                    "text": "Analysis completed successfully"
                }, {
                    "type": "json",
                    "data": {"mean": 3.0, "std": 1.58, "count": 5},
                    "description": "Statistical results"
                }]
            }],
            metadata={"processing_time": 0.4}
        )
        run.complete(output)
    
    # Create and execute run
    test_run = create_simple_run("agent-789", "Test execution")
    
    print(f"Initial status: {test_run.status.value}")
    
    # Start execution
    test_run.start()
    print(f"After start: {test_run.status.value}")
    
    # Simulate execution
    simulate_agent_execution(test_run)
    print(f"Final status: {test_run.status.value}")
    print(f"Progress: {test_run.progress:.1%}")
    print(f"Events: {len(test_run.events)}")
    print(f"Duration: {test_run.get_duration():.3f}s")
    
    # Example 4: Run manager usage
    print("\n4. Run manager usage:")
    
    manager = RunManager(max_concurrent_runs=3)
    
    # Create multiple runs
    runs = []
    for i in range(5):
        run = (RunBuilder(f"agent-{i}")
               .add_text_message("user", f"Task {i}")
               .set_priority(RunPriority.HIGH if i < 2 else RunPriority.NORMAL)
               .build())
        runs.append(run)
        manager.runs[run.run_id] = run
    
    print(f"Created {len(runs)} runs")
    print(f"Manager stats: {manager.get_statistics()}")
    
    # Example 5: Run monitoring
    print("\n5. Run monitoring:")
    
    monitor = RunMonitor()
    
    # Set up alert handler
    def alert_handler(alert_type: str, alert_data: Dict[str, Any]) -> None:
        print(f"ALERT [{alert_type}]: {alert_data['message']}")
    
    monitor.add_alert_handler(alert_handler)
    
    # Record metrics for completed run
    monitor.record_run_metrics(test_run)
    
    # Get performance summary
    summary = monitor.get_performance_summary(hours=1)
    print(f"Performance summary: {summary}")
    
    # Example 6: Serialization
    print("\n6. Serialization:")
    
    # Serialize run to JSON
    run_json = test_run.to_json()
    print(f"JSON size: {len(run_json)} characters")
    
    # Deserialize from JSON
    reconstructed_run = ACPRun.from_dict(json.loads(run_json))
    print(f"Reconstructed run ID: {reconstructed_run.run_id}")
    print(f"Status matches: {reconstructed_run.status == test_run.status}")
    print(f"Events match: {len(reconstructed_run.events) == len(test_run.events)}")
    
    # Example 7: Error handling
    print("\n7. Error handling:")
    
    error_run = create_simple_run("agent-error", "This will fail")
    error_run.start()
    
    try:
        # Simulate error
        error_run.fail("Simulated processing error", {
            "error_code": "PROC_001",
            "details": "Mock error for testing"
        })
        print(f"Error run status: {error_run.status.value}")
        print(f"Error message: {error_run.error}")
    except Exception as e:
        print(f"Exception: {e}")
    
    # Example 8: Cancellation
    print("\n8. Cancellation:")
    
    cancel_run = create_simple_run("agent-cancel", "This will be cancelled")
    cancel_run.start()
    cancel_run.cancel("User requested cancellation")
    
    print(f"Cancelled run status: {cancel_run.status.value}")
    print(f"Is cancelled: {cancel_run.is_cancelled()}")
    
    # Example 9: Validation
    print("\n9. Validation:")
    
    # Test invalid run
    try:
        invalid_input = RunInput(messages=[])  # No messages
        invalid_run = ACPRun(
            run_id="invalid",
            agent_id="",  # Empty agent ID
            input_data=invalid_input
        )
        errors = invalid_run.validate()
        print(f"Validation errors: {errors}")
    except Exception as e:
        print(f"Validation exception: {e}")
    
    # Example 10: Advanced features
    print("\n10. Advanced features:")
    
    # Create run with all features
    advanced_run = (RunBuilder("advanced-agent")
                    .add_text_message("user", "Complex task with multiple parts")
                    .add_message({
                        "role": "user",
                        "parts": [{
                            "type": "json",
                            "data": {"config": {"mode": "advanced", "iterations": 100}},
                            "description": "Configuration data"
                        }, {
                            "type": "file",
                            "filename": "data.csv",
                            "data": b"col1,col2\n1,2\n3,4",
                            "description": "Input data file"
                        }]
                    })
                    .set_priority(RunPriority.URGENT)
                    .set_timeout(600)
                    .add_parameter("max_iterations", 100)
                    .add_parameter("learning_rate", 0.01)
                    .add_context("session_id", "sess-123")
                    .add_context("user_id", "user-456")
                    .add_metadata("experiment_id", "exp-789")
                    .add_metadata("version", "2.1.0")
                    .build())
    
    print(f"Advanced run created: {advanced_run.run_id}")
    print(f"Input messages: {len(advanced_run.input_data.messages)}")
    print(f"Parameters: {advanced_run.input_data.parameters}")
    print(f"Context: {advanced_run.input_data.context}")
    print(f"Metadata: {advanced_run.metadata}")
    
    # Add event handler
    def run_event_handler(event: RunEvent) -> None:
        print(f"Event: {event.event_type.value} - {event.message}")
    
    advanced_run.add_event_handler(run_event_handler)
    
    # Simulate lifecycle
    advanced_run.start()
    advanced_run.update_progress(0.3, "Processing configuration")
    advanced_run.update_progress(0.7, "Analyzing data")
    
    # Complete with comprehensive output
    comprehensive_output = RunOutput(
        messages=[{
            "role": "agent",
            "parts": [{
                "type": "text",
                "text": "Advanced analysis completed with high confidence",
                "format": "markdown"
            }, {
                "type": "json",
                "data": {
                    "results": {"accuracy": 0.95, "precision": 0.92, "recall": 0.88},
                    "model_info": {"type": "neural_network", "layers": 3, "parameters": 1024}
                },
                "description": "Analysis results and model information"
            }]
        }],
        artifacts=[{
            "type": "model",
            "filename": "trained_model.pkl",
            "size": 2048,
            "description": "Trained machine learning model"
        }, {
            "type": "report",
            "filename": "analysis_report.pdf",
            "size": 4096,
            "description": "Detailed analysis report"
        }],
        metadata={
            "model_accuracy": 0.95,
            "training_time": 45.2,
            "data_points_processed": 10000,
            "model_version": "1.0.0"
        }
    )
    
    advanced_run.complete(comprehensive_output)
    
    print(f"Final status: {advanced_run.status.value}")
    print(f"Output messages: {len(advanced_run.output_data.messages)}")
    print(f"Artifacts: {len(advanced_run.output_data.artifacts)}")
    print(f"Total events: {len(advanced_run.events)}")
    
    # Cleanup
    manager.shutdown(wait=False)


def create_test_scenarios():
    """Create comprehensive test scenarios for ACP runs."""
    
    def test_run_lifecycle():
        """Test complete run lifecycle."""
        print("\n=== Testing Run Lifecycle ===")
        
        # Create run
        run = create_simple_run("test-agent", "Test message")
        assert run.status == RunStatus.CREATED
        print("✓ Run creation works")
        
        # Start run
        run.start()
        assert run.status == RunStatus.RUNNING
        assert run.started_at is not None
        print("✓ Run start works")
        
        # Update progress
        run.update_progress(0.5, "Halfway done")
        assert run.progress == 0.5
        print("✓ Progress update works")
        
        # Complete run
        output = RunOutput(messages=[{"role": "agent", "parts": [{"type": "text", "text": "Done"}]}])
        run.complete(output)
        assert run.status == RunStatus.COMPLETED
        assert run.completed_at is not None
        print("✓ Run completion works")
    
    def test_run_await():
        """Test await functionality."""
        print("\n=== Testing Run Await ===")
        
        run = create_simple_run("test-agent", "Test await")
        run.start()
        
        # Request await
        run.request_await("test-await", "confirmation", "Continue?")
        assert run.status == RunStatus.AWAITING
        assert "test-await" in run.get_await_requests()
        print("✓ Await request works")
        
        # Resolve await
        run.resolve_await("test-await", True)
        assert run.status == RunStatus.RUNNING
        print("✓ Await resolution works")
    
    def test_run_manager():
        """Test run manager functionality."""
        print("\n=== Testing Run Manager ===")
        
        manager = RunManager(max_concurrent_runs=2)
        
        # Create runs
        run1 = manager.create_run("agent-1", RunInput(messages=[{"role": "user", "parts": [{"type": "text", "text": "Task 1"}]}]))
        run2 = manager.create_run("agent-2", RunInput(messages=[{"role": "user", "parts": [{"type": "text", "text": "Task 2"}]}]))
        
        assert len(manager.runs) == 2
        print("✓ Run creation in manager works")
        
        # Test statistics
        stats = manager.get_statistics()
        assert stats["total_runs"] == 2
        print("✓ Manager statistics work")
        
        # Test cancellation
        cancelled = manager.cancel_run(run1.run_id, "Test cancellation")
        assert cancelled
        assert run1.status == RunStatus.CANCELLED
        print("✓ Run cancellation works")
        
        manager.shutdown(wait=False)
    
    def test_run_validation():
        """Test run validation."""
        print("\n=== Testing Run Validation ===")
        
        # Valid run
        valid_input = RunInput(messages=[{"role": "user", "parts": [{"type": "text", "text": "Valid"}]}])
        valid_run = ACPRun("valid-run", "valid-agent", valid_input)
        errors = valid_run.validate()
        assert len(errors) == 0
        print("✓ Valid run passes validation")
        
        # Invalid run
        invalid_input = RunInput(messages=[])  # No messages
        invalid_run = ACPRun("", "invalid-agent", invalid_input)  # No run ID
        errors = invalid_run.validate()
        assert len(errors) > 0
        print("✓ Invalid run fails validation")
    
    def test_run_serialization():
        """Test run serialization."""
        print("\n=== Testing Run Serialization ===")
        
        # Create complex run
        run = (RunBuilder("test-agent")
               .add_text_message("user", "Test serialization")
               .set_priority(RunPriority.HIGH)
               .set_timeout(300)
               .add_parameter("test_param", "test_value")
               .add_metadata("test_meta", "test_value")
               .build())
        
        # Start and complete run
        run.start()
        output = RunOutput(messages=[{"role": "agent", "parts": [{"type": "text", "text": "Serialization test complete"}]}])
        run.complete(output)
        
        # Serialize to JSON
        json_str = run.to_json()
        assert len(json_str) > 0
        print("✓ Run serialization works")
        
        # Deserialize from JSON
        reconstructed = ACPRun.from_dict(json.loads(json_str))
        assert reconstructed.run_id == run.run_id
        assert reconstructed.status == run.status
        assert reconstructed.agent_id == run.agent_id
        print("✓ Run deserialization works")
    
    def test_run_monitoring():
        """Test run monitoring."""
        print("\n=== Testing Run Monitoring ===")
        
        monitor = RunMonitor()
        
        # Create and complete run
        run = create_simple_run("monitor-test", "Monitor this")
        run.start()
        
        # Simulate some execution time
        import time
        time.sleep(0.01)
        
        output = RunOutput(messages=[{"role": "agent", "parts": [{"type": "text", "text": "Monitored"}]}])
        run.complete(output)
        
        # Record metrics
        monitor.record_run_metrics(run)
        assert len(monitor.metrics_history) == 1
        print("✓ Metrics recording works")
        
        # Get performance summary
        summary = monitor.get_performance_summary(hours=1)
        assert summary["total_runs"] == 1
        print("✓ Performance summary works")
    
    def test_error_handling():
        """Test error handling."""
        print("\n=== Testing Error Handling ===")
        
        # Test run failure
        run = create_simple_run("error-test", "This will fail")
        run.start()
        run.fail("Test error", {"code": "TEST_001"})
        
        assert run.status == RunStatus.FAILED
        assert run.error == "Test error"
        assert run.error_details["code"] == "TEST_001"
        print("✓ Run failure handling works")
        
        # Test run timeout
        timeout_run = create_simple_run("timeout-test", "This will timeout")
        timeout_run.start()
        timeout_run.timeout_run()
        
        assert timeout_run.status == RunStatus.TIMEOUT
        print("✓ Run timeout handling works")
        
        # Test run cancellation
        cancel_run = create_simple_run("cancel-test", "This will be cancelled")
        cancel_run.start()
        cancel_run.cancel("Test cancellation")
        
        assert cancel_run.status == RunStatus.CANCELLED
        assert cancel_run.is_cancelled()
        print("✓ Run cancellation handling works")
    
    def test_builder_patterns():
        """Test builder patterns."""
        print("\n=== Testing Builder Patterns ===")
        
        # Test simple builder
        simple = (RunBuilder("builder-test")
                  .add_text_message("user", "Simple test")
                  .build())
        
        assert len(simple.input_data.messages) == 1
        print("✓ Simple builder works")
        
        # Test complex builder
        complex_run = (RunBuilder("complex-builder")
                       .add_text_message("user", "Complex test")
                       .add_message({
                           "role": "user",
                           "parts": [{"type": "json", "data": {"test": True}}]
                       })
                       .set_priority(RunPriority.URGENT)
                       .set_timeout(600)
                       .add_parameter("param1", "value1")
                       .add_parameter("param2", 42)
                       .add_context("ctx1", "context_value")
                       .add_metadata("meta1", "metadata_value")
                       .build())
        
        assert len(complex_run.input_data.messages) == 2
        assert complex_run.priority == RunPriority.URGENT
        assert complex_run.timeout == 600
        assert complex_run.input_data.parameters["param1"] == "value1"
        assert complex_run.input_data.context["ctx1"] == "context_value"
        assert complex_run.metadata["meta1"] == "metadata_value"
        print("✓ Complex builder works")
    
    def test_utility_functions():
        """Test utility functions."""
        print("\n=== Testing Utility Functions ===")
        
        # Test simple run creation
        simple = create_simple_run("util-test", "Simple utility test")
        assert simple.agent_id == "util-test"
        assert len(simple.input_data.messages) == 1
        print("✓ Simple run utility works")
        
        # Test analysis run creation
        analysis = create_analysis_run(
            "analysis-agent",
            {"data": [1, 2, 3, 4, 5]},
            "statistical",
            {"confidence_level": 0.95}
        )
        assert analysis.input_data.parameters["confidence_level"] == 0.95
        assert analysis.metadata["analysis_type"] == "statistical"
        print("✓ Analysis run utility works")
        
        # Test file processing run creation
        file_run = create_file_processing_run(
            "file-agent",
            "test.txt",
            b"test file content",
            "Process this file"
        )
        assert len(file_run.input_data.messages) == 2
        assert file_run.metadata["task_type"] == "file_processing"
        print("✓ File processing run utility works")
    
    # Run all tests
    test_run_lifecycle()
    test_run_await()
    test_run_manager()
    test_run_validation()
    test_run_serialization()
    test_run_monitoring()
    test_error_handling()
    test_builder_patterns()
    test_utility_functions()
    
    print("\n🎉 All tests passed!")


class RunArchive:
    """Archive system for storing and retrieving completed runs."""
    
    def __init__(self, max_runs: int = 10000):
        """Initialize run archive."""
        self.max_runs = max_runs
        self.archived_runs: Dict[str, Dict[str, Any]] = {}
        self.agent_index: Dict[str, List[str]] = {}
        self.status_index: Dict[str, List[str]] = {}
        self.date_index: Dict[str, List[str]] = {}
        self._lock = threading.RLock()
        
        logger.info(f"Initialized run archive with capacity for {max_runs} runs")
    
    def archive_run(self, run: ACPRun) -> None:
        """Archive a completed run."""
        if not run.is_finished():
            raise RunError("Can only archive finished runs")
        
        with self._lock:
            # Convert run to dictionary for storage
            run_data = run.to_dict()
            self.archived_runs[run.run_id] = run_data
            
            # Update indexes
            self._update_indexes(run)
            
            # Enforce size limit
            if len(self.archived_runs) > self.max_runs:
                self._cleanup_oldest_runs()
        
        logger.debug(f"Archived run {run.run_id}")
    
    def _update_indexes(self, run: ACPRun) -> None:
        """Update search indexes."""
        # Agent index
        if run.agent_id not in self.agent_index:
            self.agent_index[run.agent_id] = []
        self.agent_index[run.agent_id].append(run.run_id)
        
        # Status index
        status_key = run.status.value
        if status_key not in self.status_index:
            self.status_index[status_key] = []
        self.status_index[status_key].append(run.run_id)
        
        # Date index (by day)
        if run.completed_at:
            date_key = run.completed_at.strftime("%Y-%m-%d")
            if date_key not in self.date_index:
                self.date_index[date_key] = []
            self.date_index[date_key].append(run.run_id)
    
    def _cleanup_oldest_runs(self) -> None:
        """Remove oldest runs to maintain size limit."""
        # Sort by completion time and remove oldest
        sorted_runs = sorted(
            self.archived_runs.items(),
            key=lambda x: x[1].get('completed_at', x[1]['created_at'])
        )
        
        runs_to_remove = len(self.archived_runs) - self.max_runs
        for i in range(runs_to_remove):
            run_id, run_data = sorted_runs[i]
            self._remove_from_indexes(run_id, run_data)
            del self.archived_runs[run_id]
    
    def _remove_from_indexes(self, run_id: str, run_data: Dict[str, Any]) -> None:
        """Remove run from indexes."""
        # Remove from agent index
        agent_id = run_data['agent_id']
        if agent_id in self.agent_index and run_id in self.agent_index[agent_id]:
            self.agent_index[agent_id].remove(run_id)
            if not self.agent_index[agent_id]:
                del self.agent_index[agent_id]
        
        # Remove from status index
        status = run_data['status']
        if status in self.status_index and run_id in self.status_index[status]:
            self.status_index[status].remove(run_id)
            if not self.status_index[status]:
                del self.status_index[status]
        
        # Remove from date index
        if run_data.get('completed_at'):
            date_key = datetime.fromisoformat(run_data['completed_at']).strftime("%Y-%m-%d")
            if date_key in self.date_index and run_id in self.date_index[date_key]:
                self.date_index[date_key].remove(run_id)
                if not self.date_index[date_key]:
                    del self.date_index[date_key]
    
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get archived run by ID."""
        return self.archived_runs.get(run_id)
    
    def search_runs(
        self,
        agent_id: Optional[str] = None,
        status: Optional[RunStatus] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search archived runs with filters."""
        with self._lock:
            candidate_run_ids = set(self.archived_runs.keys())
            
            # Filter by agent
            if agent_id:
                agent_runs = set(self.agent_index.get(agent_id, []))
                candidate_run_ids &= agent_runs
            
            # Filter by status
            if status:
                status_runs = set(self.status_index.get(status.value, []))
                candidate_run_ids &= status_runs
            
            # Filter by date range
            if date_from or date_to:
                date_runs = set()
                for date_key, run_ids in self.date_index.items():
                    date_obj = datetime.strptime(date_key, "%Y-%m-%d")
                    if date_from and date_obj < date_from:
                        continue
                    if date_to and date_obj > date_to:
                        continue
                    date_runs.update(run_ids)
                candidate_run_ids &= date_runs
            
            # Get run data and sort by completion time
            results = []
            for run_id in candidate_run_ids:
                run_data = self.archived_runs[run_id]
                results.append(run_data)
            
            # Sort by completion time (newest first)
            results.sort(
                key=lambda x: x.get('completed_at', x['created_at']),
                reverse=True
            )
            
            return results[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get archive statistics."""
        with self._lock:
            return {
                "total_archived_runs": len(self.archived_runs),
                "agents_count": len(self.agent_index),
                "status_distribution": {
                    status: len(run_ids) 
                    for status, run_ids in self.status_index.items()
                },
                "date_range": {
                    "earliest": min(self.date_index.keys()) if self.date_index else None,
                    "latest": max(self.date_index.keys()) if self.date_index else None
                },
                "max_capacity": self.max_runs,
                "usage_percentage": (len(self.archived_runs) / self.max_runs) * 100
            }


class RunScheduler:
    """Scheduler for managing run execution timing and dependencies."""
    
    def __init__(self):
        """Initialize run scheduler."""
        self.scheduled_runs: Dict[str, Dict[str, Any]] = {}
        self.dependencies: Dict[str, List[str]] = {}
        self.recurring_runs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._scheduler_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        logger.info("Initialized run scheduler")
    
    def schedule_run(
        self,
        run: ACPRun,
        scheduled_time: datetime,
        dependencies: Optional[List[str]] = None
    ) -> None:
        """Schedule a run for future execution."""
        with self._lock:
            self.scheduled_runs[run.run_id] = {
                "run": run,
                "scheduled_time": scheduled_time,
                "dependencies": dependencies or [],
                "created_at": datetime.utcnow()
            }
            
            if dependencies:
                self.dependencies[run.run_id] = dependencies
        
        logger.info(f"Scheduled run {run.run_id} for {scheduled_time}")
    
    def schedule_recurring_run(
        self,
        run_template: ACPRun,
        interval_seconds: int,
        max_executions: Optional[int] = None
    ) -> str:
        """Schedule a recurring run."""
        recurring_id = str(uuid.uuid4())
        
        with self._lock:
            self.recurring_runs[recurring_id] = {
                "template": run_template,
                "interval_seconds": interval_seconds,
                "max_executions": max_executions,
                "execution_count": 0,
                "last_execution": None,
                "next_execution": datetime.utcnow() + timedelta(seconds=interval_seconds),
                "created_at": datetime.utcnow()
            }
        
        logger.info(f"Scheduled recurring run {recurring_id} with {interval_seconds}s interval")
        return recurring_id
    
    def start_scheduler(self, run_manager: RunManager, executor_func: Callable[[ACPRun], None]) -> None:
        """Start the scheduler thread."""
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            return
        
        self._stop_event.clear()
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            args=(run_manager, executor_func),
            daemon=True
        )
        self._scheduler_thread.start()
        
        logger.info("Started run scheduler")
    
    def stop_scheduler(self) -> None:
        """Stop the scheduler thread."""
        self._stop_event.set()
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        
        logger.info("Stopped run scheduler")
    
    def _scheduler_loop(self, run_manager: RunManager, executor_func: Callable[[ACPRun], None]) -> None:
        """Main scheduler loop."""
        while not self._stop_event.is_set():
            try:
                current_time = datetime.utcnow()
                
                # Check scheduled runs
                self._check_scheduled_runs(current_time, run_manager, executor_func)
                
                # Check recurring runs
                self._check_recurring_runs(current_time, run_manager, executor_func)
                
                # Sleep for a short interval
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(5)  # Wait longer on error
    
    def _check_scheduled_runs(
        self, 
        current_time: datetime, 
        run_manager: RunManager, 
        executor_func: Callable[[ACPRun], None]
    ) -> None:
        """Check and execute scheduled runs."""
        with self._lock:
            runs_to_execute = []
            
            for run_id, schedule_info in list(self.scheduled_runs.items()):
                if current_time >= schedule_info["scheduled_time"]:
                    # Check dependencies
                    if self._dependencies_satisfied(run_id, run_manager):
                        runs_to_execute.append((run_id, schedule_info))
            
            # Execute ready runs
            for run_id, schedule_info in runs_to_execute:
                try:
                    run = schedule_info["run"]
                    
                    # Add to run manager and submit
                    run_manager.runs[run.run_id] = run
                    run_manager.submit_run(run.run_id, executor_func)
                    
                    # Remove from scheduled runs
                    del self.scheduled_runs[run_id]
                    if run_id in self.dependencies:
                        del self.dependencies[run_id]
                    
                    logger.info(f"Executed scheduled run {run_id}")
                    
                except Exception as e:
                    logger.error(f"Error executing scheduled run {run_id}: {e}")
                    # Remove failed run from schedule
                    del self.scheduled_runs[run_id]
                    if run_id in self.dependencies:
                        del self.dependencies[run_id]
    
    def _check_recurring_runs(
        self, 
        current_time: datetime, 
        run_manager: RunManager, 
        executor_func: Callable[[ACPRun], None]
    ) -> None:
        """Check and execute recurring runs."""
        with self._lock:
            for recurring_id, recurring_info in list(self.recurring_runs.items()):
                if current_time >= recurring_info["next_execution"]:
                    # Check if max executions reached
                    if (recurring_info["max_executions"] and 
                        recurring_info["execution_count"] >= recurring_info["max_executions"]):
                        del self.recurring_runs[recurring_id]
                        logger.info(f"Recurring run {recurring_id} completed max executions")
                        continue
                    
                    try:
                        # Create new run from template
                        template = recurring_info["template"]
                        new_run = ACPRun(
                            run_id=str(uuid.uuid4()),
                            agent_id=template.agent_id,
                            input_data=template.input_data,
                            priority=template.priority,
                            timeout=template.timeout,
                            metadata={
                                **(template.metadata or {}),
                                "recurring_id": recurring_id,
                                "execution_number": recurring_info["execution_count"] + 1
                            }
                        )
                        
                        # Add to run manager and submit
                        run_manager.runs[new_run.run_id] = new_run
                        run_manager.submit_run(new_run.run_id, executor_func)
                        
                        # Update recurring info
                        recurring_info["execution_count"] += 1
                        recurring_info["last_execution"] = current_time
                        recurring_info["next_execution"] = current_time + timedelta(
                            seconds=recurring_info["interval_seconds"]
                        )
                        
                        logger.info(f"Executed recurring run {recurring_id} (execution #{recurring_info['execution_count']})")
                        
                    except Exception as e:
                        logger.error(f"Error executing recurring run {recurring_id}: {e}")
    
    def _dependencies_satisfied(self, run_id: str, run_manager: RunManager) -> bool:
        """Check if run dependencies are satisfied."""
        if run_id not in self.dependencies:
            return True
        
        for dep_run_id in self.dependencies[run_id]:
            dep_run = run_manager.get_run(dep_run_id)
            if not dep_run or dep_run.status != RunStatus.COMPLETED:
                return False
        
        return True
    
    def cancel_scheduled_run(self, run_id: str) -> bool:
        """Cancel a scheduled run."""
        with self._lock:
            if run_id in self.scheduled_runs:
                del self.scheduled_runs[run_id]
                if run_id in self.dependencies:
                    del self.dependencies[run_id]
                logger.info(f"Cancelled scheduled run {run_id}")
                return True
            return False
    
    def cancel_recurring_run(self, recurring_id: str) -> bool:
        """Cancel a recurring run."""
        with self._lock:
            if recurring_id in self.recurring_runs:
                del self.recurring_runs[recurring_id]
                logger.info(f"Cancelled recurring run {recurring_id}")
                return True
            return False
    
    def get_scheduled_runs(self) -> List[Dict[str, Any]]:
        """Get list of scheduled runs."""
        with self._lock:
            return [
                {
                    "run_id": run_id,
                    "agent_id": info["run"].agent_id,
                    "scheduled_time": info["scheduled_time"].isoformat(),
                    "dependencies": info["dependencies"],
                    "created_at": info["created_at"].isoformat()
                }
                for run_id, info in self.scheduled_runs.items()
            ]
    
    def get_recurring_runs(self) -> List[Dict[str, Any]]:
        """Get list of recurring runs."""
        with self._lock:
            return [
                {
                    "recurring_id": recurring_id,
                    "agent_id": info["template"].agent_id,
                    "interval_seconds": info["interval_seconds"],
                    "execution_count": info["execution_count"],
                    "max_executions": info["max_executions"],
                    "last_execution": info["last_execution"].isoformat() if info["last_execution"] else None,
                    "next_execution": info["next_execution"].isoformat(),
                    "created_at": info["created_at"].isoformat()
                }
                for recurring_id, info in self.recurring_runs.items()
            ]


class RunPipeline:
    """Pipeline for chaining multiple runs together."""
    
    def __init__(self, pipeline_id: Optional[str] = None):
        """Initialize run pipeline."""
        self.pipeline_id = pipeline_id or str(uuid.uuid4())
        self.stages: List[Dict[str, Any]] = []
        self.pipeline_status = "created"
        self.current_stage = 0
        self.results: Dict[str, Any] = {}
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error: Optional[str] = None
        
        logger.info(f"Created run pipeline {self.pipeline_id}")
    
    def add_stage(
        self,
        stage_id: str,
        agent_id: str,
        input_template: Dict[str, Any],
        condition: Optional[Callable[[Dict[str, Any]], bool]] = None,
        retry_count: int = 0
    ) -> 'RunPipeline':
        """Add a stage to the pipeline."""
        stage = {
            "stage_id": stage_id,
            "agent_id": agent_id,
            "input_template": input_template,
            "condition": condition,
            "retry_count": retry_count,
            "current_retry": 0,
            "status": "pending",
            "run_id": None,
            "result": None,
            "error": None
        }
        
        self.stages.append(stage)
        logger.debug(f"Added stage {stage_id} to pipeline {self.pipeline_id}")
        return self
    
    def add_parallel_stages(
        self,
        stages: List[Dict[str, Any]]
    ) -> 'RunPipeline':
        """Add multiple stages that can run in parallel."""
        parallel_group = {
            "type": "parallel",
            "stages": stages,
            "status": "pending",
            "results": {}
        }
        
        self.stages.append(parallel_group)
        logger.debug(f"Added parallel group with {len(stages)} stages to pipeline {self.pipeline_id}")
        return self
    
    def execute(
        self,
        run_manager: RunManager,
        executor_func: Callable[[ACPRun], None],
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Execute the pipeline."""
        self.pipeline_status = "running"
        self.started_at = datetime.utcnow()
        self.current_stage = 0
        
        initial_context = context or {}
        
        try:
            for i, stage in enumerate(self.stages):
                self.current_stage = i
                
                if stage.get("type") == "parallel":
                    self._execute_parallel_stages(stage, run_manager, executor_func, initial_context)
                else:
                    self._execute_single_stage(stage, run_manager, executor_func, initial_context)
                
                # Update context with stage results
                if stage.get("result"):
                    initial_context.update(stage["result"])
            
            self.pipeline_status = "completed"
            self.completed_at = datetime.utcnow()
            logger.info(f"Pipeline {self.pipeline_id} completed successfully")
            
        except Exception as e:
            self.pipeline_status = "failed"
            self.completed_at = datetime.utcnow()
            self.error = str(e)
            logger.error(f"Pipeline {self.pipeline_id} failed: {e}")
            raise
    
    def _execute_single_stage(
        self,
        stage: Dict[str, Any],
        run_manager: RunManager,
        executor_func: Callable[[ACPRun], None],
        context: Dict[str, Any]
    ) -> None:
        """Execute a single stage."""
        stage_id = stage["stage_id"]
        
        # Check condition if present
        if stage["condition"] and not stage["condition"](context):
            stage["status"] = "skipped"
            logger.info(f"Skipped stage {stage_id} due to condition")
            return
        
        # Retry loop
        while stage["current_retry"] <= stage["retry_count"]:
            try:
                # Create input from template and context
                input_data = self._create_input_from_template(stage["input_template"], context)
                
                # Create and execute run
                run = ACPRun(
                    run_id=str(uuid.uuid4()),
                    agent_id=stage["agent_id"],
                    input_data=input_data,
                    metadata={
                        "pipeline_id": self.pipeline_id,
                        "stage_id": stage_id,
                        "stage_index": self.current_stage
                    }
                )
                
                stage["run_id"] = run.run_id
                stage["status"] = "running"
                
                # Execute run synchronously
                run_manager.runs[run.run_id] = run
                run.start()
                executor_func(run)
                
                # Wait for completion
                while not run.is_finished():
                    time.sleep(0.1)
                
                if run.status == RunStatus.COMPLETED:
                    stage["status"] = "completed"
                    stage["result"] = self._extract_result_from_run(run)
                    logger.info(f"Stage {stage_id} completed successfully")
                    break
                else:
                    raise Exception(f"Run failed with status {run.status.value}: {run.error}")
                
            except Exception as e:
                stage["current_retry"] += 1
                stage["error"] = str(e)
                
                if stage["current_retry"] > stage["retry_count"]:
                    stage["status"] = "failed"
                    logger.error(f"Stage {stage_id} failed after {stage['retry_count']} retries: {e}")
                    raise
                else:
                    logger.warning(f"Stage {stage_id} failed, retrying ({stage['current_retry']}/{stage['retry_count']}): {e}")
                    time.sleep(1)  # Brief delay before retry
    
    def _execute_parallel_stages(
        self,
        parallel_group: Dict[str, Any],
        run_manager: RunManager,
        executor_func: Callable[[ACPRun], None],
        context: Dict[str, Any]
    ) -> None:
        """Execute stages in parallel."""
        import concurrent.futures
        
        parallel_group["status"] = "running"
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(parallel_group["stages"])) as executor:
            futures = {}
            
            for stage in parallel_group["stages"]:
                future = executor.submit(
                    self._execute_single_stage,
                    stage,
                    run_manager,
                    executor_func,
                    context.copy()  # Each stage gets its own context copy
                )
                futures[future] = stage
            
            # Wait for all stages to complete
            for future in concurrent.futures.as_completed(futures):
                stage = futures[future]
                try:
                    future.result()  # This will raise exception if stage failed
                    parallel_group["results"][stage["stage_id"]] = stage.get("result")
                except Exception as e:
                    parallel_group["status"] = "failed"
                    logger.error(f"Parallel stage {stage['stage_id']} failed: {e}")
                    raise
        
        parallel_group["status"] = "completed"
        logger.info(f"Parallel group completed with {len(parallel_group['stages'])} stages")
    
    def _create_input_from_template(
        self,
        template: Dict[str, Any],
        context: Dict[str, Any]
    ) -> RunInput:
        """Create run input from template and context."""
        # Simple template substitution
        import json
        
        # Convert template to JSON string for substitution
        template_str = json.dumps(template)
        
        # Replace context variables (simple ${variable} format)
        for key, value in context.items():
            template_str = template_str.replace(f"${{{key}}}", str(value))
        
        # Parse back to dict
        substituted_template = json.loads(template_str)
        
        return RunInput(
            messages=substituted_template.get("messages", []),
            parameters=substituted_template.get("parameters"),
            context=substituted_template.get("context")
        )
    
    def _extract_result_from_run(self, run: ACPRun) -> Dict[str, Any]:
        """Extract result data from completed run."""
        if not run.output_data:
            return {}
        
        result = {
            "messages": run.output_data.messages,
            "metadata": run.output_data.metadata or {}
        }
        
        # Extract text content for easy access
        text_content = []
        for message in run.output_data.messages:
            for part in message.get("parts", []):
                if part.get("type") == "text":
                    text_content.append(part.get("text", ""))
        
        if text_content:
            result["text_content"] = "\n".join(text_content)
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status."""
        return {
            "pipeline_id": self.pipeline_id,
            "status": self.pipeline_status,
            "current_stage": self.current_stage,
            "total_stages": len(self.stages),
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "stages": [
                {
                    "stage_id": stage.get("stage_id", f"parallel_group_{i}"),
                    "status": stage.get("status", "pending"),
                    "agent_id": stage.get("agent_id"),
                    "run_id": stage.get("run_id"),
                    "current_retry": stage.get("current_retry", 0),
                    "max_retries": stage.get("retry_count", 0),
                    "error": stage.get("error"),
                    "type": stage.get("type", "single")
                }
                for i, stage in enumerate(self.stages)
            ]
        }
    
    def cancel(self, run_manager: RunManager) -> None:
        """Cancel the pipeline execution."""
        self.pipeline_status = "cancelled"
        self.completed_at = datetime.utcnow()
        
        # Cancel any running stages
        for stage in self.stages:
            if stage.get("run_id") and stage.get("status") == "running":
                run_manager.cancel_run(stage["run_id"], "Pipeline cancelled")
                stage["status"] = "cancelled"
        
        logger.info(f"Pipeline {self.pipeline_id} cancelled")


# Example usage and comprehensive testing
def comprehensive_run_examples():
    """Comprehensive examples showing all ACP run functionality."""
    
    print("=== Comprehensive ACP Run Examples ===\n")
    
    # 1. Basic Run Operations
    print("1. Basic Run Operations:")
    print("-" * 30)
    
    # Create a simple run
    simple_run = create_simple_run("basic-agent", "Hello, world!")
    print(f"Created run: {simple_run.run_id}")
    print(f"Agent: {simple_run.agent_id}")
    print(f"Status: {simple_run.status.value}")
    
    # Start and complete the run
    simple_run.start()
    print(f"Started run, status: {simple_run.status.value}")
    
    # Simulate processing
    simple_run.update_progress(0.5, "Processing request")
    print(f"Progress: {simple_run.progress:.1%}")
    
    # Complete with output
    output = RunOutput(
        messages=[{
            "role": "agent",
            "parts": [{"type": "text", "text": "Hello! I'm doing well, thank you for asking."}]
        }]
    )
    simple_run.complete(output)
    print(f"Completed run, status: {simple_run.status.value}")
    print(f"Duration: {simple_run.get_duration():.3f}s")
    
    # 2. Advanced Run with Builder
    print("\n2. Advanced Run with Builder:")
    print("-" * 30)
    
    advanced_run = (RunBuilder("advanced-agent")
                    .add_text_message("user", "Analyze this dataset")
                    .add_message({
                        "role": "user",
                        "parts": [{
                            "type": "json",
                            "data": {
                                "dataset": "sales_data.csv",
                                "columns": ["date", "product", "sales", "region"],
                                "rows": 10000
                            },
                            "description": "Dataset metadata"
                        }]
                    })
                    .set_priority(RunPriority.HIGH)
                    .set_timeout(600)
                    .add_parameter("analysis_type", "comprehensive")
                    .add_parameter("include_visualization", True)
                    .add_context("user_id", "user-123")
                    .add_context("session_id", "sess-456")
                    .add_metadata("experiment_id", "exp-789")
                    .build())
    
    print(f"Advanced run: {advanced_run.run_id}")
    print(f"Priority: {advanced_run.priority.value}")
    print(f"Parameters: {advanced_run.input_data.parameters}")
    print(f"Context: {advanced_run.input_data.context}")
    
    # 3. Run Manager Operations
    print("\n3. Run Manager Operations:")
    print("-" * 30)
    
    manager = RunManager(max_concurrent_runs=3)
    
    # Create multiple runs
    runs = []
    for i in range(5):
        run_input = RunInput(
            messages=[{
                "role": "user",
                "parts": [{"type": "text", "text": f"Task {i+1}"}]
            }]
        )
        run = manager.create_run(f"worker-agent-{i%3}", run_input)
        runs.append(run)
    
    print(f"Created {len(runs)} runs")
    
    # Submit runs for execution
    def mock_executor(run: ACPRun) -> None:
        """Mock executor function."""
        import time
        time.sleep(0.1)  # Simulate work
        
        output = RunOutput(
            messages=[{
                "role": "agent",
                "parts": [{"type": "text", "text": f"Completed task for run {run.run_id}"}]
            }]
        )
        run.complete(output)
    
    # Submit first 3 runs
    for run in runs[:3]:
        manager.submit_run(run.run_id, mock_executor)
    
    # Wait for completion
    import time
    time.sleep(0.5)
    
    stats = manager.get_statistics()
    print(f"Manager stats: {stats}")
    
    # 4. Run Monitoring
    print("\n4. Run Monitoring:")
    print("-" * 30)
    
    monitor = RunMonitor()
    
    # Set up alert handler
    alerts_received = []
    def test_alert_handler(alert_type: str, alert_data: Dict[str, Any]) -> None:
        alerts_received.append((alert_type, alert_data))
        print(f"ALERT: {alert_type} - {alert_data['message']}")
    
    monitor.add_alert_handler(test_alert_handler)
    
    # Record metrics for completed runs
    for run in runs:
        if run.is_finished():
            monitor.record_run_metrics(run)
    
    # Get performance summary
    summary = monitor.get_performance_summary(hours=1)
    print(f"Performance summary: {summary}")
    
    # 5. Run Scheduling
    print("\n5. Run Scheduling:")
    print("-" * 30)
    
    scheduler = RunScheduler()
    
    # Schedule a run for future execution
    future_run = create_simple_run("scheduled-agent", "This is a scheduled task")
    future_time = datetime.utcnow() + timedelta(seconds=2)
    scheduler.schedule_run(future_run, future_time)
    
    # Create a recurring run
    recurring_template = create_simple_run("recurring-agent", "Recurring task")
    recurring_id = scheduler.schedule_recurring_run(
        recurring_template,
        interval_seconds=5,
        max_executions=3
    )
    
    print(f"Scheduled run for: {future_time}")
    print(f"Recurring run ID: {recurring_id}")
    
    # Start scheduler briefly
    scheduler.start_scheduler(manager, mock_executor)
    time.sleep(3)  # Let it run for a bit
    scheduler.stop_scheduler()
    
    scheduled_runs = scheduler.get_scheduled_runs()
    recurring_runs = scheduler.get_recurring_runs()
    print(f"Scheduled runs: {len(scheduled_runs)}")
    print(f"Recurring runs: {len(recurring_runs)}")
    
    # 6. Run Pipeline
    print("\n6. Run Pipeline:")
    print("-" * 30)
    
    pipeline = RunPipeline("data-analysis-pipeline")
    
    # Add stages to pipeline
    (pipeline
     .add_stage(
         "data_validation",
         "validator-agent",
         {
             "messages": [{
                 "role": "user",
                 "parts": [{"type": "text", "text": "Validate the input data: ${input_data}"}]
             }]
         }
     )
     .add_stage(
         "data_processing",
         "processor-agent",
         {
             "messages": [{
                 "role": "user",
                 "parts": [{"type": "text", "text": "Process validated data with method: ${processing_method}"}]
             }]
         }
     )
     .add_stage(
         "result_formatting",
         "formatter-agent",
         {
             "messages": [{
                 "role": "user",
                 "parts": [{"type": "text", "text": "Format the results for presentation"}]
             }]
         }
     ))
    
    print(f"Pipeline created: {pipeline.pipeline_id}")
    print(f"Stages: {len(pipeline.stages)}")
    
    # Execute pipeline with context
    context = {
        "input_data": "sample_dataset.csv",
        "processing_method": "statistical_analysis"
    }
    
    try:
        pipeline.execute(manager, mock_executor, context)
        status = pipeline.get_status()
        print(f"Pipeline status: {status['status']}")
        print(f"Completed stages: {status['current_stage'] + 1}/{status['total_stages']}")
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
    
    # 7. Run Archive
    print("\n7. Run Archive:")
    print("-" * 30)
    
    archive = RunArchive(max_runs=1000)
    
    # Archive completed runs
    archived_count = 0
    for run in runs:
        if run.is_finished():
            archive.archive_run(run)
            archived_count += 1
    
    print(f"Archived {archived_count} runs")
    
    # Search archived runs
    search_results = archive.search_runs(
        status=RunStatus.COMPLETED,
        limit=10
    )
    print(f"Found {len(search_results)} completed runs")
    
    # Get archive statistics
    archive_stats = archive.get_statistics()
    print(f"Archive stats: {archive_stats}")
    
    # 8. Error Handling and Edge Cases
    print("\n8. Error Handling and Edge Cases:")
    print("-" * 30)
    
    # Test run failure
    failing_run = create_simple_run("failing-agent", "This will fail")
    failing_run.start()
    failing_run.fail("Simulated failure", {"error_code": "SIM_001"})
    print(f"Failed run status: {failing_run.status.value}")
    print(f"Error: {failing_run.error}")
    
    # Test run timeout
    timeout_run = create_simple_run("timeout-agent", "This will timeout")
    timeout_run.start()
    timeout_run.timeout_run()
    print(f"Timeout run status: {timeout_run.status.value}")
    
    # Test run cancellation
    cancel_run = create_simple_run("cancel-agent", "This will be cancelled")
    cancel_run.start()
    cancel_run.cancel("User requested cancellation")
    print(f"Cancelled run status: {cancel_run.status.value}")
    
    # Test validation errors
    try:
        invalid_input = RunInput(messages=[])  # No messages
        invalid_run = ACPRun("", "invalid-agent", invalid_input)  # No run ID
        errors = invalid_run.validate()
        print(f"Validation errors: {errors}")
    except Exception as e:
        print(f"Validation exception: {e}")
    
    # 9. Serialization and Persistence
    print("\n9. Serialization and Persistence:")
    print("-" * 30)
    
    # Test serialization of complex run
    complex_run = (RunBuilder("serialization-test")
                   .add_text_message("user", "Test serialization")
                   .add_message({
                       "role": "user",
                       "parts": [{
                           "type": "json",
                           "data": {"test": True, "numbers": [1, 2, 3]},
                           "description": "Test data"
                       }]
                   })
                   .set_priority(RunPriority.URGENT)
                   .add_parameter("test_param", "test_value")
                   .add_metadata("test_meta", {"nested": {"value": 42}})
                   .build())
    
    # Complete the run
    complex_run.start()
    complex_run.update_progress(0.8, "Almost done")
    output = RunOutput(
        messages=[{
            "role": "agent",
            "parts": [{
                "type": "text",
                "text": "Serialization test completed"
            }, {
                "type": "json",
                "data": {"result": "success", "score": 0.95},
                "description": "Test results"
            }]
        }],
        artifacts=[{
            "type": "report",
            "filename": "test_report.json",
            "size": 1024,
            "description": "Test report"
        }],
        metadata={"test_completed": True}
    )
    complex_run.complete(output)
    
    # Serialize to JSON
    json_str = complex_run.to_json()
    print(f"Serialized run size: {len(json_str)} characters")
    
    # Deserialize from JSON
    reconstructed = ACPRun.from_dict(json.loads(json_str))
    print(f"Deserialization successful: {reconstructed.run_id == complex_run.run_id}")
    print(f"Status preserved: {reconstructed.status == complex_run.status}")
    print(f"Events preserved: {len(reconstructed.events) == len(complex_run.events)}")
    
    # 10. Performance and Stress Testing
    print("\n10. Performance and Stress Testing:")
    print("-" * 30)
    
    # Create many runs quickly
    start_time = time.time()
    stress_runs = []
    
    for i in range(100):
        run = (RunBuilder(f"stress-agent-{i%10}")
               .add_text_message("user", f"Stress test message {i}")
               .set_priority(RunPriority.NORMAL)
               .build())
        stress_runs.append(run)
    
    creation_time = time.time() - start_time
    print(f"Created 100 runs in {creation_time:.3f}s ({100/creation_time:.1f} runs/sec)")
    
    # Test serialization performance
    start_time = time.time()
    for run in stress_runs[:10]:  # Test subset
        json_str = run.to_json()
        reconstructed = ACPRun.from_dict(json.loads(json_str))
    
    serialization_time = time.time() - start_time
    print(f"Serialized/deserialized 10 runs in {serialization_time:.3f}s")
    
    # Cleanup
    manager.shutdown(wait=False)
    
    print("\n🎉 Comprehensive ACP run examples completed!")
    print(f"Total runs created: {len(runs) + len(stress_runs) + 10}")  # Approximate
    print("All functionality demonstrated successfully.")


if __name__ == "__main__":
    # Run examples if script is executed directly
    example_run_usage()
    print("\n" + "="*50)
    comprehensive_run_examples()
    print("\n" + "="*50)
    create_test_scenarios()
