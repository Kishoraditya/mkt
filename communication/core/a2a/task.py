"""
A2A Task implementation.

Handles task lifecycle management, state tracking, and coordination
for A2A protocol communication.
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor

import logging

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task status enumeration."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class TaskPriority(Enum):
    """Task priority enumeration."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class TaskResult:
    """Represents the result of a completed task."""
    
    task_id: str
    status: TaskStatus
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    artifacts: Optional[List[Dict[str, Any]]] = None
    execution_time: Optional[float] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.artifacts is None:
            self.artifacts = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['status'] = self.status.value
        if self.timestamp:
            data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class Task:
    """
    A2A Task following Google A2A specification.
    
    Represents a unit of work with defined lifecycle and state management.
    """
    
    # Required fields
    task_id: str
    agent_id: str
    task_type: str
    
    # Optional fields
    status: TaskStatus = TaskStatus.CREATED
    priority: TaskPriority = TaskPriority.NORMAL
    input_data: Optional[Dict[str, Any]] = None
    result: Optional[TaskResult] = None
    metadata: Optional[Dict[str, Any]] = None
    
    # Timing fields
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    timeout: Optional[timedelta] = None
    
    # Progress tracking
    progress: float = 0.0
    progress_message: Optional[str] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}
        if self.timeout is None:
            self.timeout = timedelta(minutes=30)  # Default 30 minute timeout
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        data = {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "task_type": self.task_type,
            "status": self.status.value,
            "priority": self.priority.value,
            "input_data": self.input_data,
            "result": self.result.to_dict() if self.result else None,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "timeout": self.timeout.total_seconds() if self.timeout else None,
            "progress": self.progress,
            "progress_message": self.progress_message
        }
        return {k: v for k, v in data.items() if v is not None}
    
    def to_json(self) -> str:
        """Convert task to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create task from dictionary."""
        # Convert datetime strings back to datetime objects
        created_at = None
        if data.get('created_at'):
            created_at = datetime.fromisoformat(data['created_at'])
        
        started_at = None
        if data.get('started_at'):
            started_at = datetime.fromisoformat(data['started_at'])
        
        completed_at = None
        if data.get('completed_at'):
            completed_at = datetime.fromisoformat(data['completed_at'])
        
        # Convert timeout seconds to timedelta
        timeout = None
        if data.get('timeout'):
            timeout = timedelta(seconds=data['timeout'])
        
        # Convert result dictionary to TaskResult
        result = None
        # Convert result dictionary to TaskResult
        result = None
        if data.get('result'):
            result_data = data['result'].copy()
            result_data['status'] = TaskStatus(result_data['status'])
            if result_data.get('timestamp'):
                result_data['timestamp'] = datetime.fromisoformat(result_data['timestamp'])
            result = TaskResult(**result_data)
        
        return cls(
            task_id=data['task_id'],
            agent_id=data['agent_id'],
            task_type=data['task_type'],
            status=TaskStatus(data.get('status', 'created')),
            priority=TaskPriority(data.get('priority', 'normal')),
            input_data=data.get('input_data'),
            result=result,
            metadata=data.get('metadata', {}),
            created_at=created_at,
            started_at=started_at,
            completed_at=completed_at,
            timeout=timeout,
            progress=data.get('progress', 0.0),
            progress_message=data.get('progress_message')
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Task':
        """Create task from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def start(self) -> None:
        """Mark task as started."""
        if self.status != TaskStatus.CREATED:
            raise ValueError(f"Cannot start task in status {self.status.value}")
        
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.utcnow()
        logger.info(f"Task {self.task_id} started")
    
    def complete(self, result_data: Optional[Dict[str, Any]] = None, artifacts: Optional[List[Dict[str, Any]]] = None) -> None:
        """Mark task as completed with results."""
        if self.status not in [TaskStatus.RUNNING, TaskStatus.PAUSED]:
            raise ValueError(f"Cannot complete task in status {self.status.value}")
        
        execution_time = None
        if self.started_at:
            execution_time = (datetime.utcnow() - self.started_at).total_seconds()
        
        self.result = TaskResult(
            task_id=self.task_id,
            status=TaskStatus.COMPLETED,
            result_data=result_data,
            artifacts=artifacts or [],
            execution_time=execution_time
        )
        
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.progress = 100.0
        self.progress_message = "Task completed successfully"
        
        logger.info(f"Task {self.task_id} completed in {execution_time:.2f}s")
    
    def fail(self, error_message: str) -> None:
        """Mark task as failed with error message."""
        if self.status == TaskStatus.COMPLETED:
            raise ValueError("Cannot fail a completed task")
        
        execution_time = None
        if self.started_at:
            execution_time = (datetime.utcnow() - self.started_at).total_seconds()
        
        self.result = TaskResult(
            task_id=self.task_id,
            status=TaskStatus.FAILED,
            error_message=error_message,
            execution_time=execution_time
        )
        
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.progress_message = f"Task failed: {error_message}"
        
        logger.error(f"Task {self.task_id} failed: {error_message}")
    
    def cancel(self) -> None:
        """Cancel the task."""
        if self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            raise ValueError(f"Cannot cancel task in status {self.status.value}")
        
        self.status = TaskStatus.CANCELLED
        self.completed_at = datetime.utcnow()
        self.progress_message = "Task cancelled"
        
        logger.info(f"Task {self.task_id} cancelled")
    
    def pause(self) -> None:
        """Pause the task."""
        if self.status != TaskStatus.RUNNING:
            raise ValueError(f"Cannot pause task in status {self.status.value}")
        
        self.status = TaskStatus.PAUSED
        self.progress_message = "Task paused"
        
        logger.info(f"Task {self.task_id} paused")
    
    def resume(self) -> None:
        """Resume the paused task."""
        if self.status != TaskStatus.PAUSED:
            raise ValueError(f"Cannot resume task in status {self.status.value}")
        
        self.status = TaskStatus.RUNNING
        self.progress_message = "Task resumed"
        
        logger.info(f"Task {self.task_id} resumed")
    
    def update_progress(self, progress: float, message: Optional[str] = None) -> None:
        """Update task progress."""
        if not 0.0 <= progress <= 100.0:
            raise ValueError("Progress must be between 0.0 and 100.0")
        
        self.progress = progress
        if message:
            self.progress_message = message
        
        logger.debug(f"Task {self.task_id} progress: {progress:.1f}%")
    
    def is_expired(self) -> bool:
        """Check if task has exceeded its timeout."""
        if not self.started_at or not self.timeout:
            return False
        
        return datetime.utcnow() > (self.started_at + self.timeout)
    
    def get_execution_time(self) -> Optional[float]:
        """Get current execution time in seconds."""
        if not self.started_at:
            return None
        
        end_time = self.completed_at or datetime.utcnow()
        return (end_time - self.started_at).total_seconds()
    
    def validate(self) -> List[str]:
        """Validate task and return list of errors."""
        errors = []
        
        if not self.task_id:
            errors.append("task_id is required")
        if not self.agent_id:
            errors.append("agent_id is required")
        if not self.task_type:
            errors.append("task_type is required")
        
        if not 0.0 <= self.progress <= 100.0:
            errors.append("progress must be between 0.0 and 100.0")
        
        return errors


class TaskManager:
    """Manager for handling multiple A2A tasks."""
    
    def __init__(self, max_concurrent_tasks: int = 10):
        """Initialize task manager."""
        self.tasks: Dict[str, Task] = {}
        self.max_concurrent_tasks = max_concurrent_tasks
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        self.task_handlers: Dict[str, Callable] = {}
        
        logger.info(f"TaskManager initialized with max {max_concurrent_tasks} concurrent tasks")
    
    def register_handler(self, task_type: str, handler: Callable) -> None:
        """Register a handler function for a specific task type."""
        self.task_handlers[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type}")
    
    def create_task(
        self,
        agent_id: str,
        task_type: str,
        input_data: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[timedelta] = None
    ) -> Task:
        """Create a new task."""
        task_id = str(uuid.uuid4())
        
        task = Task(
            task_id=task_id,
            agent_id=agent_id,
            task_type=task_type,
            input_data=input_data,
            priority=priority,
            timeout=timeout
        )
        
        self.tasks[task_id] = task
        logger.info(f"Created task {task_id} of type {task_type} for agent {agent_id}")
        
        return task
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self.tasks.get(task_id)
    
    def get_tasks_by_agent(self, agent_id: str) -> List[Task]:
        """Get all tasks for a specific agent."""
        return [task for task in self.tasks.values() if task.agent_id == agent_id]
    
    def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """Get all tasks with a specific status."""
        return [task for task in self.tasks.values() if task.status == status]
    
    def get_running_tasks(self) -> List[Task]:
        """Get all currently running tasks."""
        return self.get_tasks_by_status(TaskStatus.RUNNING)
    
    def get_pending_tasks(self) -> List[Task]:
        """Get all pending (created) tasks."""
        return self.get_tasks_by_status(TaskStatus.CREATED)
    
    async def execute_task(self, task_id: str) -> TaskResult:
        """Execute a task asynchronously."""
        task = self.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        if task.task_type not in self.task_handlers:
            task.fail(f"No handler registered for task type: {task.task_type}")
            return task.result
        
        try:
            task.start()
            handler = self.task_handlers[task.task_type]
            
            # Execute handler in thread pool
            loop = asyncio.get_event_loop()
            result_data = await loop.run_in_executor(
                self.executor,
                handler,
                task.input_data
            )
            
            task.complete(result_data)
            
        except Exception as e:
            task.fail(str(e))
            logger.error(f"Task {task_id} execution failed: {str(e)}")
        
        return task.result
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        task = self.get_task(task_id)
        if not task:
            return False
        
        try:
            task.cancel()
            return True
        except ValueError as e:
            logger.warning(f"Cannot cancel task {task_id}: {str(e)}")
            return False
    
    def cleanup_completed_tasks(self, max_age: timedelta = timedelta(hours=24)) -> int:
        """Clean up completed tasks older than max_age."""
        cutoff_time = datetime.utcnow() - max_age
        tasks_to_remove = []
        
        for task_id, task in self.tasks.items():
            if (task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] and
                task.completed_at and task.completed_at < cutoff_time):
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del self.tasks[task_id]
        
        logger.info(f"Cleaned up {len(tasks_to_remove)} completed tasks")
        return len(tasks_to_remove)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get task manager statistics."""
        total_tasks = len(self.tasks)
        status_counts = {}
        
        for status in TaskStatus:
            status_counts[status.value] = len(self.get_tasks_by_status(status))
        
        running_tasks = self.get_running_tasks()
        avg_execution_time = 0.0
        
        if running_tasks:
            execution_times = [task.get_execution_time() for task in running_tasks if task.get_execution_time()]
            if execution_times:
                avg_execution_time = sum(execution_times) / len(execution_times)
        
        return {
            "total_tasks": total_tasks,
            "status_counts": status_counts,
            "running_tasks": len(running_tasks),
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "registered_handlers": list(self.task_handlers.keys()),
            "avg_execution_time": avg_execution_time
        }


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    # Create task manager
    manager = TaskManager(max_concurrent_tasks=5)
    
    # Register a sample handler
    def sample_handler(input_data):
        """Sample task handler."""
        import time
        time.sleep(2)  # Simulate work
        return {"processed": True, "input": input_data}
    
    manager.register_handler("sample_task", sample_handler)
    
    # Create and execute a task
    async def test_task_execution():
        task = manager.create_task(
            agent_id="agent-123",
            task_type="sample_task",
            input_data={"test": "data"},
            priority=TaskPriority.HIGH
        )
        
        print(f"Created task: {task.task_id}")
        print(f"Task JSON: {task.to_json()}")
        
        # Execute the task
        result = await manager.execute_task(task.task_id)
        print(f"Task result: {result.to_dict()}")
        
        # Get statistics
        stats = manager.get_statistics()
        print(f"Manager statistics: {json.dumps(stats, indent=2)}")
    
    # Run the test
    asyncio.run(test_task_execution())
