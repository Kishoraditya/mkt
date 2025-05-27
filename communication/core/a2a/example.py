"""
Complete A2A Protocol Example.

Demonstrates how to use the A2A client, server, and streaming components
together for agent-to-agent communication.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any

from .agent_card import AgentCard, Endpoint, Capability
from .client import A2AClient
from .server import A2AServer
from .message import MessageBuilder
from .task import TaskPriority

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExampleAgent:
    """Example A2A agent implementation."""
    
    def __init__(self, agent_id: str, name: str, port: int):
        """Initialize the example agent."""
        self.agent_id = agent_id
        self.name = name
        self.port = port
        
        # Create agent card
        self.agent_card = AgentCard(
            agent_id=agent_id,
            name=name,
            version="1.0.0",
            description=f"Example A2A agent: {name}",
            endpoints=[
                Endpoint(
                    url=f"http://localhost:{port}",
                    protocol="a2a",
                    methods=["POST", "GET"]
                )
            ],
            capabilities=[
                Capability(
                    name="text_processing",
                    description="Process and analyze text",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "operation": {"type": "string", "enum": ["analyze", "summarize", "translate"]}
                        },
                        "required": ["text", "operation"]
                    },
                    output_schema={
                        "type": "object",
                        "properties": {
                            "result": {"type": "string"},
                            "metadata": {"type": "object"}
                        }
                    }
                ),
                Capability(
                    name="math_operations",
                    description="Perform mathematical calculations",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "operation": {"type": "string"},
                            "operands": {"type": "array", "items": {"type": "number"}}
                        },
                        "required": ["operation", "operands"]
                    },
                    output_schema={
                        "type": "object",
                        "properties": {
                            "result": {"type": "number"},
                            "operation": {"type": "string"}
                        }
                    }
                )
            ]
        )
        
        # Create server
        self.server = A2AServer(self.agent_card, port=port)
        self.setup_handlers()
        
        # Create client
        self.client = A2AClient(agent_id)
    
    def setup_handlers(self):
        """Setup task and message handlers."""
        # Register task handlers
        self.server.register_task_handler("text_processing", self.handle_text_processing)
        self.server.register_task_handler("math_operations", self.handle_math_operations)
        
        # Register message handlers
        self.server.register_message_handler("greeting", self.handle_greeting_message)
        self.server.register_message_handler("query", self.handle_query_message)
    
    async def handle_text_processing(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle text processing tasks."""
        try:
            text = input_data.get("text", "")
            operation = input_data.get("operation", "analyze")
            
            logger.info(f"Processing text operation: {operation}")
            
            # Simulate processing time
            await asyncio.sleep(2)
            
            if operation == "analyze":
                result = {
                    "result": f"Analysis of '{text}': {len(text)} characters, {len(text.split())} words",
                    "metadata": {
                        "character_count": len(text),
                        "word_count": len(text.split()),
                        "processed_by": self.agent_id,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
            elif operation == "summarize":
                result = {
                    "result": f"Summary: {text[:50]}..." if len(text) > 50 else f"Summary: {text}",
                    "metadata": {
                        "original_length": len(text),
                        "summary_length": min(50, len(text)),
                        "processed_by": self.agent_id,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
            elif operation == "translate":
                result = {
                    "result": f"[TRANSLATED] {text}",  # Mock translation
                    "metadata": {
                        "source_language": "auto-detected",
                        "target_language": "english",
                        "processed_by": self.agent_id,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            logger.info(f"Text processing completed: {operation}")
            return result
            
        except Exception as e:
            logger.error(f"Error in text processing: {str(e)}")
            raise
    
    async def handle_math_operations(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle mathematical operations."""
        try:
            operation = input_data.get("operation", "")
            operands = input_data.get("operands", [])
            
            logger.info(f"Processing math operation: {operation} with operands: {operands}")
            
            # Simulate processing time
            await asyncio.sleep(1)
            
            if operation == "add":
                result_value = sum(operands)
            elif operation == "multiply":
                result_value = 1
                for operand in operands:
                    result_value *= operand
            elif operation == "average":
                result_value = sum(operands) / len(operands) if operands else 0
            elif operation == "max":
                result_value = max(operands) if operands else 0
            elif operation == "min":
                result_value = min(operands) if operands else 0
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            result = {
                "result": result_value,
                "operation": operation,
                "operands": operands,
                "processed_by": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Math operation completed: {operation} = {result_value}")
            return result
            
        except Exception as e:
            logger.error(f"Error in math operations: {str(e)}")
            raise
    
    async def handle_greeting_message(self, message, task_id, sender_id):
        """Handle greeting messages."""
        greeting_text = message.get_text_content()
        
        response_data = {
            "greeting_response": f"Hello {sender_id}! You said: '{greeting_text}'",
            "agent_name": self.name,
            "agent_id": self.agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Processed greeting from {sender_id}")
        return response_data
    
    async def handle_query_message(self, message, task_id, sender_id):
        """Handle query messages."""
        query_text = message.get_text_content()
        
        # Simple query processing
        if "capabilities" in query_text.lower():
            capabilities_info = [
                {
                    "name": cap.name,
                    "description": cap.description
                }
                for cap in self.agent_card.capabilities
            ]
            response_data = {
                "query_response": "Here are my capabilities:",
                "capabilities": capabilities_info,
                "agent_id": self.agent_id
            }
        elif "status" in query_text.lower():
            stats = self.server.get_statistics()
            response_data = {
                "query_response": "Here is my current status:",
                "status": stats,
                "agent_id": self.agent_id
            }
        else:
            response_data = {
                "query_response": f"I received your query: '{query_text}'. Try asking about 'capabilities' or 'status'.",
                "agent_id": self.agent_id
            }
        
        logger.info(f"Processed query from {sender_id}: {query_text}")
        return response_data
    
    async def start(self):
        """Start the agent server."""
        await self.server.start()
        logger.info(f"Agent {self.name} started on port {self.port}")
    
    async def stop(self):
        """Stop the agent server."""
        await self.server.stop()
        await self.client.close()
        logger.info(f"Agent {self.name} stopped")
    
    async def discover_and_communicate(self, target_url: str):
        """Discover another agent and communicate with it."""
        try:
            logger.info(f"Discovering agent at {target_url}")
            
            # Discover the target agent
            target_agent = await self.client.discover_agent(target_url)
            logger.info(f"Discovered agent: {target_agent.name} ({target_agent.agent_id})")
            
            # Send a greeting message
            greeting_builder = MessageBuilder(role="user")
            greeting_builder.add_text(f"Hello from {self.name}!")
            greeting_builder.set_metadata({"type": "greeting"})
            greeting_message = greeting_builder.build()
            
            response = await self.client.send_message(
                target_agent.agent_id,
                greeting_message
            )
            logger.info(f"Greeting response: {response.get_data_content()}")
            
            # Create and execute a text processing task
            logger.info("Creating text processing task...")
            task_result = await self.client.execute_task_and_wait(
                target_agent_id=target_agent.agent_id,
                task_type="text_processing",
                input_data={
                    "text": "This is a sample text for processing by the A2A protocol.",
                    "operation": "analyze"
                },
                priority=TaskPriority.NORMAL,
                timeout=timedelta(minutes=2)
            )
            
            logger.info(f"Text processing result: {task_result.to_dict()}")
            
            # Create and execute a math operation task
            logger.info("Creating math operation task...")
            math_result = await self.client.execute_task_and_wait(
                target_agent_id=target_agent.agent_id,
                task_type="math_operations",
                input_data={
                    "operation": "add",
                    "operands": [10, 20, 30, 40]
                },
                priority=TaskPriority.HIGH,
                timeout=timedelta(minutes=1)
            )
            
            logger.info(f"Math operation result: {math_result.to_dict()}")
            
            # Send a query message
            query_builder = MessageBuilder(role="user")
            query_builder.add_text("What are your capabilities?")
            query_builder.set_metadata({"type": "query"})
            query_message = query_builder.build()
            
            query_response = await self.client.send_message(
                target_agent.agent_id,
                query_message
            )
            logger.info(f"Query response: {query_response.get_data_content()}")
            
        except Exception as e:
            logger.error(f"Error in communication: {str(e)}")
            raise


async def run_two_agent_demo():
    """Run a demonstration with two A2A agents communicating."""
    logger.info("Starting A2A two-agent demonstration")
    
    # Create two agents
    agent1 = ExampleAgent("agent-alice", "Alice", 8080)
    agent2 = ExampleAgent("agent-bob", "Bob", 8081)
    
    try:
        # Start both agents
        await agent1.start()
        await agent2.start()
        
        # Wait a moment for servers to be ready
        await asyncio.sleep(2)
        
        # Agent 1 discovers and communicates with Agent 2
        logger.info("\n=== Agent Alice communicating with Agent Bob ===")
        await agent1.discover_and_communicate("http://localhost:8081")
        
        # Wait a moment
        await asyncio.sleep(2)
        
        # Agent 2 discovers and communicates with Agent 1
        logger.info("\n=== Agent Bob communicating with Agent Alice ===")
        await agent2.discover_and_communicate("http://localhost:8080")
        
        # Display statistics
        logger.info("\n=== Final Statistics ===")
        alice_stats = agent1.server.get_statistics()
        bob_stats = agent2.server.get_statistics()
        
        logger.info(f"Alice stats: {json.dumps(alice_stats, indent=2)}")
        logger.info(f"Bob stats: {json.dumps(bob_stats, indent=2)}")
        
        logger.info("Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise
    finally:
        # Clean up
        await agent1.stop()
        await agent2.stop()


async def run_streaming_demo():
    """Run a demonstration of streaming capabilities."""
    logger.info("Starting A2A streaming demonstration")
    
    agent = ExampleAgent("streaming-agent", "StreamingAgent", 8082)
    
    try:
        await agent.start()
        await asyncio.sleep(1)
        
        # Test streaming by creating a long-running task
        async def long_running_task(input_data):
            """Simulate a long-running task with progress updates."""
            total_steps = 10
            
            for step in range(total_steps):
                # Send progress update
                progress = (step + 1) / total_steps
                await agent.server.streaming_handler.notify_task_progress(
                    "demo-task",
                    progress,
                    f"Processing step {step + 1}/{total_steps}"
                )
                
                # Send log message
                await agent.server.streaming_handler.notify_task_log(
                    "demo-task",
                    "info",
                    f"Completed step {step + 1}"
                )
                
                await asyncio.sleep(1)
            
            return {
                "result": "Long-running task completed",
                "steps_completed": total_steps,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Register the long-running task handler
        agent.server.register_task_handler("long_running", long_running_task)
        
        logger.info("Streaming demo setup complete")
        logger.info("You can now:")
        logger.info("1. Open http://localhost:8082/.well-known/agent-card to see the agent card")
        logger.info("2. Connect to http://localhost:8082/stream/demo-task for SSE updates")
        logger.info("3. Use WebSocket at ws://localhost:8082/ws for real-time updates")
        logger.info("4. Create tasks via POST to http://localhost:8082/tasks")
        
        # Keep running for demonstration
        await asyncio.sleep(60)
        
    finally:
        await agent.stop()


async def run_performance_test():
    """Run a performance test with multiple concurrent operations."""
    logger.info("Starting A2A performance test")
    
    agent1 = ExampleAgent("perf-agent-1", "PerfAgent1", 8083)
    agent2 = ExampleAgent("perf-agent-2", "PerfAgent2", 8084)
    
    try:
        await agent1.start()
        await agent2.start()
        await asyncio.sleep(2)
        
        # Discover agent 2
        target_agent = await agent1.client.discover_agent("http://localhost:8084")
        
        # Run multiple concurrent tasks
        num_tasks = 20
        start_time = datetime.utcnow()
        
        logger.info(f"Starting {num_tasks} concurrent tasks...")
        
        tasks = []
        for i in range(num_tasks):
            if i % 2 == 0:
                # Text processing task
                task = agent1.client.execute_task_and_wait(
                    target_agent_id=target_agent.agent_id,
                    task_type="text_processing",
                    input_data={
                        "text": f"Performance test text number {i}",
                        "operation": "analyze"
                    },
                    timeout=timedelta(minutes=1)
                )
            else:
                # Math operation task
                task = agent1.client.execute_task_and_wait(
                    target_agent_id=target_agent.agent_id,
                    task_type="math_operations",
                    input_data={
                        "operation": "add",
                        "operands": [i, i*2, i*3]
                    },
                    timeout=timedelta(minutes=1)
                )
            
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        # Analyze results
        successful_tasks = sum(1 for r in results if not isinstance(r, Exception))
        failed_tasks = len(results) - successful_tasks
        
        logger.info(f"Performance test completed:")
        logger.info(f"  Total tasks: {num_tasks}")
        logger.info(f"  Successful: {successful_tasks}")
        logger.info(f"  Failed: {failed_tasks}")
        logger.info(f"  Duration: {duration:.2f} seconds")
        logger.info(f"  Tasks per second: {num_tasks / duration:.2f}")
        
        # Display final statistics
        agent1_stats = agent1.server.get_statistics()
        agent2_stats = agent2.server.get_statistics()
        
        logger.info(f"Agent 1 final stats: {json.dumps(agent1_stats, indent=2)}")
        logger.info(f"Agent 2 final stats: {json.dumps(agent2_stats, indent=2)}")
        
    finally:
        await agent1.stop()
        await agent2.stop()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        demo_type = sys.argv[1]
    else:
        demo_type = "two_agent"
    
    if demo_type == "two_agent":
        asyncio.run(run_two_agent_demo())
    elif demo_type == "streaming":
        asyncio.run(run_streaming_demo())
    elif demo_type == "performance":
        asyncio.run(run_performance_test())
    else:
        print("Usage: python example.py [two_agent|streaming|performance]")
        print("Default: two_agent")
        asyncio.run(run_two_agent_demo())
