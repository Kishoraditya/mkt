# acp

a comprehensive communication/core/acp/agent_detail.py file that fully implements the ACP (Agent Communication Protocol) specification. Here are the key features:

🎯 Core Components:
AgentDetail - Main class following ACP specification
AgentCapability - Describes what the agent can do
AgentEndpoint - Communication endpoints
AgentMetrics - Performance and usage metrics
🏗️ Agent Types Supported:
AI_ASSISTANT
TASK_EXECUTOR
DATA_PROCESSOR
SERVICE_WRAPPER
WORKFLOW_ORCHESTRATOR
CUSTOM
📊 Input/Output Types:
TEXT, IMAGE, AUDIO, VIDEO
FILE, JSON, STRUCTURED_DATA
🔧 Generator Classes:
AgentDetailGenerator - Creates different types of agent details
AgentDetailRegistry - Manages and searches agent details
✨ Key Features:
Framework-agnostic - No implementation details exposed
Rich metadata - Capabilities, metrics, deployment info
Search functionality - By type, capability, input/output types
Validation - Comprehensive error checking
Serialization - JSON support for network communication
Registry system - Centralized agent management
Scaling info - Kubernetes-ready deployment metadata
🎪 Example Generators:
AI assistants with model info
Data processors with format support
Service wrappers for external APIs
Workflow orchestrators for complex tasks
This implementation is production-ready and follows the same high-quality standards as the A2A protocol files. It's fully compatible with the ACP specification and ready for integration with the broader communication module.

a comprehensive communication/core/acp/client.py file that provides a complete ACP client implementation. Here are the key features:

🎯 Core Components:
ACPClient - Main client class for ACP communication
ACPClientPool - Pool of clients for load balancing and failover
ClientConfig - Configuration management
AuthConfig - Authentication handling
🔧 Key Features:
Agent Discovery - Fetch agent details from /.well-known/agent-detail
Run Management - Create, monitor, and cancel runs
Message Exchange - Send messages to running runs
Streaming Support - Real-time run updates
Batch Execution - Concurrent run processing
Health Checking - Monitor agent availability
🚀 Advanced Capabilities:
Retry Logic - Exponential backoff for failed requests
Authentication - Bearer tokens, API keys, Basic auth
SSL/TLS Support - Configurable certificate validation
Connection Pooling - Efficient HTTP session management
Caching - Agent detail caching for performance
Statistics - Comprehensive metrics collection
🏗️ Client Pool Features:
Load Balancing - Round-robin client selection
Failover - Automatic switching to healthy clients
Health Monitoring - Continuous health checking
Broadcast Operations - Execute on all clients
Pool Statistics - Aggregated metrics
🛡️ Error Handling:
Custom exception hierarchy
Proper HTTP status code handling
Timeout management
Connection failure recovery
📊 Monitoring & Statistics:
Request/response metrics
Success/failure rates
Active run tracking
Cache statistics
Health status monitoring
This implementation follows the same high-quality standards as the A2A client and is fully compatible with the ACP specification. It's production-ready with comprehensive error handling, monitoring, and scalability features.

a comprehensive communication/core/acp/server.py file that provides a complete ACP server implementation. Here are the key features:

🎯 Core Components:
ACPServer - Main server class with FastAPI integration
RunManager - Manages run lifecycle and execution
AuthManager - Handles authentication and authorization
RateLimiter - Prevents abuse with rate limiting
AgentHandler - Abstract base for agent implementations
🔧 Key Features:
Agent Discovery - /.well-known/agent-detail endpoint
Run Management - Create, monitor, cancel runs
Message Handling - Send messages and handle await requests
Streaming Updates - Real-time run status via SSE
Health Monitoring - Health checks and statistics
🚀 Advanced Capabilities:
FastAPI Integration - Modern async web framework
Middleware Support - CORS, compression, auth, rate limiting
Concurrent Execution - Handle multiple runs simultaneously
Background Tasks - Non-blocking run execution
Resource Management - Automatic cleanup of old runs
🛡️ Security & Reliability:
Authentication - Bearer tokens, API keys
Rate Limiting - Per-IP request limits
Input Validation - Pydantic models for API
Error Handling - Comprehensive exception handling
Resource Limits - Prevent resource exhaustion
📊 Monitoring & Operations:
Health Checks - Server health and uptime
Statistics - Run metrics and performance data
Logging - Comprehensive logging throughout
Cleanup - Automatic removal of old runs
Configuration - Flexible server configuration
🏗️ Agent Handler Types:
SimpleEchoHandler - Basic echo agent for testing
AdvancedAgentHandler - Sophisticated agent with:
Capability registration
Middleware pipeline
Await request handling
Custom processing functions
🌐 API Endpoints:
GET /.well-known/agent-detail - Agent discovery endpoint
GET /health - Health check and server status
POST /acp/runs - Create new runs
GET /acp/runs/{run_id} - Get run status and results
POST /acp/runs/{run_id}/cancel - Cancel running runs
POST /acp/runs/{run_id}/messages - Send messages to runs
GET /acp/runs/{run_id}/stream - Stream real-time updates
GET /acp/statistics - Server and run statistics
🔄 Run Lifecycle:
CREATED → RUNNING → COMPLETED/FAILED/CANCELLED
    ↓         ↓            ↓
  Start    Process    Cleanup

Copy

Apply

📝 Usage Examples:
Simple Server:
python server.py simple

Copy

Execute

Advanced Server:
python server.py

Copy

Execute

Custom Agent Handler:
class MyAgentHandler(AgentHandler):
    async def process_run(self, run, messages):
        # Custom processing logic
        return output_messages

Copy

Apply

🔧 Configuration Options:
Host/port settings
Concurrent run limits
Authentication requirements
Rate limiting
Timeout configurations
Cleanup intervals
🎯 Production Ready:
Async/await throughout
Proper resource management
Comprehensive error handling
Security best practices
Monitoring and logging
Scalable architecture
This ACP server implementation is fully compatible with the ACP specification and provides a robust foundation for building agent communication systems. It can handle both simple echo agents and sophisticated AI agents with custom capabilities.

a comprehensive communication/core/acp/await_handler.py file that implements the await mechanism for the ACP protocol. Here are the key features:

🎯 Core Components:
AwaitRequest - Data class representing await requests with validation
AwaitHandler - Abstract base class for different handler types
AwaitManager - Centralized management of await lifecycle
AwaitRequestBuilder - Builder pattern for easy creation
🔧 Handler Types:
InteractiveAwaitHandler - Prompts for real user input
AutoAwaitHandler - Provides automatic default responses
CallbackAwaitHandler - Uses registered callback functions
🚀 Key Features:
Timeout Management - Automatic timeout handling with cleanup
Response Validation - JSON schema validation support
Status Tracking - Complete lifecycle status management
Cancellation Support - Graceful cancellation of await requests
Statistics - Comprehensive metrics and monitoring
🛡️ Error Handling:
AwaitTimeoutError - For timeout scenarios
AwaitValidationError - For validation failures
AwaitCancelledError - For cancelled requests
AwaitNotFoundError - For missing requests
📊 Await Types Supported:
USER_INPUT - Free text input from user
CONFIRMATION - Boolean yes/no confirmation
CHOICE - Selection from predefined options
FILE_UPLOAD - File upload requests
APPROVAL - Approval/rejection requests
CUSTOM - Custom await types
🔄 Lifecycle Management:

```bash
PENDING → WAITING → COMPLETED/FAILED/CANCELLED/TIMEOUT
    ↓         ↓            ↓
  Create   Process    Cleanup
```

## 🎯 **Convenience Functions:**

1. **await_user_input()** - Simple text input
2. **await_confirmation()** - Boolean confirmation
3. **await_choice()** - Selection from options
4. **await_approval()** - Approval requests

### 📝 **Usage Examples:**

1. **Simple User Input:**

    ```python
    response = await await_user_input(
        manager, "run-123", "What is your name?", 
        default="Anonymous", timeout=30
    )
    ```

2. **Choice Selection:**

    ```python
    choice = await await_choice(
        manager, "run-123", "Select option:",
        ["A", "B", "C"], default="A"
    )
    ```

3. **Custom Handler:**

```python
async def custom_handler(await_request):
    # Custom processing logic
    return "Custom response"

callback_handler.register_handler(AwaitType.CUSTOM, custom_handler)
```

### 🔧 **Advanced Features:**

1. **Validation Schema:**

    ```python
    await_request = await manager.create_await(
        run_id="test",
        await_type=AwaitType.USER_INPUT,
        prompt="Enter email:",
        validation_schema={
            "type": "string",
            "format": "email"
        }
    )
    ```

2. **Metadata Support:**

    ```python
    await_request = await manager.create_await(
        # ... other params
        metadata={
            "priority": "high",
            "source": "api",
            "context": {"user_id": 123}
        }
    )
    ```

3. **Handler Registration:**

    ```python
    manager.register_handler("interactive", InteractiveAwaitHandler())
    manager.register_handler("auto", AutoAwaitHandler())
    ```

### 🛡️ **Production Features:**

1. **Resource Management** - Automatic cleanup of old requests
2. **Concurrent Safety** - Thread-safe operations
3. **Memory Efficiency** - Periodic cleanup of completed requests
4. **Error Recovery** - Graceful handling of failures
5. **Monitoring** - Comprehensive statistics and logging

### 📊 **Statistics Tracking:**

- Total await requests created
- Active await requests
- Completed/failed/cancelled/timeout counts
- Performance metrics

### 🔄 **Integration with ACP Server:**

The await handler integrates seamlessly with the ACP server implementation, allowing agents to:

1. **Pause Execution** - Stop processing and wait for input
2. **Request Information** - Ask for specific data from clients
3. **Validate Responses** - Ensure responses meet requirements
4. **Handle Timeouts** - Gracefully handle unresponsive clients
5. **Track Status** - Monitor await request progress

### 🎯 **Next Steps:**

This await handler implementation provides a complete foundation for the ACP protocol's await mechanism. It can be easily integrated with:

1. **ACP Server** - For handling await requests in runs
2. **ACP Client** - For responding to await requests
3. **Message System** - For await-related message handling
4. **Run Management** - For pausing/resuming run execution

The implementation is production-ready with comprehensive error handling, resource management, and monitoring capabilities. It supports all common await patterns while remaining flexible for custom use cases.

a comprehensive communication/core/acp/message.py file that implements the complete message system for the ACP protocol. Here are the key features:

🎯 Message Part Types:
TextPart - Plain text, markdown, or HTML content
ImagePart - Image data with metadata (dimensions, alt text)
JSONPart - Structured JSON data with optional schema validation
FilePart - File attachments with MIME type detection
BinaryPart - Raw binary data with encoding support
ReferencePart - External URI references with headers
🔧  Core Components:
ACPMessage - Main message class with multimodal support
MessageBuilder - Fluent builder pattern for easy construction
MessageValidator - Configurable validation with size limits and custom rules
MessageProcessor - Processing pipeline with preprocessors, filters, and postprocessors
MessageArchive - Storage and retrieval with indexing and search capabilities
MessageConverter - Conversion utilities (text, markdown, structured data extraction)
🚀 Key Features:
Multimodal Content - Support for text, images, files, JSON, binary data, and references
Serialization - Complete JSON serialization/deserialization with base64 encoding for binary data
Validation - Comprehensive validation with configurable rules and custom validators
Processing Pipeline - Flexible message processing with filters and transformations
Archive System - Message storage with thread/run indexing and search capabilities
Performance Optimized - Efficient handling of large messages and bulk operations
Error Handling - Robust error handling with graceful degradation
Thread Safety - Safe for concurrent operations
Extensible - Easy to add new part types and processing functions
📊 Usage Examples:

## Simple text message

msg = create_text_message(MessageRole.USER, "Hello!", run_id="run-123")

## Multimodal message

msg = (MessageBuilder(MessageRole.AGENT, "run-123")
       .add_text("Analysis complete:")
       .add_json({"result": "success", "confidence": 0.95})
       .add_reference("[https://example.com/report.pdf](https://example.com/report.pdf)")
       .build())

## Validation with custom rules

validator = (MessageValidator()
            .set_max_message_size(1024*1024)
            .add_custom_validator(business_rule_validator))

## Message processing pipeline

processor = (MessageProcessor()
            .add_preprocessor(add_metadata)
            .add_content_filter(sanitize_content)
            .add_postprocessor(log_processing))

## Archive with search

archive = MessageArchive(max_messages=1000)
archive.store(msg)
results = archive.search_messages("analysis", role=MessageRole.AGENT)

Copy

Apply

🛡️ Production Features:
Resource Management - Automatic cleanup and memory management
Size Limits - Configurable limits for messages, parts, and content
Content Filtering - Built-in content sanitization and filtering
Monitoring - Comprehensive logging and statistics
Schema Validation - JSON schema validation for structured data
MIME Type Detection - Automatic MIME type detection for files
Thread/Run Grouping - Message organization by conversation threads and runs
🔄 Integration Points:
This message implementation integrates seamlessly with:

ACP Server - For handling incoming messages in runs
ACP Client - For sending messages to other agents
Await Handler - For messages that trigger await requests
Run Management - For organizing messages within runs
Agent Detail - For capability-based message routing
🎯 Next Steps:
The message system is now complete and ready for integration. The next logical components to implement would be:

communication/core/acp/run.py - Run lifecycle management
communication/core/acp/client.py - ACP client implementation
communication/core/acp/server.py - ACP server implementation

1. **Core Run Classes:**
   - `RunStatus` enum for tracking run states
   - `RunPriority` enum for prioritization
   - `RunInput` and `RunOutput` dataclasses for structured data
   - `RunEvent` for lifecycle tracking
   - `ACPRun` main class with full lifecycle management

2. **Advanced Management:**
   - `RunManager` for concurrent run execution
   - `RunMonitor` for performance tracking and alerting
   - `RunScheduler` for time-based and recurring execution
   - `RunArchive` for long-term storage and search
   - `RunPipeline` for chaining multiple runs

3. **Builder Pattern:**
   - `RunBuilder` for fluent run construction
   - Utility functions for common run types

4. **Monitoring and Analytics:**
   - Real-time performance metrics
   - Alert system for anomalies
   - Historical data analysis
   - Resource usage tracking

5. **Serialization:**
   - Complete JSON serialization/deserialization
   - Preservation of all run state and metadata
   - Support for complex nested data structures

6. **Error Handling:**
   - Comprehensive validation
   - Graceful failure handling
   - Retry mechanisms
   - Timeout management

7. **Concurrency Support:**
   - Thread-safe operations
   - Concurrent execution management
   - Resource limiting
   - Deadlock prevention

8. **Testing and Examples:**
   - Comprehensive test scenarios
   - Performance benchmarks
   - Usage examples
   - Edge case handling

The implementation follows ACP protocol specifications while providing a robust, production-ready foundation for agent run management. It integrates seamlessly with the broader communication module and supports all required ACP features including the await mechanism, message handling, and agent detail management.
