# core

Abstract Base Classes: Foundation for all protocol implementations
Protocol Factory: Registry and factory for creating protocol-specific instances
Message Pipeline: Extensible message processing with middleware support
Connection Management: Centralized connection tracking across protocols
Event System: Pub/sub event bus for communication events
Metrics Collection: Built-in monitoring and performance tracking
Configuration Management: Centralized configuration with protocol-specific settings
Error Handling: Comprehensive exception hierarchy
Utility Functions: Common helpers for validation, serialization, etc.
Global State Management: Singleton instances for shared resources
Next Steps:
Now that we have the base foundation, we should proceed with:

Update existing protocol implementations to inherit from these base classes
Implement the remaining protocol components (ACP, remaining A2A, ANP client)
Create the API layer that uses these base classes
Build comprehensive tests using the base interfaces
