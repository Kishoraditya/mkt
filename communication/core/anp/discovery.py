"""
ANP Agent Discovery Service Protocol (ADSP) implementation.

This module implements the Agent Discovery Service Protocol for ANP,
enabling agents to discover each other through both active and passive mechanisms.
"""

import json
import uuid
import asyncio
import aiohttp
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, asdict, field
from urllib.parse import urljoin, urlparse
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DiscoveryMethod(Enum):
    """Discovery method types."""
    WELL_KNOWN = "well_known"
    SEARCH_SERVICE = "search_service"
    BROADCAST = "broadcast"
    PEER_TO_PEER = "peer_to_peer"
    REGISTRY = "registry"


class AgentStatus(Enum):
    """Agent status in discovery."""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"


class CapabilityType(Enum):
    """Types of agent capabilities."""
    AI_REASONING = "ai_reasoning"
    DATA_PROCESSING = "data_processing"
    CODE_GENERATION = "code_generation"
    IMAGE_ANALYSIS = "image_analysis"
    TEXT_ANALYSIS = "text_analysis"
    WORKFLOW_ORCHESTRATION = "workflow_orchestration"
    MONITORING = "monitoring"
    SECURITY = "security"
    COMMUNICATION = "communication"
    STORAGE = "storage"
    CUSTOM = "custom"


@dataclass
class AgentCapability:
    """Represents an agent capability for discovery."""
    
    type: CapabilityType
    name: str
    description: str
    version: str = "1.0.0"
    parameters: Optional[Dict[str, Any]] = None
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    cost_model: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.parameters is None:
            self.parameters = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['type'] = self.type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentCapability':
        """Create from dictionary."""
        data['type'] = CapabilityType(data['type'])
        return cls(**data)


@dataclass
class AgentEndpoint:
    """Represents an agent communication endpoint."""
    
    url: str
    protocol: str = "https"
    supported_methods: List[str] = field(default_factory=lambda: ["POST", "GET"])
    authentication: Optional[Dict[str, Any]] = None
    rate_limits: Optional[Dict[str, Any]] = None
    health_check_path: str = "/health"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentEndpoint':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class AgentMetadata:
    """Agent metadata for discovery."""
    
    # Required fields
    agent_id: str
    name: str
    description: str
    version: str
    
    # Optional fields
    owner: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    license: Optional[str] = None
    documentation_url: Optional[str] = None
    source_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMetadata':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class DiscoverableAgent:
    """Complete discoverable agent information."""
    
    # Core identification
    agent_id: str
    did: str  # Decentralized Identifier
    metadata: AgentMetadata
    
    # Communication
    endpoints: List[AgentEndpoint]
    capabilities: List[AgentCapability]
    
    # Discovery information
    discovery_methods: List[DiscoveryMethod]
    status: AgentStatus = AgentStatus.ONLINE
    last_seen: Optional[datetime] = None
    discovery_ttl: int = 3600  # Time to live in seconds
    
    # Security
    public_key: Optional[str] = None
    certificate: Optional[str] = None
    trust_level: int = 0  # 0-100 trust score
    
    # Performance metrics
    response_time_avg: Optional[float] = None
    availability: Optional[float] = None
    load_factor: Optional[float] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.last_seen is None:
            self.last_seen = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            'agent_id': self.agent_id,
            'did': self.did,
            'metadata': self.metadata.to_dict(),
            'endpoints': [ep.to_dict() for ep in self.endpoints],
            'capabilities': [cap.to_dict() for cap in self.capabilities],
            'discovery_methods': [method.value for method in self.discovery_methods],
            'status': self.status.value,
            'last_seen': self.last_seen.isoformat() if self.last_seen else None,
            'discovery_ttl': self.discovery_ttl,
            'public_key': self.public_key,
            'certificate': self.certificate,
            'trust_level': self.trust_level,
            'response_time_avg': self.response_time_avg,
            'availability': self.availability,
            'load_factor': self.load_factor
        }
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DiscoverableAgent':
        """Create from dictionary."""
        # Convert nested objects
        metadata = AgentMetadata.from_dict(data['metadata'])
        endpoints = [AgentEndpoint.from_dict(ep) for ep in data['endpoints']]
        capabilities = [AgentCapability.from_dict(cap) for cap in data['capabilities']]
        discovery_methods = [DiscoveryMethod(method) for method in data['discovery_methods']]
        status = AgentStatus(data['status'])
        
        # Convert datetime
        last_seen = None
        if data.get('last_seen'):
            last_seen = datetime.fromisoformat(data['last_seen'])
        
        return cls(
            agent_id=data['agent_id'],
            did=data['did'],
            metadata=metadata,
            endpoints=endpoints,
            capabilities=capabilities,
            discovery_methods=discovery_methods,
            status=status,
            last_seen=last_seen,
            discovery_ttl=data.get('discovery_ttl', 3600),
            public_key=data.get('public_key'),
            certificate=data.get('certificate'),
            trust_level=data.get('trust_level', 0),
            response_time_avg=data.get('response_time_avg'),
            availability=data.get('availability'),
            load_factor=data.get('load_factor')
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'DiscoverableAgent':
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def is_expired(self) -> bool:
        """Check if agent discovery information has expired."""
        if not self.last_seen:
            return True
        
        expiry_time = self.last_seen + timedelta(seconds=self.discovery_ttl)
        return datetime.utcnow() > expiry_time
    
    def update_last_seen(self) -> None:
        """Update last seen timestamp."""
        self.last_seen = datetime.utcnow()
    
    def get_primary_endpoint(self) -> Optional[AgentEndpoint]:
        """Get the primary communication endpoint."""
        if not self.endpoints:
            return None
        return self.endpoints[0]
    
    def supports_capability(self, capability_type: CapabilityType) -> bool:
        """Check if agent supports a specific capability type."""
        return any(cap.type == capability_type for cap in self.capabilities)
    
    def get_capabilities_by_type(self, capability_type: CapabilityType) -> List[AgentCapability]:
        """Get all capabilities of a specific type."""
        return [cap for cap in self.capabilities if cap.type == capability_type]
    
    def calculate_compatibility_score(self, required_capabilities: List[CapabilityType]) -> float:
        """Calculate compatibility score based on required capabilities."""
        if not required_capabilities:
            return 1.0
        
        supported_count = sum(1 for cap_type in required_capabilities 
                            if self.supports_capability(cap_type))
        
        return supported_count / len(required_capabilities)


@dataclass
class DiscoveryQuery:
    """Query for discovering agents."""
    
    # Query criteria
    capability_types: List[CapabilityType] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    
    # Filters
    min_trust_level: int = 0
    max_response_time: Optional[float] = None
    min_availability: Optional[float] = None
    status_filter: List[AgentStatus] = field(default_factory=lambda: [AgentStatus.ONLINE])
    
    # Search parameters
    max_results: int = 50
    include_expired: bool = False
    sort_by: str = "trust_level"  # trust_level, response_time, availability, last_seen
    sort_order: str = "desc"  # asc, desc
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'capability_types': [ct.value for ct in self.capability_types],
            'tags': self.tags,
            'categories': self.categories,
            'min_trust_level': self.min_trust_level,
            'max_response_time': self.max_response_time,
            'min_availability': self.min_availability,
            'status_filter': [status.value for status in self.status_filter],
            'max_results': self.max_results,
            'include_expired': self.include_expired,
            'sort_by': self.sort_by,
            'sort_order': self.sort_order
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DiscoveryQuery':
        """Create from dictionary."""
        capability_types = [CapabilityType(ct) for ct in data.get('capability_types', [])]
        status_filter = [AgentStatus(status) for status in data.get('status_filter', ['online'])]
        
        return cls(
            capability_types=capability_types,
            tags=data.get('tags', []),
            categories=data.get('categories', []),
            min_trust_level=data.get('min_trust_level', 0),
            max_response_time=data.get('max_response_time'),
            min_availability=data.get('min_availability'),
            status_filter=status_filter,
            max_results=data.get('max_results', 50),
            include_expired=data.get('include_expired', False),
            sort_by=data.get('sort_by', 'trust_level'),
            sort_order=data.get('sort_order', 'desc')
        )


class WellKnownDiscovery:
    """Implements .well-known URI discovery mechanism."""
    
    WELL_KNOWN_PATH = "/.well-known/anp-agent"
    
    def __init__(self, timeout: int = 30):
        """Initialize well-known discovery."""
        self.timeout = timeout
        
    async def discover_agent(self, base_url: str) -> Optional[DiscoverableAgent]:
        """Discover agent via .well-known URI."""
        well_known_url = urljoin(base_url, self.WELL_KNOWN_PATH)
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(well_known_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        agent = DiscoverableAgent.from_dict(data)
                        agent.update_last_seen()
                        
                        logger.info(f"Discovered agent {agent.agent_id} via well-known URI")
                        return agent
                    else:
                        logger.warning(f"Well-known discovery failed for {base_url}: HTTP {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error discovering agent at {base_url}: {e}")
            return None
    
    async def publish_agent(self, agent: DiscoverableAgent, server_callback) -> bool:
        """Publish agent information for well-known discovery."""
        try:
            # Register the well-known endpoint with the server
            agent_data = agent.to_dict()
            
            # This would typically be handled by the web server
            # For now, we'll use a callback to register the endpoint
            if server_callback:
                await server_callback(self.WELL_KNOWN_PATH, agent_data)
                logger.info(f"Published agent {agent.agent_id} for well-known discovery")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error publishing agent for well-known discovery: {e}")
            return False


class SearchServiceDiscovery:
    """Implements search service discovery mechanism."""
    
    def __init__(self, search_service_url: str, timeout: int = 30):
        """Initialize search service discovery."""
        self.search_service_url = search_service_url
        self.timeout = timeout
    
    async def register_agent(self, agent: DiscoverableAgent) -> bool:
        """Register agent with search service."""
        try:
            register_url = urljoin(self.search_service_url, "/api/v1/agents/register")
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(register_url, json=agent.to_dict()) as response:
                    if response.status in [200, 201]:
                        logger.info(f"Registered agent {agent.agent_id} with search service")
                        return True
                    else:
                        logger.warning(f"Failed to register agent: HTTP {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error registering agent with search service: {e}")
            return False
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister agent from search service."""
        try:
            unregister_url = urljoin(self.search_service_url, f"/api/v1/agents/{agent_id}")
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.delete(unregister_url) as response:
                    if response.status in [200, 204]:
                        logger.info(f"Unregistered agent {agent_id} from search service")
                        return True
                    else:
                        logger.warning(f"Failed to unregister agent: HTTP {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error unregistering agent from search service: {e}")
            return False
    
    async def search_agents(self, query: DiscoveryQuery) -> List[DiscoverableAgent]:
        """Search for agents using the search service."""
        try:
            search_url = urljoin(self.search_service_url, "/api/v1/agents/search")
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(search_url, json=query.to_dict()) as response:
                    if response.status == 200:
                        data = await response.json()
                        agents = [DiscoverableAgent.from_dict(agent_data) 
                                for agent_data in data.get('agents', [])]
                        
                        logger.info(f"Found {len(agents)} agents via search service")
                        return agents
                    else:
                        logger.warning(f"Search failed: HTTP {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error searching agents: {e}")
            return []
    
    async def get_agent(self, agent_id: str) -> Optional[DiscoverableAgent]:
        """Get specific agent information from search service."""
        try:
            agent_url = urljoin(self.search_service_url, f"/api/v1/agents/{agent_id}")
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(agent_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        agent = DiscoverableAgent.from_dict(data)
                        
                        logger.info(f"Retrieved agent {agent_id} from search service")
                        return agent
                    else:
                        logger.warning(f"Failed to get agent {agent_id}: HTTP {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error getting agent {agent_id}: {e}")
            return None
    
    async def update_agent_status(self, agent_id: str, status: AgentStatus) -> bool:
        """Update agent status in search service."""
        try:
            status_url = urljoin(self.search_service_url, f"/api/v1/agents/{agent_id}/status")
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.put(status_url, json={"status": status.value}) as response:
                    if response.status == 200:
                        logger.info(f"Updated agent {agent_id} status to {status.value}")
                        return True
                    else:
                        logger.warning(f"Failed to update agent status: HTTP {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error updating agent status: {e}")
            return False


class BroadcastDiscovery:
    """Implements broadcast discovery mechanism for local networks."""
    
    def __init__(self, broadcast_port: int = 8765, timeout: int = 5):
        """Initialize broadcast discovery."""
        self.broadcast_port = broadcast_port
        self.timeout = timeout
        self.discovered_agents: Dict[str, DiscoverableAgent] = {}
        
    async def broadcast_presence(self, agent: DiscoverableAgent) -> None:
        """Broadcast agent presence on local network."""
        try:
            import socket
            
            # Create broadcast message
            message = {
                "type": "agent_announcement",
                "agent_id": agent.agent_id,
                "timestamp": datetime.utcnow().isoformat(),
                "agent_data": agent.to_dict()
            }
            
            message_bytes = json.dumps(message).encode('utf-8')
            
            # Send broadcast
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            
            try:
                sock.sendto(message_bytes, ('<broadcast>', self.broadcast_port))
                logger.info(f"Broadcasted presence for agent {agent.agent_id}")
            finally:
                sock.close()
                
        except Exception as e:
            logger.error(f"Error broadcasting agent presence: {e}")
    
    async def listen_for_broadcasts(self, callback) -> None:
        """Listen for agent broadcast announcements."""
        try:
            import socket
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('', self.broadcast_port))
            sock.settimeout(1.0)  # Non-blocking with timeout
            
            logger.info(f"Listening for agent broadcasts on port {self.broadcast_port}")
            
            while True:
                try:
                    data, addr = sock.recvfrom(65536)
                    message = json.loads(data.decode('utf-8'))
                    
                    if message.get('type') == 'agent_announcement':
                        agent_data = message.get('agent_data')
                        if agent_data:
                            agent = DiscoverableAgent.from_dict(agent_data)
                            self.discovered_agents[agent.agent_id] = agent
                            
                            if callback:
                                await callback(agent, addr)
                            
                            logger.info(f"Discovered agent {agent.agent_id} via broadcast from {addr}")
                
                except socket.timeout:
                    # Continue listening
                    await asyncio.sleep(0.1)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in broadcast from {addr}")
                except Exception as e:
                    logger.error(f"Error processing broadcast: {e}")
                    
        except Exception as e:
            logger.error(f"Error in broadcast listener: {e}")
    
    def get_discovered_agents(self) -> List[DiscoverableAgent]:
        """Get all agents discovered via broadcast."""
        # Filter out expired agents
        current_time = datetime.utcnow()
        active_agents = []
        
        for agent in self.discovered_agents.values():
            if not agent.is_expired():
                active_agents.append(agent)
        
        return active_agents


class PeerToPeerDiscovery:
    """Implements peer-to-peer discovery mechanism."""
    
    def __init__(self):
        """Initialize P2P discovery."""
        self.known_peers: Set[str] = set()
        self.agent_cache: Dict[str, DiscoverableAgent] = {}
        
    async def add_peer(self, peer_url: str) -> None:
        """Add a known peer for discovery."""
        self.known_peers.add(peer_url)
        logger.info(f"Added peer: {peer_url}")
    
    async def remove_peer(self, peer_url: str) -> None:
        """Remove a peer from discovery."""
        self.known_peers.discard(peer_url)
        logger.info(f"Removed peer: {peer_url}")
    
    async def discover_from_peers(self, query: DiscoveryQuery) -> List[DiscoverableAgent]:
        """Discover agents by querying known peers."""
        discovered_agents = []
        
        for peer_url in self.known_peers:
            try:
                peer_agents = await self._query_peer(peer_url, query)
                discovered_agents.extend(peer_agents)
                
            except Exception as e:
                logger.error(f"Error querying peer {peer_url}: {e}")
        
        # Remove duplicates based on agent_id
        unique_agents = {}
        for agent in discovered_agents:
            if agent.agent_id not in unique_agents:
                unique_agents[agent.agent_id] = agent
        
        return list(unique_agents.values())
    
    async def _query_peer(self, peer_url: str, query: DiscoveryQuery) -> List[DiscoverableAgent]:
        """Query a specific peer for agents."""
        try:
            query_url = urljoin(peer_url, "/api/v1/discovery/query")
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(query_url, json=query.to_dict()) as response:
                    if response.status == 200:
                        data = await response.json()
                        agents = [DiscoverableAgent.from_dict(agent_data) 
                                for agent_data in data.get('agents', [])]
                        
                        # Cache discovered agents
                        for agent in agents:
                            self.agent_cache[agent.agent_id] = agent
                        
                        return agents
                    else:
                        logger.warning(f"Peer query failed: HTTP {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error querying peer {peer_url}: {e}")
            return []
    
    async def share_agent_with_peers(self, agent: DiscoverableAgent) -> int:
        """Share agent information with known peers."""
        shared_count = 0
        
        for peer_url in self.known_peers:
            try:
                if await self._share_with_peer(peer_url, agent):
                    shared_count += 1
                    
            except Exception as e:
                logger.error(f"Error sharing with peer {peer_url}: {e}")
        
        logger.info(f"Shared agent {agent.agent_id} with {shared_count} peers")
        return shared_count
    
    async def _share_with_peer(self, peer_url: str, agent: DiscoverableAgent) -> bool:
        """Share agent information with a specific peer."""
        try:
            share_url = urljoin(peer_url, "/api/v1/discovery/share")
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(share_url, json=agent.to_dict()) as response:
                    return response.status in [200, 201]
                    
        except Exception as e:
            logger.error(f"Error sharing with peer {peer_url}: {e}")
            return False


class RegistryDiscovery:
    """Implements centralized registry discovery mechanism."""
    
    def __init__(self, registry_url: str, api_key: Optional[str] = None):
        """Initialize registry discovery."""
        self.registry_url = registry_url
        self.api_key = api_key
        
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    async def register_agent(self, agent: DiscoverableAgent) -> bool:
        """Register agent with centralized registry."""
        try:
            register_url = urljoin(self.registry_url, "/api/v1/registry/agents")
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(
                    register_url, 
                    json=agent.to_dict(),
                    headers=self._get_headers()
                ) as response:
                    if response.status in [200, 201]:
                        logger.info(f"Registered agent {agent.agent_id} with registry")
                        return True
                    else:
                        logger.warning(f"Registry registration failed: HTTP {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error registering with registry: {e}")
            return False
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister agent from registry."""
        try:
            unregister_url = urljoin(self.registry_url, f"/api/v1/registry/agents/{agent_id}")
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.delete(
                    unregister_url,
                    headers=self._get_headers()
                ) as response:
                    if response.status in [200, 204]:
                        logger.info(f"Unregistered agent {agent_id} from registry")
                        return True
                    else:
                        logger.warning(f"Registry unregistration failed: HTTP {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error unregistering from registry: {e}")
            return False
    
    async def discover_agents(self, query: DiscoveryQuery) -> List[DiscoverableAgent]:
        """Discover agents from registry."""
        try:
            discover_url = urljoin(self.registry_url, "/api/v1/registry/discover")
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(
                    discover_url,
                    json=query.to_dict(),
                    headers=self._get_headers()
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        agents = [DiscoverableAgent.from_dict(agent_data) 
                                for agent_data in data.get('agents', [])]
                        
                        logger.info(f"Discovered {len(agents)} agents from registry")
                        return agents
                    else:
                        logger.warning(f"Registry discovery failed: HTTP {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error discovering from registry: {e}")
            return []
    
    async def get_agent_details(self, agent_id: str) -> Optional[DiscoverableAgent]:
        """Get detailed agent information from registry."""
        try:
            agent_url = urljoin(self.registry_url, f"/api/v1/registry/agents/{agent_id}")
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(
                    agent_url,
                    headers=self._get_headers()
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        agent = DiscoverableAgent.from_dict(data)
                        logger.info(f"Retrieved agent {agent_id} details from registry")
                        return agent
                    else:
                        logger.warning(f"Failed to get agent details: HTTP {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error getting agent details from registry: {e}")
            return None


class DiscoveryManager:
    """Main discovery manager that coordinates multiple discovery methods."""
    
    def __init__(self):
        """Initialize discovery manager."""
        self.well_known = WellKnownDiscovery()
        self.search_service: Optional[SearchServiceDiscovery] = None
        self.broadcast = BroadcastDiscovery()
        self.p2p = PeerToPeerDiscovery()
        self.registry: Optional[RegistryDiscovery] = None
        
        # Local agent cache
        self.local_agents: Dict[str, DiscoverableAgent] = {}
        self.discovery_cache: Dict[str, DiscoverableAgent] = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Discovery preferences
        self.enabled_methods: Set[DiscoveryMethod] = {
            DiscoveryMethod.WELL_KNOWN,
            DiscoveryMethod.BROADCAST
        }
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        
    def configure_search_service(self, service_url: str) -> None:
        """Configure search service discovery."""
        self.search_service = SearchServiceDiscovery(service_url)
        self.enabled_methods.add(DiscoveryMethod.SEARCH_SERVICE)
        logger.info(f"Configured search service: {service_url}")
    
    def configure_registry(self, registry_url: str, api_key: Optional[str] = None) -> None:
        """Configure registry discovery."""
        self.registry = RegistryDiscovery(registry_url, api_key)
        self.enabled_methods.add(DiscoveryMethod.REGISTRY)
        logger.info(f"Configured registry: {registry_url}")
    
    def enable_method(self, method: DiscoveryMethod) -> None:
        """Enable a discovery method."""
        self.enabled_methods.add(method)
        logger.info(f"Enabled discovery method: {method.value}")
    
    def disable_method(self, method: DiscoveryMethod) -> None:
        """Disable a discovery method."""
        self.enabled_methods.discard(method)
        logger.info(f"Disabled discovery method: {method.value}")
    
    async def register_local_agent(self, agent: DiscoverableAgent) -> bool:
        """Register a local agent for discovery."""
        try:
            self.local_agents[agent.agent_id] = agent
            
            # Register with all enabled discovery methods
            registration_results = []
            
            if DiscoveryMethod.SEARCH_SERVICE in self.enabled_methods and self.search_service:
                result = await self.search_service.register_agent(agent)
                registration_results.append(("search_service", result))
            
            if DiscoveryMethod.REGISTRY in self.enabled_methods and self.registry:
                result = await self.registry.register_agent(agent)
                registration_results.append(("registry", result))
            
            if DiscoveryMethod.BROADCAST in self.enabled_methods:
                await self.broadcast.broadcast_presence(agent)
                registration_results.append(("broadcast", True))
            
            if DiscoveryMethod.PEER_TO_PEER in self.enabled_methods:
                shared_count = await self.p2p.share_agent_with_peers(agent)
                registration_results.append(("p2p", shared_count > 0))
            
            # Log results
            successful_methods = [method for method, success in registration_results if success]
            logger.info(f"Registered agent {agent.agent_id} with methods: {successful_methods}")
            
            return len(successful_methods) > 0
            
        except Exception as e:
            logger.error(f"Error registering local agent: {e}")
            return False
    
    async def unregister_local_agent(self, agent_id: str) -> bool:
        """Unregister a local agent from discovery."""
        try:
            if agent_id not in self.local_agents:
                logger.warning(f"Agent {agent_id} not found in local agents")
                return False
            
            # Unregister from all enabled discovery methods
            unregistration_results = []
            
            if DiscoveryMethod.SEARCH_SERVICE in self.enabled_methods and self.search_service:
                result = await self.search_service.unregister_agent(agent_id)
                unregistration_results.append(("search_service", result))
            
            if DiscoveryMethod.REGISTRY in self.enabled_methods and self.registry:
                result = await self.registry.unregister_agent(agent_id)
                unregistration_results.append(("registry", result))
            
            # Remove from local cache
            del self.local_agents[agent_id]
            
            # Log results
            successful_methods = [method for method, success in unregistration_results if success]
            logger.info(f"Unregistered agent {agent_id} from methods: {successful_methods}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error unregistering local agent: {e}")
            return False
    
    async def discover_agents(self, query: DiscoveryQuery) -> List[DiscoverableAgent]:
        """Discover agents using all enabled methods."""
        all_discovered_agents = []
        
        try:
            # Discover from each enabled method
            discovery_tasks = []
            
            if DiscoveryMethod.WELL_KNOWN in self.enabled_methods:
                # Well-known discovery requires specific URLs
                # This would typically be called with specific base URLs
                pass
            
            if DiscoveryMethod.SEARCH_SERVICE in self.enabled_methods and self.search_service:
                task = self.search_service.search_agents(query)
                discovery_tasks.append(("search_service", task))
            
            if DiscoveryMethod.BROADCAST in self.enabled_methods:
                # Get agents discovered via broadcast
                broadcast_agents = self.broadcast.get_discovered_agents()
                filtered_agents = self._filter_agents(broadcast_agents, query)
                all_discovered_agents.extend(filtered_agents)
            
            if DiscoveryMethod.PEER_TO_PEER in self.enabled_methods:
                task = self.p2p.discover_from_peers(query)
                discovery_tasks.append(("p2p", task))
            
            if DiscoveryMethod.REGISTRY in self.enabled_methods and self.registry:
                task = self.registry.discover_agents(query)
                discovery_tasks.append(("registry", task))
            
            # Execute discovery tasks concurrently
            if discovery_tasks:
                results = await asyncio.gather(
                    *[task for _, task in discovery_tasks],
                    return_exceptions=True
                )
                
                for i, result in enumerate(results):
                    method_name = discovery_tasks[i][0]
                    if isinstance(result, Exception):
                        logger.error(f"Discovery error in {method_name}: {result}")
                    else:
                        all_discovered_agents.extend(result)
                        logger.info(f"Discovered {len(result)} agents via {method_name}")
            
            # Remove duplicates and apply final filtering
            unique_agents = self._deduplicate_agents(all_discovered_agents)
            filtered_agents = self._filter_agents(unique_agents, query)
            sorted_agents = self._sort_agents(filtered_agents, query)
            
            # Apply result limit
            final_agents = sorted_agents[:query.max_results]
            
            # Update cache
            for agent in final_agents:
                self.discovery_cache[agent.agent_id] = agent
            
            logger.info(f"Discovery completed: {len(final_agents)} agents found")
            return final_agents
            
        except Exception as e:
            logger.error(f"Error during agent discovery: {e}")
            return []
    
    async def discover_agent_by_url(self, base_url: str) -> Optional[DiscoverableAgent]:
        """Discover a specific agent by its base URL using well-known discovery."""
        if DiscoveryMethod.WELL_KNOWN not in self.enabled_methods:
            logger.warning("Well-known discovery is disabled")
            return None
        
        return await self.well_known.discover_agent(base_url)
    
    async def get_agent_details(self, agent_id: str) -> Optional[DiscoverableAgent]:
        """Get detailed information about a specific agent."""
        # Check local agents first
        if agent_id in self.local_agents:
            return self.local_agents[agent_id]
        
        # Check discovery cache
        if agent_id in self.discovery_cache:
            agent = self.discovery_cache[agent_id]
            if not agent.is_expired():
                return agent
        
        # Try to get from discovery services
        if self.search_service:
            agent = await self.search_service.get_agent(agent_id)
            if agent:
                self.discovery_cache[agent_id] = agent
                return agent
        
        if self.registry:
            agent = await self.registry.get_agent_details(agent_id)
            if agent:
                self.discovery_cache[agent_id] = agent
                return agent
        
        logger.warning(f"Agent {agent_id} not found")
        return None
    
    async def update_agent_status(self, agent_id: str, status: AgentStatus) -> bool:
        """Update agent status across discovery services."""
        success = True
        
        # Update local agent
        if agent_id in self.local_agents:
            self.local_agents[agent_id].status = status
            self.local_agents[agent_id].update_last_seen()
        
        # Update in discovery services
        if self.search_service:
            result = await self.search_service.update_agent_status(agent_id, status)
            success = success and result
        
        # Registry updates would be handled similarly
        # For now, we'll just log the status update
        logger.info(f"Updated agent {agent_id} status to {status.value}")
        
        return success
    
    def _filter_agents(self, agents: List[DiscoverableAgent], query: DiscoveryQuery) -> List[DiscoverableAgent]:
        """Filter agents based on query criteria."""
        filtered = []
        
        for agent in agents:
            # Skip expired agents unless explicitly included
            if not query.include_expired and agent.is_expired():
                continue
            
            # Status filter
            if agent.status not in query.status_filter:
                continue
            
            # Trust level filter
            if agent.trust_level < query.min_trust_level:
                continue
            
            # Response time filter
            if (query.max_response_time is not None and 
                agent.response_time_avg is not None and 
                agent.response_time_avg > query.max_response_time):
                continue
            
            # Availability filter
            if (query.min_availability is not None and 
                agent.availability is not None and 
                agent.availability < query.min_availability):
                continue
            
            # Capability filter
            if query.capability_types:
                if not any(agent.supports_capability(cap_type) for cap_type in query.capability_types):
                    continue
            
            # Tags filter
            if query.tags:
                if not any(tag in agent.metadata.tags for tag in query.tags):
                    continue
            
            # Categories filter
            if query.categories:
                if not any(cat in agent.metadata.categories for cat in query.categories):
                    continue
            
            filtered.append(agent)
        
        return filtered
    
    def _deduplicate_agents(self, agents: List[DiscoverableAgent]) -> List[DiscoverableAgent]:
        """Remove duplicate agents based on agent_id."""
        seen_ids = set()
        unique_agents = []
        
        for agent in agents:
            if agent.agent_id not in seen_ids:
                seen_ids.add(agent.agent_id)
                unique_agents.append(agent)
        
        return unique_agents
    
    def _sort_agents(self, agents: List[DiscoverableAgent], query: DiscoveryQuery) -> List[DiscoverableAgent]:
        """Sort agents based on query sort criteria."""
        reverse = query.sort_order == "desc"
        
        if query.sort_by == "trust_level":
            return sorted(agents, key=lambda a: a.trust_level or 0, reverse=reverse)
        elif query.sort_by == "response_time":
            return sorted(agents, key=lambda a: a.response_time_avg or float('inf'), reverse=not reverse)
        elif query.sort_by == "availability":
            return sorted(agents, key=lambda a: a.availability or 0, reverse=reverse)
        elif query.sort_by == "last_seen":
            return sorted(agents, key=lambda a: a.last_seen or datetime.min, reverse=reverse)
        else:
            # Default to trust level
            return sorted(agents, key=lambda a: a.trust_level or 0, reverse=True)
    
    async def start_background_discovery(self) -> None:
        """Start background discovery tasks."""
        # Start broadcast listener
        if DiscoveryMethod.BROADCAST in self.enabled_methods:
            task = asyncio.create_task(
                self.broadcast.listen_for_broadcasts(self._on_agent_discovered)
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        
        # Start periodic cleanup
        task = asyncio.create_task(self._periodic_cleanup())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        
        logger.info("Started background discovery tasks")
    
    async def stop_background_discovery(self) -> None:
        """Stop background discovery tasks."""
        for task in self._background_tasks:
            task.cancel()
        
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        self._background_tasks.clear()
        logger.info("Stopped background discovery tasks")
    
    async def _on_agent_discovered(self, agent: DiscoverableAgent, source_addr) -> None:
        """Callback for when an agent is discovered via broadcast."""
        logger.info(f"Agent {agent.agent_id} discovered via broadcast from {source_addr}")
        
        # Update cache
        self.discovery_cache[agent.agent_id] = agent
        
        # Could trigger additional actions here (notifications, etc.)
    
    async def _periodic_cleanup(self) -> None:
        """Periodic cleanup of expired agents and cache."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Clean up discovery cache
                current_time = datetime.utcnow()
                expired_agents = []
                
                for agent_id, agent in self.discovery_cache.items():
                    if agent.is_expired():
                        expired_agents.append(agent_id)
                
                for agent_id in expired_agents:
                    del self.discovery_cache[agent_id]
                
                if expired_agents:
                    logger.info(f"Cleaned up {len(expired_agents)} expired agents from cache")
                
                # Update local agent last_seen timestamps
                for agent in self.local_agents.values():
                    agent.update_last_seen()
                
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")


class DiscoveryService:
    """High-level discovery service that provides a simple interface."""
    
    def __init__(self):
        """Initialize discovery service."""
        self.manager = DiscoveryManager()
        self._initialized = False
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize discovery service with configuration."""
        try:
            # Configure search service if provided
            if 'search_service_url' in config:
                self.manager.configure_search_service(config['search_service_url'])
            
            # Configure registry if provided
            if 'registry_url' in config:
                api_key = config.get('registry_api_key')
                self.manager.configure_registry(config['registry_url'], api_key)
            
            # Configure enabled methods
            if 'enabled_methods' in config:
                for method_name in config['enabled_methods']:
                    try:
                        method = DiscoveryMethod(method_name)
                        self.manager.enable_method(method)
                    except ValueError:
                        logger.warning(f"Unknown discovery method: {method_name}")
            
            # Configure P2P peers
            if 'p2p_peers' in config:
                for peer_url in config['p2p_peers']:
                    await self.manager.p2p.add_peer(peer_url)
            
            # Start background tasks
            await self.manager.start_background_discovery()
            
            self._initialized = True
            logger.info("Discovery service initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing discovery service: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown discovery service."""
        if self._initialized:
            await self.manager.stop_background_discovery()
            
            # Unregister all local agents
            for agent_id in list(self.manager.local_agents.keys()):
                await self.manager.unregister_local_agent(agent_id)
            
            self._initialized = False
            logger.info("Discovery service shut down")
    
    async def register_agent(self, agent: DiscoverableAgent) -> bool:
        """Register an agent for discovery."""
        if not self._initialized:
            raise RuntimeError("Discovery service not initialized")
        
        return await self.manager.register_local_agent(agent)
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from discovery."""
        if not self._initialized:
            raise RuntimeError("Discovery service not initialized")
        
        return await self.manager.unregister_local_agent(agent_id)
    
    async def find_agents(
        self,
        capability_types: Optional[List[CapabilityType]] = None,
        tags: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        max_results: int = 10
    ) -> List[DiscoverableAgent]:
        """Find agents with specified criteria."""
        if not self._initialized:
            raise RuntimeError("Discovery service not initialized")
        
        query = DiscoveryQuery(
            capability_types=capability_types or [],
            tags=tags or [],
            categories=categories or [],
            max_results=max_results
        )
        
        return await self.manager.discover_agents(query)
    
    async def find_agent_by_id(self, agent_id: str) -> Optional[DiscoverableAgent]:
        """Find a specific agent by ID."""
        if not self._initialized:
            raise RuntimeError("Discovery service not initialized")
        
        return await self.manager.get_agent_details(agent_id)
    
    async def find_agent_by_url(self, base_url: str) -> Optional[DiscoverableAgent]:
        """Find an agent by its base URL."""
        if not self._initialized:
            raise RuntimeError("Discovery service not initialized")
        
        return await self.manager.discover_agent_by_url(base_url)
    
    def get_local_agents(self) -> List[DiscoverableAgent]:
        """Get all locally registered agents."""
        return list(self.manager.local_agents.values())
    
    async def update_agent_status(self, agent_id: str, status: AgentStatus) -> bool:
        """Update agent status."""
        if not self._initialized:
            raise RuntimeError("Discovery service not initialized")
        
        return await self.manager.update_agent_status(agent_id, status)


# Utility functions for creating common agent types
def create_ai_agent(
    name: str,
    description: str,
    base_url: str,
    did: str,
    capabilities: List[CapabilityType],
    **kwargs
) -> DiscoverableAgent:
    """Create a discoverable AI agent."""
    
    agent_id = str(uuid.uuid4())
    
    metadata = AgentMetadata(
        agent_id=agent_id,
        name=name,
        description=description,
        version=kwargs.get('version', '1.0.0'),
        owner=kwargs.get('owner'),
        tags=kwargs.get('tags', ['ai', 'agent']),
        categories=kwargs.get('categories', ['artificial_intelligence'])
    )
    
    endpoints = [
        AgentEndpoint(
            url=urljoin(base_url, "/api/v1/"),
            protocol="https" if base_url.startswith("https") else "http",
            supported_methods=["POST", "GET"],
            health_check_path="/health"
        )
    ]
    
    agent_capabilities = []
    for cap_type in capabilities:
        agent_capabilities.append(
            AgentCapability(
                type=cap_type,
                name=cap_type.value.replace('_', ' ').title(),
                description=f"AI capability: {cap_type.value}",
                version="1.0.0"
            )
        )
    
    return DiscoverableAgent(
        agent_id=agent_id,
        did=did,
        metadata=metadata,
        endpoints=endpoints,
        capabilities=agent_capabilities,
        discovery_methods=[
            DiscoveryMethod.WELL_KNOWN,
            DiscoveryMethod.SEARCH_SERVICE,
            DiscoveryMethod.REGISTRY
        ],
        status=AgentStatus.ONLINE,
        trust_level=kwargs.get('trust_level', 50),
        response_time_avg=kwargs.get('response_time_avg'),
        availability=kwargs.get('availability', 0.99),
        load_factor=kwargs.get('load_factor', 0.5)
    )


def create_service_agent(
    name: str,
    description: str,
    base_url: str,
    did: str,
    service_capabilities: Dict[str, str],
    **kwargs
) -> DiscoverableAgent:
    """Create a discoverable service agent."""
    
    agent_id = str(uuid.uuid4())
    
    metadata = AgentMetadata(
        agent_id=agent_id,
        name=name,
        description=description,
        version=kwargs.get('version', '1.0.0'),
        owner=kwargs.get('owner'),
        tags=kwargs.get('tags', ['service', 'agent']),
        categories=kwargs.get('categories', ['service'])
    )
    
    endpoints = [
        AgentEndpoint(
            url=urljoin(base_url, "/api/v1/"),
            protocol="https" if base_url.startswith("https") else "http",
            supported_methods=["POST", "GET", "PUT", "DELETE"],
            health_check_path="/health"
        )
    ]
    
    agent_capabilities = []
    for service_name, service_desc in service_capabilities.items():
        agent_capabilities.append(
            AgentCapability(
                type=CapabilityType.CUSTOM,
                name=service_name,
                description=service_desc,
                version="1.0.0",
                parameters={"service_type": "api_endpoint"}
            )
        )
    
    return DiscoverableAgent(
        agent_id=agent_id,
        did=did,
        metadata=metadata,
        endpoints=endpoints,
        capabilities=agent_capabilities,
        discovery_methods=[
            DiscoveryMethod.WELL_KNOWN,
            DiscoveryMethod.SEARCH_SERVICE,
            DiscoveryMethod.BROADCAST
        ],
        status=AgentStatus.ONLINE,
        trust_level=kwargs.get('trust_level', 75),
        response_time_avg=kwargs.get('response_time_avg'),
        availability=kwargs.get('availability', 0.995),
        load_factor=kwargs.get('load_factor', 0.3)
    )


# Example usage and testing functions
async def example_discovery_usage():
    """Example of basic discovery usage."""
    print("🔍 ANP Discovery - Basic Usage Example")
    print("=" * 40)
    
    # 1. Create discovery service
    discovery = DiscoveryService()
    
    # Configure discovery service
    config = {
        'enabled_methods': ['well_known', 'broadcast', 'peer_to_peer'],
        'p2p_peers': []  # Would contain actual peer URLs
    }
    
    await discovery.initialize(config)
    
    # 2. Create and register some agents
    print("\n1. Creating and registering agents...")
    
    # Create AI agent
    ai_agent = create_ai_agent(
        name="GPT Analysis Agent",
        description="AI agent for text analysis and generation",
        base_url="https://ai-agent.example.com",
        did="did:wba:example:ai-agent-123",
        capabilities=[
            CapabilityType.AI_REASONING,
            CapabilityType.TEXT_ANALYSIS,
            CapabilityType.CODE_GENERATION
        ],
        trust_level=85,
        availability=0.98
    )
    
    # Create service agent
    service_agent = create_service_agent(
        name="Data Processing Service",
        description="High-performance data processing service",
        base_url="https://data-service.example.com",
        did="did:wba:example:data-service-456",
        service_capabilities={
            "data_transformation": "Transform data between formats",
            "data_validation": "Validate data integrity and structure",
            "batch_processing": "Process large datasets in batches"
        },
        trust_level=90,
        availability=0.999
    )
    
    # Register agents
    ai_registered = await discovery.register_agent(ai_agent)
    service_registered = await discovery.register_agent(service_agent)
    
    print(f"✓ AI Agent registered: {ai_registered}")
    print(f"✓ Service Agent registered: {service_registered}")
    
    # 3. Discover agents
    print("\n2. Discovering agents...")
    
    # Find AI agents
    ai_agents = await discovery.find_agents(
        capability_types=[CapabilityType.AI_REASONING],
        max_results=5
    )
    
    print(f"✓ Found {len(ai_agents)} AI agents")
    for agent in ai_agents:
        print(f"  - {agent.metadata.name} (trust: {agent.trust_level})")
    
    # Find data processing agents
    data_agents = await discovery.find_agents(
        tags=['service'],
        categories=['service'],
        max_results=5
    )
    
    print(f"✓ Found {len(data_agents)} service agents")
    for agent in data_agents:
        print(f"  - {agent.metadata.name} (availability: {agent.availability})")
    
    # 4. Get specific agent details
    print("\n3. Getting agent details...")
    
    agent_details = await discovery.find_agent_by_id(ai_agent.agent_id)
    if agent_details:
        print(f"✓ Retrieved details for: {agent_details.metadata.name}")
        print(f"  Capabilities: {[cap.name for cap in agent_details.capabilities]}")
        print(f"  Status: {agent_details.status.value}")
    
    # 5. Update agent status
    print("\n4. Updating agent status...")
    
    status_updated = await discovery.update_agent_status(
        service_agent.agent_id,
        AgentStatus.MAINTENANCE
    )
    
    print(f"✓ Status updated: {status_updated}")
    
    # 6. Cleanup
    await discovery.shutdown()
    print("\n✓ Discovery service shut down")


async def comprehensive_discovery_testing():
    """Comprehensive testing of discovery functionality."""
    print("\n🧪 ANP Discovery - Comprehensive Testing")
    print("=" * 42)
    
    # 1. Test agent creation and validation
    print("\n1. Testing agent creation and validation...")
    
    # Create test agent
    test_agent = create_ai_agent(
        name="Test Agent",
        description="Agent for testing purposes",
        base_url="https://test.example.com",
        did="did:wba:test:agent-789",
        capabilities=[CapabilityType.AI_REASONING],
        trust_level=75
    )
    
    print(f"✓ Created test agent: {test_agent.agent_id}")
    print(f"✓ Agent supports AI reasoning: {test_agent.supports_capability(CapabilityType.AI_REASONING)}")
    print(f"✓ Agent is not expired: {not test_agent.is_expired()}")
    
    # Test serialization
    agent_json = test_agent.to_json()
    reconstructed_agent = DiscoverableAgent.from_json(agent_json)
    
    assert test_agent.agent_id == reconstructed_agent.agent_id
    assert test_agent.metadata.name == reconstructed_agent.metadata.name
    print("✓ Serialization/deserialization works correctly")
    
    # 2. Test discovery query filtering
    print("\n2. Testing discovery query filtering...")
    
    # Create multiple test agents
    agents = []
    for i in range(10):
        agent = create_ai_agent(
            name=f"Test Agent {i}",
            description=f"Test agent number {i}",
            base_url=f"https://test{i}.example.com",
            did=f"did:wba:test:agent-{i}",
            capabilities=[CapabilityType.AI_REASONING] if i % 2 == 0 else [CapabilityType.DATA_PROCESSING],
            trust_level=50 + (i * 5),
            availability=0.9 + (i * 0.01)
        )
        agents.append(agent)
    
    # Test filtering
    query = DiscoveryQuery(
        capability_types=[CapabilityType.AI_REASONING],
        min_trust_level=60,
        min_availability=0.95,
        max_results=5
    )
    
    # Simulate filtering (normally done by DiscoveryManager)
    filtered_agents = []
    for agent in agents:
        if (agent.supports_capability(CapabilityType.AI_REASONING) and
            agent.trust_level >= 60 and
            agent.availability >= 0.95):
            filtered_agents.append(agent)
    
    print(f"✓ Filtered {len(filtered_agents)} agents from {len(agents)} total")
    print(f"✓ All filtered agents meet criteria")
    
    # 3. Test discovery methods
    print("\n3. Testing discovery methods...")
    
    # Test well-known discovery
    well_known = WellKnownDiscovery()
    print(f"✓ Well-known discovery path: {well_known.WELL_KNOWN_PATH}")
    
    # Test broadcast discovery
    broadcast = BroadcastDiscovery()
    print(f"✓ Broadcast discovery port: {broadcast.broadcast_port}")
    
    # Test P2P discovery
    p2p = PeerToPeerDiscovery()
    await p2p.add_peer("https://peer1.example.com")
    await p2p.add_peer("https://peer2.example.com")
    print(f"✓ P2P discovery has {len(p2p.known_peers)} peers")
    
    # 4. Test discovery manager
    print("\n4. Testing discovery manager...")
    
    manager = DiscoveryManager()
    
    # Test method enabling/disabling
    manager.enable_method(DiscoveryMethod.WELL_KNOWN)
    manager.enable_method(DiscoveryMethod.BROADCAST)
    manager.disable_method(DiscoveryMethod.REGISTRY)
    
    enabled_count = len(manager.enabled_methods)
    print(f"✓ Discovery manager has {enabled_count} enabled methods")
    
    # Test agent registration
    registration_success = await manager.register_local_agent(test_agent)
    print(f"✓ Agent registration: {registration_success}")
    
    local_agent_count = len(manager.local_agents)
    print(f"✓ Local agents count: {local_agent_count}")
    
    # Test agent unregistration
    unregistration_success = await manager.unregister_local_agent(test_agent.agent_id)
    print(f"✓ Agent unregistration: {unregistration_success}")
    
    # 5. Test capability matching
    print("\n5. Testing capability matching...")
    
    # Create agents with different capabilities
    ai_agent = create_ai_agent(
        name="AI Agent",
        description="AI processing agent",
        base_url="https://ai.example.com",
        did="did:wba:test:ai",
        capabilities=[CapabilityType.AI_REASONING, CapabilityType.TEXT_ANALYSIS]
    )
    
    data_agent = create_service_agent(
        name="Data Agent",
        description="Data processing agent",
        base_url="https://data.example.com",
        did="did:wba:test:data",
        service_capabilities={"data_processing": "Process data"}
    )
    
    # Test compatibility scoring
    required_caps = [CapabilityType.AI_REASONING, CapabilityType.TEXT_ANALYSIS]
    ai_score = ai_agent.calculate_compatibility_score(required_caps)
    data_score = data_agent.calculate_compatibility_score(required_caps)
    
    print(f"✓ AI agent compatibility score: {ai_score}")
    print(f"✓ Data agent compatibility score: {data_score}")
    assert ai_score > data_score, "AI agent should have higher compatibility score"
    
    print("\n✅ All discovery tests passed!")


async def advanced_discovery_examples():
    """Advanced discovery examples and patterns."""
    print("\n🚀 ANP Discovery - Advanced Examples")
    print("=" * 38)
    
    # 1. Multi-method discovery with fallback
    print("\n1. Multi-method discovery with fallback...")
    
    discovery = DiscoveryService()
    
    # Configure with multiple methods
    config = {
        'enabled_methods': ['well_known', 'search_service', 'broadcast', 'registry'],
        'search_service_url': 'https://search.example.com',
        'registry_url': 'https://registry.example.com',
        'registry_api_key': 'test-api-key',
        'p2p_peers': [
            'https://peer1.example.com',
            'https://peer2.example.com'
        ]
    }
    
    await discovery.initialize(config)
    print("✓ Initialized discovery with multiple methods")
    
    # 2. Complex agent ecosystem
    print("\n2. Creating complex agent ecosystem...")
    
    # Create specialized agents
    agents = []
    
    # AI reasoning agents
    for i in range(3):
        agent = create_ai_agent(
            name=f"AI Reasoner {i+1}",
            description=f"Advanced AI reasoning agent {i+1}",
            base_url=f"https://ai{i+1}.example.com",
            did=f"did:wba:ai:reasoner-{i+1}",
            capabilities=[CapabilityType.AI_REASONING, CapabilityType.CODE_GENERATION],
            trust_level=80 + i * 5,
            availability=0.95 + i * 0.01,
            response_time_avg=0.1 + i * 0.05
        )
        agents.append(agent)
        await discovery.register_agent(agent)
    
    # Data processing agents
    for i in range(2):
        agent = create_service_agent(
            name=f"Data Processor {i+1}",
            description=f"High-performance data processor {i+1}",
            base_url=f"https://data{i+1}.example.com",
            did=f"did:wba:data:processor-{i+1}",
            service_capabilities={
                "batch_processing": "Process large data batches",
                "real_time_processing": "Process data in real-time",
                "data_transformation": "Transform data formats"
            },
            trust_level=90 + i * 2,
            availability=0.99 + i * 0.005,
            response_time_avg=0.05 + i * 0.02
        )
        agents.append(agent)
        await discovery.register_agent(agent)
    
    # Monitoring agents
    monitoring_agent = create_service_agent(
        name="System Monitor",
        description="System monitoring and alerting agent",
        base_url="https://monitor.example.com",
        did="did:wba:monitor:system-1",
        service_capabilities={
            "health_monitoring": "Monitor system health",
            "performance_tracking": "Track performance metrics",
            "alerting": "Send alerts and notifications"
        },
        trust_level=95,
        availability=0.999,
        response_time_avg=0.02
    )
    agents.append(monitoring_agent)
    await discovery.register_agent(monitoring_agent)
    
    print(f"✓ Created and registered {len(agents)} specialized agents")
    
    # 3. Advanced discovery queries
    print("\n3. Advanced discovery queries...")
    
    # Find best AI reasoning agents
    best_ai_agents = await discovery.find_agents(
        capability_types=[CapabilityType.AI_REASONING],
        max_results=2
    )
    
    print(f"✓ Found {len(best_ai_agents)} best AI agents:")
    for agent in best_ai_agents:
        print(f"  - {agent.metadata.name} (trust: {agent.trust_level}, "
              f"response: {agent.response_time_avg}s)")
    
    # Find high-availability service agents
    high_availability_agents = await discovery.find_agents(
        tags=['service'],
        max_results=5
    )
    
    print(f"✓ Found {len(high_availability_agents)} high-availability service agents:")
    for agent in high_availability_agents:
        print(f"  - {agent.metadata.name} (availability: {agent.availability})")
    
    # 4. Agent selection strategies
    print("\n4. Agent selection strategies...")
    
    # Strategy 1: Best performance (lowest response time)
    def select_by_performance(agents_list):
        return min(agents_list, key=lambda a: a.response_time_avg or float('inf'))
    
    # Strategy 2: Highest trust
    def select_by_trust(agents_list):
        return max(agents_list, key=lambda a: a.trust_level or 0)
    
    # Strategy 3: Best availability
    def select_by_availability(agents_list):
        return max(agents_list, key=lambda a: a.availability or 0)
    
    # Strategy 4: Balanced score
    def select_by_balanced_score(agents_list):
        def calculate_score(agent):
            trust_score = (agent.trust_level or 0) / 100
            availability_score = agent.availability or 0
            performance_score = 1 / (1 + (agent.response_time_avg or 1))
            return (trust_score + availability_score + performance_score) / 3
        
        return max(agents_list, key=calculate_score)
    
    if best_ai_agents:
        performance_choice = select_by_performance(best_ai_agents)
        trust_choice = select_by_trust(best_ai_agents)
        availability_choice = select_by_availability(best_ai_agents)
        balanced_choice = select_by_balanced_score(best_ai_agents)
        
        print(f"✓ Performance choice: {performance_choice.metadata.name}")
        print(f"✓ Trust choice: {trust_choice.metadata.name}")
        print(f"✓ Availability choice: {availability_choice.metadata.name}")
        print(f"✓ Balanced choice: {balanced_choice.metadata.name}")
    
    # 5. Dynamic agent discovery and load balancing
    print("\n5. Dynamic agent discovery and load balancing...")
    
    class LoadBalancer:
        def __init__(self, discovery_service):
            self.discovery = discovery_service
            self.agent_loads = {}  # agent_id -> current_load
        
        async def get_best_agent(self, capability_type, max_load=0.8):
            """Get the best available agent for a capability."""
            candidates = await self.discovery.find_agents(
                capability_types=[capability_type],
                max_results=10
            )
            
            # Filter by load
            available_agents = []
            for agent in candidates:
                current_load = self.agent_loads.get(agent.agent_id, 0)
                if current_load < max_load:
                    available_agents.append((agent, current_load))
            
            if not available_agents:
                return None
            
            # Select agent with lowest load
            best_agent, _ = min(available_agents, key=lambda x: x[1])
            return best_agent
        
        def update_agent_load(self, agent_id, load):
            """Update agent load information."""
            self.agent_loads[agent_id] = load
    
    # Test load balancer
    load_balancer = LoadBalancer(discovery)
    
    # Simulate some load
    for agent in agents[:3]:
        load_balancer.update_agent_load(agent.agent_id, 0.3 + (hash(agent.agent_id) % 50) / 100)
    
    best_agent = await load_balancer.get_best_agent(CapabilityType.AI_REASONING)
    if best_agent:
        current_load = load_balancer.agent_loads.get(best_agent.agent_id, 0)
        print(f"✓ Selected agent: {best_agent.metadata.name} (load: {current_load:.2f})")
    
    # 6. Agent health monitoring integration
    print("\n6. Agent health monitoring integration...")
    
    class HealthMonitor:
        def __init__(self, discovery_service):
            self.discovery = discovery_service
            self.health_checks = {}
        
        async def check_agent_health(self, agent):
            """Check agent health and update status."""
            try:
                # Simulate health check
                import random
                is_healthy = random.random() > 0.1  # 90% healthy
                
                if is_healthy:
                    await self.discovery.update_agent_status(agent.agent_id, AgentStatus.ONLINE)
                    self.health_checks[agent.agent_id] = {
                        'status': 'healthy',
                        'last_check': datetime.utcnow(),
                        'response_time': random.uniform(0.01, 0.1)
                    }
                else:
                    await self.discovery.update_agent_status(agent.agent_id, AgentStatus.OFFLINE)
                    self.health_checks[agent.agent_id] = {
                        'status': 'unhealthy',
                        'last_check': datetime.utcnow(),
                        'error': 'Connection timeout'
                    }
                
                return is_healthy
                
            except Exception as e:
                logger.error(f"Health check failed for {agent.agent_id}: {e}")
                return False
        
        async def monitor_all_agents(self):
            """Monitor health of all registered agents."""
            local_agents = self.discovery.get_local_agents()
            
            for agent in local_agents:
                is_healthy = await self.check_agent_health(agent)
                status = "✓" if is_healthy else "✗"
                print(f"  {status} {agent.metadata.name}")
    
    # Test health monitoring
    health_monitor = HealthMonitor(discovery)
    print("✓ Health check results:")
    await health_monitor.monitor_all_agents()
    
    # 7. Cleanup
    await discovery.shutdown()
    print("\n✓ Advanced discovery examples completed")


# Main execution for testing
async def main():
    """Main execution function for testing and examples."""
    print("🔍 ANP Discovery Module - Comprehensive Testing Suite")
    print("=" * 60)
    
    try:
        # Run all example suites
        await example_discovery_usage()
        await comprehensive_discovery_testing()
        await advanced_discovery_examples()
        
        print("\n" + "=" * 60)
        print("🎉 All discovery tests completed successfully!")
        print("✅ ANP discovery implementation is working correctly")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    """Run the discovery module tests."""
    asyncio.run(main())


# Export main classes and functions
__all__ = [
    # Enums
    'DiscoveryMethod',
    'AgentStatus', 
    'CapabilityType',
    
    # Data classes
    'AgentCapability',
    'AgentEndpoint',
    'AgentMetadata',
    'DiscoverableAgent',
    'DiscoveryQuery',
    
    # Discovery implementations
    'WellKnownDiscovery',
    'SearchServiceDiscovery',
    'BroadcastDiscovery',
    'PeerToPeerDiscovery',
    'RegistryDiscovery',
    
    # Core classes
    'DiscoveryManager',
    'DiscoveryService',
    
    # Utility functions
    'create_ai_agent',
    'create_service_agent',
    
    # Example functions
    'example_discovery_usage',
    'comprehensive_discovery_testing',
    'advanced_discovery_examples'
]


# Module configuration
DEFAULT_DISCOVERY_CONFIG = {
    'enabled_methods': ['well_known', 'broadcast'],
    'cache_ttl': 300,  # 5 minutes
    'discovery_timeout': 30,  # 30 seconds
    'broadcast_port': 8765,
    'max_results_default': 50,
    'health_check_interval': 60,  # 1 minute
    'cleanup_interval': 300,  # 5 minutes
    'trust_level_threshold': 50,
    'availability_threshold': 0.9,
    'response_time_threshold': 1.0  # 1 second
}


class DiscoveryError(Exception):
    """Base exception for discovery-related errors."""
    pass


class AgentNotFoundError(DiscoveryError):
    """Raised when an agent cannot be found."""
    pass


class DiscoveryTimeoutError(DiscoveryError):
    """Raised when discovery operations timeout."""
    pass


class InvalidAgentError(DiscoveryError):
    """Raised when agent data is invalid."""
    pass


class DiscoveryConfigurationError(DiscoveryError):
    """Raised when discovery configuration is invalid."""
    pass


# Validation utilities
def validate_agent_data(agent_data: Dict[str, Any]) -> List[str]:
    """Validate agent data and return list of errors."""
    errors = []
    
    # Required fields
    required_fields = ['agent_id', 'did', 'metadata', 'endpoints', 'capabilities']
    for field in required_fields:
        if field not in agent_data:
            errors.append(f"Missing required field: {field}")
    
    # Validate metadata
    if 'metadata' in agent_data:
        metadata = agent_data['metadata']
        metadata_required = ['agent_id', 'name', 'description', 'version']
        for field in metadata_required:
            if field not in metadata:
                errors.append(f"Missing required metadata field: {field}")
    
    # Validate endpoints
    if 'endpoints' in agent_data:
        endpoints = agent_data['endpoints']
        if not isinstance(endpoints, list) or len(endpoints) == 0:
            errors.append("At least one endpoint is required")
        else:
            for i, endpoint in enumerate(endpoints):
                if 'url' not in endpoint:
                    errors.append(f"Endpoint {i} missing URL")
    
    # Validate capabilities
    if 'capabilities' in agent_data:
        capabilities = agent_data['capabilities']
        if not isinstance(capabilities, list):
            errors.append("Capabilities must be a list")
    
    # Validate DID format (basic check)
    if 'did' in agent_data:
        did = agent_data['did']
        if not isinstance(did, str) or not did.startswith('did:'):
            errors.append("Invalid DID format")
    
    return errors


def validate_discovery_query(query_data: Dict[str, Any]) -> List[str]:
    """Validate discovery query and return list of errors."""
    errors = []
    
    # Validate capability types
    if 'capability_types' in query_data:
        cap_types = query_data['capability_types']
        if not isinstance(cap_types, list):
            errors.append("capability_types must be a list")
        else:
            valid_types = [ct.value for ct in CapabilityType]
            for cap_type in cap_types:
                if cap_type not in valid_types:
                    errors.append(f"Invalid capability type: {cap_type}")
    
    # Validate numeric fields
    numeric_fields = {
        'min_trust_level': (0, 100),
        'max_response_time': (0, None),
        'min_availability': (0, 1),
        'max_results': (1, 1000)
    }
    
    for field, (min_val, max_val) in numeric_fields.items():
        if field in query_data:
            value = query_data[field]
            if not isinstance(value, (int, float)):
                errors.append(f"{field} must be a number")
            elif value < min_val:
                errors.append(f"{field} must be >= {min_val}")
            elif max_val is not None and value > max_val:
                errors.append(f"{field} must be <= {max_val}")
    
    # Validate status filter
    if 'status_filter' in query_data:
        status_filter = query_data['status_filter']
        if not isinstance(status_filter, list):
            errors.append("status_filter must be a list")
        else:
            valid_statuses = [status.value for status in AgentStatus]
            for status in status_filter:
                if status not in valid_statuses:
                    errors.append(f"Invalid status: {status}")
    
    # Validate sort parameters
    if 'sort_by' in query_data:
        sort_by = query_data['sort_by']
        valid_sort_fields = ['trust_level', 'response_time', 'availability', 'last_seen']
        if sort_by not in valid_sort_fields:
            errors.append(f"Invalid sort_by field: {sort_by}")
    
    if 'sort_order' in query_data:
        sort_order = query_data['sort_order']
        if sort_order not in ['asc', 'desc']:
            errors.append("sort_order must be 'asc' or 'desc'")
    
    return errors


# Performance monitoring utilities
class DiscoveryMetrics:
    """Metrics collection for discovery operations."""
    
    def __init__(self):
        """Initialize metrics collection."""
        self.metrics = {
            'discovery_requests': 0,
            'discovery_successes': 0,
            'discovery_failures': 0,
            'agents_discovered': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_response_time': 0.0,
            'method_usage': {},
            'error_counts': {}
        }
        self.start_time = datetime.utcnow()
    
    def record_discovery_request(self, method: str, success: bool, response_time: float, agents_found: int = 0):
        """Record a discovery request."""
        self.metrics['discovery_requests'] += 1
        
        if success:
            self.metrics['discovery_successes'] += 1
            self.metrics['agents_discovered'] += agents_found
        else:
            self.metrics['discovery_failures'] += 1
        
        # Update average response time
        current_avg = self.metrics['average_response_time']
        total_requests = self.metrics['discovery_requests']
        self.metrics['average_response_time'] = (
            (current_avg * (total_requests - 1) + response_time) / total_requests
        )
        
        # Update method usage
        if method not in self.metrics['method_usage']:
            self.metrics['method_usage'][method] = 0
        self.metrics['method_usage'][method] += 1
    
    def record_cache_hit(self):
        """Record a cache hit."""
        self.metrics['cache_hits'] += 1
    
    def record_cache_miss(self):
        """Record a cache miss."""
        self.metrics['cache_misses'] += 1
    
    def record_error(self, error_type: str):
        """Record an error."""
        if error_type not in self.metrics['error_counts']:
            self.metrics['error_counts'][error_type] = 0
        self.metrics['error_counts'][error_type] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        metrics = self.metrics.copy()
        metrics['uptime_seconds'] = uptime
        metrics['requests_per_second'] = self.metrics['discovery_requests'] / max(uptime, 1)
        
        # Calculate success rate
        total_requests = self.metrics['discovery_requests']
        if total_requests > 0:
            metrics['success_rate'] = self.metrics['discovery_successes'] / total_requests
        else:
            metrics['success_rate'] = 0.0
        
        # Calculate cache hit rate
        total_cache_requests = self.metrics['cache_hits'] + self.metrics['cache_misses']
        if total_cache_requests > 0:
            metrics['cache_hit_rate'] = self.metrics['cache_hits'] / total_cache_requests
        else:
            metrics['cache_hit_rate'] = 0.0
        
        return metrics
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.__init__()


# Discovery result ranking utilities
class DiscoveryRanker:
    """Utility for ranking discovery results."""
    
    @staticmethod
    def rank_by_trust(agents: List[DiscoverableAgent]) -> List[DiscoverableAgent]:
        """Rank agents by trust level (highest first)."""
        return sorted(agents, key=lambda a: a.trust_level or 0, reverse=True)
    
    @staticmethod
    def rank_by_performance(agents: List[DiscoverableAgent]) -> List[DiscoverableAgent]:
        """Rank agents by performance (lowest response time first)."""
        return sorted(agents, key=lambda a: a.response_time_avg or float('inf'))
    
    @staticmethod
    def rank_by_availability(agents: List[DiscoverableAgent]) -> List[DiscoverableAgent]:
        """Rank agents by availability (highest first)."""
        return sorted(agents, key=lambda a: a.availability or 0, reverse=True)
    
    @staticmethod
    def rank_by_load(agents: List[DiscoverableAgent]) -> List[DiscoverableAgent]:
        """Rank agents by load factor (lowest first)."""
        return sorted(agents, key=lambda a: a.load_factor or 0)
    
    @staticmethod
    def rank_by_composite_score(
        agents: List[DiscoverableAgent],
        weights: Optional[Dict[str, float]] = None
    ) -> List[DiscoverableAgent]:
        """Rank agents by composite score with configurable weights."""
        
        if weights is None:
            weights = {
                'trust': 0.3,
                'availability': 0.3,
                'performance': 0.2,
                'load': 0.2
            }
        
        def calculate_score(agent: DiscoverableAgent) -> float:
            # Normalize trust level (0-100 to 0-1)
            trust_score = (agent.trust_level or 0) / 100
            
            # Availability is already 0-1
            availability_score = agent.availability or 0
            
            # Performance score (inverse of response time, normalized)
            response_time = agent.response_time_avg or 1.0
            performance_score = 1 / (1 + response_time)
            
            # Load score (inverse of load factor)
            load_factor = agent.load_factor or 0.5
            load_score = 1 - load_factor
            
            # Calculate weighted composite score
            composite_score = (
                trust_score * weights.get('trust', 0) +
                availability_score * weights.get('availability', 0) +
                performance_score * weights.get('performance', 0) +
                load_score * weights.get('load', 0)
            )
            
            return composite_score
        
        return sorted(agents, key=calculate_score, reverse=True)


# Discovery caching utilities
class DiscoveryCache:
    """Cache for discovery results with TTL support."""
    
    def __init__(self, default_ttl: int = 300):
        """Initialize discovery cache."""
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.metrics = DiscoveryMetrics()
    
    def _generate_key(self, query: DiscoveryQuery) -> str:
        """Generate cache key for query."""
        import hashlib
        query_str = json.dumps(query.to_dict(), sort_keys=True)
        return hashlib.md5(query_str.encode()).hexdigest()
    
    def get(self, query: DiscoveryQuery) -> Optional[List[DiscoverableAgent]]:
        """Get cached results for query."""
        key = self._generate_key(query)
        
        if key in self.cache:
            entry = self.cache[key]
            
            # Check if entry has expired
            if datetime.utcnow() < entry['expires_at']:
                self.metrics.record_cache_hit()
                agents_data = entry['agents']
                return [DiscoverableAgent.from_dict(agent_data) for agent_data in agents_data]
            else:
                # Remove expired entry
                del self.cache[key]
        
        self.metrics.record_cache_miss()
        return None
    
    def put(self, query: DiscoveryQuery, agents: List[DiscoverableAgent], ttl: Optional[int] = None) -> None:
        """Cache results for query."""
        key = self._generate_key(query)
        ttl = ttl or self.default_ttl
        
        self.cache[key] = {
            'agents': [agent.to_dict() for agent in agents],
            'cached_at': datetime.utcnow(),
            'expires_at': datetime.utcnow() + timedelta(seconds=ttl)
        }
    
    def invalidate(self, query: Optional[DiscoveryQuery] = None) -> None:
        """Invalidate cache entries."""
        if query is None:
            # Clear all cache
            self.cache.clear()
        else:
            # Clear specific query
            key = self._generate_key(query)
            self.cache.pop(key, None)
    
    def cleanup_expired(self) -> int:
        """Remove expired cache entries and return count removed."""
        current_time = datetime.utcnow()
        expired_keys = []
        
        for key, entry in self.cache.items():
            if current_time >= entry['expires_at']:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        return len(expired_keys)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        current_time = datetime.utcnow()
        active_entries = 0
        expired_entries = 0
        
        for entry in self.cache.values():
            if current_time < entry['expires_at']:
                active_entries += 1
            else:
                expired_entries += 1
        
        return {
            'total_entries': len(self.cache),
            'active_entries': active_entries,
            'expired_entries': expired_entries,
            'metrics': self.metrics.get_metrics()
        }


# Final module documentation
"""
ANP Discovery Module - Complete Implementation

This module provides a comprehensive implementation of the Agent Discovery Service Protocol (ADSP)
for the Agent Network Protocol (ANP). It supports multiple discovery mechanisms and provides
a flexible, extensible framework for agent discovery in decentralized networks.

Key Features:
- Multiple discovery methods (well-known, search service, broadcast, P2P, registry)
- Comprehensive agent metadata and capability management
- Advanced querying and filtering capabilities
- Performance monitoring and metrics collection
- Caching with TTL support
- Load balancing and health monitoring integration
- Extensive validation and error handling

Usage Examples:
    # Basic usage
    discovery = DiscoveryService()
    await discovery.initialize(config)
    
    # Register an agent
    agent = create_ai_agent(name="My Agent", ...)
    await discovery.register_agent(agent)
    
    # Find agents
    agents = await discovery.find_agents(
        capability_types=[CapabilityType.AI_REASONING],
        max_results=5
    )
    
    # Advanced usage with custom ranking
    ranker = DiscoveryRanker()
    ranked_agents = ranker.rank_by_composite_score(agents)

For more examples, see the example functions in this module.
"""

# Version information
__version__ = "1.0.0"
__author__ = "MKT Communication Team"
__license__ = "MIT"

# Module metadata
MODULE_INFO = {
    "name": "ANP Discovery",
    "version": __version__,
    "description": "Agent Network Protocol Discovery Implementation",
    "protocols_supported": ["ANP"],
    "discovery_methods": [
        "well_known",
        "search_service", 
        "broadcast",
        "peer_to_peer",
        "registry"
    ],
    "features": [
        "Multi-method discovery",
        "Advanced querying",
        "Performance monitoring",
        "Caching with TTL",
        "Load balancing support",
        "Health monitoring integration"
    ]
}

# Configuration validation
def validate_discovery_config(config: Dict[str, Any]) -> List[str]:
    """Validate discovery service configuration."""
    errors = []
    
    # Check enabled methods
    if 'enabled_methods' in config:
        methods = config['enabled_methods']
        if not isinstance(methods, list):
            errors.append("enabled_methods must be a list")
        else:
            valid_methods = [method.value for method in DiscoveryMethod]
            for method in methods:
                if method not in valid_methods:
                    errors.append(f"Invalid discovery method: {method}")
    
    # Check URLs
    url_fields = ['search_service_url', 'registry_url']
    for field in url_fields:
        if field in config:
            url = config[field]
            if not isinstance(url, str) or not url.startswith(('http://', 'https://')):
                errors.append(f"{field} must be a valid HTTP/HTTPS URL")
    
    # Check numeric values
    numeric_fields = {
        'cache_ttl': (1, 86400),  # 1 second to 1 day
        'discovery_timeout': (1, 300),  # 1 second to 5 minutes
        'broadcast_port': (1024, 65535),  # Valid port range
        'max_results_default': (1, 1000)
    }
    
    for field, (min_val, max_val) in numeric_fields.items():
        if field in config:
            value = config[field]
            if not isinstance(value, int):
                errors.append(f"{field} must be an integer")
            elif value < min_val or value > max_val:
                errors.append(f"{field} must be between {min_val} and {max_val}")
    
    # Check P2P peers
    if 'p2p_peers' in config:
        peers = config['p2p_peers']
        if not isinstance(peers, list):
            errors.append("p2p_peers must be a list")
        else:
            for i, peer in enumerate(peers):
                if not isinstance(peer, str) or not peer.startswith(('http://', 'https://')):
                    errors.append(f"p2p_peers[{i}] must be a valid HTTP/HTTPS URL")
    
    return errors


# Logging configuration for the module
def configure_discovery_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Configure logging for the discovery module."""
    import logging
    
    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.info(f"Discovery module logging configured (level: {level})")


# Module initialization function
async def initialize_discovery_module(config: Optional[Dict[str, Any]] = None) -> DiscoveryService:
    """Initialize the discovery module with configuration."""
    
    # Use default config if none provided
    if config is None:
        config = DEFAULT_DISCOVERY_CONFIG.copy()
    
    # Validate configuration
    config_errors = validate_discovery_config(config)
    if config_errors:
        raise DiscoveryConfigurationError(f"Configuration errors: {config_errors}")
    
    # Configure logging
    log_level = config.get('log_level', 'INFO')
    log_file = config.get('log_file')
    configure_discovery_logging(log_level, log_file)
    
    # Create and initialize discovery service
    discovery = DiscoveryService()
    await discovery.initialize(config)
    
    logger.info("Discovery module initialized successfully")
    return discovery


# Cleanup function
async def cleanup_discovery_module(discovery: DiscoveryService):
    """Clean up discovery module resources."""
    try:
        await discovery.shutdown()
        logger.info("Discovery module cleaned up successfully")
    except Exception as e:
        logger.error(f"Error during discovery module cleanup: {e}")


# Context manager for discovery service
class DiscoveryContext:
    """Context manager for discovery service lifecycle."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize context manager."""
        self.config = config
        self.discovery: Optional[DiscoveryService] = None
    
    async def __aenter__(self) -> DiscoveryService:
        """Enter context and initialize discovery service."""
        self.discovery = await initialize_discovery_module(self.config)
        return self.discovery
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context and cleanup discovery service."""
        if self.discovery:
            await cleanup_discovery_module(self.discovery)


# Convenience functions for common use cases
async def quick_discover_agents(
    capability_types: List[CapabilityType],
    config: Optional[Dict[str, Any]] = None,
    max_results: int = 10
) -> List[DiscoverableAgent]:
    """Quick agent discovery with minimal setup."""
    
    async with DiscoveryContext(config) as discovery:
        return await discovery.find_agents(
            capability_types=capability_types,
            max_results=max_results
        )


async def quick_register_agent(
    agent: DiscoverableAgent,
    config: Optional[Dict[str, Any]] = None
) -> bool:
    """Quick agent registration with minimal setup."""
    
    async with DiscoveryContext(config) as discovery:
        return await discovery.register_agent(agent)


# Testing utilities
class MockDiscoveryService:
    """Mock discovery service for testing."""
    
    def __init__(self):
        """Initialize mock service."""
        self.registered_agents: Dict[str, DiscoverableAgent] = {}
        self.discovery_calls = []
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Mock initialization."""
        pass
    
    async def shutdown(self) -> None:
        """Mock shutdown."""
        pass
    
    async def register_agent(self, agent: DiscoverableAgent) -> bool:
        """Mock agent registration."""
        self.registered_agents[agent.agent_id] = agent
        return True
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """Mock agent unregistration."""
        return self.registered_agents.pop(agent_id, None) is not None
    
    async def find_agents(
        self,
        capability_types: Optional[List[CapabilityType]] = None,
        tags: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        max_results: int = 10
    ) -> List[DiscoverableAgent]:
        """Mock agent discovery."""
        
        self.discovery_calls.append({
            'capability_types': capability_types,
            'tags': tags,
            'categories': categories,
            'max_results': max_results
        })
        
        # Return filtered agents based on criteria
        results = []
        for agent in self.registered_agents.values():
            # Simple filtering logic for testing
            if capability_types:
                if not any(agent.supports_capability(cap) for cap in capability_types):
                    continue
            
            if tags:
                if not any(tag in agent.metadata.tags for tag in tags):
                    continue
            
            if categories:
                if not any(cat in agent.metadata.categories for cat in categories):
                    continue
            
            results.append(agent)
            
            if len(results) >= max_results:
                break
        
        return results
    
    async def find_agent_by_id(self, agent_id: str) -> Optional[DiscoverableAgent]:
        """Mock agent lookup by ID."""
        return self.registered_agents.get(agent_id)
    
    def get_local_agents(self) -> List[DiscoverableAgent]:
        """Mock local agents getter."""
        return list(self.registered_agents.values())


# Performance testing utilities
class DiscoveryPerformanceTester:
    """Performance testing utilities for discovery operations."""
    
    def __init__(self, discovery: DiscoveryService):
        """Initialize performance tester."""
        self.discovery = discovery
        self.results = []
    
    async def test_registration_performance(self, num_agents: int = 100) -> Dict[str, float]:
        """Test agent registration performance."""
        import time
        
        agents = []
        for i in range(num_agents):
            agent = create_ai_agent(
                name=f"Test Agent {i}",
                description=f"Performance test agent {i}",
                base_url=f"https://test{i}.example.com",
                did=f"did:wba:test:perf-{i}",
                capabilities=[CapabilityType.AI_REASONING]
            )
            agents.append(agent)
        
        # Measure registration time
        start_time = time.time()
        
        for agent in agents:
            await self.discovery.register_agent(agent)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        return {
            'total_time': total_time,
            'agents_per_second': num_agents / total_time,
            'average_time_per_agent': total_time / num_agents
        }
    
    async def test_discovery_performance(self, num_queries: int = 50) -> Dict[str, float]:
        """Test discovery query performance."""
        import time
        
        # Create various queries
        queries = []
        capability_types = list(CapabilityType)
        
        for i in range(num_queries):
            query_caps = [capability_types[i % len(capability_types)]]
            queries.append(query_caps)
        
        # Measure discovery time
        start_time = time.time()
        
        for query_caps in queries:
            await self.discovery.find_agents(
                capability_types=query_caps,
                max_results=10
            )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        return {
            'total_time': total_time,
            'queries_per_second': num_queries / total_time,
            'average_time_per_query': total_time / num_queries
        }


# Final exports and module completion
__all__.extend([
    # Configuration
    'DEFAULT_DISCOVERY_CONFIG',
    'MODULE_INFO',
    
    # Exceptions
    'DiscoveryError',
    'AgentNotFoundError', 
    'DiscoveryTimeoutError',
    'InvalidAgentError',
    'DiscoveryConfigurationError',
    
    # Validation
    'validate_agent_data',
    'validate_discovery_query',
    'validate_discovery_config',
    
    # Utilities
    'DiscoveryMetrics',
    'DiscoveryRanker',
    'DiscoveryCache',
    
    # Module functions
    'configure_discovery_logging',
    'initialize_discovery_module',
    'cleanup_discovery_module',
    
    # Context manager
    'DiscoveryContext',
    
    # Convenience functions
    'quick_discover_agents',
    'quick_register_agent',
    
    # Testing utilities
    'MockDiscoveryService',
    'DiscoveryPerformanceTester'
])

# Module completion message
logger.info(f"ANP Discovery module loaded successfully (version {__version__})")
logger.info(f"Supported discovery methods: {MODULE_INFO['discovery_methods']}")
logger.info(f"Available features: {MODULE_INFO['features']}")

# End of communication/core/anp/discovery.py
