"""
Django management command to set up protocol configurations.
"""

import json
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings

from communication.config import get_config
from communication.models.a2a import A2AAgentCard
from communication.models.acp import ACPAgentDetail
from communication.models.anp import ANPAgent


class Command(BaseCommand):
    """Management command to set up protocol configurations."""
    
    help = 'Set up protocol configurations for A2A, ACP, and ANP'
    
    def add_arguments(self, parser):
        """Add command arguments."""
        parser.add_argument(
            '--protocol',
            type=str,
            choices=['a2a', 'acp', 'anp', 'all'],
            default='all',
            help='Protocol to set up (default: all)'
        )
        
        parser.add_argument(
            '--agent-id',
            type=str,
            help='Agent ID for the local agent'
        )
        
        parser.add_argument(
            '--agent-name',
            type=str,
            help='Agent name for the local agent'
        )
        
        parser.add_argument(
            '--host',
            type=str,
            default='localhost',
            help='Host for the agent endpoints'
        )
        
        parser.add_argument(
            '--port-a2a',
            type=int,
            default=8080,
            help='Port for A2A protocol'
        )
        
        parser.add_argument(
            '--port-acp',
            type=int,
            default=8081,
            help='Port for ACP protocol'
        )
        
        parser.add_argument(
            '--port-anp',
            type=int,
            default=8082,
            help='Port for ANP protocol'
        )
        
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force overwrite existing configurations'
        )
        
        parser.add_argument(
            '--config-file',
            type=str,
            help='JSON file with protocol configurations'
        )
    
    def handle(self, *args, **options):
        """Handle the command execution."""
        try:
            config = get_config()
            
            # Load configuration from file if provided
            if options['config_file']:
                self._load_config_from_file(options['config_file'], options)
            
            # Set default values
            agent_id = options.get('agent_id') or f"mkt-agent-{settings.SECRET_KEY[:8]}"
            agent_name = options.get('agent_name') or "MKT Communication Agent"
            host = options['host']
            
            # Set up protocols
            if options['protocol'] in ['a2a', 'all']:
                self._setup_a2a(agent_id, agent_name, host, options['port_a2a'], options['force'])
            
            if options['protocol'] in ['acp', 'all']:
                self._setup_acp(agent_id, agent_name, host, options['port_acp'], options['force'])
            
            if options['protocol'] in ['anp', 'all']:
                self._setup_anp(agent_id, agent_name, host, options['port_anp'], options['force'])
            
            self.stdout.write(
                self.style.SUCCESS(f"Successfully set up {options['protocol']} protocol(s)")
            )
            
        except Exception as e:
            raise CommandError(f"Error setting up protocols: {str(e)}")
    
    def _load_config_from_file(self, config_file, options):
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update options with config file data
            for key, value in config_data.items():
                if key not in options or options[key] is None:
                    options[key] = value
                    
        except FileNotFoundError:
            raise CommandError(f"Configuration file not found: {config_file}")
        except json.JSONDecodeError as e:
            raise CommandError(f"Invalid JSON in configuration file: {e}")
    
    def _setup_a2a(self, agent_id, agent_name, host, port, force):
        """Set up A2A protocol configuration."""
        self.stdout.write("Setting up A2A protocol...")
        
        # Check if agent card already exists
        existing_card = A2AAgentCard.objects.filter(agent_id=agent_id).first()
        if existing_card and not force:
            self.stdout.write(
                self.style.WARNING(f"A2A agent card already exists for {agent_id}. Use --force to overwrite.")
            )
            return
        
        # Create or update agent card
        agent_card_data = {
            'agent_id': agent_id,
            'name': agent_name,
            'description': f"MKT Communication Agent - A2A Protocol",
            'version': '1.0.0',
            'endpoints': [
                {
                    'name': 'tasks',
                    'url': f"http://{host}:{port}/tasks",
                    'methods': ['POST', 'GET']
                },
                {
                    'name': 'messages',
                    'url': f"http://{host}:{port}/messages",
                    'methods': ['POST', 'GET']
                },
                {
                    'name': 'stream',
                    'url': f"http://{host}:{port}/stream",
                    'methods': ['GET']
                }
            ],
            'capabilities': [
                {
                    'name': 'text_processing',
                    'description': 'Process text messages',
                    'input_types': ['text/plain', 'application/json'],
                    'output_types': ['text/plain', 'application/json']
                },
                {
                    'name': 'task_management',
                    'description': 'Manage tasks and workflows',
                    'input_types': ['application/json'],
                    'output_types': ['application/json']
                }
            ],
            'authentication': {
                'type': 'bearer',
                'required': True
            },
            'rate_limits': {
                'requests_per_minute': 100,
                'concurrent_tasks': 10
            }
        }
        
        if existing_card:
            # Update existing card
            for key, value in agent_card_data.items():
                setattr(existing_card, key, value)
            existing_card.save()
            self.stdout.write(f"Updated A2A agent card for {agent_id}")
        else:
            # Create new card
            A2AAgentCard.objects.create(**agent_card_data)
            self.stdout.write(f"Created A2A agent card for {agent_id}")
    
    def _setup_acp(self, agent_id, agent_name, host, port, force):
        """Set up ACP protocol configuration."""
        self.stdout.write("Setting up ACP protocol...")
        
        # Check if agent detail already exists
        existing_detail = ACPAgentDetail.objects.filter(agent_id=agent_id).first()
        if existing_detail and not force:
            self.stdout.write(
                self.style.WARNING(f"ACP agent detail already exists for {agent_id}. Use --force to overwrite.")
            )
            return
        
        # Create or update agent detail
        agent_detail_data = {
            'agent_id': agent_id,
            'name': agent_name,
            'description': f"MKT Communication Agent - ACP Protocol",
            'version': '1.0.0',
            'base_url': f"http://{host}:{port}",
            'capabilities': [
                {
                    'name': 'text_processing',
                    'description': 'Process text messages and generate responses',
                    'input_schema': {
                        'type': 'object',
                        'properties': {
                            'text': {'type': 'string'},
                            'context': {'type': 'object'}
                        },
                        'required': ['text']
                    },
                    'output_schema': {
                        'type': 'object',
                        'properties': {
                            'response': {'type': 'string'},
                            'metadata': {'type': 'object'}
                        }
                    }
                },
                {
                    'name': 'data_analysis',
                    'description': 'Analyze structured data',
                    'input_schema': {
                        'type': 'object',
                        'properties': {
                            'data': {'type': 'array'},
                            'analysis_type': {'type': 'string'}
                        },
                        'required': ['data']
                    },
                    'output_schema': {
                        'type': 'object',
                        'properties': {
                            'results': {'type': 'object'},
                            'insights': {'type': 'array'}
                        }
                    }
                }
            ],
            'authentication_required': True,
            'max_run_duration': 3600,
            'supported_message_types': ['text', 'json', 'file']
        }
        
        if existing_detail:
            # Update existing detail
            for key, value in agent_detail_data.items():
                setattr(existing_detail, key, value)
            existing_detail.save()
            self.stdout.write(f"Updated ACP agent detail for {agent_id}")
        else:
            # Create new detail
            ACPAgentDetail.objects.create(**agent_detail_data)
            self.stdout.write(f"Created ACP agent detail for {agent_id}")
    
    def _setup_anp(self, agent_id, agent_name, host, port, force):
        """Set up ANP protocol configuration."""
        self.stdout.write("Setting up ANP protocol...")
        
        # Check if ANP agent already exists
        existing_agent = ANPAgent.objects.filter(agent_id=agent_id).first()
        if existing_agent and not force:
            self.stdout.write(
                self.style.WARNING(f"ANP agent already exists for {agent_id}. Use --force to overwrite.")
            )
            return
        
        # Generate DID and keys for ANP
        from communication.core.anp.did import DIDManager
        did_manager = DIDManager()
        did_document = did_manager.create_did(agent_id)
        
        # Create or update ANP agent
        anp_agent_data = {
            'agent_id': agent_id,
            'name': agent_name,
            'description': f"MKT Communication Agent - ANP Protocol",
            'did': did_document['id'],
            'did_document': did_document,
            'base_url': f"http://{host}:{port}",
            'supported_protocols': ['a2a', 'acp'],
            'discovery_endpoints': [
                f"http://{host}:{port}/.well-known/agent",
                f"http://{host}:{port}/discovery"
            ],
            'encryption_enabled': True,
            'public_key': did_document['verificationMethod'][0]['publicKeyJwk']
        }
        
        if existing_agent:
            # Update existing agent
            for key, value in anp_agent_data.items():
                setattr(existing_agent, key, value)
            existing_agent.save()
            self.stdout.write(f"Updated ANP agent for {agent_id}")
        else:
            # Create new agent
            ANPAgent.objects.create(**anp_agent_data)
            self.stdout.write(f"Created ANP agent for {agent_id}")
        
        # Save DID document to file for reference
        did_file = f"anp_did_{agent_id}.json"
        with open(did_file, 'w') as f:
            json.dump(did_document, f, indent=2)
        self.stdout.write(f"DID document saved to {did_file}")
