"""
Django management command to generate agent cards/details for protocols.
"""

import json
import os
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings

from communication.models.a2a import A2AAgentCard
from communication.models.acp import ACPAgentDetail
from communication.models.anp import ANPAgent


class Command(BaseCommand):
    """Management command to generate agent cards and details."""
    
    help = 'Generate agent cards/details for communication protocols'
    
    def add_arguments(self, parser):
        """Add command arguments."""
        parser.add_argument(
            'protocol',
            type=str,
            choices=['a2a', 'acp', 'anp'],
            help='Protocol to generate agent card/detail for'
        )
        
        parser.add_argument(
            '--agent-id',
            type=str,
            help='Agent ID to generate card/detail for'
        )
        
        parser.add_argument(
            '--output',
            type=str,
            help='Output file path (default: stdout)'
        )
        
        parser.add_argument(
            '--format',
            type=str,
            choices=['json', 'yaml'],
            default='json',
            help='Output format (default: json)'
        )
        
        parser.add_argument(
            '--pretty',
            action='store_true',
            help='Pretty print the output'
        )
    
    def handle(self, *args, **options):
        """Handle the command execution."""
        try:
            protocol = options['protocol']
            agent_id = options.get('agent_id')
            
            # Generate the appropriate document
            if protocol == 'a2a':
                document = self._generate_a2a_card(agent_id)
            elif protocol == 'acp':
                document = self._generate_acp_detail(agent_id)
            elif protocol == 'anp':
                document = self._generate_anp_document(agent_id)
            else:
                raise CommandError(f"Unsupported protocol: {protocol}")
            
            # Format the output
            if options['format'] == 'json':
                output = json.dumps(
                    document, 
                    indent=2 if options['pretty'] else None,
                    ensure_ascii=False
                )
            elif options['format'] == 'yaml':
                import yaml
                output = yaml.dump(document, default_flow_style=False)
            
            # Write to file or stdout
            if options['output']:
                with open(options['output'], 'w') as f:
                    f.write(output)
                self.stdout.write(
                    self.style.SUCCESS(f"Generated {protocol} document saved to {options['output']}")
                )
            else:
                self.stdout.write(output)
                
        except Exception as e:
            raise CommandError(f"Error generating agent document: {str(e)}")
    
    def _generate_a2a_card(self, agent_id):
        """Generate A2A agent card."""
        if agent_id:
            try:
                agent_card = A2AAgentCard.objects.get(agent_id=agent_id)
                return agent_card.to_dict()
            except A2AAgentCard.DoesNotExist:
                raise CommandError(f"A2A agent card not found for agent_id: {agent_id}")
        else:
            # List all agent cards
            cards = A2AAgentCard.objects.all()
            if not cards:
                raise CommandError("No A2A agent cards found")
            
            return {
                "agent_cards": [card.to_dict() for card in cards]
            }
    
    def _generate_acp_detail(self, agent_id):
        """Generate ACP agent detail."""
        if agent_id:
            try:
                agent_detail = ACPAgentDetail.objects.get(agent_id=agent_id)
                return agent_detail.to_dict()
            except ACPAgentDetail.DoesNotExist:
                raise CommandError(f"ACP agent detail not found for agent_id: {agent_id}")
        else:
            # List all agent details
            details = ACPAgentDetail.objects.all()
            if not details:
                raise CommandError("No ACP agent details found")
            
            return {
                "agent_details": [detail.to_dict() for detail in details]
            }
    
    def _generate_anp_document(self, agent_id):
        """Generate ANP agent document."""
        if agent_id:
            try:
                anp_agent = ANPAgent.objects.get(agent_id=agent_id)
                return {
                    "agent_info": anp_agent.to_dict(),
                    "did_document": anp_agent.did_document
                }
            except ANPAgent.DoesNotExist:
                raise CommandError(f"ANP agent not found for agent_id: {agent_id}")
        else:
            # List all ANP agents
            agents = ANPAgent.objects.all()
            if not agents:
                raise CommandError("No ANP agents found")
            
            return {
                "anp_agents": [
                    {
                        "agent_info": agent.to_dict(),
                        "did_document": agent.did_document
                    }
                    for agent in agents
                ]
            }
