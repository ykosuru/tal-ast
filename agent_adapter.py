#!/usr/bin/env python3
"""
Agent Adapter Interface
Shows how to integrate the agent-agnostic hybrid modernizer with ANY coding agent

Build your own adapter for: Aider, Cursor, OpenHands, Claude Code, or custom agents
"""

from pathlib import Path
from typing import Dict, Any, Protocol
from abc import ABC, abstractmethod
import subprocess
import json


class CodingAgent(Protocol):
    """
    Protocol (interface) that any coding agent adapter must implement
    """
    
    def process_task(self, task: Any) -> Dict[str, Any]:
        """
        Process a modernization task
        
        Args:
            task: ModernizationTask from HybridModernizer
            
        Returns:
            Dict with:
                - success: bool
                - output_file: str (path to generated code)
                - agent_response: Any (agent-specific output)
        """
        ...


class BaseAgent(ABC):
    """Base class for agent adapters"""
    
    @abstractmethod
    def process_task(self, task: Any) -> Dict[str, Any]:
        """Process a modernization task"""
        pass
    
    def extract_task_info(self, task: Any) -> Dict[str, str]:
        """Extract common info from task"""
        return {
            'procedure_name': task.procedure_name,
            'source_file': task.source_file_path,
            'context_file': task.context_file_path,
            'prompt': task.short_prompt,
            'target_language': task.target_language
        }


# ============================================================================
# Example Adapter 1: Aider
# ============================================================================

class AiderAgent(BaseAgent):
    """Adapter for Aider coding agent"""
    
    def __init__(self, auto_commit: bool = False):
        self.auto_commit = auto_commit
    
    def process_task(self, task: Any) -> Dict[str, Any]:
        """Process task with Aider"""
        info = self.extract_task_info(task)
        
        cmd = [
            "aider",
            "--message", info['prompt'],
            info['source_file'],
            info['context_file']
        ]
        
        if self.auto_commit:
            cmd.insert(1, "--yes")
        
        print(f"🤖 Running Aider for {info['procedure_name']}...")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Determine output file (Aider modifies source_file in place)
            output_file = Path(info['source_file']).with_suffix(f".{info['target_language'].lower()}")
            
            return {
                'success': result.returncode == 0,
                'output_file': str(output_file),
                'agent_response': {
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'returncode': result.returncode
                }
            }
        
        except FileNotFoundError:
            return {
                'success': False,
                'error': 'Aider not installed. Install with: pip install aider-chat',
                'manual_command': ' '.join(cmd)
            }
        
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Aider timed out after 5 minutes'
            }


# ============================================================================
# Example Adapter 2: Cursor (Manual/Interactive)
# ============================================================================

class CursorAgent(BaseAgent):
    """Adapter for Cursor IDE (manual interaction)"""
    
    def process_task(self, task: Any) -> Dict[str, Any]:
        """Provide instructions for Cursor"""
        info = self.extract_task_info(task)
        
        print(f"\n📝 Cursor Instructions for {info['procedure_name']}:")
        print(f"   1. Open: {info['source_file']}")
        print(f"   2. Open: {info['context_file']}")
        print(f"   3. In Cursor, ask:")
        print(f"      'Modernize this to {info['target_language']} following context.md'")
        
        output_file = Path(info['source_file']).with_suffix(f".{info['target_language'].lower()}")
        
        return {
            'success': True,
            'output_file': str(output_file),
            'agent_response': {
                'instructions': [
                    f"Open {info['source_file']}",
                    f"Open {info['context_file']}",
                    f"Ask Cursor to modernize to {info['target_language']}"
                ]
            }
        }


# ============================================================================
# Example Adapter 3: API-based Agent (Claude, OpenAI, etc.)
# ============================================================================

class APIAgent(BaseAgent):
    """Generic adapter for API-based coding agents"""
    
    def __init__(self, api_endpoint: str, api_key: str = None):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
    
    def process_task(self, task: Any) -> Dict[str, Any]:
        """Process task via API"""
        info = self.extract_task_info(task)
        
        # Build API request
        request_payload = {
            'source_code': task.source_code,
            'context': task.context_markdown,
            'prompt': info['prompt'],
            'target_language': info['target_language'],
            'metadata': {
                'procedure_name': info['procedure_name'],
                'coupling_score': task.coupling_score,
                'call_depth': task.call_depth
            }
        }
        
        print(f"🌐 Calling API for {info['procedure_name']}...")
        
        # Example: Make API request (you'd implement actual API call)
        # response = requests.post(
        #     self.api_endpoint,
        #     json=request_payload,
        #     headers={'Authorization': f'Bearer {self.api_key}'}
        # )
        
        # For demo purposes, show what would be sent
        print(f"   API Endpoint: {self.api_endpoint}")
        print(f"   Payload size: {len(json.dumps(request_payload))} bytes")
        
        # Simulate response
        output_file = Path(info['source_file']).with_suffix(f".{info['target_language'].lower()}")
        
        return {
            'success': True,
            'output_file': str(output_file),
            'agent_response': {
                'api_endpoint': self.api_endpoint,
                'payload': request_payload
            }
        }


# ============================================================================
# Example Adapter 4: Custom Local Agent
# ============================================================================

class CustomAgent(BaseAgent):
    """Adapter for custom/local coding agent"""
    
    def __init__(self, command: str):
        """
        Args:
            command: Command template with {source}, {context}, {prompt} placeholders
                    Example: "my-agent --source {source} --context {context} --prompt '{prompt}'"
        """
        self.command_template = command
    
    def process_task(self, task: Any) -> Dict[str, Any]:
        """Process task with custom agent"""
        info = self.extract_task_info(task)
        
        # Format command
        cmd = self.command_template.format(
            source=info['source_file'],
            context=info['context_file'],
            prompt=info['prompt'],
            target=info['target_language']
        )
        
        print(f"⚙️  Running custom agent for {info['procedure_name']}...")
        print(f"   Command: {cmd}")
        
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            output_file = Path(info['source_file']).with_suffix(f".{info['target_language'].lower()}")
            
            return {
                'success': result.returncode == 0,
                'output_file': str(output_file),
                'agent_response': {
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'command': cmd
                }
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


# ============================================================================
# Agent Orchestrator
# ============================================================================

class AgentOrchestrator:
    """
    Orchestrates modernization tasks across multiple agents
    Supports parallel processing, retries, and agent selection
    """
    
    def __init__(self, agent: BaseAgent):
        self.agent = agent
    
    def process_single(self, task: Any) -> Dict[str, Any]:
        """Process single modernization task"""
        return self.agent.process_task(task)
    
    def process_batch(self, tasks: list) -> list:
        """Process multiple tasks sequentially"""
        results = []
        
        for i, task in enumerate(tasks, 1):
            print(f"\n{'='*70}")
            print(f"Processing {i}/{len(tasks)}: {task.procedure_name}")
            print(f"{'='*70}")
            
            result = self.agent.process_task(task)
            results.append({
                'task': task.procedure_name,
                'result': result
            })
        
        return results
    
    def process_parallel(self, tasks: list, max_workers: int = 4) -> list:
        """Process multiple tasks in parallel"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(self.agent.process_task, task): task 
                for task in tasks
            }
            
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append({
                        'task': task.procedure_name,
                        'result': result
                    })
                except Exception as e:
                    results.append({
                        'task': task.procedure_name,
                        'result': {
                            'success': False,
                            'error': str(e)
                        }
                    })
        
        return results


# ============================================================================
# Example Usage
# ============================================================================

def main():
    """Example: How to use the agent-agnostic system"""
    
    print("""
Agent Adapter Examples
Shows how to integrate ANY coding agent with the hybrid modernizer

Example 1: Using Aider
    from hybrid_modernizer import HybridModernizer
    from agent_adapter import AiderAgent, AgentOrchestrator
    
    # Prepare modernization task
    modernizer = HybridModernizer(kg_json_path="tal_kg.json")
    task = modernizer.prepare_modernization("PROCESS_WIRE_TRANSFER", "Python")
    
    # Process with Aider
    agent = AiderAgent(auto_commit=True)
    orchestrator = AgentOrchestrator(agent)
    result = orchestrator.process_single(task)

Example 2: Using Cursor (manual)
    agent = CursorAgent()
    orchestrator = AgentOrchestrator(agent)
    result = orchestrator.process_single(task)

Example 3: Using API-based agent
    agent = APIAgent(
        api_endpoint="https://your-api.com/modernize",
        api_key="your-key"
    )
    orchestrator = AgentOrchestrator(agent)
    result = orchestrator.process_single(task)

Example 4: Batch processing entire subsystem
    modernizer = HybridModernizer(kg_json_path="tal_kg.json")
    result = modernizer.prepare_subsystem(
        entry_procedures=["PROCESS_WIRE_TRANSFER"],
        exclude_utilities={"LOG_MESSAGE"}
    )
    
    # Process all tasks with your agent
    agent = AiderAgent()
    orchestrator = AgentOrchestrator(agent)
    results = orchestrator.process_batch(result['tasks'])
    
    # Or process in parallel
    results = orchestrator.process_parallel(result['tasks'], max_workers=4)

Example 5: Custom agent
    agent = CustomAgent(
        command="my-agent --input {source} --context {context}"
    )
    orchestrator = AgentOrchestrator(agent)
    result = orchestrator.process_single(task)

Key Benefits:
- Agent-agnostic: Switch agents without changing modernization logic
- Scalable: Parallel processing for large codebases
- Extensible: Easy to add new agent adapters
- Testable: Can mock agents for testing
""")


if __name__ == "__main__":
    main()
