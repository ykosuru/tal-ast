#!/usr/bin/env python3
"""
End-to-End Example: Agent-Agnostic Modernization
Shows complete workflow from KG to modernized code
"""

from pathlib import Path
from hybrid_modernizer import HybridModernizer
from agent_adapter import (
    AiderAgent,
    CursorAgent,
    APIAgent,
    CustomAgent,
    AgentOrchestrator
)


def example1_single_procedure():
    """Example 1: Modernize a single procedure"""
    
    print("="*70)
    print("EXAMPLE 1: Single Procedure Modernization")
    print("="*70)
    
    # Step 1: Initialize modernizer with your KG
    modernizer = HybridModernizer(kg_json_path="tal_knowledge_graph.json")
    
    # Step 2: Prepare modernization task (agent-agnostic)
    task = modernizer.prepare_modernization(
        procedure_name="PROCESS_WIRE_TRANSFER",
        target_language="Python"
    )
    
    print(f"\n✅ Task prepared:")
    print(f"   Procedure: {task.procedure_name}")
    print(f"   Source: {task.source_file_path}")
    print(f"   Context: {task.context_file_path}")
    print(f"   Coupling: {task.coupling_score:.2f}")
    print(f"   Warnings: {task.num_warnings}")
    
    # Step 3: Choose your agent and process
    # Option A: Use Aider
    agent = AiderAgent(auto_commit=True)
    orchestrator = AgentOrchestrator(agent)
    result = orchestrator.process_single(task)
    
    print(f"\n📊 Result:")
    print(f"   Success: {result['success']}")
    print(f"   Output: {result.get('output_file')}")
    
    return result


def example2_subsystem_batch():
    """Example 2: Modernize entire subsystem in batch"""
    
    print("="*70)
    print("EXAMPLE 2: Subsystem Batch Modernization")
    print("="*70)
    
    # Step 1: Initialize modernizer
    modernizer = HybridModernizer(kg_json_path="tal_knowledge_graph.json")
    
    # Step 2: Prepare all tasks for subsystem
    subsystem = modernizer.prepare_subsystem(
        entry_procedures=[
            "PROCESS_WIRE_TRANSFER",
            "VALIDATE_WIRE",
            "OFAC_SCREENING"
        ],
        exclude_utilities={
            "LOG_MESSAGE",
            "FORMAT_DATE",
            "GET_TIMESTAMP"
        },
        target_language="Python"
    )
    
    print(f"\n✅ Prepared {len(subsystem['tasks'])} tasks")
    
    # Step 3: Process all tasks with your agent
    agent = AiderAgent()
    orchestrator = AgentOrchestrator(agent)
    
    # Sequential processing
    results = orchestrator.process_batch(subsystem['tasks'])
    
    # Report
    successes = sum(1 for r in results if r['result']['success'])
    print(f"\n📊 Batch Results:")
    print(f"   Total: {len(results)}")
    print(f"   Success: {successes}")
    print(f"   Failed: {len(results) - successes}")
    
    return results


def example3_parallel_processing():
    """Example 3: Parallel processing for speed"""
    
    print("="*70)
    print("EXAMPLE 3: Parallel Modernization")
    print("="*70)
    
    # Prepare tasks
    modernizer = HybridModernizer(kg_json_path="tal_knowledge_graph.json")
    subsystem = modernizer.prepare_subsystem(
        entry_procedures=["MAIN_PROCEDURE"],
        target_language="Python"
    )
    
    print(f"\n✅ Prepared {len(subsystem['tasks'])} tasks")
    
    # Process in parallel (4 workers)
    agent = AiderAgent()
    orchestrator = AgentOrchestrator(agent)
    
    results = orchestrator.process_parallel(
        subsystem['tasks'],
        max_workers=4
    )
    
    print(f"\n📊 Parallel Results:")
    print(f"   Processed: {len(results)} procedures")
    
    return results


def example4_custom_agent():
    """Example 4: Using your own custom agent"""
    
    print("="*70)
    print("EXAMPLE 4: Custom Agent Integration")
    print("="*70)
    
    # Prepare task
    modernizer = HybridModernizer(kg_json_path="tal_knowledge_graph.json")
    task = modernizer.prepare_modernization("PROCESS_WIRE_TRANSFER", "Python")
    
    # Use your custom agent command
    agent = CustomAgent(
        command="my-modernizer --source {source} --context {context} --target {target}"
    )
    
    orchestrator = AgentOrchestrator(agent)
    result = orchestrator.process_single(task)
    
    return result


def example5_export_to_json():
    """Example 5: Export tasks to JSON for external processing"""
    
    print("="*70)
    print("EXAMPLE 5: Export Tasks to JSON")
    print("="*70)
    
    # Prepare tasks
    modernizer = HybridModernizer(kg_json_path="tal_knowledge_graph.json")
    subsystem = modernizer.prepare_subsystem(
        entry_procedures=["PROCESS_WIRE_TRANSFER"],
        target_language="Python"
    )
    
    # Export each task to JSON
    task_dir = Path("./modernization_tasks")
    task_dir.mkdir(exist_ok=True)
    
    for task in subsystem['tasks']:
        output_file = task_dir / f"{task.procedure_name}.json"
        modernizer.export_task_to_json(task, output_file)
    
    print(f"\n✅ Exported {len(subsystem['tasks'])} tasks to {task_dir}")
    print(f"\n   Now you can:")
    print(f"   1. Process tasks with any external system")
    print(f"   2. Queue tasks in a job system")
    print(f"   3. Distribute across multiple machines")
    print(f"   4. Track progress in a database")
    
    return task_dir


def example6_api_based_agent():
    """Example 6: Using an API-based coding agent"""
    
    print("="*70)
    print("EXAMPLE 6: API-Based Agent")
    print("="*70)
    
    # Prepare task
    modernizer = HybridModernizer(kg_json_path="tal_knowledge_graph.json")
    task = modernizer.prepare_modernization("PROCESS_WIRE_TRANSFER", "Python")
    
    # Use API agent
    agent = APIAgent(
        api_endpoint="https://your-modernization-api.com/v1/modernize",
        api_key="your-api-key"
    )
    
    orchestrator = AgentOrchestrator(agent)
    result = orchestrator.process_single(task)
    
    return result


def example7_staged_modernization():
    """Example 7: Staged modernization by coupling score"""
    
    print("="*70)
    print("EXAMPLE 7: Staged Modernization")
    print("="*70)
    
    # Prepare subsystem
    modernizer = HybridModernizer(kg_json_path="tal_knowledge_graph.json")
    subsystem = modernizer.prepare_subsystem(
        entry_procedures=["MAIN_PROCEDURE"],
        target_language="Python"
    )
    
    # Group tasks by coupling
    low_coupling = [t for t in subsystem['tasks'] if t.coupling_score < 0.3]
    med_coupling = [t for t in subsystem['tasks'] if 0.3 <= t.coupling_score < 0.7]
    high_coupling = [t for t in subsystem['tasks'] if t.coupling_score >= 0.7]
    
    print(f"\n📊 Task Distribution:")
    print(f"   Low coupling:    {len(low_coupling)} (easy wins)")
    print(f"   Medium coupling: {len(med_coupling)} (manageable)")
    print(f"   High coupling:   {len(high_coupling)} (complex)")
    
    # Stage 1: Low coupling procedures
    print(f"\n🔄 STAGE 1: Low coupling procedures")
    agent = AiderAgent()
    orchestrator = AgentOrchestrator(agent)
    stage1_results = orchestrator.process_batch(low_coupling)
    
    # Stage 2: Medium coupling
    print(f"\n🔄 STAGE 2: Medium coupling procedures")
    stage2_results = orchestrator.process_batch(med_coupling)
    
    # Stage 3: High coupling
    print(f"\n🔄 STAGE 3: High coupling procedures")
    stage3_results = orchestrator.process_batch(high_coupling)
    
    print(f"\n✅ All stages complete!")
    
    return {
        'stage1': stage1_results,
        'stage2': stage2_results,
        'stage3': stage3_results
    }


def example8_production_pipeline():
    """Example 8: Complete production pipeline"""
    
    print("="*70)
    print("EXAMPLE 8: Production Pipeline")
    print("="*70)
    
    import json
    
    # Configuration
    config = {
        'kg_file': 'tal_knowledge_graph.json',
        'entry_procedures': ['PROCESS_WIRE_TRANSFER', 'VALIDATE_WIRE'],
        'exclude_utilities': {'LOG_MESSAGE', 'FORMAT_DATE', 'GET_TIMESTAMP'},
        'target_language': 'Python',
        'agent': 'aider',
        'parallel_workers': 4,
        'output_dir': './production_output',
        'task_export_dir': './production_tasks'
    }
    
    print(f"\n📋 Configuration:")
    print(json.dumps(config, indent=2))
    
    # Step 1: Prepare modernization
    print(f"\n1️⃣  Preparing modernization tasks...")
    modernizer = HybridModernizer(kg_json_path=config['kg_file'])
    
    subsystem = modernizer.prepare_subsystem(
        entry_procedures=config['entry_procedures'],
        exclude_utilities=set(config['exclude_utilities']),
        target_language=config['target_language'],
        output_dir=Path(config['output_dir']),
        context_dir=Path(config['output_dir']) / 'context'
    )
    
    print(f"   ✓ Prepared {len(subsystem['tasks'])} tasks")
    
    # Step 2: Export tasks for tracking
    print(f"\n2️⃣  Exporting tasks to JSON...")
    task_dir = Path(config['task_export_dir'])
    task_dir.mkdir(exist_ok=True, parents=True)
    
    for task in subsystem['tasks']:
        output_file = task_dir / f"{task.procedure_name}.json"
        modernizer.export_task_to_json(task, output_file)
    
    print(f"   ✓ Exported to {task_dir}")
    
    # Step 3: Process with agent
    print(f"\n3️⃣  Processing with {config['agent']}...")
    
    if config['agent'] == 'aider':
        agent = AiderAgent(auto_commit=True)
    elif config['agent'] == 'cursor':
        agent = CursorAgent()
    else:
        agent = CustomAgent(command=config['agent'])
    
    orchestrator = AgentOrchestrator(agent)
    
    # Use parallel or sequential based on config
    if config['parallel_workers'] > 1:
        results = orchestrator.process_parallel(
            subsystem['tasks'],
            max_workers=config['parallel_workers']
        )
    else:
        results = orchestrator.process_batch(subsystem['tasks'])
    
    # Step 4: Generate report
    print(f"\n4️⃣  Generating report...")
    
    report = {
        'config': config,
        'summary': {
            'total_tasks': len(subsystem['tasks']),
            'successful': sum(1 for r in results if r['result']['success']),
            'failed': sum(1 for r in results if not r['result']['success'])
        },
        'results': results
    }
    
    report_file = Path(config['output_dir']) / 'modernization_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"   ✓ Report saved to {report_file}")
    
    # Step 5: Summary
    print(f"\n{'='*70}")
    print("PRODUCTION PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"  Total tasks: {report['summary']['total_tasks']}")
    print(f"  Successful:  {report['summary']['successful']}")
    print(f"  Failed:      {report['summary']['failed']}")
    print(f"  Output dir:  {config['output_dir']}")
    print(f"{'='*70}\n")
    
    return report


def main():
    """Run examples"""
    
    print("""
Agent-Agnostic Modernization System - Examples

Choose an example:
    1. Single procedure modernization
    2. Subsystem batch processing
    3. Parallel processing
    4. Custom agent
    5. Export to JSON
    6. API-based agent
    7. Staged modernization
    8. Production pipeline

All examples are agent-agnostic - you can swap agents without changing code!
""")
    
    # Uncomment to run specific examples:
    # example1_single_procedure()
    # example2_subsystem_batch()
    # example3_parallel_processing()
    # example4_custom_agent()
    # example5_export_to_json()
    # example6_api_based_agent()
    # example7_staged_modernization()
    # example8_production_pipeline()


if __name__ == "__main__":
    main()
