#!/usr/bin/env python3
"""
HYBRID RAG + LLM AGENT WITH CRITIQUE
Combines codebase patterns with LLM knowledge, then critiques the result

Workflow:
1. Search local patterns (RAG)
2. Fill gaps with LLM knowledge
3. Generate hybrid implementation
4. CRITIQUE the implementation
5. Refine based on critique
6. Final validation
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class CritiqueLevel(Enum):
    """Severity levels for critique feedback"""
    CRITICAL = "critical"    # Must fix
    MAJOR = "major"          # Should fix
    MINOR = "minor"          # Nice to have
    PASSED = "passed"        # Looks good

@dataclass
class CritiqueFeedback:
    """Structured critique feedback"""
    level: CritiqueLevel
    category: str  # functionality, security, performance, style, etc.
    issue: str
    suggestion: str
    line_reference: Optional[str] = None

@dataclass
class CritiqueReport:
    """Complete critique report"""
    goal_alignment_score: float  # 0-1, how well it meets the goal
    completeness_score: float    # 0-1, how complete the implementation is
    quality_score: float          # 0-1, code quality
    feedback_items: List[CritiqueFeedback]
    passes_critique: bool
    summary: str
    suggested_improvements: List[str]

class CritiqueAgent:
    """Agent that critiques generated code"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.min_acceptable_score = 0.7  # Minimum score to pass
    
    def analyze_goal_alignment(self, goal: str, code: str, analysis: Dict) -> Dict:
        """Check if code meets the original goal"""
        
        prompt = f"""Analyze if this code meets the specified goal:

GOAL: {goal}

REQUIRED COMPONENTS (from analysis):
{json.dumps(analysis.get('required_components', []), indent=2)}

TECHNICAL REQUIREMENTS:
{json.dumps(analysis.get('technical_requirements', []), indent=2)}

GENERATED CODE:
{code}

Provide a JSON response:
{{
    "goal_alignment_score": 0.0-1.0,
    "missing_features": ["feature1", ...],
    "implemented_features": ["feature1", ...],
    "unnecessary_additions": ["feature1", ...],
    "alignment_notes": "explanation"
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a code reviewer. Analyze goal alignment."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            
            return json.loads(content)
            
        except Exception as e:
            return {
                "goal_alignment_score": 0.5,
                "missing_features": [],
                "implemented_features": [],
                "unnecessary_additions": [],
                "alignment_notes": f"Analysis failed: {e}"
            }
    
    def analyze_code_quality(self, code: str) -> Dict:
        """Analyze code quality and best practices"""
        
        prompt = f"""Review this code for quality and best practices:

{code}

Provide a JSON response:
{{
    "quality_score": 0.0-1.0,
    "security_issues": [
        {{"severity": "critical/major/minor", "issue": "description", "line": "reference"}}
    ],
    "performance_issues": [
        {{"severity": "critical/major/minor", "issue": "description", "suggestion": "fix"}}
    ],
    "style_issues": [
        {{"severity": "minor", "issue": "description"}}
    ],
    "best_practices_violations": [
        {{"issue": "description", "suggestion": "improvement"}}
    ],
    "positive_aspects": ["good practice 1", ...],
    "error_handling_adequate": true/false
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert code reviewer. Be thorough but fair."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            
            return json.loads(content)
            
        except Exception as e:
            return {
                "quality_score": 0.5,
                "security_issues": [],
                "performance_issues": [],
                "style_issues": [],
                "best_practices_violations": [],
                "positive_aspects": [],
                "error_handling_adequate": False
            }
    
    def check_completeness(self, goal: str, code: str, tests: str) -> Dict:
        """Check if implementation is complete"""
        
        prompt = f"""Check if this implementation is complete:

GOAL: {goal}

CODE:
{code[:2000]}...

TESTS:
{tests[:1000] if tests else "No tests provided"}

Evaluate:
1. Are all main functions implemented?
2. Is error handling present?
3. Are edge cases covered?
4. Is documentation adequate?
5. Are tests comprehensive?

Return JSON:
{{
    "completeness_score": 0.0-1.0,
    "has_main_functionality": true/false,
    "has_error_handling": true/false,
    "has_documentation": true/false,
    "has_tests": true/false,
    "missing_components": ["component1", ...],
    "completeness_notes": "explanation"
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Evaluate code completeness."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            
            return json.loads(content)
            
        except Exception as e:
            return {
                "completeness_score": 0.5,
                "has_main_functionality": True,
                "has_error_handling": False,
                "has_documentation": False,
                "has_tests": bool(tests),
                "missing_components": [],
                "completeness_notes": "Analysis failed"
            }
    
    def generate_critique_report(self, goal: str, code: str, tests: str, analysis: Dict) -> CritiqueReport:
        """Generate comprehensive critique report"""
        print("\n🔍 CRITIQUING IMPLEMENTATION...")
        print("-" * 50)
        
        # 1. Check goal alignment
        print("   📎 Checking goal alignment...")
        alignment = self.analyze_goal_alignment(goal, code, analysis)
        
        # 2. Check code quality
        print("   🎨 Analyzing code quality...")
        quality = self.analyze_code_quality(code)
        
        # 3. Check completeness
        print("   ✅ Verifying completeness...")
        completeness = self.check_completeness(goal, code, tests)
        
        # Build feedback items
        feedback_items = []
        
        # Add critical issues
        for issue in quality.get('security_issues', []):
            if issue['severity'] == 'critical':
                feedback_items.append(CritiqueFeedback(
                    level=CritiqueLevel.CRITICAL,
                    category='security',
                    issue=issue['issue'],
                    suggestion=f"Fix security vulnerability: {issue['issue']}",
                    line_reference=issue.get('line')
                ))
        
        # Add missing features
        for feature in alignment.get('missing_features', []):
            feedback_items.append(CritiqueFeedback(
                level=CritiqueLevel.MAJOR,
                category='functionality',
                issue=f"Missing required feature: {feature}",
                suggestion=f"Implement {feature} as specified in requirements"
            ))
        
        # Add performance issues
        for issue in quality.get('performance_issues', []):
            level = CritiqueLevel.MAJOR if issue['severity'] == 'major' else CritiqueLevel.MINOR
            feedback_items.append(CritiqueFeedback(
                level=level,
                category='performance',
                issue=issue['issue'],
                suggestion=issue.get('suggestion', 'Optimize for better performance')
            ))
        
        # Add best practice violations
        for violation in quality.get('best_practices_violations', [])[:3]:  # Top 3
            feedback_items.append(CritiqueFeedback(
                level=CritiqueLevel.MINOR,
                category='best_practices',
                issue=violation['issue'],
                suggestion=violation.get('suggestion', 'Follow best practices')
            ))
        
        # Calculate overall scores
        goal_score = alignment.get('goal_alignment_score', 0.5)
        quality_score = quality.get('quality_score', 0.5)
        complete_score = completeness.get('completeness_score', 0.5)
        
        # Weighted average
        overall_score = (goal_score * 0.4 + quality_score * 0.3 + complete_score * 0.3)
        
        # Determine if it passes
        has_critical = any(f.level == CritiqueLevel.CRITICAL for f in feedback_items)
        passes = overall_score >= self.min_acceptable_score and not has_critical
        
        # Generate summary
        if passes:
            summary = f"✅ Implementation meets requirements (Score: {overall_score:.1%})"
        else:
            summary = f"⚠️ Implementation needs improvements (Score: {overall_score:.1%})"
        
        # Suggested improvements
        improvements = []
        if not completeness.get('has_error_handling'):
            improvements.append("Add comprehensive error handling")
        if not completeness.get('has_documentation'):
            improvements.append("Add docstrings and comments")
        if alignment.get('missing_features'):
            improvements.append(f"Implement missing features: {', '.join(alignment['missing_features'][:3])}")
        
        return CritiqueReport(
            goal_alignment_score=goal_score,
            completeness_score=complete_score,
            quality_score=quality_score,
            feedback_items=feedback_items,
            passes_critique=passes,
            summary=summary,
            suggested_improvements=improvements
        )
    
    def display_critique_report(self, report: CritiqueReport):
        """Display critique report in readable format"""
        print("\n" + "=" * 60)
        print("📋 CRITIQUE REPORT")
        print("=" * 60)
        
        # Scores
        print("\n📊 SCORES:")
        print(f"   Goal Alignment:  {'🟢' if report.goal_alignment_score >= 0.7 else '🟡' if report.goal_alignment_score >= 0.5 else '🔴'} {report.goal_alignment_score:.1%}")
        print(f"   Completeness:    {'🟢' if report.completeness_score >= 0.7 else '🟡' if report.completeness_score >= 0.5 else '🔴'} {report.completeness_score:.1%}")
        print(f"   Code Quality:    {'🟢' if report.quality_score >= 0.7 else '🟡' if report.quality_score >= 0.5 else '🔴'} {report.quality_score:.1%}")
        
        # Summary
        print(f"\n📝 SUMMARY: {report.summary}")
        
        # Critical issues
        critical_issues = [f for f in report.feedback_items if f.level == CritiqueLevel.CRITICAL]
        if critical_issues:
            print("\n🚨 CRITICAL ISSUES:")
            for issue in critical_issues:
                print(f"   ❌ {issue.category}: {issue.issue}")
                print(f"      → {issue.suggestion}")
        
        # Major issues
        major_issues = [f for f in report.feedback_items if f.level == CritiqueLevel.MAJOR]
        if major_issues:
            print("\n⚠️ MAJOR ISSUES:")
            for issue in major_issues[:5]:  # Show top 5
                print(f"   ⚠️ {issue.category}: {issue.issue}")
                print(f"      → {issue.suggestion}")
        
        # Improvements
        if report.suggested_improvements:
            print("\n💡 SUGGESTED IMPROVEMENTS:")
            for imp in report.suggested_improvements:
                print(f"   • {imp}")
        
        # Decision
        print("\n" + "=" * 60)
        if report.passes_critique:
            print("✅ CODE PASSES CRITIQUE")
        else:
            print("❌ CODE NEEDS REVISION")
        print("=" * 60)
    
    def suggest_fixes(self, code: str, report: CritiqueReport) -> str:
        """Generate specific fixes for critical issues"""
        
        if report.passes_critique:
            return code
        
        critical_and_major = [
            f for f in report.feedback_items 
            if f.level in [CritiqueLevel.CRITICAL, CritiqueLevel.MAJOR]
        ]
        
        if not critical_and_major:
            return code
        
        print("\n🔧 Generating fixes for issues...")
        
        issues_text = "\n".join([
            f"- {f.category}: {f.issue} (Suggestion: {f.suggestion})"
            for f in critical_and_major[:5]
        ])
        
        prompt = f"""Fix these issues in the code:

ISSUES TO FIX:
{issues_text}

CURRENT CODE:
{code}

Generate the fixed code that addresses all these issues.
Return only the complete fixed code."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Fix the code based on critique feedback."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=3000
            )
            
            fixed_code = response.choices[0].message.content
            
            if "```python" in fixed_code:
                fixed_code = fixed_code.split("```python")[1].split("```")[0]
            elif "```" in fixed_code:
                fixed_code = fixed_code.split("```")[1].split("```")[0]
            
            print("   ✅ Generated fixed code")
            return fixed_code.strip()
            
        except Exception as e:
            print(f"   ❌ Failed to generate fixes: {e}")
            return code


class HybridAgentWithCritique:
    """Main hybrid agent with critique capability"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.documents = {}
        self.code_snippets = {}
        self.critique_agent = CritiqueAgent(api_key)
        self.max_refinement_attempts = 2
        
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
    
    def load_documents(self, directory: str):
        """Load documents from directory"""
        print(f"\n📂 Scanning {directory}...")
        
        path = Path(directory)
        files_loaded = 0
        
        for ext in ['.py', '.js', '.java', '.cpp', '.go']:
            for file in path.rglob(f'*{ext}'):
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        self.documents[str(file)] = content
                        self._extract_code_elements(content, str(file))
                        files_loaded += 1
                except:
                    pass
        
        print(f"✅ Loaded {files_loaded} files")
    
    def _extract_code_elements(self, content: str, filename: str):
        """Extract functions and classes"""
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if line.strip().startswith('def ') or line.strip().startswith('class '):
                name = line.split()[1].split('(')[0].split(':')[0]
                snippet = '\n'.join(lines[i:min(i+30, len(lines))])
                self.code_snippets[name] = {
                    'file': filename,
                    'content': snippet
                }
    
    def analyze_goal(self, goal: str) -> Dict:
        """Analyze the goal"""
        prompt = f"""Analyze this goal and identify requirements:

{goal}

Return JSON with:
{{
    "required_components": [...],
    "patterns_needed": [...],
    "technical_requirements": [...],
    "success_criteria": [...]
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Analyze requirements. Return JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            
            return json.loads(content)
        except:
            return {
                "required_components": [],
                "patterns_needed": [],
                "technical_requirements": [],
                "success_criteria": []
            }
    
    def search_local_patterns(self, analysis: Dict) -> List[Dict]:
        """Search local codebase"""
        relevant = []
        terms = analysis.get('required_components', []) + analysis.get('patterns_needed', [])
        
        for term in terms:
            for name, snippet in self.code_snippets.items():
                if term.lower() in name.lower() or term.lower() in snippet['content'].lower():
                    relevant.append({
                        'name': name,
                        'content': snippet['content']
                    })
        
        return relevant[:5]
    
    def generate_implementation(self, goal: str, local_patterns: List[Dict], analysis: Dict) -> Tuple[str, str]:
        """Generate implementation combining local and LLM knowledge"""
        
        local_context = "\n\n".join([
            f"--- {p['name']} ---\n{p['content'][:300]}"
            for p in local_patterns
        ])
        
        prompt = f"""Implement this goal using local patterns where available and best practices:

Goal: {goal}

Requirements: {json.dumps(analysis, indent=2)}

Local patterns found:
{local_context if local_context else "No specific patterns found"}

Generate complete Python implementation:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Generate code combining local patterns and best practices."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
                max_tokens=2500
            )
            
            code = response.choices[0].message.content
            
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0]
            elif "```" in code:
                code = code.split("```")[1].split("```")[0]
            
            # Generate tests
            test_response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Generate comprehensive tests."},
                    {"role": "user", "content": f"Generate pytest tests for:\n\n{code}"}
                ],
                temperature=0.5,
                max_tokens=1500
            )
            
            tests = test_response.choices[0].message.content
            
            if "```python" in tests:
                tests = tests.split("```python")[1].split("```")[0]
            elif "```" in tests:
                tests = tests.split("```")[1].split("```")[0]
            
            return code.strip(), tests.strip()
            
        except Exception as e:
            print(f"Error: {e}")
            return "", ""
    
    def run_with_critique(self, goal: str):
        """Run the complete workflow with critique"""
        print("\n" + "=" * 60)
        print("🚀 HYBRID AGENT WITH CRITIQUE")
        print("=" * 60)
        print(f"\n🎯 Goal: {goal}")
        
        # Step 1: Analyze goal
        print("\n📊 Analyzing requirements...")
        analysis = self.analyze_goal(goal)
        
        # Step 2: Search local patterns
        print("🔍 Searching local codebase...")
        local_patterns = self.search_local_patterns(analysis)
        print(f"   Found {len(local_patterns)} relevant patterns")
        
        # Step 3: Generate initial implementation
        print("\n🤖 Generating implementation...")
        code, tests = self.generate_implementation(goal, local_patterns, analysis)
        
        if not code:
            print("❌ Failed to generate code")
            return None
        
        print(f"   ✅ Generated {len(code.splitlines())} lines of code")
        print(f"   ✅ Generated {len(tests.splitlines())} lines of tests")
        
        # Step 4: Critique the implementation
        refinement_count = 0
        final_code = code
        final_tests = tests
        
        while refinement_count < self.max_refinement_attempts:
            # Generate critique
            report = self.critique_agent.generate_critique_report(
                goal, final_code, final_tests, analysis
            )
            
            # Display critique
            self.critique_agent.display_critique_report(report)
            
            if report.passes_critique:
                print("\n🎉 Implementation passes all checks!")
                break
            
            # Ask user what to do
            print("\n" + "=" * 60)
            print("OPTIONS:")
            print("1. Accept as-is")
            print("2. Auto-fix issues")
            print("3. Regenerate from scratch")
            print("4. Cancel")
            
            choice = input("\nChoice (1-4): ").strip()
            
            if choice == '1':
                break
            elif choice == '2':
                print("\n🔧 Attempting to fix issues...")
                final_code = self.critique_agent.suggest_fixes(final_code, report)
                refinement_count += 1
            elif choice == '3':
                print("\n🔄 Regenerating implementation...")
                final_code, final_tests = self.generate_implementation(goal, local_patterns, analysis)
                refinement_count += 1
            else:
                print("❌ Cancelled")
                return None
        
        # Final output
        print("\n" + "=" * 60)
        print("✅ FINAL IMPLEMENTATION")
        print("=" * 60)
        
        print("\n📝 Code:")
        print("-" * 40)
        print(final_code)
        
        print("\n🧪 Tests:")
        print("-" * 40)
        print(final_tests)
        
        # Save option
        save = input("\n💾 Save? (y/n): ").lower()
        
        if save == 'y':
            base = input("Base filename (default: critiqued): ").strip() or "critiqued"
            
            with open(f"{base}.py", 'w') as f:
                f.write(f'"""Goal: {goal}"""\n\n')
                f.write(final_code)
            
            with open(f"test_{base}.py", 'w') as f:
                f.write(f'"""Tests for: {goal}"""\n\n')
                f.write("import pytest\n")
                f.write(f"from {base} import *\n\n")
                f.write(final_tests)
            
            print(f"✅ Saved to {base}.py and test_{base}.py")
        
        return {
            'goal': goal,
            'code': final_code,
            'tests': final_tests,
            'analysis': analysis
        }


def main():
    """Main function"""
    print("=" * 70)
    print("🤖 HYBRID AGENT WITH CRITIQUE")
    print("   Generate → Critique → Refine → Validate")
    print("=" * 70)
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n❌ Set OPENAI_API_KEY first")
        return
    
    agent = HybridAgentWithCritique(api_key)
    
    # Load codebase
    print("\n📂 Load codebase:")
    print("1. Current directory")
    print("2. Specific directory")
    print("3. Skip")
    
    choice = input("\n>>> ").strip()
    
    if choice == '1':
        agent.load_documents(".")
    elif choice == '2':
        path = input("Path: ").strip()
        if os.path.exists(path):
            agent.load_documents(path)
    
    # Examples
    examples = [
        "Create a user authentication system with JWT tokens",
        "Build a rate limiter with token bucket algorithm",
        "Implement a caching layer for database queries",
        "Create a REST API with input validation",
        "Build a task queue with retry logic"
    ]
    
    print("\n💡 Examples:")
    for i, ex in enumerate(examples, 1):
        print(f"{i}. {ex}")
    
    print("\nEnter number or custom goal (or 'quit'):")
    
    while True:
        choice = input("\n🎯 Goal: ").strip()
        
        if choice.lower() in ['quit', 'q']:
            break
        
        if choice.isdigit() and 1 <= int(choice) <= len(examples):
            goal = examples[int(choice) - 1]
        elif choice:
            goal = choice
        else:
            continue
        
        agent.run_with_critique(goal)

if __name__ == "__main__":
    main()
