#!/usr/bin/env python3
"""
Visualize Decision Rules from Trained ACE ML Model

This script extracts and visualizes the decision rules learned by the 
RandomForest model for predicting ACE error/repair codes.

Usage:
    python visualize_rules.py --model-dir ./models_9x --output rules_9x.html
    python visualize_rules.py --model-dir ./models_8x --output rules_8x.html --top 10
"""

import argparse
import json
import pickle
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import numpy as np

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from sklearn.tree import export_text, plot_tree
    HAS_SKLEARN_VIZ = True
except ImportError:
    HAS_SKLEARN_VIZ = False


def load_model(model_dir: str) -> Tuple[Any, List[str], List[str]]:
    """Load model and metadata."""
    model_path = Path(model_dir)
    
    with open(model_path / 'model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open(model_path / 'training_info.json', 'r') as f:
        info = json.load(f)
    
    feature_names = info.get('feature_names', [])
    class_names = info.get('class_names', [])
    
    return model, feature_names, class_names


def extract_tree_rules(tree, feature_names: List[str], class_name: str, 
                       max_depth: int = 5) -> List[Dict]:
    """Extract decision rules from a single decision tree."""
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != -2 else "undefined!"
        for i in tree_.feature
    ]
    
    rules = []
    
    def recurse(node, path, depth):
        if depth > max_depth:
            return
            
        if tree_.feature[node] != -2:  # Not a leaf
            name = feature_name[node]
            threshold = tree_.threshold[node]
            
            # Left branch (feature <= threshold)
            left_path = path + [(name, '<=', threshold)]
            recurse(tree_.children_left[node], left_path, depth + 1)
            
            # Right branch (feature > threshold)
            right_path = path + [(name, '>', threshold)]
            recurse(tree_.children_right[node], right_path, depth + 1)
        else:
            # Leaf node - check if it predicts positive
            values = tree_.value[node][0]
            if len(values) >= 2 and values[1] > values[0]:  # Positive class wins
                confidence = values[1] / sum(values)
                samples = int(sum(values))
                if confidence > 0.5 and samples >= 3:  # Meaningful rule
                    rules.append({
                        'conditions': path,
                        'confidence': confidence,
                        'samples': samples,
                        'class': class_name
                    })
    
    recurse(0, [], 0)
    return rules


def simplify_condition(cond: Tuple[str, str, float]) -> str:
    """Convert condition tuple to readable string."""
    feature, op, threshold = cond
    
    # Boolean features (threshold ~0.5 means True/False)
    if threshold > 0.4 and threshold < 0.6:
        if op == '<=':
            return f"{feature} = False"
        else:
            return f"{feature} = True"
    
    # Numeric features
    return f"{feature} {op} {threshold:.2f}"


def get_feature_importance_per_class(model, feature_names: List[str], 
                                     class_names: List[str]) -> Dict[str, List[Tuple[str, float]]]:
    """Get feature importance for each class."""
    importance_by_class = {}
    
    # Check if it's a multi-output model
    if hasattr(model, 'estimators_'):
        for idx, (estimator, class_name) in enumerate(zip(model.estimators_, class_names)):
            if hasattr(estimator, 'feature_importances_'):
                importances = estimator.feature_importances_
                # Get top features
                top_indices = np.argsort(importances)[-15:][::-1]
                top_features = [(feature_names[i], importances[i]) 
                               for i in top_indices if importances[i] > 0.01]
                importance_by_class[class_name] = top_features
    
    return importance_by_class


def extract_rules_for_class(model, class_idx: int, feature_names: List[str],
                            class_name: str, max_rules: int = 10) -> List[Dict]:
    """Extract top rules for a specific class."""
    all_rules = []
    
    if hasattr(model, 'estimators_'):
        estimator = model.estimators_[class_idx]
        
        # For RandomForest, sample from trees
        if hasattr(estimator, 'estimators_'):
            trees = estimator.estimators_[:10]  # Sample first 10 trees
            for tree in trees:
                rules = extract_tree_rules(tree, feature_names, class_name)
                all_rules.extend(rules)
        else:
            # Single tree
            rules = extract_tree_rules(estimator, feature_names, class_name)
            all_rules.extend(rules)
    
    # Deduplicate and sort by confidence * samples
    seen = set()
    unique_rules = []
    for rule in all_rules:
        key = tuple(sorted(str(c) for c in rule['conditions']))
        if key not in seen:
            seen.add(key)
            rule['score'] = rule['confidence'] * np.log1p(rule['samples'])
            unique_rules.append(rule)
    
    unique_rules.sort(key=lambda x: x['score'], reverse=True)
    return unique_rules[:max_rules]


def generate_text_report(model, feature_names: List[str], class_names: List[str],
                        top_classes: int = 20, top_rules: int = 5) -> str:
    """Generate text report of decision rules."""
    lines = []
    lines.append("=" * 80)
    lines.append("ACE ML MODEL - DECISION RULES VISUALIZATION")
    lines.append("=" * 80)
    lines.append("")
    
    # Get feature importance per class
    importance_by_class = get_feature_importance_per_class(model, feature_names, class_names)
    
    # Sort classes by something meaningful (e.g., code number)
    sorted_classes = sorted(enumerate(class_names), key=lambda x: x[1])
    
    for class_idx, class_name in sorted_classes[:top_classes]:
        lines.append("")
        lines.append("-" * 80)
        lines.append(f"CODE: {class_name}")
        lines.append("-" * 80)
        
        # Feature importance
        if class_name in importance_by_class:
            lines.append("")
            lines.append("Top Predictive Features:")
            for feat, imp in importance_by_class[class_name][:10]:
                bar = "█" * int(imp * 50)
                lines.append(f"  {feat:<45} {imp:.3f} {bar}")
        
        # Decision rules
        rules = extract_rules_for_class(model, class_idx, feature_names, class_name, top_rules)
        
        if rules:
            lines.append("")
            lines.append("Decision Rules:")
            for i, rule in enumerate(rules, 1):
                lines.append(f"")
                lines.append(f"  Rule {i} (confidence: {rule['confidence']:.1%}, samples: {rule['samples']})")
                lines.append(f"    IF:")
                for cond in rule['conditions']:
                    simplified = simplify_condition(cond)
                    lines.append(f"      {simplified}")
                lines.append(f"    THEN: Predict {class_name}")
        else:
            lines.append("")
            lines.append("  (No strong decision rules extracted)")
    
    return "\n".join(lines)


def generate_html_report(model, feature_names: List[str], class_names: List[str],
                         top_classes: int = 30, top_rules: int = 5) -> str:
    """Generate HTML report with visualizations."""
    
    importance_by_class = get_feature_importance_per_class(model, feature_names, class_names)
    
    html = """<!DOCTYPE html>
<html>
<head>
    <title>ACE ML Model - Decision Rules</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
        h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
        h2 { color: #4CAF50; margin-top: 30px; }
        .code-card { background: white; border-radius: 8px; padding: 20px; margin: 20px 0;
                     box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .code-name { font-size: 1.4em; font-weight: bold; color: #333; }
        .code-8xxx { border-left: 4px solid #f44336; }
        .code-9xxx { border-left: 4px solid #2196F3; }
        .feature-bar { background: #e0e0e0; border-radius: 4px; height: 20px; margin: 5px 0; }
        .feature-fill { background: #4CAF50; height: 100%; border-radius: 4px; }
        .feature-row { display: flex; align-items: center; margin: 5px 0; }
        .feature-name { width: 350px; font-family: monospace; font-size: 0.9em; }
        .feature-value { width: 60px; text-align: right; font-size: 0.9em; color: #666; }
        .rule-box { background: #f9f9f9; border-radius: 4px; padding: 15px; margin: 10px 0;
                    border: 1px solid #ddd; }
        .rule-header { font-weight: bold; color: #666; margin-bottom: 10px; }
        .condition { font-family: monospace; padding: 3px 8px; margin: 2px;
                     background: #e3f2fd; border-radius: 3px; display: inline-block; }
        .condition-false { background: #ffebee; }
        .condition-true { background: #e8f5e9; }
        .arrow { color: #4CAF50; font-weight: bold; margin: 10px 0; }
        .confidence { color: #666; font-size: 0.9em; }
        .legend { display: flex; gap: 20px; margin: 20px 0; }
        .legend-item { display: flex; align-items: center; gap: 5px; }
        .legend-box { width: 20px; height: 20px; border-radius: 3px; }
        .toc { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .toc a { color: #4CAF50; text-decoration: none; }
        .toc a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>🎯 ACE ML Model - Decision Rules</h1>
    
    <div class="legend">
        <div class="legend-item">
            <div class="legend-box" style="background: #f44336;"></div>
            <span>8XXX Validation Errors</span>
        </div>
        <div class="legend-item">
            <div class="legend-box" style="background: #2196F3;"></div>
            <span>9XXX Repair Codes</span>
        </div>
    </div>
    
    <div class="toc">
        <strong>Quick Navigation:</strong><br>
"""
    
    # Table of contents
    sorted_classes = sorted(enumerate(class_names), key=lambda x: x[1])
    for class_idx, class_name in sorted_classes[:top_classes]:
        html += f'        <a href="#{class_name}">{class_name}</a> | \n'
    
    html += """
    </div>
"""
    
    # Generate cards for each class
    for class_idx, class_name in sorted_classes[:top_classes]:
        base_code = class_name.split('_')[0]
        css_class = 'code-8xxx' if base_code.startswith('8') else 'code-9xxx'
        
        html += f"""
    <div class="code-card {css_class}" id="{class_name}">
        <div class="code-name">{class_name}</div>
"""
        
        # Feature importance
        if class_name in importance_by_class and importance_by_class[class_name]:
            html += """
        <h3>📊 Top Predictive Features</h3>
"""
            max_imp = max(imp for _, imp in importance_by_class[class_name])
            for feat, imp in importance_by_class[class_name][:8]:
                width = (imp / max_imp) * 100
                html += f"""
        <div class="feature-row">
            <div class="feature-name">{feat}</div>
            <div class="feature-bar" style="width: 200px;">
                <div class="feature-fill" style="width: {width}%;"></div>
            </div>
            <div class="feature-value">{imp:.3f}</div>
        </div>
"""
        
        # Decision rules
        rules = extract_rules_for_class(model, class_idx, feature_names, class_name, top_rules)
        
        if rules:
            html += """
        <h3>🔀 Decision Rules</h3>
"""
            for i, rule in enumerate(rules, 1):
                html += f"""
        <div class="rule-box">
            <div class="rule-header">Rule {i}</div>
            <div><strong>IF:</strong></div>
"""
                for cond in rule['conditions']:
                    simplified = simplify_condition(cond)
                    css = 'condition-true' if '= True' in simplified or '>' in simplified else 'condition-false'
                    html += f'            <span class="condition {css}">{simplified}</span>\n'
                
                html += f"""
            <div class="arrow">↓ THEN predict {class_name}</div>
            <div class="confidence">Confidence: {rule['confidence']:.1%} | Samples: {rule['samples']}</div>
        </div>
"""
        else:
            html += """
        <p><em>No strong decision rules extracted (model relies on feature combinations)</em></p>
"""
        
        html += """
    </div>
"""
    
    html += """
</body>
</html>
"""
    return html


def generate_mermaid_flowchart(model, feature_names: List[str], class_name: str,
                                class_idx: int, max_depth: int = 4) -> str:
    """Generate Mermaid flowchart for a single class."""
    rules = extract_rules_for_class(model, class_idx, feature_names, class_name, max_rules=3)
    
    if not rules:
        return f"```mermaid\nflowchart TD\n    A[No strong rules for {class_name}]\n```"
    
    lines = ["```mermaid", "flowchart TD"]
    
    node_id = 0
    for rule_idx, rule in enumerate(rules[:2]):  # Top 2 rules
        conditions = rule['conditions']
        prev_node = None
        
        for i, cond in enumerate(conditions[:max_depth]):
            simplified = simplify_condition(cond)
            current_node = f"N{node_id}"
            node_id += 1
            
            if '= True' in simplified or '>' in simplified:
                lines.append(f"    {current_node}{{{simplified}}}")
            else:
                lines.append(f"    {current_node}[{simplified}]")
            
            if prev_node:
                lines.append(f"    {prev_node} --> {current_node}")
            
            prev_node = current_node
        
        # Add prediction node
        pred_node = f"P{rule_idx}"
        lines.append(f"    {pred_node}(({class_name}))")
        lines.append(f"    {prev_node} --> {pred_node}")
        lines.append(f"    style {pred_node} fill:#4CAF50,color:#fff")
    
    lines.append("```")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Visualize ML model decision rules')
    parser.add_argument('--model-dir', required=True, help='Model directory')
    parser.add_argument('--output', default='rules.html', help='Output file (html or txt)')
    parser.add_argument('--format', choices=['html', 'text', 'mermaid'], default='html',
                       help='Output format')
    parser.add_argument('--top', type=int, default=30, help='Number of top classes to show')
    parser.add_argument('--code', default=None, help='Specific code to analyze')
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_dir}...")
    model, feature_names, class_names = load_model(args.model_dir)
    print(f"Loaded model with {len(feature_names)} features and {len(class_names)} classes")
    
    if args.code:
        # Single code analysis
        if args.code in class_names:
            class_idx = class_names.index(args.code)
            if args.format == 'mermaid':
                output = generate_mermaid_flowchart(model, feature_names, args.code, class_idx)
            else:
                # Just show this one code
                class_names_filtered = [args.code]
                output = generate_text_report(model, feature_names, class_names_filtered, 1, 10)
            print(output)
        else:
            print(f"Code {args.code} not found. Available: {class_names[:10]}...")
        return
    
    if args.format == 'html':
        print("Generating HTML report...")
        output = generate_html_report(model, feature_names, class_names, args.top)
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Saved to {args.output}")
    
    elif args.format == 'text':
        print("Generating text report...")
        output = generate_text_report(model, feature_names, class_names, args.top)
        if args.output.endswith('.txt'):
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"Saved to {args.output}")
        else:
            print(output)
    
    elif args.format == 'mermaid':
        print("Generating Mermaid flowcharts...")
        lines = ["# Decision Rule Flowcharts\n"]
        sorted_classes = sorted(enumerate(class_names), key=lambda x: x[1])
        for class_idx, class_name in sorted_classes[:args.top]:
            lines.append(f"\n## {class_name}\n")
            lines.append(generate_mermaid_flowchart(model, feature_names, class_name, class_idx))
        
        output = "\n".join(lines)
        with open(args.output.replace('.html', '.md'), 'w') as f:
            f.write(output)
        print(f"Saved to {args.output.replace('.html', '.md')}")


if __name__ == '__main__':
    main()
