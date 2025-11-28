#!/usr/bin/env python3
"""
Extract human-readable rules for 8XXX error codes.

Usage:
    python extract_rules.py --model-dir ./models_8x_dt --data-dir ./raw_data --output rules_8xxx.md
"""

import argparse
import json
import pickle
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np

# Try to import sklearn for decision tree extraction
try:
    from sklearn.tree import _tree
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def extract_dt_rules(model, feature_names: List[str], class_names: List[str], 
                     max_depth: int = 5) -> Dict[str, List[str]]:
    """Extract rules from Decision Tree model."""
    rules_by_code = defaultdict(list)
    
    if not HAS_SKLEARN:
        return rules_by_code
    
    # Handle MultiOutputClassifier
    if hasattr(model, 'estimators_'):
        for idx, estimator in enumerate(model.estimators_):
            if idx < len(class_names):
                code = class_names[idx]
                if code.startswith('8'):
                    tree_rules = _extract_tree_rules(estimator, feature_names, max_depth)
                    rules_by_code[code].extend(tree_rules)
    elif hasattr(model, 'tree_'):
        tree_rules = _extract_tree_rules(model, feature_names, max_depth)
        # Assign to first class
        if class_names:
            rules_by_code[class_names[0]].extend(tree_rules)
    
    return rules_by_code


def extract_rf_rules(model_dir: str, max_depth: int = 4, min_samples: int = 50) -> Dict[str, List[dict]]:
    """
    Extract rules from Random Forest model.
    
    Random Forest = many decision trees. We extract top rules from each tree
    and find consensus rules that appear across multiple trees.
    """
    from collections import Counter
    
    # Load model
    model_path = Path(model_dir) / 'model.pkl'
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Handle dict-wrapped model
    if isinstance(model_data, dict):
        model = model_data.get('model')
        feature_names = model_data.get('feature_names', [])
        class_names = model_data.get('class_names', [])
    else:
        model = model_data
        feature_names = []
        class_names = []
    
    # Load class names from label encoder if needed
    if not class_names:
        label_path = Path(model_dir) / 'label_encoder.pkl'
        if label_path.exists():
            with open(label_path, 'rb') as f:
                label_data = pickle.load(f)
                class_names = label_data.get('code_to_idx', {}).keys()
                class_names = list(class_names)
    
    print(f"Model type: {type(model).__name__}")
    print(f"Features: {len(feature_names)}")
    print(f"Classes: {len(class_names)}")
    
    rules_by_code = defaultdict(list)
    
    # MultiOutputClassifier wraps one classifier per output
    if hasattr(model, 'estimators_'):
        for idx, estimator in enumerate(model.estimators_):
            if idx >= len(class_names):
                continue
            
            code = class_names[idx]
            if not code.startswith('8'):
                continue
            
            # Random Forest has multiple trees
            if hasattr(estimator, 'estimators_'):
                # This is a RandomForest/GradientBoosting
                all_rules = []
                for tree in estimator.estimators_[:20]:  # Sample first 20 trees
                    tree_rules = _extract_tree_paths(tree, feature_names, max_depth)
                    all_rules.extend(tree_rules)
                
                # Find consensus rules (appear in multiple trees)
                rule_counts = Counter([r['condition'] for r in all_rules])
                for condition, count in rule_counts.most_common(5):
                    if count >= 2:  # Appears in at least 2 trees
                        rules_by_code[code].append({
                            'condition': condition,
                            'tree_count': count,
                            'confidence': 'high' if count >= 5 else 'medium'
                        })
            
            elif hasattr(estimator, 'tree_'):
                # Single decision tree
                tree_rules = _extract_tree_paths(estimator, feature_names, max_depth)
                for r in tree_rules[:5]:
                    rules_by_code[code].append(r)
    
    return rules_by_code


def _extract_tree_paths(tree_model, feature_names: List[str], max_depth: int) -> List[dict]:
    """Extract decision paths from a single tree that lead to positive predictions."""
    if not HAS_SKLEARN:
        return []
    
    tree_ = tree_model.tree_
    
    if len(feature_names) == 0:
        feature_names = [f'f{i}' for i in range(tree_.n_features)]
    
    rules = []
    
    def recurse(node, depth, conditions):
        if depth > max_depth:
            return
        
        # Leaf node
        if tree_.feature[node] == _tree.TREE_UNDEFINED:
            value = tree_.value[node]
            n_samples = sum(value[0])
            
            if len(value[0]) >= 2:
                pos_rate = value[0][1] / n_samples if n_samples > 0 else 0
                
                # Only keep high-confidence positive predictions
                if pos_rate > 0.7 and n_samples >= 10 and conditions:
                    # Simplify conditions - keep only important ones
                    simplified = _simplify_conditions(conditions)
                    if simplified:
                        rules.append({
                            'condition': simplified,
                            'confidence': pos_rate,
                            'samples': int(n_samples)
                        })
            return
        
        feat_idx = tree_.feature[node]
        if feat_idx >= len(feature_names):
            return
            
        name = feature_names[feat_idx]
        threshold = tree_.threshold[node]
        
        # Skip non-informative features
        skip_features = ['amount', 'count', 'length', 'freq', 'encoded']
        if any(s in name.lower() for s in skip_features):
            # Still recurse but don't add condition
            recurse(tree_.children_left[node], depth + 1, conditions)
            recurse(tree_.children_right[node], depth + 1, conditions)
            return
        
        # For boolean features (threshold ~0.5)
        if 0.4 < threshold < 0.6:
            # Left = False, Right = True
            recurse(tree_.children_left[node], depth + 1, 
                   conditions + [(name, '=', 'False')])
            recurse(tree_.children_right[node], depth + 1, 
                   conditions + [(name, '=', 'True')])
        else:
            recurse(tree_.children_left[node], depth + 1, 
                   conditions + [(name, '<=', f'{threshold:.2f}')])
            recurse(tree_.children_right[node], depth + 1, 
                   conditions + [(name, '>', f'{threshold:.2f}')])
    
    recurse(0, 0, [])
    return rules


def _simplify_conditions(conditions: List[tuple]) -> str:
    """Simplify and format conditions."""
    # Keep only boolean conditions set to True
    true_conditions = [c[0] for c in conditions if c[1] == '=' and c[2] == 'True']
    false_conditions = [c[0] for c in conditions if c[1] == '=' and c[2] == 'False']
    
    parts = []
    for feat in true_conditions[:4]:  # Max 4 conditions
        parts.append(f"{feat} = True")
    for feat in false_conditions[:2]:  # Max 2 negative conditions
        parts.append(f"{feat} = False")
    
    if parts:
        return " AND ".join(parts)
    return None


def extract_rules_from_model(model_dir: str, output_file: str):
    """Extract rules from Random Forest model and format as markdown."""
    print(f"Loading model from {model_dir}...")
    
    rules_by_code = extract_rf_rules(model_dir)
    
    if not rules_by_code:
        print("No rules extracted. Try --detailed mode with data analysis instead.")
        return
    
    descriptions = get_code_descriptions()
    
    lines = [
        "# 8XXX Validation Rules - Extracted from Random Forest Model",
        "",
        "## Overview",
        "",
        "These rules were extracted by analyzing decision paths in the Random Forest model.",
        "Rules that appear across multiple trees are marked as high confidence.",
        "",
        "---",
        ""
    ]
    
    for code in sorted(rules_by_code.keys()):
        rules = rules_by_code[code]
        base_code = code.split('_')[0]
        desc = descriptions.get(base_code, 'Unknown')
        
        lines.append(f"## {code}")
        lines.append("")
        lines.append(f"**Description:** {desc}")
        lines.append("")
        lines.append("**Discovered Rules:**")
        lines.append("")
        
        for i, rule in enumerate(rules, 1):
            cond = rule.get('condition', '')
            conf = rule.get('confidence', 'medium')
            trees = rule.get('tree_count', 1)
            
            if isinstance(conf, float):
                conf_str = f"{conf*100:.0f}%"
            else:
                conf_str = conf
            
            lines.append(f"**Rule {i}** (confidence: {conf_str}, trees: {trees})")
            lines.append("```")
            lines.append(f"IF {cond}")
            lines.append(f"THEN predict {code}")
            lines.append("```")
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Rules written to {output_file}")


def _extract_tree_rules(tree_model, feature_names: List[str], max_depth: int) -> List[str]:
    """Extract rules from a single decision tree."""
    rules = []
    tree_ = tree_model.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    def recurse(node, depth, conditions):
        if depth > max_depth:
            return
        
        # Leaf node
        if tree_.feature[node] == _tree.TREE_UNDEFINED:
            # Check if this leaf predicts positive class
            value = tree_.value[node]
            if len(value[0]) >= 2 and value[0][1] > value[0][0]:
                # Positive prediction
                confidence = value[0][1] / sum(value[0])
                if confidence > 0.6 and conditions:
                    rule = " AND ".join(conditions)
                    rules.append(f"IF {rule} THEN predict (conf: {confidence:.2f})")
            return
        
        name = feature_name[node]
        threshold = tree_.threshold[node]
        
        # Left child (<=)
        if tree_.children_left[node] != _tree.TREE_LEAF:
            left_cond = conditions + [f"{name} <= {threshold:.4f}"]
            recurse(tree_.children_left[node], depth + 1, left_cond)
        
        # Right child (>)
        if tree_.children_right[node] != _tree.TREE_LEAF:
            right_cond = conditions + [f"{name} > {threshold:.4f}"]
            recurse(tree_.children_right[node], depth + 1, right_cond)
    
    recurse(0, 0, [])
    return rules


def analyze_data_patterns(data_dir: str, code_series: str = '8') -> Dict[str, Dict]:
    """Analyze raw data to find feature patterns for each code."""
    from data_pipeline import IFMLDataPipeline
    
    pipeline = IFMLDataPipeline()
    pipeline.load_combined_files(data_dir, '*.json')
    
    print(f"Loaded {len(pipeline.records)} records")
    
    # Collect feature values when each code fires
    code_patterns = defaultdict(lambda: {
        'count': 0,
        'feature_values': defaultdict(list),
        'common_conditions': []
    })
    
    # Track feature presence for each code
    for rec in pipeline.records:
        # Get 8XXX codes from this record
        codes = [c for c in rec.error_codes_only if c.startswith(code_series)]
        
        if not codes:
            continue
        
        features = rec.request_features
        
        for code in codes:
            base_code = code.split('_')[0]
            code_patterns[code]['count'] += 1
            
            # Track key feature values
            for key, value in features.items():
                if isinstance(value, bool) and value:
                    code_patterns[code]['feature_values'][key].append(value)
                elif isinstance(value, (int, float)) and value > 0:
                    code_patterns[code]['feature_values'][key].append(value)
    
    return code_patterns


def derive_rules_from_patterns(code_patterns: Dict, min_support: float = 0.7) -> Dict[str, List[str]]:
    """Derive IF-THEN rules from observed patterns."""
    rules_by_code = {}
    
    for code, data in sorted(code_patterns.items()):
        count = data['count']
        if count < 10:
            continue
        
        rules = []
        conditions = []
        
        # Find features that are TRUE in most cases
        for feature, values in data['feature_values'].items():
            if len(values) >= count * min_support:
                # This feature is present in 70%+ of cases
                if all(isinstance(v, bool) for v in values):
                    conditions.append(f"{feature} = True")
                elif all(isinstance(v, (int, float)) for v in values):
                    avg_val = np.mean(values)
                    min_val = np.min(values)
                    if min_val > 0:
                        conditions.append(f"{feature} > 0")
        
        if conditions:
            # Sort by relevance (validation features first)
            priority_keywords = ['iban', 'nch', 'bic', 'valid', 'needs', 'missing', 'domestic', 'international']
            
            def priority(cond):
                cond_lower = cond.lower()
                for i, kw in enumerate(priority_keywords):
                    if kw in cond_lower:
                        return i
                return 100
            
            conditions.sort(key=priority)
            
            # Take top conditions
            top_conditions = conditions[:5]
            rule = f"IF {' AND '.join(top_conditions)} THEN {code}"
            rules.append(rule)
        
        if rules:
            rules_by_code[code] = rules
    
    return rules_by_code


def get_code_descriptions() -> Dict[str, str]:
    """Return descriptions for 8XXX codes."""
    return {
        '8001': 'Invalid BIC format',
        '8004': 'Missing required IBAN for IBAN country',
        '8022': 'Invalid IBAN format',
        '8026': 'Inconsistent NCH/routing info',
        '8852': 'Missing beneficiary account',
        '8894': 'Invalid IBAN checksum',
        '8895': 'Invalid NCH/ABA routing number',
        '8896': 'Missing NCH for domestic payment',
    }


def format_rules_markdown(rules_by_code: Dict[str, List[str]], 
                          code_patterns: Dict,
                          output_file: str):
    """Format rules as markdown documentation."""
    descriptions = get_code_descriptions()
    
    lines = [
        "# 8XXX Validation Error Codes - Discovered Rules",
        "",
        "## Overview",
        "",
        "These rules were extracted from analyzing payment data patterns.",
        "Each rule shows conditions that commonly trigger the error code.",
        "",
        "---",
        ""
    ]
    
    # Sort by code
    for code in sorted(rules_by_code.keys()):
        rules = rules_by_code[code]
        base_code = code.split('_')[0]
        desc = descriptions.get(base_code, 'Unknown')
        count = code_patterns.get(code, {}).get('count', 0)
        
        lines.append(f"## {code}")
        lines.append("")
        lines.append(f"**Description:** {desc}")
        lines.append(f"**Occurrences in data:** {count}")
        lines.append("")
        lines.append("**Rules:**")
        lines.append("```")
        for rule in rules:
            lines.append(rule)
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # Write file
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Rules written to {output_file}")


def extract_detailed_rules(data_dir: str, output_file: str):
    """Extract detailed rules by analyzing feature correlations."""
    from data_pipeline import IFMLDataPipeline
    
    pipeline = IFMLDataPipeline()
    pipeline.load_combined_files(data_dir, '*.json')
    
    print(f"Analyzing {len(pipeline.records)} records...")
    
    # Define which features are relevant for each code
    code_feature_map = {
        '8004': ['needs_iban', 'has_iban', 'is_international', 'iban_country', 'country'],
        '8022': ['iban_valid_format', 'has_iban', 'iban_checksum_valid'],
        '8026': ['nch_sources', 'has_nch', 'nch_valid', 'has_adr_bank_id'],
        '8852': ['has_account', 'account_length'],
        '8894': ['iban_checksum_valid', 'iban_valid_format', 'has_iban'],
        '8895': ['nch_valid', 'fedaba_checksum_valid', 'nch_validation_applicable', 'is_domestic', 'has_nch'],
        '8896': ['has_nch', 'is_domestic', 'nch_validation_applicable'],
    }
    
    # Party prefixes
    prefixes = ['orig', 'send', 'dbt', 'cdt', 'intm', 'bnf', 'ordi', 'acwi']
    
    # Collect statistics
    code_stats = defaultdict(lambda: {
        'total': 0,
        'feature_when_true': defaultdict(int),
        'feature_when_false': defaultdict(int),
    })
    
    for rec in pipeline.records:
        features = rec.request_features
        codes = set(c.split('_')[0] for c in rec.error_codes_only if c.startswith('8'))
        
        for base_code, relevant_features in code_feature_map.items():
            has_code = base_code in codes
            
            for prefix in prefixes:
                for feat in relevant_features:
                    full_feat = f"{prefix}_{feat}"
                    if full_feat in features:
                        val = features[full_feat]
                        if isinstance(val, bool) and val:
                            if has_code:
                                code_stats[base_code]['feature_when_true'][full_feat] += 1
                            else:
                                code_stats[base_code]['feature_when_false'][full_feat] += 1
            
            if has_code:
                code_stats[base_code]['total'] += 1
    
    # Generate rules
    lines = [
        "# 8XXX Validation Error Codes - Detailed Rules",
        "",
        "## Extraction Method",
        "",
        "Rules extracted by analyzing feature correlations when codes fire vs don't fire.",
        "",
        "---",
        ""
    ]
    
    descriptions = get_code_descriptions()
    
    for base_code in sorted(code_stats.keys()):
        stats = code_stats[base_code]
        total = stats['total']
        
        if total < 10:
            continue
        
        desc = descriptions.get(base_code, 'Unknown')
        
        lines.append(f"## {base_code} - {desc}")
        lines.append("")
        lines.append(f"**Total occurrences:** {total}")
        lines.append("")
        
        # Find discriminative features
        discriminative = []
        for feat, count_true in stats['feature_when_true'].items():
            count_false = stats['feature_when_false'].get(feat, 0)
            
            # Calculate lift
            if count_false > 0:
                ratio = (count_true / total) / (count_false / (len(pipeline.records) - total + 1))
            else:
                ratio = float('inf') if count_true > 0 else 0
            
            support = count_true / total
            
            if support > 0.5 and ratio > 2:
                discriminative.append((feat, support, ratio))
        
        # Sort by support
        discriminative.sort(key=lambda x: -x[1])
        
        if discriminative:
            lines.append("**Key Conditions (support > 50%, lift > 2):**")
            lines.append("")
            lines.append("| Feature | Support | Lift |")
            lines.append("|---------|---------|------|")
            for feat, support, ratio in discriminative[:10]:
                ratio_str = f"{ratio:.1f}" if ratio != float('inf') else "∞"
                lines.append(f"| `{feat}` | {support*100:.0f}% | {ratio_str}x |")
            lines.append("")
            
            # Generate rule
            top_features = [f[0] for f in discriminative[:3]]
            if top_features:
                rule_parts = [f"{f} = True" for f in top_features]
                lines.append("**Derived Rule:**")
                lines.append("```")
                lines.append(f"IF {' AND '.join(rule_parts)}")
                lines.append(f"THEN predict {base_code}")
                lines.append("```")
                lines.append("")
        else:
            lines.append("*No strongly discriminative features found*")
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    # Write file
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Detailed rules written to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Extract 8XXX rules')
    parser.add_argument('--model-dir', default=None, help='Model directory (for RF/DT rules)')
    parser.add_argument('--data-dir', default=None, help='Raw data directory')
    parser.add_argument('--output', default='rules_8xxx.md', help='Output file')
    parser.add_argument('--detailed', action='store_true', help='Extract detailed rules with statistics')
    parser.add_argument('--from-model', action='store_true', help='Extract rules from Random Forest model')
    
    args = parser.parse_args()
    
    if args.from_model:
        if not args.model_dir:
            print("Error: --model-dir required when using --from-model")
            return
        extract_rules_from_model(args.model_dir, args.output)
    
    elif args.detailed:
        if not args.data_dir:
            print("Error: --data-dir required when using --detailed")
            return
        extract_detailed_rules(args.data_dir, args.output)
    
    else:
        if not args.data_dir:
            print("Error: --data-dir required")
            return
        # Analyze data patterns
        print("Analyzing data patterns...")
        code_patterns = analyze_data_patterns(args.data_dir, '8')
        
        print(f"Found {len(code_patterns)} unique 8XXX codes")
        
        # Derive rules from patterns
        print("Deriving rules...")
        rules = derive_rules_from_patterns(code_patterns)
        
        print(f"Generated rules for {len(rules)} codes")
        
        # Format and save
        format_rules_markdown(rules, code_patterns, args.output)


if __name__ == '__main__':
    main()
