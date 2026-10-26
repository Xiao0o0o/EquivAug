#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2 (GSM8K): Equivalence Transformation.

Parse each arithmetic expression from Step 1 into an AST and apply
equivalence-preserving rewrite rules (commutativity, associativity,
distributivity, fraction decomposition, ...; see Table 1 of the paper).
Every variant is evaluated and kept only if its value equals the original value.

- Input:  JSONL from Step 1 (fields: "question", "answer", "expr").
- Output: Same records with "expr_variants" and "expr_variants_with_depth".

Usage:
    python ast_transformation.py --input train_expr.jsonl --output train_expr_variants.jsonl \
        --max_transformations 3 --max_variants 5
"""

import argparse
import re
import json
import copy
from typing import List, Dict, Any, Optional, Tuple, Set
import random
from collections import defaultdict

# Import the base classes (keep them unchanged)
class ASTNode:
    """AST Node class that supports both numeric and symbolic values"""
    def __init__(self, node_type, value=None, operator=None):
        self.type = node_type  # 'BinaryOp', 'Number', 'Variable', 'Assignment'
        self.value = value
        self.operator = operator
        self.left = None
        self.right = None
    
    def __repr__(self):
        if self.type == 'Number':
            return f"Number({self.value})"
        elif self.type == 'Variable':
            return f"Variable({self.value})"
        elif self.type == 'BinaryOp':
            return f"BinaryOp({self.operator})"
        elif self.type == 'Assignment':
            return f"Assignment(=)"
        return f"{self.type}({self.operator or self.value})"
    
    def copy(self):
        """Deep copy of node"""
        return copy.deepcopy(self)
    
    def to_expression(self):
        """Convert AST back to expression string"""
        if self.type == 'Number':
            # Handle integers and floats
            if isinstance(self.value, (int, float)):
                if float(self.value).is_integer():
                    return str(int(self.value))
                return str(self.value)
            return str(self.value)
        elif self.type == 'Variable':
            return str(self.value)
        elif self.type == 'BinaryOp':
            left_expr = self.left.to_expression() if self.left else "0"
            right_expr = self.right.to_expression() if self.right else "0"
            
            # Add parentheses for clarity
            if self.left and self.left.type == 'BinaryOp':
                if self._needs_parentheses(self.left.operator, self.operator, True):
                    left_expr = f"({left_expr})"
            
            if self.right and self.right.type == 'BinaryOp':
                if self._needs_parentheses(self.right.operator, self.operator, False):
                    right_expr = f"({right_expr})"
            
            return f"{left_expr}{self.operator}{right_expr}"
        elif self.type == 'Assignment':
            left_expr = self.left.to_expression() if self.left else ""
            right_expr = self.right.to_expression() if self.right else ""
            return f"{left_expr}={right_expr}"
        return ""
    
    def _needs_parentheses(self, child_op, parent_op, is_left):
        """Determine if parentheses are needed based on operator precedence"""
        precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
        
        if child_op not in precedence or parent_op not in precedence:
            return False
        
        child_prec = precedence[child_op]
        parent_prec = precedence[parent_op]
        
        # Lower precedence always needs parentheses
        if child_prec < parent_prec:
            return True
        
        # Same precedence: need parentheses for right associative cases
        if child_prec == parent_prec:
            # For subtraction and division, right side needs parentheses
            if not is_left and parent_op in ['-', '/']:
                return True
        
        return False

class ExpressionParser:
    def __init__(self, expression):
        self.expression = expression.replace(' ', '')
        self.pos = 0
        self.tokens = self._tokenize()
        self.token_pos = 0
    
    def _tokenize(self):
        token_pattern = r'(\d*\.\d+|\d+|[a-zA-Z_][a-zA-Z0-9_]*|\+|\-|\*|/|\(|\)|=)'
        tokens = re.findall(token_pattern, self.expression)
        return tokens
    
    def _current_token(self):
        if self.token_pos < len(self.tokens):
            return self.tokens[self.token_pos]
        return None
    
    def _consume_token(self):
        token = self._current_token()
        self.token_pos += 1
        return token
    
    def parse(self):
        return self._parse_assignment()
    
    def _parse_assignment(self):
        left = self._parse_expression()
        
        if self._current_token() == '=':
            self._consume_token()
            right = self._parse_expression()
            
            node = ASTNode('Assignment', operator='=')
            node.left = left
            node.right = right
            return node
        
        return left
    
    def _parse_expression(self):
        left = self._parse_term()
        
        while self._current_token() in ['+', '-']:
            op = self._consume_token()
            right = self._parse_term()
            
            node = ASTNode('BinaryOp', operator=op)
            node.left = left
            node.right = right
            left = node
        
        return left
    
    def _parse_term(self):
        left = self._parse_factor()
        
        while self._current_token() in ['*', '/']:
            op = self._consume_token()
            right = self._parse_factor()
            
            node = ASTNode('BinaryOp', operator=op)
            node.left = left
            node.right = right
            left = node
        
        return left
    
    def _parse_factor(self):
        token = self._current_token()
        
        if token == '(':
            self._consume_token()
            node = self._parse_expression()
            if self._current_token() == ')':
                self._consume_token()
            return node
        elif token == '-':
            self._consume_token()
            node = ASTNode('BinaryOp', operator='-')
            node.left = ASTNode('Number', value=0)
            node.right = self._parse_factor()
            return node
        elif token and re.match(r'^(\d*\.\d+|\d+)$', token):
            self._consume_token()
            return ASTNode('Number', value=float(token))
        elif token and re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', token):
            self._consume_token()
            return ASTNode('Variable', value=token)
        else:
            raise ValueError(f"Unexpected token: {token}")

class ExpressionEvaluator:
    """Evaluator for mathematical expressions"""
    
    @staticmethod
    def evaluate(node: ASTNode, variables: Dict[str, float] = None) -> float:
        if variables is None:
            variables = {}
        
        if node.type == 'Number':
            return float(node.value)
        
        elif node.type == 'Variable':
            if node.value in variables:
                return float(variables[node.value])
            else:
                raise ValueError(f"Variable '{node.value}' not defined")
        
        elif node.type == 'BinaryOp':
            left_val = ExpressionEvaluator.evaluate(node.left, variables)
            right_val = ExpressionEvaluator.evaluate(node.right, variables)
            
            if node.operator == '+':
                return left_val + right_val
            elif node.operator == '-':
                return left_val - right_val
            elif node.operator == '*':
                return left_val * right_val
            elif node.operator == '/':
                if right_val == 0:
                    raise ValueError("Division by zero")
                return left_val / right_val
            else:
                raise ValueError(f"Unknown operator: {node.operator}")
        
        elif node.type == 'Assignment':
            return ExpressionEvaluator.evaluate(node.right, variables)
        
        else:
            raise ValueError(f"Unknown node type: {node.type}")
    
    @staticmethod
    def evaluate_expression(expr: str, variables: Dict[str, float] = None) -> float:
        parser = ExpressionParser(expr)
        ast = parser.parse()
        return ExpressionEvaluator.evaluate(ast, variables)

class ImprovedASTTransformer:
    """Improved AST transformer with controlled transformation depth and priority"""
    
    def __init__(self):
        # Initialize rule counters
        self.rule_usage = defaultdict(int)
        
        # Define transformation rules with priority order
        # Priority 1: Rules 6, 10, 11, 13, 18, 19, 20, 21, 22, 23, 24
        # Priority 2: Rules 7, 8, 9, 12, 14, 15, 16, 17
        # Priority 3: Rules 0, 1, 2, 3, 4, 5
        self.transformation_rules = [
            # Priority 1 (highest)
            (6, self.distributive, "a*(b+c) -> a*b + a*c"),
            (10, self.reverse_distributive, "a*b + a*c -> a*(b+c)"),
            (11, self.reverse_division_chain, "a÷(b×c) -> a÷b÷c"),
            (13, self.reverse_subtraction_chain, "a-(b+c) -> a-b-c"),
            (18, self.distributive_subtraction, "a*(b-c) -> a*b - a*c"),
            (19, self.division_distribute_add, "(a+b)/c -> a/c + b/c"),
            (20, self.common_denominator_add, "a/b + c/b -> (a+c)/b"),
            (21, self.reverse_distributive_sub, "a*b - a*c -> a*(b-c)"),
            (22, self.division_distribute_sub, "(a-b)/c -> a/c - b/c"),
            (23, self.common_denominator_sub, "a/b - c/b -> (a-c)/b"),
            (24, self.distributive_left, "(a+b)*c -> a*c + b*c"),
            
            # Priority 2
            (7, self.division_chain, "a÷b÷c -> a÷(b×c)"),
            (8, self.mul_div_reorder, "a×b÷c -> a÷c×b"),
            (9, self.subtraction_chain, "a-b-c -> a-(b+c)"),
            (12, self.reverse_mul_div_reorder, "a÷c×b -> a×b÷c"),
            (14, self.div_div_to_div_mul, "(a/b)/c -> a/(b*c)"),
            (15, self.mul_div_to_mul_over_div, "a*(b/c) -> (a*b)/c"),
            (16, self.div_to_mul_div, "a/(b/c) -> (a*c)/b"),
            (17, self.div_mul_to_mul_over, "(a/b)*c -> (a*c)/b"),
            
            # Priority 3 (lowest)
            (0, self.commutative_add, "a+b -> b+a"),
            (1, self.commutative_mul, "a*b -> b*a"),
            (2, self.associative_add, "(a+b)+c -> a+(b+c)"),
            (3, self.associative_mul, "(a*b)*c -> a*(b*c)"),
            (4, self.reverse_associative_add, "a+(b+c) -> (a+b)+c"),
            (5, self.reverse_associative_mul, "a*(b*c) -> (a*b)*c"),
        ]


    def _is_binop(self, node, op):
        return node and node.type == 'BinaryOp' and node.operator == op

    def _flatten_op(self, node, op):
        if self._is_binop(node, op):
            return self._flatten_op(node.left, op) + self._flatten_op(node.right, op)
        return [node.copy()]

    def _build_chain(self, op, items):
        if not items:
            return ASTNode('Number', value=1.0 if op == '*' else 0.0)
        cur = items[0].copy()
        for it in items[1:]:
            nn = ASTNode('BinaryOp', operator=op)
            nn.left = cur
            nn.right = it.copy()
            cur = nn
        return cur

    def _canon(self, node):
        if node is None:
            return ('None',)
        if node.type == 'Number':
            return ('N', round(float(node.value), 10))
        if node.type == 'Variable':
            return ('V', str(node.value))
        if node.type == 'BinaryOp' and node.operator in ('+', '*'):
            items = self._flatten_op(node, node.operator)
            sigs = [self._canon(it) for it in items]
            sigs.sort()
            return (node.operator, tuple(sigs))
        if node.type == 'BinaryOp':
            return (node.operator, self._canon(node.left), self._canon(node.right))
        return (node.type, node.operator, node.value)

    def _nodes_equal_comm_assoc(self, a, b):
        return self._canon(a) == self._canon(b)



    def reset_rule_usage(self):
        """Reset the rule usage counters"""
        self.rule_usage = defaultdict(int)
    
    def get_rule_statistics(self):
        """Get formatted rule usage statistics"""
        stats = []
        for rule_id, _, description in self.transformation_rules:
            count = self.rule_usage.get(rule_id, 0)
            stats.append(f"Rule {rule_id:2d}: {count:5d} times - {description}")
        return "\n".join(stats)
    
    def find_all_transformations(self, root: ASTNode, max_transformations: int = 1, 
                                  max_variants: int = 10) -> List[Dict[str, Any]]:
        """
        Find all possible transformations with controlled depth.
        Each variant records which depth level it was generated at.
        
        Args:
            root: The AST root node
            max_transformations: Maximum number of transformations to apply (depth)
            max_variants: Maximum number of unique variants to generate
        
        Returns:
            List of dicts: [{"expr": str, "depth": int}, ...]
        """
        original_expr = root.to_expression()
        seen_exprs = set([original_expr])  # Track seen expressions to avoid duplicates
        all_variants = []  # List of {"expr": str, "depth": int}
        
        # Start with the original expression
        current_level = [root.copy()]
        
        # Apply transformations level by level
        for depth in range(1, max_transformations + 1):
            next_level = []
            
            for current_ast in current_level:
                # Find all possible single-step transformations from this AST
                single_step_variants = self._find_single_step_transformations(current_ast)
                
                for variant_ast, rule_id in single_step_variants:
                    variant_expr = variant_ast.to_expression()
                    
                    # Check for duplicates (including original)
                    if variant_expr not in seen_exprs:
                        seen_exprs.add(variant_expr)
                        next_level.append(variant_ast)
                        self.rule_usage[rule_id] += 1
                        
                        all_variants.append({
                            "expr": variant_expr,
                            "depth": depth
                        })
                        
                        # Stop if we have enough variants
                        if len(all_variants) >= max_variants:
                            break
                
                if len(all_variants) >= max_variants:
                    break
            
            # Move to next level
            current_level = next_level
            
            # Stop if no new transformations found or enough variants
            if not current_level or len(all_variants) >= max_variants:
                break
        
        return all_variants[:max_variants]
    
    def _find_single_step_transformations(self, root: ASTNode) -> List[Tuple[ASTNode, int]]:
        """
        Find all possible single-step transformations from current AST
        
        Returns:
            List of (transformed_ast, rule_id) tuples
        """
        transformations = []
        
        # Collect all subtrees with random order for subtree selection
        subtrees = self._collect_all_subtrees(root)
        random.shuffle(subtrees)  # Randomize subtree order
        
        # Try each rule in priority order (no shuffle for rules)
        for rule_id, rule_func, _ in self.transformation_rules:
            for node, parent, is_left in subtrees:
                # Try to apply the rule
                transformed = rule_func(node)
                
                if transformed is not node:  # Transformation successful
                    # Create a new AST with this transformation
                    new_root = root.copy()
                    
                    # Find and replace the corresponding node in the new AST
                    if parent is None:
                        new_root = transformed
                    else:
                        # Need to find the corresponding parent in the new tree
                        new_subtrees = self._collect_all_subtrees(new_root)
                        for new_node, new_parent, new_is_left in new_subtrees:
                            if self._nodes_match_position(node, parent, is_left,
                                                         new_node, new_parent, new_is_left):
                                if new_parent:
                                    if new_is_left:
                                        new_parent.left = transformed.copy()
                                    else:
                                        new_parent.right = transformed.copy()
                                break
                    
                    transformations.append((new_root, rule_id))
        
        return transformations
    
    def _nodes_match_position(self, node1, parent1, is_left1, node2, parent2, is_left2):
        """Check if two nodes are at the same position in their respective trees"""
        if is_left1 != is_left2:
            return False

        if parent1 is None and parent2 is None:
            return self._nodes_equal(node1, node2)

        if (parent1 is None) != (parent2 is None):
            return False

        return self._nodes_equal(parent1, parent2) and self._nodes_equal(node1, node2)
    
    def _collect_all_subtrees(self, root: ASTNode) -> List[Tuple[ASTNode, Optional[ASTNode], bool]]:
        """
        Collect all subtrees in the AST
        Returns list of (node, parent, is_left_child) tuples
        """
        subtrees = []
        
        def traverse(node, parent=None, is_left=True):
            if node is None:
                return
            
            subtrees.append((node, parent, is_left))
            
            if hasattr(node, 'left'):
                traverse(node.left, node, True)
            if hasattr(node, 'right'):
                traverse(node.right, node, False)
        
        traverse(root)
        return subtrees
    
    def _nodes_equal(self, node1, node2):
        """Check if two nodes are structurally equal"""
        if node1 is None and node2 is None:
            return True
        if node1 is None or node2 is None:
            return False
        if node1.type != node2.type:
            return False
        if node1.type == 'Number':
            return abs(node1.value - node2.value) < 1e-9
        if node1.type == 'Variable':
            return node1.value == node2.value
        if node1.type == 'BinaryOp':
            return (node1.operator == node2.operator and
                    self._nodes_equal(node1.left, node2.left) and
                    self._nodes_equal(node1.right, node2.right))
        return False
    
    # =====================================================================
    # All transformation rules
    # =====================================================================

    def commutative_add(self, node):
        """Rule 0: X+Y -> Y+X"""
        if node and node.type == 'BinaryOp' and node.operator == '+':
            new_node = node.copy()
            new_node.left, new_node.right = new_node.right, new_node.left
            return new_node
        return node
    
    def commutative_mul(self, node):
        """Rule 1: X*Y -> Y*X"""
        if node and node.type == 'BinaryOp' and node.operator == '*':
            new_node = node.copy()
            new_node.left, new_node.right = new_node.right, new_node.left
            return new_node
        return node
    
    def associative_add(self, node):
        """Rule 2: (X+Y)+Z -> X+(Y+Z)"""
        if (node and node.type == 'BinaryOp' and node.operator == '+' and 
            node.left and node.left.type == 'BinaryOp' and node.left.operator == '+'):
            new_node = ASTNode('BinaryOp', operator='+')
            new_node.left = node.left.left.copy()
            inner = ASTNode('BinaryOp', operator='+')
            inner.left = node.left.right.copy()
            inner.right = node.right.copy()
            new_node.right = inner
            return new_node
        return node
    
    def associative_mul(self, node):
        """Rule 3: (X*Y)*Z -> X*(Y*Z)"""
        if (node and node.type == 'BinaryOp' and node.operator == '*' and 
            node.left and node.left.type == 'BinaryOp' and node.left.operator == '*'):
            new_node = ASTNode('BinaryOp', operator='*')
            new_node.left = node.left.left.copy()
            inner = ASTNode('BinaryOp', operator='*')
            inner.left = node.left.right.copy()
            inner.right = node.right.copy()
            new_node.right = inner
            return new_node
        return node
    
    def reverse_associative_add(self, node):
        """Rule 4: X+(Y+Z) -> (X+Y)+Z"""
        if (node and node.type == 'BinaryOp' and node.operator == '+' and 
            node.right and node.right.type == 'BinaryOp' and node.right.operator == '+'):
            new_node = ASTNode('BinaryOp', operator='+')
            inner = ASTNode('BinaryOp', operator='+')
            inner.left = node.left.copy()
            inner.right = node.right.left.copy()
            new_node.left = inner
            new_node.right = node.right.right.copy()
            return new_node
        return node
    
    def reverse_associative_mul(self, node):
        """Rule 5: X*(Y*Z) -> (X*Y)*Z"""
        if (node and node.type == 'BinaryOp' and node.operator == '*' and 
            node.right and node.right.type == 'BinaryOp' and node.right.operator == '*'):
            new_node = ASTNode('BinaryOp', operator='*')
            inner = ASTNode('BinaryOp', operator='*')
            inner.left = node.left.copy()
            inner.right = node.right.left.copy()
            new_node.left = inner
            new_node.right = node.right.right.copy()
            return new_node
        return node

    def distributive(self, node):
        """Rule 6: X*(Y+Z) -> X*Y+X*Z"""
        if not self._is_binop(node, '*'):
            return node

        factors = self._flatten_op(node, '*')  
        idx = next((i for i, f in enumerate(factors) if self._is_binop(f, '+')), None)
        if idx is None:
            return node

        add_factor = factors[idx]
        other_factors = [factors[j] for j in range(len(factors)) if j != idx]
        A = self._build_chain('*', other_factors)  

        new_node = ASTNode('BinaryOp', operator='+')
        left_mul = ASTNode('BinaryOp', operator='*');  left_mul.left = A.copy(); left_mul.right = add_factor.left.copy()
        right_mul = ASTNode('BinaryOp', operator='*'); right_mul.left = A.copy(); right_mul.right = add_factor.right.copy()
        new_node.left = left_mul; new_node.right = right_mul
        return new_node
    
    def division_chain(self, node):
        """Rule 7: a÷b÷c -> a÷(b×c)"""
        if (node and node.type == 'BinaryOp' and node.operator == '/' and 
            node.left and node.left.type == 'BinaryOp' and node.left.operator == '/'):
            new_node = ASTNode('BinaryOp', operator='/')
            new_node.left = node.left.left.copy()
            mul_node = ASTNode('BinaryOp', operator='*')
            mul_node.left = node.left.right.copy()
            mul_node.right = node.right.copy()
            new_node.right = mul_node
            return new_node
        return node
    
    def mul_div_reorder(self, node):
        """Rule 8: a×b÷c -> a÷c×b"""
        if (node and node.type == 'BinaryOp' and node.operator == '/' and 
            node.left and node.left.type == 'BinaryOp' and node.left.operator == '*'):
            outer = ASTNode('BinaryOp', operator='*')
            inner = ASTNode('BinaryOp', operator='/')
            inner.left = node.left.left.copy()
            inner.right = node.right.copy()
            outer.left = inner
            outer.right = node.left.right.copy()
            return outer
        return node
    
    def subtraction_chain(self, node):
        """Rule 9: a-b-c -> a-(b+c)"""
        if (node and node.type == 'BinaryOp' and node.operator == '-' and 
            node.left and node.left.type == 'BinaryOp' and node.left.operator == '-'):
            new_node = ASTNode('BinaryOp', operator='-')
            new_node.left = node.left.left.copy()
            add_node = ASTNode('BinaryOp', operator='+')
            add_node.left = node.left.right.copy()
            add_node.right = node.right.copy()
            new_node.right = add_node
            return new_node
        return node
    
    def reverse_distributive(self, node):
        """Rule 10: X*Y+X*Z -> X*(Y+Z)"""
        if not (self._is_binop(node, '+')):
            return node

        from collections import Counter, defaultdict
        L_factors = self._flatten_op(node.left, '*') if self._is_binop(node.left, '*') else [node.left.copy()]
        R_factors = self._flatten_op(node.right, '*') if self._is_binop(node.right, '*') else [node.right.copy()]

        sig = lambda n: self._canon(n)
        cntL = Counter(sig(x) for x in L_factors)
        cntR = Counter(sig(x) for x in R_factors)
        common = cntL & cntR  

        if not common:
            return node  

        taken = defaultdict(int)
        X_nodes = []
        for f in L_factors:
            s = sig(f)
            if s in common and taken[s] < common[s]:
                X_nodes.append(f.copy())
                taken[s] += 1

        def remove_counts(factors, counts):
            used = defaultdict(int)
            rem = []
            for f in factors:
                s = sig(f)
                if s in counts and used[s] < counts[s]:
                    used[s] += 1  
                else:
                    rem.append(f.copy())
            return rem

        Y_nodes = remove_counts(L_factors, common)
        Z_nodes = remove_counts(R_factors, common)

        X = self._build_chain('*', X_nodes)  
        Y = self._build_chain('*', Y_nodes) if Y_nodes else ASTNode('Number', value=1.0)
        Z = self._build_chain('*', Z_nodes) if Z_nodes else ASTNode('Number', value=1.0)

        mul = ASTNode('BinaryOp', operator='*')
        add = ASTNode('BinaryOp', operator='+')
        add.left = Y; add.right = Z
        mul.left = X; mul.right = add
        return mul
    
    def reverse_division_chain(self, node):
        """Rule 11: a÷(b×c) -> a÷b÷c"""
        if (node and node.type == 'BinaryOp' and node.operator == '/' and 
            node.right and node.right.type == 'BinaryOp' and node.right.operator == '*'):
            outer = ASTNode('BinaryOp', operator='/')
            inner = ASTNode('BinaryOp', operator='/')
            inner.left = node.left.copy()
            inner.right = node.right.left.copy()
            outer.left = inner
            outer.right = node.right.right.copy()
            return outer
        return node
    
    def reverse_mul_div_reorder(self, node):
        """Rule 12: a÷c×b -> a×b÷c"""
        if (node and node.type == 'BinaryOp' and node.operator == '*' and 
            node.left and node.left.type == 'BinaryOp' and node.left.operator == '/'):
            outer = ASTNode('BinaryOp', operator='/')
            inner = ASTNode('BinaryOp', operator='*')
            inner.left = node.left.left.copy()
            inner.right = node.right.copy()
            outer.left = inner
            outer.right = node.left.right.copy()
            return outer
        return node
    
    def reverse_subtraction_chain(self, node):
        """Rule 13: a-(b+c) -> a-b-c"""
        if (node and node.type == 'BinaryOp' and node.operator == '-' and 
            node.right and node.right.type == 'BinaryOp' and node.right.operator == '+'):
            outer = ASTNode('BinaryOp', operator='-')
            inner = ASTNode('BinaryOp', operator='-')
            inner.left = node.left.copy()
            inner.right = node.right.left.copy()
            outer.left = inner
            outer.right = node.right.right.copy()
            return outer
        return node
    
    def div_div_to_div_mul(self, node):
        """Rule 14: (a/b)/c -> a/(b*c)"""
        if (node and node.type == 'BinaryOp' and node.operator == '/' and 
            node.left and node.left.type == 'BinaryOp' and node.left.operator == '/'):
            new_node = ASTNode('BinaryOp', operator='/')
            new_node.left = node.left.left.copy()
            mul_node = ASTNode('BinaryOp', operator='*')
            mul_node.left = node.left.right.copy()
            mul_node.right = node.right.copy()
            new_node.right = mul_node
            return new_node
        return node
    
    def mul_div_to_mul_over_div(self, node):
        """Rule 15: a*(b/c) -> (a*b)/c"""
        if (node and node.type == 'BinaryOp' and node.operator == '*' and 
            node.right and node.right.type == 'BinaryOp' and node.right.operator == '/'):
            new_node = ASTNode('BinaryOp', operator='/')
            mul_node = ASTNode('BinaryOp', operator='*')
            mul_node.left = node.left.copy()
            mul_node.right = node.right.left.copy()
            new_node.left = mul_node
            new_node.right = node.right.right.copy()
            return new_node
        return node
    
    def div_to_mul_div(self, node):
        """Rule 16: a/(b/c) -> (a*c)/b"""
        if (node and node.type == 'BinaryOp' and node.operator == '/' and 
            node.right and node.right.type == 'BinaryOp' and node.right.operator == '/'):
            new_node = ASTNode('BinaryOp', operator='/')
            mul_node = ASTNode('BinaryOp', operator='*')
            mul_node.left = node.left.copy()
            mul_node.right = node.right.right.copy()
            new_node.left = mul_node
            new_node.right = node.right.left.copy()
            return new_node
        return node
    
    def div_mul_to_mul_over(self, node):
        """Rule 17: (a/b)*c -> (a*c)/b"""
        if (node and node.type == 'BinaryOp' and node.operator == '*' and 
            node.left and node.left.type == 'BinaryOp' and node.left.operator == '/'):
            new_node = ASTNode('BinaryOp', operator='/')
            mul_node = ASTNode('BinaryOp', operator='*')
            mul_node.left = node.left.left.copy()
            mul_node.right = node.right.copy()
            new_node.left = mul_node
            new_node.right = node.left.right.copy()
            return new_node
        return node
    
    def distributive_subtraction(self, node):
        """Rule 18: a*(b-c) -> a*b - a*c"""
        if not self._is_binop(node, '*'):
            return node

        factors = self._flatten_op(node, '*')
        idx = next((i for i, f in enumerate(factors) if self._is_binop(f, '-')), None)
        if idx is None:
            return node

        sub_factor = factors[idx]
        other_factors = [factors[j] for j in range(len(factors)) if j != idx]
        A = self._build_chain('*', other_factors)

        new_node = ASTNode('BinaryOp', operator='-')
        left_mul = ASTNode('BinaryOp', operator='*');  left_mul.left = A.copy(); left_mul.right = sub_factor.left.copy()
        right_mul = ASTNode('BinaryOp', operator='*'); right_mul.left = A.copy(); right_mul.right = sub_factor.right.copy()
        new_node.left = left_mul; new_node.right = right_mul
        return new_node
    
    def division_distribute_add(self, node):
        """Rule 19: (a+b)/c -> a/c + b/c"""
        if (node and node.type == 'BinaryOp' and node.operator == '/' and 
            node.left and node.left.type == 'BinaryOp' and node.left.operator == '+'):
            new_node = ASTNode('BinaryOp', operator='+')
            left_div = ASTNode('BinaryOp', operator='/')
            left_div.left = node.left.left.copy()
            left_div.right = node.right.copy()
            right_div = ASTNode('BinaryOp', operator='/')
            right_div.left = node.left.right.copy()
            right_div.right = node.right.copy()
            new_node.left = left_div
            new_node.right = right_div
            return new_node
        return node
    
    def common_denominator_add(self, node):
        """Rule 20: a/b + c/b -> (a+c)/b"""
        if not (self._is_binop(node, '+') and
                self._is_binop(node.left, '/') and
                self._is_binop(node.right, '/')):
            return node

        denom_L = node.left.right
        denom_R = node.right.right

        if not self._nodes_equal_comm_assoc(denom_L, denom_R):
            return node

        new_node = ASTNode('BinaryOp', operator='/')
        add_node = ASTNode('BinaryOp', operator='+')
        add_node.left = node.left.left.copy()
        add_node.right = node.right.left.copy()
        new_node.left = add_node
        new_node.right = node.left.right.copy()
        return new_node

    # =====================================================================
    # NEW RULES 21-24
    # =====================================================================

    def reverse_distributive_sub(self, node):
        """Rule 21: a*b - a*c -> a*(b-c)  (subtraction version of reverse_distributive)"""
        if not self._is_binop(node, '-'):
            return node

        from collections import Counter, defaultdict
        L_factors = self._flatten_op(node.left, '*') if self._is_binop(node.left, '*') else [node.left.copy()]
        R_factors = self._flatten_op(node.right, '*') if self._is_binop(node.right, '*') else [node.right.copy()]

        sig = lambda n: self._canon(n)
        cntL = Counter(sig(x) for x in L_factors)
        cntR = Counter(sig(x) for x in R_factors)
        common = cntL & cntR

        if not common:
            return node

        taken = defaultdict(int)
        X_nodes = []
        for f in L_factors:
            s = sig(f)
            if s in common and taken[s] < common[s]:
                X_nodes.append(f.copy())
                taken[s] += 1

        def remove_counts(factors, counts):
            used = defaultdict(int)
            rem = []
            for f in factors:
                s = sig(f)
                if s in counts and used[s] < counts[s]:
                    used[s] += 1
                else:
                    rem.append(f.copy())
            return rem

        Y_nodes = remove_counts(L_factors, common)
        Z_nodes = remove_counts(R_factors, common)

        X = self._build_chain('*', X_nodes)
        Y = self._build_chain('*', Y_nodes) if Y_nodes else ASTNode('Number', value=1.0)
        Z = self._build_chain('*', Z_nodes) if Z_nodes else ASTNode('Number', value=1.0)

        mul = ASTNode('BinaryOp', operator='*')
        sub = ASTNode('BinaryOp', operator='-')
        sub.left = Y; sub.right = Z
        mul.left = X; mul.right = sub
        return mul

    def division_distribute_sub(self, node):
        """Rule 22: (a-b)/c -> a/c - b/c"""
        if (node and node.type == 'BinaryOp' and node.operator == '/' and
            node.left and node.left.type == 'BinaryOp' and node.left.operator == '-'):
            new_node = ASTNode('BinaryOp', operator='-')
            left_div = ASTNode('BinaryOp', operator='/')
            left_div.left = node.left.left.copy()
            left_div.right = node.right.copy()
            right_div = ASTNode('BinaryOp', operator='/')
            right_div.left = node.left.right.copy()
            right_div.right = node.right.copy()
            new_node.left = left_div
            new_node.right = right_div
            return new_node
        return node

    def common_denominator_sub(self, node):
        """Rule 23: a/b - c/b -> (a-c)/b"""
        if not (self._is_binop(node, '-') and
                self._is_binop(node.left, '/') and
                self._is_binop(node.right, '/')):
            return node

        denom_L = node.left.right
        denom_R = node.right.right

        if not self._nodes_equal_comm_assoc(denom_L, denom_R):
            return node

        new_node = ASTNode('BinaryOp', operator='/')
        sub_node = ASTNode('BinaryOp', operator='-')
        sub_node.left = node.left.left.copy()
        sub_node.right = node.right.left.copy()
        new_node.left = sub_node
        new_node.right = node.left.right.copy()
        return new_node

    def distributive_left(self, node):
        """Rule 24: (a+b)*c -> a*c + b*c  (left-side distributive)"""
        if not self._is_binop(node, '*'):
            return node

        # Check if left child is addition
        if not (node.left and node.left.type == 'BinaryOp' and node.left.operator == '+'):
            return node

        # Collect all multiplication factors on the right side (excluding the left add)
        # For (a+b)*c*d, we want C = c*d, then a*C + b*C
        # But we need to be careful: the left child with '+' might not be found by flatten
        # since flatten only flattens '*'. So we handle it directly.
        
        add_node = node.left
        # Build the "rest" of the multiplication chain (everything except the add_node)
        # For simple case: (a+b)*c, rest = c
        # For (a+b)*c*d parsed as ((a+b)*c)*d, flatten gives [add_node, c, d]
        factors = self._flatten_op(node, '*')
        # Find and remove the add_node from factors
        idx = next((i for i, f in enumerate(factors) if self._is_binop(f, '+')), None)
        if idx is None:
            return node
        
        add_factor = factors[idx]
        other_factors = [factors[j] for j in range(len(factors)) if j != idx]
        C = self._build_chain('*', other_factors)

        new_node = ASTNode('BinaryOp', operator='+')
        left_mul = ASTNode('BinaryOp', operator='*')
        left_mul.left = add_factor.left.copy()
        left_mul.right = C.copy()
        right_mul = ASTNode('BinaryOp', operator='*')
        right_mul.left = add_factor.right.copy()
        right_mul.right = C.copy()
        new_node.left = left_mul
        new_node.right = right_mul
        return new_node


def extract_answer_from_text(answer_text: str) -> float:
    """
    Extract the numerical answer from the answer text
    The answer is after the #### marker
    """
    if '####' in answer_text:
        answer_part = answer_text.split('####')[-1].strip()
        # Remove any commas and extract the number
        answer_part = answer_part.replace(',', '')
        # Try to extract a number using regex
        match = re.search(r'-?\d+\.?\d*', answer_part)
        if match:
            return float(match.group())
    return None

def process_jsonl_with_evaluation(input_path: str, output_path: str = None, 
                                 max_transformations: int = 1, max_variants: int = 5):
    """
    Process JSONL file with improved transformation and evaluation.
    
    Each variant now includes a "depth" field indicating which transformation
    depth level it was generated at.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        max_transformations: Maximum transformation depth (1 = single transformation, 2 = two levels, etc.)
        max_variants: Maximum number of variants to generate per expression
    """
    if output_path is None:
        output_path = input_path.replace('.jsonl', '_improved_v3.jsonl')
    
    transformer = ImprovedASTTransformer()
    evaluator = ExpressionEvaluator()
    processed_data = []
    stats = {
        'success': 0, 
        'error': 0, 
        'total_variants': 0,
        'total_variants_after_filter': 0,
        'variants_filtered_out': 0,
        'evaluation_matches': 0,
        'evaluation_mismatches': 0,
        'evaluation_errors': 0,
        'depth_distribution': defaultdict(int),  # count variants per depth
    }
    
    print(f"Processing: {input_path}")
    print(f"Output will be saved to: {output_path}")
    print(f"Max transformation depth: {max_transformations}")
    print(f"Max variants per expression: {max_variants}")
    print("-" * 60)
    
    with open(input_path, 'r', encoding='utf-8') as infile:
        for line_num, line in enumerate(infile, 1):
            try:
                # Parse JSON line
                data = json.loads(line.strip())
                
                # Get original expression
                original_expr = data.get('expr', '')
                if not original_expr:
                    print(f"Line {line_num}: No 'expr' field found, skipping")
                    processed_data.append(data)
                    continue
                
                # Get expected answer
                expected_answer = extract_answer_from_text(data.get('answer', ''))
                
                # Parse and transform
                try:
                    parser = ExpressionParser(original_expr)
                    ast = parser.parse()
                    
                    # Generate variants using improved method with depth control
                    # Now returns List[Dict] with {"expr": str, "depth": int}
                    variant_dicts = transformer.find_all_transformations(
                        ast, 
                        max_transformations=max_transformations,
                        max_variants=max_variants
                    )
                    
                    # Evaluate original expression
                    original_value = None
                    try:
                        original_value = evaluator.evaluate(ast)
                        data['calculated_value'] = original_value
                        
                        # Compare with expected answer
                        if expected_answer is not None:
                            if abs(original_value - expected_answer) < 1e-6:
                                data['answer_match'] = True
                                stats['evaluation_matches'] += 1
                            else:
                                data['answer_match'] = False
                                data['expected'] = expected_answer
                                stats['evaluation_mismatches'] += 1
                                print(f"Line {line_num}: Answer mismatch - Expected: {expected_answer}, Got: {original_value}")
                    except Exception as e:
                        data['evaluation_error'] = str(e)
                        stats['evaluation_errors'] += 1

                    # ========== Skip transformation if the original expression is wrong ==========
                    if not data.get('answer_match', False):
                        data['expr_original'] = original_expr
                        data['expr_variants'] = []
                        data['expr_variants_with_depth'] = []
                        data['expr_variants_final'] = []
                        data['skipped'] = 'original expression does not match ground truth'
                        stats['success'] += 1  # still counted as processed
                        processed_data.append(data)
                        continue
                    # ================================================
                    
                    stats['total_variants'] += len(variant_dicts)
                    
                    # Evaluate each variant and filter out incorrect ones
                    valid_variant_dicts = []
                    for vd in variant_dicts:
                        try:
                            variant_parser = ExpressionParser(vd["expr"])
                            variant_ast = variant_parser.parse()
                            variant_value = evaluator.evaluate(variant_ast)
                            
                            # Filter: only keep if matches original value
                            if original_value is not None and abs(variant_value - original_value) < 1e-6:
                                vd["value"] = variant_value
                                valid_variant_dicts.append(vd)
                                stats['depth_distribution'][vd["depth"]] += 1
                            else:
                                stats['variants_filtered_out'] += 1
                        except:
                            stats['variants_filtered_out'] += 1
                    
                    stats['total_variants_after_filter'] += len(valid_variant_dicts)
                    
                    # Write to data
                    data['expr_original'] = original_expr
                    data['transformation_depth'] = max_transformations
                    
                    # expr_variants: list of strings (just expressions)
                    data['expr_variants'] = [vd["expr"] for vd in valid_variant_dicts]
                    
                    # expr_variants_with_depth: list of dicts with expr, depth, value
                    data['expr_variants_with_depth'] = valid_variant_dicts
                    
                    # expr_variants_final: formatted strings "expr=value (depth=N)"
                    expr_variants_final = []
                    for vd in valid_variant_dicts:
                        val = vd.get("value")
                        depth = vd.get("depth")
                        if val is not None:
                            expr_variants_final.append(f"{vd['expr']}={val} (depth={depth})")
                        else:
                            expr_variants_final.append(f"{vd['expr']} (depth={depth})")
                    data['expr_variants_final'] = expr_variants_final
                    
                    stats['success'] += 1
                    
                    if line_num <= 5:  # Show first 5 examples
                        print(f"\nExample {line_num}:")
                        print(f"  Original: {original_expr}")
                        print(f"  Value: {original_value if original_value is not None else 'N/A'}")
                        print(f"  Valid variants ({len(valid_variant_dicts)}):")
                        for vd in valid_variant_dicts[:3]:
                            print(f"    depth={vd['depth']}: {vd['expr']}")
                        if len(valid_variant_dicts) > 3:
                            print(f"    ... and {len(valid_variant_dicts) - 3} more")
                    
                except Exception as e:
                    print(f"Line {line_num}: Error processing expression '{original_expr}': {e}")
                    data['expr_original'] = original_expr
                    data['expr_variants'] = []
                    data['expr_variants_with_depth'] = []
                    data['expr_variants_final'] = []
                    data['error'] = str(e)
                    stats['error'] += 1
                
                processed_data.append(data)
                
            except json.JSONDecodeError as e:
                print(f"Line {line_num}: JSON decode error: {e}")
                stats['error'] += 1
                continue
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for data in processed_data:
            json.dump(data, outfile, ensure_ascii=False)
            outfile.write('\n')
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print(f"Total lines processed: {len(processed_data)}")
    print(f"Successful transformations: {stats['success']}")
    print(f"Errors: {stats['error']}")
    print(f"\nVariant Statistics:")
    print(f"  Total variants generated (before filter): {stats['total_variants']}")
    print(f"  Variants filtered out (incorrect):        {stats['variants_filtered_out']}")
    print(f"  Total valid variants (after filter):      {stats['total_variants_after_filter']}")
    print(f"  Average valid variants per expression:    {stats['total_variants_after_filter']/max(stats['success'], 1):.2f}")
    print(f"\nVariants by Depth:")
    for depth in sorted(stats['depth_distribution'].keys()):
        count = stats['depth_distribution'][depth]
        print(f"  Depth {depth}: {count} variants")
    print(f"\nEvaluation Results:")
    print(f"  Matching answers: {stats['evaluation_matches']}")
    print(f"  Mismatched answers: {stats['evaluation_mismatches']}")
    print(f"  Evaluation errors: {stats['evaluation_errors']}")
    
    print(f"\n{'='*60}")
    print("Rule Usage Statistics:")
    print(transformer.get_rule_statistics())
    print(f"{'='*60}")
    
    print(f"\nOutput saved to: {output_path}")
    print("=" * 60)
    
    return processed_data, stats, transformer.rule_usage

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply equivalence transformations to GSM8K expression ASTs.")
    parser.add_argument("--input", required=True, help="JSONL from Step 1 (must contain 'expr' and 'answer')")
    parser.add_argument("--output", required=True, help="Output JSONL with expression variants")
    parser.add_argument("--max_transformations", type=int, default=3,
                        help="Transformation depth: number of rules applied sequentially")
    parser.add_argument("--max_variants", type=int, default=5,
                        help="Maximum number of variants per expression")
    args = parser.parse_args()

    process_jsonl_with_evaluation(
        input_path=args.input,
        output_path=args.output,
        max_transformations=args.max_transformations,
        max_variants=args.max_variants,
    )
