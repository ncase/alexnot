"""
CUT DOWN CODE
- Cut down selectors
- Don't Repeat Yourself
"""

import re
import random
import itertools
from typing import Dict, List

# permute_code_blocks: programs with this many blocks or fewer get every ordering, longer ones get a random sample
MAX_BLOCKS_FOR_ALL_ORDERINGS:int = 4
RANDOM_ORDERINGS_FOR_MANY_BLOCKS:int = 23  # = 4! - 1, the most a 4-block program proposes

# Suggest deleting blocks of code
def delete_blocks(train_pairs:List[Dict], parent_program_results:List[Dict]) -> set:
    candidates = set()
    for program_result in parent_program_results:
        program = program_result['program']
        
        # Split program into lines (filter out empty lines)
        lines = [line for line in program.split('\n') if line.strip()]
        
        # Find all for-loop blocks and their indices
        for_blocks = []
        for i, line in enumerate(lines):
            if line.strip().startswith('for '):
                # Find end of this for-block
                base_indent = len(line) - len(line.lstrip())
                end_of_block = i
                for j in range(i+1, len(lines)):
                    next_line = lines[j]
                    next_indent = len(next_line) - len(next_line.lstrip())
                    if next_indent > base_indent:
                        end_of_block = j
                    else:
                        break
                for_blocks.append((i, end_of_block))
        
        # For each for-block, create a candidate with that block deleted
        for start, end in for_blocks:
            # Create new program without this for-block
            new_lines = lines[:start] + lines[end+1:]
            new_program = '\n'.join(new_lines)
            
            # Add to candidates
            candidates.add( new_program.strip() )
    
    return candidates

# Delete random lines
def delete_lines(train_pairs:List[Dict], parent_program_results:List[Dict]) -> set:
    candidates = set()
    for program_result in parent_program_results:
        program = program_result['program']
    
        # Split program into lines (keep all lines, even empty ones)
        lines = program.split('\n')
        
        # For each line, create a candidate with that line deleted
        for i in range(len(lines)):
            # Create new program without the i-th line
            new_lines = lines[:i] + lines[i+1:]
            new_program = '\n'.join(new_lines)
            
            # Add to candidates
            candidates.add( new_program.strip() )
    
    return candidates

# Delete random arguments
def delete_arguments(train_pairs:List[Dict], parent_program_results:List[Dict]) -> set:
    candidates = set()
    for program_result in parent_program_results:
        program = program_result['program']
        
        # Find all function calls with arguments: function_name(arg_A=val_A,arg_B=val_B,arg_C=val_C)
        # Pattern breakdown:
        #   \w+           - function name (word characters)
        #   \(            - opening parenthesis
        #   ([^)]+)       - capture group 1: non-empty arguments inside parentheses
        #   \)            - closing parenthesis
        pattern = r'\w+\(([^)]+)\)'
        matches = list(re.finditer(pattern, program))
        
        for match in matches:
            args_str = match.group(1)
            
            # Parse arguments: split by comma and strip whitespace
            args = [arg.strip() for arg in args_str.split(',') if arg.strip()]
            
            # Skip if no arguments (nothing to delete)
            if len(args) == 0:
                continue
                
            # For each argument, create a variant with that argument removed
            for i in range(len(args)):
                # Create new args list without the i-th argument
                new_args = args[:i] + args[i+1:]
                
                # Reconstruct the function call
                new_args_str = ','.join(new_args)
                replacement = program[:match.start()] + f'{program[match.start():match.start()+match.group().find("(")]}({new_args_str})' + program[match.end():]
                
                # Add to candidates
                candidates.add( replacement.strip() )

    return candidates

# Suggest every reordering of a program's top-level blocks (a for-loop with its body, or a one-line call).
# Order matters when shapes overlap: shapes made later are drawn on top, so reordering blocks changes who covers whom
def permute_code_blocks(train_pairs:list[dict], parent_program_results:list[dict]) -> set:
    candidates:set = set()
    for program_result in parent_program_results:

        # Group the lines into top-level blocks: an unindented line starts a new block, indented lines join the current one
        blocks:list[list[str]] = []
        for line in program_result['program'].split('\n'):
            if not line.strip():
                continue
            if line[0] in ' \t' and blocks:
                blocks[-1].append(line)
            else:
                blocks.append([line])

        # Up to 4 blocks: every ordering (at most 4! = 24). More: a random 23 orderings, since N! blows up fast
        block_count:int = len(blocks)
        identity_order:tuple = tuple(range(block_count))
        if block_count <= MAX_BLOCKS_FOR_ALL_ORDERINGS:
            orderings:set[tuple] = set(itertools.permutations(identity_order))
        else:
            # 5+ blocks have at least 5! - 1 = 119 non-identity orderings, so collecting 23 distinct ones ends quickly
            orderings = {identity_order}
            while len(orderings) < RANDOM_ORDERINGS_FOR_MANY_BLOCKS + 1:
                orderings.add(tuple(random.sample(identity_order, block_count)))

        # The identity is just the parent itself (and a 1-block program has nothing else, so it proposes nothing)
        orderings.discard(identity_order)
        for ordering in orderings:
            candidates.add( '\n'.join(line for block_index in ordering for line in blocks[block_index]) )

    return candidates
