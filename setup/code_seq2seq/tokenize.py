import sys
import os
import json
import builtins
import io
import keyword
import token
from tokenize import tokenize
import ast
import pickle as pkl
import random
import astor


def t_rename_fields(the_ast, all_sites=False):
    """
    all_sites=True: a single, randomly selected, referenced field 
    (self.field in Python) has its name replaced by a hole
    all_sites=False: all possible fields are selected
    """
    changed = False

    # Going to need parent info
    for node in ast.walk(the_ast):
        for child in ast.iter_child_nodes(node):
            child.parent = node

    candidates = []
    for node in ast.walk(the_ast):
        if isinstance(node, ast.Name) and node.id == 'self':
            if isinstance(node.parent, ast.Attribute):
                if isinstance(node.parent.parent, ast.Call) and node.parent.parent.func == node.parent:
                    continue
                if node.parent.attr not in [ c.attr for c in candidates ]:
                    candidates.append(node.parent)

    if len(candidates) == 0:
        return False, the_ast

    if not all_sites:
        selected = [random.choice(candidates)]
    else:
        selected = candidates

    to_rename = []
    for cnt, selected_node in enumerate(selected, start=1):
        for node in ast.walk(the_ast):
            if isinstance(node, ast.Name) and node.id == 'self':
                if isinstance(node.parent, ast.Attribute) and node.parent.attr == selected_node.attr:
                    if isinstance(node.parent.parent, ast.Call) and node.parent.parent.func == node.parent:
                        continue
                    to_rename.append((node.parent, cnt))

    for node, idx in to_rename:
        changed = True
        node.attr = 'VAR' + str(idx)

    return changed, the_ast


def t_rename_parameters(the_ast, all_sites=False):
    """
    Parameters get replaced by holes.
    """
    changed = False
    candidates = []
    for node in ast.walk(the_ast):
        if isinstance(node, ast.arg):
            if node.arg != 'self' and node.arg not in [ c.arg for c in candidates ]:
                # print(node.arg, node.lineno)
                candidates.append(node)

    if len(candidates) == 0:
        return False, the_ast

    if not all_sites:
        selected = [random.choice(candidates)]
    else:
        selected = candidates

    parameter_defs = {}
    for cnt, s in enumerate(selected, start=1):
        parameter_defs[s.arg] = cnt

    to_rename = []
    for node in ast.walk(the_ast):
        if isinstance(node, ast.Name) and node.id in parameter_defs:
            to_rename.append((node, parameter_defs[node.id]))
        elif isinstance(node, ast.arg) and node.arg in parameter_defs:
            to_rename.append((node, parameter_defs[node.arg]))

    for node, idx in to_rename:
        changed = True
        if hasattr(node, 'arg'):
            node.arg = 'VAR' + str(idx)
        else:
            node.id = 'VAR' + str(idx)

    return changed, the_ast


def t_rename_local_variables(the_ast, all_sites=False):
    """
    Local variables get replaced by holes.
    """
    changed = False
    candidates = []
    for node in ast.walk(the_ast):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            if node.id not in [ c.id for c in candidates ]:
                # print(node.id, node.lineno)
                candidates.append(node)

    if len(candidates) == 0:
        return False, the_ast

    if not all_sites:
        selected = [random.choice(candidates)]
    else:
        selected = candidates

    local_var_defs = {}
    for cnt, s in enumerate(selected, start=1):
        local_var_defs[s.id] = cnt

    to_rename = []
    for node in ast.walk(the_ast):
        if isinstance(node, ast.Name) and node.id in local_var_defs:
            to_rename.append((node, local_var_defs[node.id]))

    for node, idx in to_rename:
        changed = True
        node.id = 'VAR' + str(idx)

    return changed, the_ast

def _tokenize_programs(programs, max_tokens=500):
    sequences = []
    tokens = keyword.kwlist + dir(builtins)
    for program in programs:
        sequence = []
        try:
          changed, result = t_rename_local_variables(
            ast.parse(program),
            all_sites=True
          )
          changed, result = t_rename_fields(
            ast.parse(astor.to_source(result)),
            all_sites=True
          )
          changed, result = t_rename_parameters(
            ast.parse(astor.to_source(result)),
            all_sites=True
          )
          program = astor.to_source(result)
        except Exception as ex:
          # import traceback
          # traceback.print_exc()
          sequences.append("CHARS")
          continue
        for typ, text, _, _, _ in tokenize(io.BytesIO(program.encode('utf-8')).readline):
            # print("{}::{}".format(typ, text))
            if typ is token.STRING:
                sequence.append("CHARS")
            elif typ is token.NUMBER:
                sequence.append("NUM")
            elif typ is token.NEWLINE or typ is token.NL:
                sequence.append("NEWLINE")
            elif typ is token.INDENT:
                sequence.append("INDENT")
            elif typ is token.DEDENT:
                sequence.append("DEDENT")
            else:
                # Uncontrolled number of newlines and tabs 
                # will cause dataloader to crash
                if text != '\n' and text != '\t':
                    sequence.append(text)
        
        sequence_str = " ".join(sequence)
        sequences.append(sequence_str[:max_tokens])
    return sequences


def transform_data(src_path, dest_path=None, debug=False):
  with open(src_path, 'r') as fp:
    files = fp.readlines()
  
  files = [fi[:-1] for fi in files]

  if debug:
    files = files[:10]
  
  print('Files loaded..\n {}'.format(json.dumps(files[:3], indent=2)))
  
  all_src, all_src_names = [], []
  for f in files:
    with open(f, 'r') as fp:
      src = fp.read()
      all_src.append(src)
      name = f.split("/")[-2]+"_"+f.split("/")[-1]
      all_src_names.append(name)

  tokenized_programs = _tokenize_programs(all_src)
  '''
  for p in all_src:
    print(len(p))
  for t in tokenized_programs:
    print(len(t))
  '''
  if dest_path is not None:
    with open(dest_path, 'w') as fp:
        for n, p in zip(all_src_names, tokenized_programs):
            fp.write("{}\t{}\t{}\n".format(n, p, p))
        fp.truncate(fp.tell() - len(os.linesep))
    
    print('Done dumping tokenized programs to {}'.format(dest_path))
  else:
      return tokenized_programs


if __name__ == '__main__':
  debug           = (sys.argv[1] == 'True')
  train_file_path = sys.argv[2]
  test_file_path  = sys.argv[3]
  train_dest_path = sys.argv[4]
  test_dest_path  = sys.argv[5]

  dataset = transform_data(train_file_path, train_dest_path, debug)
  dataset = transform_data(test_file_path, test_dest_path, debug)
