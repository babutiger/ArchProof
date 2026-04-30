"""ONNX graph traversal utilities."""

import onnx
from typing import Dict, List, Set, Tuple


def build_adjacency(graph: onnx.GraphProto) -> Tuple[Dict[str, onnx.NodeProto], Dict[str, List[str]], Dict[str, List[str]]]:
    """Build node lookup + forward/backward adjacency from ONNX graph.

    Returns:
        node_by_output: maps tensor name -> producing node
        forward: maps node output name -> list of consumer node output names
        backward: maps node output name -> list of producer node output names
    """
    node_by_output: Dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for out in node.output:
            node_by_output[out] = node

    forward: Dict[str, List[str]] = {}
    backward: Dict[str, List[str]] = {}

    for node in graph.node:
        node_id = node.output[0] if node.output else node.name
        backward[node_id] = []
        for inp in node.input:
            if inp in node_by_output:
                parent_id = node_by_output[inp].output[0]
                backward[node_id].append(parent_id)
                forward.setdefault(parent_id, []).append(node_id)

    return node_by_output, forward, backward


def get_output_names(graph: onnx.GraphProto) -> Set[str]:
    """Get the set of graph output tensor names."""
    return {o.name for o in graph.output}


def is_reachable(forward: Dict[str, List[str]], start: str, targets: Set[str], visited: Set[str] = None) -> bool:
    """Check if any target is reachable from start via forward edges (G4)."""
    if visited is None:
        visited = set()
    if start in visited:
        return False
    visited.add(start)
    if start in targets:
        return True
    for child in forward.get(start, []):
        if is_reachable(forward, child, targets, visited):
            return True
    return False


def find_gate_candidates(graph: onnx.GraphProto, gate_whitelist: Set[str]) -> List[onnx.NodeProto]:
    """Find all nodes whose op_type is in the gate whitelist (G1)."""
    return [node for node in graph.node if node.op_type in gate_whitelist]


def find_branch_nodes(graph: onnx.GraphProto, gate_node: onnx.NodeProto,
                      forward: Dict[str, List[str]]) -> List[str]:
    """Find candidate branch nodes downstream of a gate node.

    A branch node is a direct data-dependent child of the gate's output
    that feeds into the rest of the graph (potential dormant path).
    """
    branches = []
    gate_id = gate_node.output[0] if gate_node.output else None
    if gate_id is None:
        return branches
    for child_id in forward.get(gate_id, []):
        branches.append(child_id)
    return branches
