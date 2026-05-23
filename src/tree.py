from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Any, Optional
import src.read as read

# Precision for rounding floats in node IDs (number of decimal places)
_ID_PRECISION = 4

def _r(val):              
    """Round a numeric value for use in node IDs."""
    return round(float(val), _ID_PRECISION)               #the function of this is to never let reounding errors influence node id's.


@dataclass
class Node:
    name: str
    stage: int
    parent: Optional[str]
    info: Dict[str, Any]      # hvilke stokastiske variabler som blir kjent i noden
    cond_prob: float          # betinget sannsynlighet gitt foreldrenoden


def build_scenario_tree(input_data: Dict[str, Any]) -> Dict[str, Any]:

    CM_up = input_data["CM_up"]
    CM_down = input_data["CM_down"]
    DA = input_data["DA"]
    EAM_up = input_data["EAM_up"]
    EAM_down = input_data["EAM_down"]
    wind_speed = input_data["wind_speed"]
    probabilities = input_data.get("probabilities", {})
    

    print("Read parameters from parquet.")
    print (CM_up, CM_down, DA, EAM_up, EAM_down, wind_speed)

    imb = ["up", "down"] # Imbalance er enten EAM_up eller EAM_down, med 50% sannsynlighet hver

    """
    Bygger scenariotre for:
      - Stage 1: root (før alt er kjent)
      - Stage 2: CM-priser (CM_up, CM_down)
      - Stage 3: DA-pris
      - Stage 4: EAM-priser + vind
      - Stage 5: imbalance pris (imb)

    Input kan være lister, numpy-arrays, etc.
    Antall alternativer i hver liste kan være vilkårlig.
    """

    nodes: Dict[str, Node] = {}
    children: Dict[Optional[str], List[str]] = {}

    def add_node(name, stage, parent, info, cond_prob):
        nodes[name] = Node(name, stage, parent, info, cond_prob)
        children.setdefault(parent, []).append(name)

    # --- Rotnode (stage 1) ---
    root = "root"
    add_node(root, stage=1, parent=None, info={}, cond_prob=1.0)
    print("[INFO] Added root node.")
    def normalize_probs(values, probs, name):
        if not values:
            raise ValueError(f"Missing values for {name}")
        if probs is None:
            return [1.0 / len(values) for _ in values]
        if len(probs) != len(values):
            raise ValueError(f"Probabilities for {name} must match values length")
        total = float(sum(probs))
        if total == 0:
            return [1.0 / len(values) for _ in values]
        return [float(value) / total for value in probs]

    p_cm_up = normalize_probs(CM_up, probabilities.get("CM_up"), "CM_up")
    p_cm_down = normalize_probs(CM_down, probabilities.get("CM_down"), "CM_down")
    p_da_probs = normalize_probs(DA, probabilities.get("DA"), "DA")
    p_eam_up_probs = normalize_probs(EAM_up, probabilities.get("EAM_up"), "EAM_up")
    p_eam_down_probs = normalize_probs(EAM_down, probabilities.get("EAM_down"), "EAM_down")
    p_wind_probs = normalize_probs(wind_speed, probabilities.get("wind_speed"), "wind_speed")

    # --- Stage 2: CM (alle kombinasjoner av CM_up og CM_down) ---
    n_CM_up = len(CM_up)
    n_CM_down = len(CM_down)

    stage2_nodes: List[str] = []
    for i, p_up in enumerate(CM_up):
        for j, p_down in enumerate(CM_down):
            name = f"u_({_r(p_up)},{_r(p_down)})"
            info = {"CM_up": p_up, "CM_down": p_down}
            cond_prob = p_cm_up[i] * p_cm_down[j]
            add_node(name, stage=2, parent=root, info=info, cond_prob=cond_prob)
            stage2_nodes.append(name)
    print("[INFO] Added stage 2 CM nodes.")

    # --- Stage 3: DA (for hver CM-node alle DA-alternativer) ---
    stage3_nodes: List[str] = []
    for parent_u in stage2_nodes:
        # Extract parent CM values from the parent node's info
        parent_info = nodes[parent_u].info
        for idx, p_da in enumerate(DA):
            name = f"v_({_r(parent_info['CM_up'])},{_r(parent_info['CM_down'])}|{_r(p_da)})"
            info = {"DA": p_da}
            add_node(name, stage=3, parent=parent_u, info=info, cond_prob=p_da_probs[idx])
            stage3_nodes.append(name)
    print("[INFO] Added stage 3 DA nodes.")

    # --- Stage 4: EAM + vind (alle kombinasjoner) ---
    stage4_nodes: List[str] = []
    for parent_v in stage3_nodes:
        # Build the path prefix from the parent v-node
        parent_v_info = nodes[parent_v].info
        grandparent_u = nodes[parent_v].parent
        gp_info = nodes[grandparent_u].info
        path_prefix = f"{_r(gp_info['CM_up'])},{_r(gp_info['CM_down'])}|{_r(parent_v_info['DA'])}"

        for i_eam_up, p_eup in enumerate(EAM_up):
            for i_eam_down, p_edown in enumerate(EAM_down):
                for i_wind, w in enumerate(wind_speed):
                    for i in imb:

                        # Vi antar at imbalance prisen er enten EAM_up eller EAM_down, med 50% sannsynlighet hver
                        if i == "up":
                            p_imb = p_eup
                        elif i == "down":
                            p_imb = p_edown

                        name = f"w_({path_prefix}|{_r(p_eup)},{_r(p_edown)},{_r(w)},{i})"
                        info = {
                            "EAM_up": p_eup,
                            "EAM_down": p_edown,
                            "wind_speed": w,
                            "imb": p_imb
                        }
                        stage4_prob = (
                            p_eam_up_probs[i_eam_up]
                            * p_eam_down_probs[i_eam_down]
                            * p_wind_probs[i_wind]
                            * 0.5
                        )
                        add_node(
                            name,
                            stage=4,
                            parent=parent_v,
                            info=info,
                            cond_prob=stage4_prob,
                        )
                        stage4_nodes.append(name)
    print("[INFO] Added stage 4 EAM + wind nodes.")


    
    # --- Bygg scenarier (én per løvnode) ---
    scenarios = []
    for leaf in stage4_nodes:
        path = []
        values: Dict[str, Any] = {}
        prob = 1.0
        cur = leaf

        # gå opp treet til roten
        while cur is not None:
            node = nodes[cur]
            prob *= node.cond_prob
            values.update(node.info)
            path.append(cur)
            cur = node.parent

        path.reverse()  # root -> ... -> leaf

        scenarios.append(
            {
                "leaf": leaf,
                "probability": prob,   # total sannsynlighet for scenariet
                "path": path,          # nodene langs denne historien
                "values": values,      # realiserte verdier (CM, DA, EAM, vind)
            }
        )

    tree = {
        "root": root,
        "nodes": nodes,        # dict: navn -> Node
        "children": children,  # dict: parent -> liste med barnenoder
        "leaves": stage4_nodes,
        "scenarios": scenarios,
    }
    return tree


def build_sets_from_tree(tree):
    """
    Input:
        tree: output fra build_scenario_tree()

    Output:
        U: set med alle stage-2 noder
        V: dict: V[u] = set med stage-3 noder barn av u
        W: dict: W[v] = set med stage-4 noder barn av v
        S: hele settet av noder i stage 2, 3 og 4
    """

    nodes = tree["nodes"]
    children = tree["children"]

    # --- 𝒰: scenarier i stage 2 ---
    U = {name for name, n in nodes.items() if n.stage == 2}

    # --- 𝒱(u): scenarier i stage 3 etter u ---
    V = {u: set(children.get(u, [])) for u in U}

    # --- 𝒲(v): scenarier i stage 4 etter v ---
    # Finn alle stage-3 noder:
    V_all = set().union(*V.values())
    W = {v: set(children.get(v, [])) for v in V_all}

    # --- 𝒮 = U ∪ V_all ∪ W_all ---
    W_all = set().union(*W.values()) if W else set()
    S = U.union(V_all).union(W_all)

    return U, V, W, S


def build_scenario_bundles(input_data: dict, n_per_bundle: int, num_bundles: int, seed: int = 30) -> List[Dict[str, Any]]:
    """
    Builds and stores a collection of scenario bundles (small scenario trees).

    Each bundle is a full scenario tree built by build_scenario_tree, using
    incrementing seeds for reproducibility.

    Input:
        input_data:   dictionary containing input parameters (CM_up, CM_down, DA, EAM_up, EAM_down, wind_speed)
        n_per_bundle: number of scenarios per bundle
        num_bundles:  how many scenario trees (bundles) to generate
        seed:         base seed for the first bundle; incremented by 1 for each subsequent bundle

    Output:
        B: list of scenario tree dicts, each with the same structure as
           returned by build_scenario_tree
    """
    B: List[Dict[str, Any]] = []

    for b in range(num_bundles):
        current_seed = seed + b
        bundle_data = read.get_bundle_data(input_data, n_per_bundle, current_seed)  # get the same data structure but with different random seed for each bundle

        
        tree = build_scenario_tree(bundle_data)
        B.append(tree)
        print(f"[INFO] Built bundle {b + 1}/{num_bundles} (seed={current_seed})")

    return B


def build_index_sets(U, V_all, W_all, M_u, M_v, M_w, M):
    """
    Build index sets for (m,s) and (m,w).

    Returns:
        idx_ms : list of (m, s) for all valid market-stage combinations
        idx_mw : list of (m, w) for all m in M and all w in W_all for d_{m,w}
    """

    idx_ms = []

    # Stage 2: CM markets (m in M_u, s in U)
    for u in U:
        for m in M_u:
            idx_ms.append((m, u))

    # Stage 3: DA market (m in M_v, s in V_all)
    for v in V_all:
        for m in M_v:
            idx_ms.append((m, v))

    # Stage 4: EAM markets (m in M_w, s in W_all)
    for w in W_all:
        for m in M_w:
            idx_ms.append((m, w))

    # d_{m,w}: only scenarios w, but all products m
    idx_mw = []
    for w in W_all:
        for m in M:
            idx_mw.append((m, w))

    
    # DA
    idx_DA = []
    for v in V_all:
        for m in M_v:
            idx_DA.append((m, v))

    # mFRR
    idx_mFRR = []
    for u in U:
        for m in M_u:
            idx_mFRR.append((m, u))
    
    for w in W_all:
        for m in M_w:
            idx_mFRR.append((m, w))
    


    return idx_ms, idx_mw, idx_DA, idx_mFRR