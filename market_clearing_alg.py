import networkx as nx
import matplotlib.pyplot as plt
from vis_bipartite_graph import draw_graph

# Basic utilities (parts, formatting, matrices)
# ==========================================================================================================
def get_parts(G):
    """Return (sellers, buyers) as sorted lists of string IDs."""
    sellers = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 0]
    buyers  = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 1]
    if not sellers or not buyers:
        nodes_sorted = sorted(G.nodes(), key=lambda x: int(x))
        n = len(nodes_sorted) // 2
        sellers = [str(i) for i in range(0, n)]
        buyers  = [str(i) for i in range(n, 2 * n)]
    return sorted(sellers, key=int), sorted(buyers, key=int)

def fmt(x):
    """Format numeric values as ints; None -> '-'."""
    if x is None: return "-"
    return str(int(round(float(x))))

def edge_valuation(G, b, s):
    """Return valuation on edge (b,s) (int) or None if no edge/attr."""
    data = G.get_edge_data(b, s) or G.get_edge_data(s, b)
    if not data or "valuation" not in data:
        return None
    return int(data["valuation"])

def get_valuations(G):
    """Return buyers × sellers matrix of valuations (missing edges = 0)."""
    sellers, buyers = get_parts(G)

    valuations = []
    for b in buyers:
        row = []
        for s in sellers:
            if G.has_edge(s, b):
                row.append(G.edges[s, b]["valuation"])
            else:
                row.append(0)
        valuations.append(row)
    return valuations

def get_prices(G, sellers):
    """Return {seller_id: int(price)} with default 0."""
    return {s: int(G.nodes[s].get("price", 0)) for s in sellers}

def build_valuation_matrix(G, buyers, sellers):
    """Return [(buyer, [payoff or None for each seller])]"""
    rows = []
    for b in buyers:
        vals = [edge_valuation(G, b, s) for s in sellers]
        rows.append((b, vals))
    return rows

def build_payoff_matrix(prices, val_rows, sellers):
    """
    prices: dict seller->price
    val_rows: [(buyer, [valuation or None])]
    Returns [(buyer, [payoff or None])]
    """
    rows = []
    for b, vals in val_rows:
        pays = [None if v is None else (v - int(prices[sellers[j]])) for j, v in enumerate(vals)]
        rows.append((b, pays))
    return rows

def print_matrix(title, buyers, sellers, rows):
    """Pretty-print a buyer x seller matrix."""
    print(f"\n== {title} ==")
    header = " ".join(f"{s:>6}" for s in sellers)
    print(f"{'Buyer':>6} | {header}")
    print("-" * (8 + 7 * len(sellers)))
    for b, vals in rows:
        line = " ".join(f"{fmt(x):>6}" for x in vals)
        print(f"{b:>6} | {line}")
        
# Dumping (terminal output)
# =========================================================================================================

def dump_market_state(
    G,
    *,
    round_no: int | None = None,
    energy: int | None = None,
    preferred_edges: list[tuple[str, str]] | None = None,
    matching: list[tuple[str, str]] | None = None,
    constricted_buyers: list[str] | None = None,
    constricted_sellers: list[str] | None = None,
    status: str | None = None,  # e.g., "cleared", "stopped", "constricted"
):
    """Unified terminal dump: prices, valuation/payoff tables, 
    and optional round/energy, preferred edges, matching, 
    constricted sets, and status."""
    sellers, buyers = get_parts(G)
    prices = get_prices(G, sellers)
    val_rows = build_valuation_matrix(G, buyers, sellers)
    pay_rows = build_payoff_matrix(prices, val_rows, sellers)

    print("\n\n\n===================================================================================================")
    if round_no is not None:
        print(f"--- Round {round_no} ---")
    if energy is not None:
        print(f"Current market energy: {int(energy)}")

    print("\n== Seller Prices ==")
    for s in sellers:
        price_int = int(prices[s])
        print(f"Seller {s}: price = {price_int}")

    print_matrix("Valuations (buyer × seller)", buyers, sellers, val_rows)
    print_matrix("Payoffs = valuation − price(seller)", buyers, sellers, pay_rows)

    if preferred_edges is not None:
        print("\nPreferred edges:", sorted(list(preferred_edges)))
    if matching is not None:
        print(f"Matching size: {len(matching)} / {len(sellers)}")
        print("Matching pairs:", sorted(list(matching)))

    if constricted_buyers:
        b_sorted = sorted(constricted_buyers, key=int)
        print("Constricted buyers:", b_sorted)
    if constricted_sellers:
        s_sorted = sorted(constricted_sellers, key=int)
        print("Constricted sellers:", s_sorted)

    if status == "cleared":
        print("\nMarket cleared ✅")
    elif status == "stopped":
        print("\nStopped (no constricted sellers) ⚠️")
    elif status == "constricted":
        print("\nConstricted market detected ⚠️")

# utilities for preferred graph / matching / constricted set
# =========================================================================================================

def _utility(G, b, s):
    """u(b,s) = valuation(b,s) - price(s); None if edge missing."""
    data = G.get_edge_data(b, s) or G.get_edge_data(s, b)
    if not data or "valuation" not in data:
        return None
    val = int(data["valuation"])
    price = int(G.nodes[s].get("price", 0))
    return val - price

def build_preferred_graph(G):
    """
    Build the preferred-seller graph for current prices:
    for each buyer b, keep edge(s) to seller(s) s that maximize u(b,s).
    """
    sellers, buyers = get_parts(G)
    P = nx.Graph()
    P.add_nodes_from(G.nodes(data=True))

    for b in buyers:
        best = None
        keep = []
        for s in G.neighbors(b):
            if s not in sellers:
                continue
            u = _utility(G, b, s)
            if u is None:
                continue
            if best is None or u > best:
                best, keep = u, [s]
            elif u == best:
                keep.append(s)
        for s in keep:
            P.add_edge(b, s)
    return P

def maximum_matching(P, sellers, buyers):
    """
    Maximum matching on the preferred graph.
    Returns a set of pairs (buyer, seller).
    """
    M = nx.algorithms.bipartite.maximum_matching(P, top_nodes=buyers)
    return {(b, a) for b, a in M.items() if b in buyers}

def buyer_constricted_set_with_best(P, matching, sellers, buyers, G):
    """
    Alternating BFS from unmatched buyers.
    Returns:
      - constricted_buyers: set of buyers reachable via alternating paths
      - best_options: dict buyer -> {
            'best_sellers': [seller ids],    # neighbors in P (ties allowed)
            'best_utility': int            # u(b,s) for any s in best_sellers
        }
    """
    matched_b = {b for (b, s) in matching}
    start = [b for b in buyers if b not in matched_b]

    from collections import deque
    q = deque(start)
    seen_b, seen_s = set(start), set()
    matched_s_to_b = {s: b for (b, s) in matching}

    while q:
        b = q.popleft()
        for s in P.neighbors(b):
            if s in seen_s:
                continue
            seen_s.add(s)
            if s in matched_s_to_b:
                b2 = matched_s_to_b[s]
                if b2 not in seen_b:
                    seen_b.add(b2)
                    q.append(b2)

    # For each constricted buyer, report best seller(s) and utility.
    # In the preferred graph P, neighbors of b are exactly the best sellers.
    best_options = {}
    for b in seen_b:
        best_sellers = sorted(list(P.neighbors(b)), key=int)
        if best_sellers:
            # utilities are equal for all neighbors in P; compute once
            u_sample = _utility(G, b, best_sellers[0])
        else:
            u_sample = None
        best_options[b] = {
            'best_sellers': best_sellers,
            'best_utility': u_sample,
        }

    return seen_b, best_options


def find_constricted_or_perfect(G):
    """
    Determine whether the current market (at current prices) has:
      - a perfect matching on the preferred-seller graph, OR
      - a buyer-side constricted set; for each constricted buyer, include best option(s).

    Returns dict:
      {
        'is_perfect': bool,
        'preferred_graph': P,
        'preferred_edges': list[(b,s)],
        'matching': set[(b,s)],
        'constricted_buyers': set[str],         # empty if perfect
        'best_options': dict[str, {'best_sellers': [...], 'best_utility': int or None}]
      }
    """
    sellers, buyers = get_parts(G)
    P = build_preferred_graph(G)
    pref_edges = list(P.edges())
    M = maximum_matching(P, sellers, buyers)

    if len(M) == len(sellers):  # perfect
        return {
            'is_perfect': True,
            'preferred_graph': P,
            'preferred_edges': pref_edges,
            'matching': M,
            'constricted_buyers': set(),
            'best_options': {},
        }

    B_constricted, best_opts = buyer_constricted_set_with_best(P, M, sellers, buyers, G)
    return {
        'is_perfect': False,
        'preferred_graph': P,
        'preferred_edges': pref_edges,
        'matching': M,
        'constricted_buyers': B_constricted,
        'best_options': best_opts,
    }

def get_total_potential_energy(G):
    """
    Returns the total potential energy of the market:
    the sum of all valuations over all buyer-seller edges.
    """
    total = 0
    for _, _, data in G.edges(data=True):
        if "valuation" in data:
            total += int(data["valuation"])
    return total

def _constricted_sets(P, matching, sellers, buyers):
    """
    Alternating BFS from unmatched buyers on the preferred graph P.
    Returns (buyers_reachable, sellers_reachable).
    """
    matched_b = {b for (b, s) in matching}
    start = [b for b in buyers if b not in matched_b]

    from collections import deque
    q = deque(start)
    seen_b, seen_s = set(start), set()
    matched_s_to_b = {s: b for (b, s) in matching}

    while q:
        b = q.popleft()
        for s in P.neighbors(b):         # preferred edge b -> s
            if s in seen_s:
                continue
            seen_s.add(s)
            if s in matched_s_to_b:      # follow matched edge s -> b'
                b2 = matched_s_to_b[s]
                if b2 not in seen_b:
                    seen_b.add(b2)
                    q.append(b2)

    return seen_b, seen_s  # buyers in alternating tree, sellers in alternating tree

# Clearing loops (non-interactive / interactive)
# =========================================================================================================

#Returns the perfect graph (not interactive)
def run_clearing_until_equilibrium(
    G,
    price_step: int = 1,
    max_rounds: int = 1000,
    verbose: bool = True,
    show_progress: bool = True,
):
    """
    Automatic (non-interactive) version of market clearing.
    Repeats until a perfect matching is found or max_rounds reached.

    Returns:
        prices : list of seller prices aligned with visual order
        payoffs : 2D list of buyer payoffs [[u(b1,s1), u(b1,s2), ...], ...]
    """
    sellers, buyers = get_parts(G)

    current_energy = get_total_potential_energy(G)

    r = 0
    while current_energy > 0:
        r += 1
        P = build_preferred_graph(G)
        M = maximum_matching(P, sellers, buyers)
        if r >= max_rounds:
            dump_market_state(
                G,
                round_no=r,
                energy=current_energy,
                preferred_edges=list(P.edges()),
                matching=list(M),
                status="stopped",
            )
            return
        current_energy = sum(
            max(0, G.edges[s, b]["valuation"] - int(G.nodes[s].get("price", 0)))
            for s, b in G.edges()
        )

        if verbose:
            dump_market_state(
                G,
                round_no=r,
                energy=current_energy,
                preferred_edges=list(P.edges()),
                matching=list(M),
            )

        if len(M) == len(sellers):
            break

        constr_b, constr_s = _constricted_sets(P, M, sellers, buyers)
        if not constr_s:
            if verbose:
                dump_market_state(
                    G,
                    round_no=r,
                    energy=current_energy,
                    preferred_edges=list(P.edges()),
                    matching=list(M),
                    constricted_buyers=list(constr_b),
                    constricted_sellers=list(constr_s),
                    status="stopped",
                )
            break

        for s in constr_s:
            G.nodes[s]["price"] = int(G.nodes[s].get("price", 0)) + int(price_step)

    # final summary (always via dump)
    final_prices = [int(G.nodes[s].get("price", 0)) for s in sellers]
    price_dict = dict(zip(sellers, final_prices))
    P_final = build_preferred_graph(G)
    pref_edges_final = list(P_final.edges())
    final_M = maximum_matching(P_final, sellers, buyers)

    if verbose:
        final_status = "cleared" if len(final_M) == len(sellers) else None
        dump_market_state(
            G,
            preferred_edges=pref_edges_final,
            matching=list(final_M),
            status=final_status,
        )
    sellers, buyers = get_parts(G)
    price_lookup = {s: int(G.nodes[s].get("price", 0)) for s in sellers}
    payoffs_matrix = []
    for b in buyers:
        row = []
        for s in sellers:
            if G.has_edge(s, b):
                val = int(G.edges[s, b]["valuation"])
                row.append(val - price_lookup[s])
            else:
                row.append(0)
        row = row
        payoffs_matrix.append(row)

    return final_prices, payoffs_matrix, list(final_M), buyers, pref_edges_final


# ======== interactive loop ========

def run_interactive_clearing(
    G,
    price_step: int = 1,
    ask_each_round: bool = True,
    max_rounds: int = 1000,
    plot: bool = True,
):
    """
    Runs the market-clearing loop:
      - build preferred graph
      - max matching
      - if not perfect: optionally show graph and ask user, raise prices, repeat.

    All prints happen here. We print tables each round via dump_market_state(G).
    """
    sellers, buyers = get_parts(G)
    prev_fig = None
    

    def _compute_payoffs():
        """Compute buyer payoffs (valuation - price)."""
        payoffs = []
        for b in buyers:
            row = []
            for s in sellers:
                if G.has_edge(s, b):
                    val = G.edges[s, b]["valuation"]
                    price = int(G.nodes[s].get("price", 0))
                    row.append(val - price)
                else:
                    row.append(0)
            payoffs.append(row)
        return payoffs

    current_energy = get_total_potential_energy(G)
    r = 0

    while current_energy > 0:
        P = build_preferred_graph(G)
        M = maximum_matching(P, sellers, buyers)
        r += 1
        if r >= max_rounds:
            dump_market_state(
                G,
                round_no=r,
                energy=current_energy,
                preferred_edges=list(P.edges()),
                matching=list(M),
                status="stopped",
            )
            return
        current_energy = sum(
            max(0, G.edges[s, b]["valuation"] - int(G.nodes[s].get("price", 0)))
            for s, b in G.edges()
        )

        constr_b, constr_s = _constricted_sets(P, M, sellers, buyers)

        dump_market_state(
            G,
            round_no=r,
            energy=current_energy,
            preferred_edges=list(P.edges()),
            matching=list(M),
            constricted_buyers=list(constr_b),
            constricted_sellers=list(constr_s),
        )

        if len(M) == len(sellers):
            # final dump with status
            dump_market_state(
                G,
                preferred_edges=list(P.edges()),
                matching=list(M),
                status="cleared",
            )
            if plot:
                if prev_fig is not None:
                    plt.close(prev_fig)
                payoffs = _compute_payoffs()
                prices = get_prices(G, sellers)
                draw_graph(
                    G,
                    title="Final Market Equilibrium",
                    highlight_edges=list(P.edges()),
                    highlight_edges_green=list(M),
                    seller_prices=[prices[s] for s in sellers],
                    buyer_payoffs=payoffs,
                )
                plt.ioff(); plt.show()
            return

        if plot:
            if prev_fig is not None:
                plt.close(prev_fig)
            payoffs = _compute_payoffs()
            prices = get_prices(G, sellers)
            prev_fig = draw_graph(
                G,
                title=f"Market Clearing – Round {r}",
                highlight_edges=list(P.edges()),
                seller_prices=[prices[s] for s in sellers],
                buyer_payoffs=payoffs,
            )

        if ask_each_round:
            try:
                resp = input("Proceed to next round? [Enter = yes, n/q = no] ").strip().lower()
            except EOFError:
                resp = ""
            if resp in {"q", "n", "no"}:
                dump_market_state(G, status="stopped")
                return

        if not constr_s:
            dump_market_state(
                G,
                round_no=r,
                energy=current_energy,
                preferred_edges=list(P.edges()),
                matching=list(M),
                constricted_buyers=list(constr_b),
                constricted_sellers=list(constr_s),
                status="stopped",
            )
            return

        for s in constr_s:
            G.nodes[s]["price"] = int(G.nodes[s].get("price", 0)) + int(price_step)
