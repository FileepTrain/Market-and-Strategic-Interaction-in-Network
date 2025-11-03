import math
import networkx as nx
import matplotlib as plt
from mo_alter_vis_bi import draw_graph


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
    """Format numbers compactly; None -> '-'."""
    if x is None:
        return "-"
    if isinstance(x, (int, float)) and math.isclose(x, round(x), abs_tol=1e-9):
        return f"{int(round(x))}"
    return f"{float(x):.2f}"

def edge_valuation(G, b, s):
    """Return valuation on edge (b,s) (float) or None if no edge/attr."""
    data = G.get_edge_data(b, s) or G.get_edge_data(s, b)
    if not data or "valuation" not in data:
        return None
    return float(data["valuation"])

def get_prices(G, sellers):
    """Return dict seller->price (float, default 0)."""
    return {s: float(G.nodes[s].get("price", 0)) for s in sellers}

def build_valuation_matrix(G, buyers, sellers):
    """
    Returns list of rows: [(buyer, [v(b,s) or None for s in sellers]), ...]
    """
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
        pays = [None if v is None else (v - prices[sellers[j]]) for j, v in enumerate(vals)]
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

def dump_market_state(G):
    """
    Terminal output:
      - prices per seller
      - valuation matrix
      - payoff matrix (valuation - price)
    """
    sellers, buyers = get_parts(G)
    prices = get_prices(G, sellers)
    val_rows = build_valuation_matrix(G, buyers, sellers)
    pay_rows = build_payoff_matrix(prices, val_rows, sellers)

    print("\n== Seller Prices ==")
    for s in sellers:
        print(f"Seller {s}: price = {fmt(prices[s])}")

    print_matrix("Valuations (buyer × seller)", buyers, sellers, val_rows)
    print_matrix("Payoffs = valuation − price(seller)", buyers, sellers, pay_rows)

# ---------- utilities for preferred graph / matching / constricted set ----------

def _utility(G, b, s):
    """u(b,s) = valuation(b,s) - price(s); None if edge missing."""
    data = G.get_edge_data(b, s) or G.get_edge_data(s, b)
    if not data or "valuation" not in data:
        return None
    val = float(data["valuation"])
    price = float(G.nodes[s].get("price", 0))
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
            'best_utility': float            # u(b,s) for any s in best_sellers
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
        'best_options': dict[str, {'best_sellers': [...], 'best_utility': float or None}]
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

# ======== NEW: constricted sets (buyers + sellers) ========

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


# ======== NEW: pretty print for each round ========

def _print_round(r, sellers, buyers, prices, matching, constr_b, constr_s, best_opts):
    print(f"\n--- Round {r} ---")
    print("Prices:", {s: prices[s] for s in sellers})
    print(f"Matching size: {len(matching)} / {len(sellers)}")
    if constr_b:
        b_sorted = sorted(constr_b, key=int)
        print("Constricted buyers:", b_sorted)
        for b in b_sorted:
            info = best_opts.get(b, {'best_sellers': [], 'best_utility': None})
            print(f"  Buyer {b}: best sellers = {info['best_sellers']}, best utility = {info['best_utility']}")
    if constr_s:
        print("Constricted sellers:", sorted(constr_s, key=int))


#Returns the perfect graph (not interactive)
def run_clearing_until_equilibrium(
    G,
    price_step: float = 1.0,
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

    def _best_options_for(b, P):
        best_sellers = sorted(list(P.neighbors(b)), key=int)
        u = _utility(G, b, best_sellers[0]) if best_sellers else None
        return {'best_sellers': best_sellers, 'best_utility': u}

    for r in range(1, max_rounds + 1):
        P = build_preferred_graph(G)
        M = maximum_matching(P, sellers, buyers)

        #if verbose:
            #print(f"\n--- Round {r} ---")
            #print(f"Matching pairs: {sorted(list(M))}")

        if len(M) == len(sellers):
            #if verbose:
                #print("Market cleared ✅")
            break  # done

        constr_b, constr_s = _constricted_sets(P, M, sellers, buyers)
        
        if not constr_s:
            if verbose:
                print("No constricted sellers identified; stopping.")
            break

        # Raise prices on constricted sellers
        for s in constr_s:
            G.nodes[s]["price"] = float(G.nodes[s].get("price", 0.0)) + float(price_step)

    # --- Compute final outputs ---
    final_prices = [float(G.nodes[s].get("price", 0.0)) for s in sellers]
    price_dict = dict(zip(sellers, final_prices))  # for quick lookup

    # Buyer payoffs matrix (valuation - final price)
    payoffs_matrix = []
    for b in buyers:
        row = []
        for s in sellers:
            if G.has_edge(s, b):
                val = G.edges[s, b]["valuation"]
                payoff = val - price_dict[s]
            else:
                payoff = 0
            row.append(payoff)
        payoffs_matrix.append(row)

    if verbose:
        print("\nFinal Prices:", dict(zip(sellers, final_prices)))
        print("Final Payoffs (buyers × sellers):")
        for b, row in zip(buyers, payoffs_matrix):
            print(f"  Buyer {b}: {row}")

    #Returns a list of the final prices, the 2D matrix of buyer payoffs, and the list of edges for a perfect match
    return final_prices, payoffs_matrix, list(M)

# ======== NEW: interactive loop ========

def run_interactive_clearing(
    G,
    price_step: float = 1.0,
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

    def _best_options_for(b, P):
        # neighbors in P are exactly b's best sellers (ties allowed)
        best_sellers = sorted(list(P.neighbors(b)), key=int)
        u = _utility(G, b, best_sellers[0]) if best_sellers else None
        return {"best_sellers": best_sellers, "best_utility": u}

    def _compute_payoffs():
        """Compute buyer payoffs (valuation - price)."""
        payoffs = []
        for b in buyers:
            row = []
            for s in sellers:
                if G.has_edge(s, b):
                    val = G.edges[s, b]["valuation"]
                    price = float(G.nodes[s].get("price", 0.0))
                    row.append(val - price)
                else:
                    row.append(0)
            payoffs.append(row)
        return payoffs

    for r in range(1, max_rounds + 1):
        # Preferred graph + matching
        P = build_preferred_graph(G)
        M = maximum_matching(P, sellers, buyers)

        # Show current state (prices, valuation & payoff matrices)
        dump_market_state(G)

        # Also show current preferred edges and matching summary
        print(f"\n--- Round {r} ---")
        prices = get_prices(G, sellers)
        print("Prices:", {s: prices[s] for s in sellers})
        print(f"Preferred edges: {sorted(list(P.edges()))}")
        print(f"Matching size: {len(M)} / {len(sellers)}")
        print(f"Matching pairs: {sorted(list(M))}")

        # Plot graph if requested
        if plot:
            payoffs = _compute_payoffs()
            draw_graph(
                G,
                title=f"Market Clearing – Round {r}",
                highlight_edges=list(M),
                seller_prices=[prices[s] for s in sellers],
                buyer_payoffs=payoffs,
            )

        # Perfect?
        if len(M) == len(sellers):
            print("Market cleared ✅")
            # Draw final graph one last time
            if plot:
                payoffs = _compute_payoffs()
                draw_graph(
                    G,
                    title="Final Market Equilibrium",
                    highlight_edges=list(M),
                    seller_prices=[prices[s] for s in sellers],
                    buyer_payoffs=payoffs,
                )
            return

        # Not perfect → compute constricted sets
        constr_b, constr_s = _constricted_sets(P, M, sellers, buyers)
        best_opts = {b: _best_options_for(b, P) for b in constr_b}

        # Print constricted info
        if constr_b:
            b_sorted = sorted(constr_b, key=int)
            print("Constricted buyers:", b_sorted)
            for b in b_sorted:
                info = best_opts.get(b, {"best_sellers": [], "best_utility": None})
                print(
                    f"  Buyer {b}: best sellers = {info['best_sellers']}, "
                    f"best utility = {info['best_utility']}"
                )
        if constr_s:
            print("Constricted sellers:", sorted(constr_s, key=int))

        # Ask user before continuing
        if ask_each_round:
            try:
                resp = input("Proceed to next round? [Enter = yes, q = quit] ").strip().lower()
            except EOFError:
                resp = ""  # auto-continue if input not available
            if resp in {"q", "n", "no"}:
                print("Stopped by user.")
                return

        # Raise prices on constricted sellers
        if not constr_s:
            print("No constricted sellers identified; stopping to avoid a loop.")
            return

        for s in constr_s:
            G.nodes[s]["price"] = float(G.nodes[s].get("price", 0.0)) + float(price_step)
