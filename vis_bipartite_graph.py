# vis_bipartite_graph.py
import math
import networkx as nx
import matplotlib.pyplot as plt

# ---------- helpers: data & layout ----------

def get_parts(G):
    """Return (sellers, buyers) as sorted lists of node IDs (strings)."""
    sellers = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 0]
    buyers  = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 1]
    if not sellers or not buyers:
        nodes_sorted = sorted(G.nodes(), key=lambda x: int(x))
        n = len(nodes_sorted) // 2
        sellers = [str(i) for i in range(0, n)]
        buyers  = [str(i) for i in range(n, 2 * n)]
    return sorted(sellers, key=int), sorted(buyers, key=int)

def make_positions(sellers, buyers, x_left=0.0, x_right=1.0):
    """
    Place sellers at x_left (left column) and buyers at x_right (right column).
    Smaller IDs at the TOP (so we reverse for plotting).
    Returns: dict node->(x, y)
    """
    pos = {}
    for i, s in enumerate(reversed(sellers)):
        pos[s] = (x_left, i)
    for i, b in enumerate(reversed(buyers)):
        pos[b] = (x_right, i)
    return pos

def fmt_num(x):
    """Format int-like as int; otherwise 2 decimals. None -> '0' (for placeholders)."""
    if x is None:
        return "0"
    if isinstance(x, (int, float)) and math.isclose(x, round(x), rel_tol=0, abs_tol=1e-9):
        return str(int(round(x)))
    return f"{float(x):.2f}"

def collect_edge_labels(G):
    """Return a dict {(u,v): 'valuation'} for edges with 'valuation'."""
    labels = {}
    for u, v, data in G.edges(data=True):
        if "valuation" in data:
            labels[(u, v)] = fmt_num(data["valuation"])
    return labels

# ---------- helpers: drawing primitives ----------

def draw_nodes(ax, G, pos, sellers, buyers):
    nx.draw_networkx_nodes(G, pos, nodelist=sellers, node_color="#87CEFA",
                           node_shape="s", node_size=700, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=buyers, node_color="#90EE90",
                           node_shape="o", node_size=700, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={n: n for n in sellers}, font_size=9, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={n: n for n in buyers},  font_size=9, ax=ax)

def draw_edges(ax, G, pos, highlight_edges=None):
    nx.draw_networkx_edges(G, pos, edge_color="gray", width=1.2, ax=ax)
    if highlight_edges:
        nx.draw_networkx_edges(G, pos, edgelist=highlight_edges,
                               edge_color="crimson", width=2.5, style="dashed", ax=ax)

def draw_edge_valuations(ax, G, pos):
    labels = collect_edge_labels(G)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels,
                                 font_size=8, label_pos=0.75, rotate=False, ax=ax)

def draw_prices(ax, pos, sellers, G, prices_override=None, dx=-0.22):
    """
    Draw price text to the LEFT of each seller.
    If prices_override is provided, it must align with sellers order (list),
    or be a dict {seller_id: price}.
    """
    # normalize override to a list aligned with sellers
    if prices_override is None:
        aligned = [G.nodes[s].get("price", 0) for s in sellers]
    elif isinstance(prices_override, dict):
        aligned = [prices_override.get(s, 0) for s in sellers]
    else:
        if len(prices_override) != len(sellers):
            raise ValueError("Length of prices_override must match number of sellers.")
        aligned = list(prices_override)

    for s, p in zip(sellers, aligned):
        x, y = pos[s]
        ax.text(x + dx, y, fmt_num(p), fontsize=8, ha="right", va="center", color="black")


def draw_payoff_vectors(ax, pos, buyers, sellers, G=None,
                        payoff_provider=None, payoffs_override=None, dx=0.22):
    """
    Draw payoff text to the RIGHT of each buyer.

    Highlights the maximum payoff (ties included) in bold red.
    Accepts either:
      - payoffs_override: list aligned with buyers order OR dict {buyer_id: payoff_list}
      - payoff_provider: callable(buyer, sellers)
    """
    for i, b in enumerate(buyers):
        # Get payoff vector
        if payoffs_override is not None:
            if isinstance(payoffs_override, dict):
                vec = payoffs_override.get(b, [0]*len(sellers))
            else:
                vec = payoffs_override[i]
        elif payoff_provider is not None:
            vec = payoff_provider(b, sellers)
        else:
            vec = [0]*len(sellers)

        # Determine maxima
        if isinstance(vec, (list, tuple)) and vec:
            max_val = max(vec)
            highlighted_indices = [j for j, v in enumerate(vec) if v == max_val and max_val != 0]

            # Draw each payoff value inline with commas
            x, y = pos[b]
            offset = 0.0
            for j, v in enumerate(vec):
                s_val = fmt_num(v)
                is_highlight = j in highlighted_indices
                

                # draw number
                ax.text(x + dx + offset, y, s_val,
                        fontsize=8, ha="left", va="center",
                        color="crimson" if is_highlight else "black",
                        fontweight="bold" if is_highlight else "normal")

                # move right a little
                offset += 0.05

        else:
            # Single or empty payoff
            x, y = pos[b]
            ax.text(x + dx, y, fmt_num(vec), fontsize=8,
                    ha="left", va="center", color="black")

    

def draw_headers(ax, pos, x_left=0.0, x_right=1.0,
                 label_price="Price", label_sellers="Sellers",
                 label_buyers="Buyers", label_payoffs="Payoffs",
                 price_dx=-0.22, payoffs_dx=0.22):
    """Draw column headers above the top row."""
    y_values = [y for _, y in pos.values()]
    y_top = max(y_values) + 0.7
    ax.text(x_left + price_dx, y_top, label_price,   fontsize=11, fontweight="bold",
            ha="right", va="bottom", color="black")
    ax.text(x_left,           y_top, label_sellers, fontsize=11, fontweight="bold",
            ha="center", va="bottom", color="black")
    ax.text(x_right,          y_top, label_buyers,  fontsize=11, fontweight="bold",
            ha="center", va="bottom", color="black")
    ax.text(x_right + payoffs_dx, y_top, label_payoffs, fontsize=11, fontweight="bold",
            ha="left", va="bottom", color="black")


def pad_axes(ax, pos, x_left=0.0, x_right=1.0, pad_x=0.35, pad_top=1.2, pad_bottom=0.5):
    """Pad x/y limits so side annotations aren’t clipped."""
    y_vals = [y for _, y in pos.values()]
    ax.set_xlim(x_left - pad_x, x_right + pad_x)
    ax.set_ylim(min(y_vals) - pad_bottom, max(y_vals) + pad_top)

def _normalize_prices(sellers, seller_prices):
    """
    Return a list of prices aligned to `sellers` order without mutating G.
    Accepts either:
      - list of prices (same length as sellers, already ordered), or
      - dict {seller_id: price}
    """
    if seller_prices is None:
        return None
    if isinstance(seller_prices, dict):
        return [seller_prices.get(s, 0) for s in sellers]
    # assume list/tuple
    if len(seller_prices) != len(sellers):
        raise ValueError("Length of seller_prices must match number of sellers.")
    return list(seller_prices)


# ---------- public API ----------

def draw_graph(G, title="Bipartite Market Graph", highlight_edges=None,
               payoff_provider=None, seller_prices=None, buyer_payoffs=None):
    """
    Draw a bipartite market graph with side prices (left) and payoff vectors (right).

    Parameters
    ----------
    G : networkx.Graph
    title : str
    highlight_edges : list[tuple], optional
        Edges to highlight (e.g., preferred or matched).
    payoff_provider : callable(buyer:str, sellers:list[str]) -> list
        Returns payoff vector for a buyer aligned with sellers. If None, zeros are used.
    seller_prices: list|dict|None
        If list, it must align with the visual seller order.
        If dict, keys are seller IDs; values are prices.
        This does NOT mutate G; it's just for display.
    buyer_prices: 2D array|None
        Must be a 2D array with rows for buyers (in order) and col for sellers (in order)
    
    """
    sellers, buyers = get_parts(G)
    pos = make_positions(sellers, buyers, x_left=0.0, x_right=1.0)

    # normalize input prices
    prices_aligned = _normalize_prices(sellers, seller_prices)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)
    fig.subplots_adjust(top=0.90)

    draw_nodes(ax, G, pos, sellers, buyers)
    draw_edges(ax, G, pos, highlight_edges=highlight_edges)
    draw_edge_valuations(ax, G, pos)

    # left side: seller prices
    draw_prices(ax, pos, sellers, G=G, prices_override=prices_aligned, dx=-0.22)

    # right side: buyer payoffs
    draw_payoff_vectors(
        ax, pos, buyers, sellers,
        G=G,
        payoff_provider=payoff_provider,
        payoffs_override=buyer_payoffs,  # <-- this line fixes it
        dx=0.22
    )

    draw_headers(ax, pos, x_left=0.0, x_right=1.0,
                 label_price="Price", label_sellers="Sellers",
                 label_buyers="Buyers", label_payoffs="Payoffs",
                 price_dx=-0.22, payoffs_dx=0.22)

    pad_axes(ax, pos, x_left=0.0, x_right=1.0, pad_x=0.35, pad_top=1.2, pad_bottom=0.5)
    ax.axis("off")
    plt.tight_layout()
    plt.show()
