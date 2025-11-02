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

def draw_prices(ax, pos, sellers, G, dx=-0.22):
    """Draw price text to the LEFT of each seller."""
    for s in sellers:
        price = fmt_num(G.nodes[s].get("price", 0))
        x, y = pos[s]
        ax.text(x + dx, y, price, fontsize=8, ha="right", va="center", color="black")

def draw_payoff_vectors(ax, pos, buyers, sellers, payoff_provider, dx=0.22):
    """
    Draw payoff vector (comma-separated) to the RIGHT of each buyer.
    payoff_provider(buyer, sellers_list) -> list of numbers/strings.
    """
    for b in buyers:
        vec = payoff_provider(b, sellers)
        text = ", ".join(fmt_num(v) for v in vec)
        x, y = pos[b]
        ax.text(x + dx, y, text, fontsize=8, ha="left", va="center", color="black")

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

# ---------- public API ----------

def draw_graph(G, title="Bipartite Market Graph", highlight_edges=None,
               payoff_provider=None):
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
    """
    sellers, buyers = get_parts(G)
    pos = make_positions(sellers, buyers, x_left=0.0, x_right=1.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)
    fig.subplots_adjust(top=0.90)

    draw_nodes(ax, G, pos, sellers, buyers)
    draw_edges(ax, G, pos, highlight_edges=highlight_edges)
    draw_edge_valuations(ax, G, pos)

    # left: prices; right: payoff vectors
    draw_prices(ax, pos, sellers, G, dx=-0.22)

    if payoff_provider is None:
        # default: zero vector with length = #sellers
        payoff_provider = lambda b, S: [0] * len(S)
    draw_payoff_vectors(ax, pos, buyers, sellers, payoff_provider, dx=0.22)

    draw_headers(ax, pos, x_left=0.0, x_right=1.0,
                 label_price="Price", label_sellers="Sellers",
                 label_buyers="Buyers", label_payoffs="Payoffs",
                 price_dx=-0.22, payoffs_dx=0.22)

    pad_axes(ax, pos, x_left=0.0, x_right=1.0, pad_x=0.35, pad_top=1.2, pad_bottom=0.5)

    ax.axis("off")
    plt.tight_layout()
    plt.show()
