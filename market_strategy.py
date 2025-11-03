import networkx as nx
import argparse
from vis_bipartite_graph import draw_graph
from market_clearing_alg import  run_interactive_clearing
import matplotlib.pyplot as plt

# File Handeling Funcitons
# ====================================================================================================
"""To take in a graph .gml file, and check that the input ID values are ints
Input: either a local file with just the name, or the file's entire path
Output: the graph that corresponds to the the file name
"""
def load_gml(path: str):
    print(f"load_gml called with path={path}")
    try:
        G = nx.read_gml(path)
    except Exception as e:
        raise nx.NetworkXError(f"Failed to read GML file: {e}")
    
    #Normalize node labels to strings
    G = nx.relabel_nodes(G, lambda n: str(n), copy=True)
    
    #Check node attributes
    for node, data in G.nodes(data=True):
        for key, value in data.items():
            try:
                float(value)
            except Exception:
                raise nx.NetworkXError(f"Node {node} has non-numeric attribute '{key}': {value}")
            
    #Check edge attributes
    for u, v, data in G.edges(data=True):
        for key, value in data.items():
            try:
                float(value)
            except Exception:
                raise nx.NetworkXError(f"Edge ({u},{v}) has non-numeric attribute '{key}': {value}")
            
    return G


# Functions Outlined from Directions
# ====================================================================================================

#def plot(G, edge_dict, total_time, total_potential_engergy, title, ax):
    #Import from vis_direct_graph.py
    #draw_graph(G, edge_dict, total_time, total_potential_engergy, title, ax)


# Arg Parser
# ====================================================================================================
"""To take all the arguments in the command line, save relevant information needed to compute the functions, and make some checks that it follows the input instructions
"""
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Traffic Analysis")
    
    parser.add_argument(
        "input",
        help="Path to input file"
    )
    
    parser.add_argument(
        "--plot",
        action="store_true",
        help="VPlot the bipartite graph and per-round preference graph."
    )
    
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="Show step-by-step output for each round."
    )

    return parser

# Main
# ====================================================================================================
"""Builds the parser, and calls the functions that correspond to said argument. It also checks for additional erros such as requiring a .gml file, ensuring that the input .gml file is properly made, the node ids are digits and within range, etc.
"""
def main():
    #To figure out the different possible errors: https://pelegm-networkx.readthedocs.io/en/latest/_modules/networkx/readwrite/gml.html
    #NetworkXError:
    #ValueError:
    #   didn't apply a stringizer or destringizer when writting to the gml file pulling from
    #NetworkXError:
    #   input can't be parsed(value is not string with stringizer is None)
    #UnicodeDecodeError:
    #   type of NetworkXError where input isn't ASCII-encoded

    #Into regular calls (check that it is a .gml file)
    
    parser = build_parser()
    args = parser.parse_args()

    G = None            # local graph variable -> all functions

    #Ensure the paths (input and output end with .gml)
    if args.input and not args.input.endswith(".gml"):
        parser.error("Input file must be a .gml file")

     #Ensure the file exists
    from pathlib import Path
    if not Path(args.input).is_file():
        parser.error(f"input file not found: {args.input}")
        
    try:
        G = load_gml(args.input)
    except (nx.NetworkXError, ValueError, UnicodeDecodeError) as err:
        parser.error(f"Error reading {args.input}: {err}")

    # Check if graph has nodes
    if G.number_of_nodes() == 0:
        parser.error("graph has no nodes")
    if G.number_of_edges() == 0:
        parser.error("graph has no edges")
    
    if args.plot:
        print("\nPlotting bipartite graph...")
        draw_graph(G, title="Initial Bipartite Market Graph")
    
    if args.interactive:
    # one call does everything: prints, prompts, price updates, recompute loop
        run_interactive_clearing(
            G,
            price_step=1.0,
            ask_each_round=True,
            max_rounds=200,
            plot=args.plot,   # will highlight preferred edges each round if --plot
        )
        
if __name__ == "__main__":
    main()