"""
draw_graph.py  —  Print the graph exactly as LangGraph sees it.

    ./venv/bin/python draw_graph.py            mermaid (paste into a README)
    ./venv/bin/python draw_graph.py ascii      terminal diagram
    ./venv/bin/python draw_graph.py edges      plain list of nodes and edges
    ./venv/bin/python draw_graph.py png        render to output/graph.png

This is generated from the compiled graph, not hand-drawn, so it can never
drift out of date with build_graph().
"""
import sys

import agent


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "mermaid"
    graph = agent.build_graph().get_graph()

    if mode == "mermaid":
        print(graph.draw_mermaid())

    elif mode == "ascii":
        print(graph.draw_ascii())

    elif mode == "edges":
        nodes = [n for n in graph.nodes if not n.startswith("__")]
        print(f"{len(nodes)} nodes:")
        for n in nodes:
            print(f"    {n}")
        print(f"\n{len(graph.edges)} edges:")
        for e in graph.edges:
            label = f"  [{e.data}]" if getattr(e, "data", None) else ""
            arrow = "-.->" if getattr(e, "conditional", False) else "--->"
            print(f"    {e.source:<22}{arrow} {e.target}{label}")

    elif mode == "png":
        import helpers as H
        path = H.OUTPUT_DIR / "graph.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(graph.draw_mermaid_png())
        print(f"wrote {path}")

    else:
        print(__doc__)


if __name__ == "__main__":
    main()
