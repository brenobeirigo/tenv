import matplotlib.pyplot as plt
import osmnx as ox
import os


def plot_region_neighbors(
    G,
    region_centers,
    sorted_neighbors,
    path=None,
    show=False,
    file_format="png",
    max_neighbors=1000,
    replace=False,
):
    """Plot all region centers and associated nodes

    Parameters
    ----------
    G : networkx
        Transportation network
    region_centers : dict
        Region center ids for each max. trip duration
    sorted_neighbors : dict
        for each max. duration, the region center id and list of
        (id, distance) tuples.
    path : str, optional
        Target folder, by default None
    show : bool, optional
        If True, show plot, by default False
    file_format : str, optional
        Plot file format (e.g., pdf, png, svg), by default "png"
    max_neighbors: int
        Max. number of closest neighbors
    """

    nodes = list(G.nodes())

    map_features = {
        "regular_node": {
            "color": "#999999",
            "size": 0.5,
            "edge_color": "#999999",
        },
        "region_center": {
            "color": "#FFFFFF",
            "size": 15,
            "edge_color": "#FF0000",
        },
    }
    # Loop max. reachable distances of region centers
    for max_dist, centers in region_centers.items():
        if path:
            filename = (
                f"{path}/"
                f"max_dist_{max_dist:03}_"
                f"neighbors_{max_neighbors:03}.{file_format}"
            )

            # If file exists and should not be replaced, skip
            if os.path.isfile(filename) and not replace:
                continue

        all_paths = list()
        # List of ods to plot lines from centers to reachable nodes
        all_tuples = list()
        # Set of centers associated to nodes
        centers_from_nodes = set()

        # Color and size of regular nodes in map
        node_color = [map_features["regular_node"]["color"] for n in G.nodes()]
        node_size = [map_features["regular_node"]["size"] for n in G.nodes()]
        edge_color = [
            map_features["regular_node"]["edge_color"] for n in G.nodes()
        ]
        for o in centers:
            if max_dist not in sorted_neighbors:
                continue

            for i, n in enumerate(sorted_neighbors[max_dist][o]):

                # Does not create self edge
                if o == n:
                    continue

                # Only query a maximum number of neighbors
                if i > max_neighbors:
                    break

                # Change region center features
                node_color[nodes.index(o)] = map_features["region_center"][
                    "color"
                ]
                node_size[nodes.index(o)] = map_features["region_center"][
                    "size"
                ]
                edge_color[nodes.index(o)] = map_features["region_center"][
                    "edge_color"
                ]

                all_paths.append([o, n])
                all_tuples.append(
                    (
                        (G.nodes[o]["x"], G.nodes[o]["y"]),
                        (G.nodes[n]["x"], G.nodes[n]["y"]),
                    )
                )
                centers_from_nodes.add(o)

        # TODO Warning! Plot graph routes was modified to support line
        # plotting
        fig, ax = ox.plot_graph_routes(
            G,
            all_paths,
            use_geom=False,
            orig_dest_points=all_tuples,
            route_linewidth=5,
            edge_linewidth=0.5,
            node_color=node_color,
            fig_height=10,
            node_edgecolor=edge_color,
            node_zorder=1000,
            node_size=node_size,
            route_alpha=0.5,
            save=False,
            show=False,
            close=False,
        )

        # print(
        #     f"{centers}({len(centers)}) "
        #     f"\n {centers_from_nodes}({len(centers)})"
        # )

        ax.title.set_text(
            f"Reachable from centers in {(max_dist/60):.1f} min\n"
            f"(Max. neighbors = {max_neighbors})"
        )

        if path:
            fig.savefig(filename, dpi=300, bbox_inches="tight")

        if show:
            plt.show()


def plot_regions(
    G,
    region_centers,
    region_id_dict,
    path=None,
    show=False,
    file_format="png",
    replace=False,
):
    """Plot all region centers and associated nodes

    Parameters
    ----------
    G : networkx
        Transportation network
    region_centers : dict
        Region center ids for each max. trip duration
    region_id_dict : dict
        Region id of each node considering max. trip durations
    path : str, optional
        Target folder, by default None
    show : bool, optional
        If True, show plot, by default False
    file_format : str, optional
        Plot file format (e.g., pdf, png, svg), by default "png"
    """

    nodes = list(G.nodes())

    map_features = {
        "regular_node": {
            "color": "#999999",
            "size": 0.5,
            "edge_color": "#999999",
        },
        "region_center": {
            "color": "#FFFFFF",
            "size": 15,
            "edge_color": "#FF0000",
        },
    }
    # Loop max. reachable distances of region centers
    for max_dist, centers in region_centers.items():

        n_centers = len(region_centers[max_dist])

        if path:

            filename = (
                f"{path}/"
                f"max_dist_{max_dist:03}_centers_{n_centers:03}.{file_format}"
            )

            # If file exists and should not be replaced, skip
            if os.path.isfile(filename) and not replace:
                continue

        all_paths = list()
        # List of ods to plot lines from centers to reachable nodes
        all_tuples = list()
        # Set of centers associated to nodes
        centers_from_nodes = set()

        # Color and size of regular nodes in map
        node_color = [map_features["regular_node"]["color"] for n in G.nodes()]
        node_size = [map_features["regular_node"]["size"] for n in G.nodes()]
        edge_color = [
            map_features["regular_node"]["edge_color"] for n in G.nodes()
        ]

        for n in nodes:
            # The closest region center to n whitin distance "max_dist"
            o = region_id_dict[n][max_dist]

            # Change region center features
            node_color[nodes.index(o)] = map_features["region_center"]["color"]
            node_size[nodes.index(o)] = map_features["region_center"]["size"]
            edge_color[nodes.index(o)] = map_features["region_center"][
                "edge_color"
            ]

            all_paths.append([o, n])
            all_tuples.append(
                (
                    (G.nodes[o]["x"], G.nodes[o]["y"]),
                    (G.nodes[n]["x"], G.nodes[n]["y"]),
                )
            )
            centers_from_nodes.add(o)

        # TODO Warning! Plot graph routes was modified to support line
        # plotting
        fig, ax = ox.plot_graph_routes(
            G,
            all_paths,
            use_geom=False,
            orig_dest_points=all_tuples,
            route_linewidth=0.5,
            edge_linewidth=0.5,
            node_color=node_color,
            fig_height=10,
            node_edgecolor=edge_color,
            node_zorder=1000,
            node_size=node_size,
            route_alpha=0.3,
            save=False,
            show=False,
            close=False,
        )

        # print(
        #     f"{centers}({len(centers)}) "
        #     f"\n {centers_from_nodes}({len(centers)})"
        # )

        ax.title.set_text(
            f"Reachable from centers in {(max_dist/60):>4.1f} min\n"
            f"(#centers = {n_centers:>3})"
        )

        if path:
            fig.savefig(filename, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
