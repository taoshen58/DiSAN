
def transform_tree_to_parent_index(tree_structure):
    # 0. get parent node starting index (how many token this tree have)
    def recursive_count_leaf_node(tree):
        if len(tree.children_nodes) > 0:
            leaf_number = sum([recursive_count_leaf_node(node) for node in tree.children_nodes])
            return leaf_number
        else:
            return 1
    parent_node_start_index = recursive_count_leaf_node(tree_structure) + 1

    # 1. assign index for leaf node and non-leaf node separately
    def recursive_assign_index_for_tree(tree, patent_index, non_leaf_index, leaf_index):
        if len(tree.children_nodes) > 0: # non-leaf
            # non-leaf node
            tree.node_index = non_leaf_index
            tree.parent_index = patent_index
            # for its children nodes
            now_non_leaf_index = non_leaf_index + 1
            now_leaf_node_index = leaf_index
            for child_node in tree.children_nodes:
                now_non_leaf_index,now_leaf_node_index = recursive_assign_index_for_tree(
                    child_node, tree.node_index, now_non_leaf_index, now_leaf_node_index)
            return now_non_leaf_index, now_leaf_node_index
        else:
            # leaf node
            tree.node_index = leaf_index
            tree.parent_index = patent_index
            return non_leaf_index, leaf_index + 1

    recursive_assign_index_for_tree(tree_structure, 0, parent_node_start_index, 1)

    # 2. get leaf_node_index_seq for all_nodes
    def recursive_gene_leaf_indices(tree):
        if len(tree.children_nodes) > 0:  # non-leaf
            tree.leaf_node_index_seq = []
            for child_node in tree.children_nodes:
                tree.leaf_node_index_seq += recursive_gene_leaf_indices(child_node)
            return tree.leaf_node_index_seq
        else:
            tree.leaf_node_index_seq = [tree.node_index]
            return tree.leaf_node_index_seq
    recursive_gene_leaf_indices(tree_structure)

    # 3. get all node as list
    def recursive_get_all_nodes(tree):
        if len(tree.children_nodes) > 0:  # non-leaf
            nodes = [tree]
            for child_node in tree.children_nodes:
                nodes += recursive_get_all_nodes(child_node)
            return nodes
        else:
            return [tree]
    all_nodes = recursive_get_all_nodes(tree_structure)

    # 4. sort for all nodes
    all_nodes_sorted = list(sorted(all_nodes, key=lambda node: node.node_index))



    return tree_structure, all_nodes_sorted








