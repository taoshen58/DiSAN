

class TreeNode(object):
    def __init__(self, is_leaf, tag=None, token=None):
        self.tag = tag
        self.is_leaf = is_leaf
        self.token = token
        self.children_nodes = []
        # for transformation
        self.parent_index = None
        self.node_index = None
        self.leaf_node_index_seq = []


def recursive_build_penn_format(seq):
    if seq[0] == '(' and seq[-1] == ')' and len(seq[1:-1]) > 2:
        node = TreeNode(False, tag=seq[1])
        children_seqs = []
        children_seq = []
        counter = 0
        for token in seq[2:-1]:
            children_seq.append(token)
            if token == '(':  counter += 1
            elif token == ')':  counter -= 1
            if counter == 0:
                children_seqs.append(children_seq)
                children_seq = []
        node.children_nodes = [recursive_build_penn_format(children_seq) for children_seq in children_seqs]
        return node
    else:
        new_seq = seq[1:-1]
        assert len(new_seq) == 2, seq
        node = TreeNode(True, tag=new_seq[0], token=new_seq[1])
        return node


def recursive_build_binary(seq):
    if seq[0] == '(' and seq[-1] == ')' and len(seq):
        node = TreeNode(is_leaf=False)
        children_seqs = []
        children_seq = []
        counter = 0
        for token in seq[1:-1]:
            children_seq.append(token)
            if token == '(':
                counter += 1
            elif token == ')':
                counter -= 1
            if counter == 0:
                children_seqs.append(children_seq)
                children_seq = []
        node.children_nodes = [recursive_build_binary(children_seq) for children_seq in children_seqs]
        return node
    else:
        assert len(seq) == 1, seq
        node = TreeNode(is_leaf=True, token=seq[0])
        return node


def check_tree(tree, layer):
    if len(tree.children_nodes) > 0:
        now_str = '%snon_leaf: %s:%s, %s:%s\n' % \
                  ('\t'* layer, tree.tag, tree.token, tree.node_index, tree.parent_index)
        s = ''.join([check_tree(node, layer+1) for node in tree.children_nodes])
        return now_str + s
    else:
        return '%sleaf: %s:%s, %s:%s\n' % ('\t'* layer, tree.tag, tree.token, tree.node_index, tree.parent_index)


def tokenize_str_format_tree(tree_str):

    # 1. spilt by ' '
    raw_token_list = tree_str.split(' ')
    # 2. split when find  '(' or ')'
    token_list = []
    for token in raw_token_list:
        new_token_list = []
        idx_in_token = 0
        for idx_char, char in enumerate(token):
            if char == '(' or char == ')':
                if idx_char > idx_in_token:
                    new_token_list.append(token[idx_in_token: idx_char])
                new_token_list.append(char)
                idx_in_token = idx_char + 1
        if idx_in_token < len(token):
            new_token_list.append(token[idx_in_token:])
        token_list += new_token_list
    return token_list





