
def shift_reduce_constituency_forest(node_and_parent_idx_pairs):

    def get_idx_node_parent_pair(node_1base_idx):
        for idx_input,(node_idx,parent_idx) in enumerate(node_and_parent_idx_pairs):
            if node_idx == node_1base_idx:
                return idx_input,node_idx,parent_idx,
        raise RuntimeError( 'cannot find the node %d in node_and_parent_idx_pairs %s'\
               %(node_1base_idx,str(node_and_parent_idx_pairs)))



    node_num = len(node_and_parent_idx_pairs)
    root_node_num = sum([1 for _,p in node_and_parent_idx_pairs if p == 0])
    shifted = [0] * node_num
    children = []
    parents = []
    used = []  # 0 for un-used, 1 for used
    op_stack = []  # to restore the operation as the output

    while True:
        # check enough 0
        now_root_num = sum([1 for p in parents if p == 0])
        if now_root_num == root_node_num:
            break

        # check whether reduce
        do_reduce = False
        try:
            last_idx = parents[-1]
            now_children_num = sum([1 for u, p_idx in zip(used, parents) if u == 0 and p_idx == last_idx])
            fact_children_num = sum([1 for _, p_idx in node_and_parent_idx_pairs if p_idx == last_idx])
            if now_children_num == fact_children_num: do_reduce = True
            #if last_idx == 0:do_reduce = False#FIXME :???WHY
        except IndexError:
            pass # len(parents) == 0
        # reduce or shift
        if do_reduce:
            reduce_idx = parents[-1] # reduce for reduce_idx node
            reduce_parent_idx = get_idx_node_parent_pair(reduce_idx)[2]  # its parent node index
            # mark used and collect reduce_idxs
            reduce_idxs = []
            for idx_n,p in enumerate(parents): # idx_n: index in stack mat; p:corresponding parent node idx
                if used[idx_n] == 0 and p == reduce_idx:
                    used[idx_n] = 1
                    reduce_idxs.append(idx_n)

            shifted[get_idx_node_parent_pair(reduce_idx)[0]] = 1
            children.append(reduce_idx)
            parents.append(reduce_parent_idx)
            used.append(0)
            op_stack.append((2, reduce_idx, reduce_idxs))


        else: # do shift
            # get pointer
            pointer = 0
            for idx_s in range(node_num):
                if shifted[idx_s] == 0:
                    pointer = idx_s
                    shifted[idx_s] = 1
                    break
            children.append(node_and_parent_idx_pairs[pointer][0])
            parents.append(node_and_parent_idx_pairs[pointer][1])
            used.append(0)
            # op_stack.append(-child_and_parent_idx[pointer][0])
            op_stack.append((1, node_and_parent_idx_pairs[pointer][0], []))
    assert len(op_stack) == len(node_and_parent_idx_pairs)
    return op_stack


def shift_reduce_constitucy(parent_idx_seq):
    '''
    This file is implemented to solve: tree sequence to constituency transition sequence:

    input is a list of father node idx of constituency tree (1 based)
    output is a list of element which is a one of 
    [
        1. 1 for reduce and 2 for shift
        2. 1 based node index
        2. a list of 0-based index to reduce 
    ]
    @ author: xx
    @ Email: xx@xxx
    '''
    node_num = len(parent_idx_seq) # all node number in the parsing tree
    child_and_parent_idx = [(child_idx+1, parent_idx) # list of (child_idx, parent_idx) pair
                            for child_idx,parent_idx in enumerate(parent_idx_seq)]
    # runtime variable:
    shifted = [0] * node_num # 0 for shifted and 0 for un-shifted
    children = []
    parents = []
    used = [] # 0 for un-used, 1 for used
    op_stack = [] # to restore the operation as the output

    while True:
        # check whether reduce
        do_reduce = False
        try:
            last_idx = parents[-1]
            # count
            count=sum([1 for u,p_idx in zip(used,parents) if u==0 and p_idx==last_idx ])
            # check if count satisfy the number of 'last_idx''s children num
            children_num = sum([1 for _,p_idx in child_and_parent_idx if p_idx==last_idx])
            if count == children_num:
                do_reduce = True
        except IndexError:
            pass # len(parents) == 0
        # reduce or shift
        if do_reduce:
            reduce_idx = parents[-1]
            reduce_parent_idx = child_and_parent_idx[reduce_idx-1][1]
            # mark used
            reduce_idxs = []
            for idx_n,p in enumerate(parents):
                if used[idx_n] == 0 and p == reduce_idx:
                    used[idx_n] = 1
                    reduce_idxs.append(idx_n)

            shifted[reduce_idx-1] = 1
            children.append(reduce_idx)
            parents.append(reduce_parent_idx)
            used.append(0)
            op_stack.append((2,reduce_idx,reduce_idxs))
            if reduce_parent_idx == 0:
                break
        else: # do shift
            # get pointer
            pointer = 0
            for idx_s in range(node_num):
                if shifted[idx_s] == 0:
                    pointer = idx_s
                    shifted[idx_s] = 1
                    break

            children.append(child_and_parent_idx[pointer][0])
            parents.append(child_and_parent_idx[pointer][1])
            used.append(0)
            #op_stack.append(-child_and_parent_idx[pointer][0])
            op_stack.append((1,child_and_parent_idx[pointer][0],[] ))

    return op_stack




if __name__ == '__main__':
    seq_str = '19 19 22 23 24 25 26 27 27 28 30 31 31 32 33 34 35 35 20 21 0 20 22 23 24 25 28 29 26 29 30 21 32 33 34'
    seq = list(map(int,seq_str.split(' ')))

    print(shift_reduce_constitucy(seq))


