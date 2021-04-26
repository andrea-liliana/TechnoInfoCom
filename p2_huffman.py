import heapq
from io import StringIO


class Node:
    def __init__(self, left_child=None, right_child=None, weight=None, symbol=None):
        self.left_child = left_child
        self.right_child = right_child

        if self.has_both_children():
            assert weight is None and symbol is None
            self.weight = self.left_child.weight + self.right_child.weight
            self.symbol = None
        else:
            assert weight > 0 and symbol is not None, f"Weight={weight}, symbol={symbol}"
            self.weight = weight
            self.symbol = symbol

        assert (left_child is None and right_child is None) or self.has_both_children()
        self.code = None

    def has_both_children(self):
        return self.left_child is not None and self.right_child is not None

    def __eq__(self, other):
        return self.weight == other.weight

    def __lt__(self, other):
        return self.weight < other.weight


def build_huffman_tree(symbols_cnts: dict):
    # Create leaves of the tree
    nodes = []
    for symbol, cnt in symbols_cnts.items():
        nodes.append((cnt, Node(None, None, cnt, symbol)))

    # Order leaves by weights, heapq is a min-heap
    heapq.heapify(nodes)

    # Build the tree bottom up
    while len(nodes) > 1:
        # Pop the two nodes with the lowest weights
        left = heapq.heappop(nodes)[1]
        right = heapq.heappop(nodes)[1]

        new_node = Node(left, right)
        heapq.heappush(nodes, (new_node.weight, new_node))

    # return the remaining node which is the top node
    # of the tree
    return nodes[0][1]


def compute_leaves_codes(node: Node, prefix=""):
    if node.has_both_children():
        a = compute_leaves_codes(node.left_child, prefix + "0")
        b = compute_leaves_codes(node.right_child, prefix + "1")
        return a+b
    else:
        assert node.left_child is None and node.right_child is None
        node.code = prefix
        return [node]


def build_codebooks(top_node):
    # Affect a code to each leaf node
    d = compute_leaves_codes(top_node, "")

    # Build maps from/to symbol to/from Huffman codes
    code_map = dict()
    decode_map = dict()
    for node in sorted(d, key=lambda n: n.weight):
        #print(f"{node.symbol} {node.weight:5d} {node.code}")
        code_map[node.symbol] = node.code
        decode_map[node.code] = node.symbol

    return code_map, decode_map


def encode(symbol_iter, code_map):
    """ Convert a serie of symbols into a serie of
    corresponding codewords (expected to be string
    representation of binary codes, eg 001100).

    - symbol_iter : an iterator which will give all
    the symbols of the data to compress, on by one,
    in order.
    - code_map : map from symbol to codeword.

    Note that data end detection rely on the iterator end
    (here it's detected by Python). So we don't add an
    additionaly symbol to represent the end of file.
    """

    file_str = StringIO()
    for symbol in symbol_iter:
        file_str.write(code_map[symbol])

    return file_str.getvalue()


def decode_one_symbol(compressed, decode_map):
    prefix = ""
    for c in compressed:
        assert c in (True, False), f"Unexpected char : {c}"
        if c:
            prefix += "1"
        else:
            prefix += "0"

        if prefix in decode_map:
            return len(prefix), decode_map[prefix]

    raise Exception("EOF unexpected")

def decode(compressed, decode_map, nb_symbols = 2**31):
    # File end is detected by file size. See remark in
    # encode() funtcion.

    ns = 0
    prefix = ""
    file_str = StringIO()
    for c in compressed:
        prefix += c

        if prefix in decode_map:
            file_str.write(decode_map[prefix])
            prefix = ""

            if ns < nb_symbols-1:
                ns += 1
            else:
                break

    return file_str.getvalue()
