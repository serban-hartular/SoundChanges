import dataclasses
from enum import Enum
from typing import List, Tuple, Callable


def simple_string_distance(s : str, t : str) -> int:
    m = len(s)
    n = len(t)
    v0 = list(range(n+1))
    v1 = [0]*(n+1)
    for i in range(m):
        v1[0] = i + 1
        for j in range(n):
            # // calculating costs for A[i + 1][j + 1]
            deletionCost = v0[j + 1] + 1
            insertionCost = v1[j] + 1
            substitutionCost = v0[j] if s[i] == t[j] else v0[j]+1
            v1[j + 1] = min(deletionCost, insertionCost, substitutionCost)
        v0, v1 = v1, v0 # swap
    return v0[n]

Position = Tuple[int, int]

BOS = '^'
EOS = '$'
EMPTY = '∅'

class Operator(Enum):
    NOP = 'nop'
    INS = 'ins'
    DEL = 'del'
    SUB = 'sub'

@dataclasses.dataclass
class Operation:
    op : Operator
    d_in : str
    d_out : str
    str_rep : str = ''
    def __post_init__(self):
        self.str_rep = self.op.value
        if self.op == Operator.NOP:
            if self.d_in != self.d_out: raise Exception('NOP means no change!')
            return
        elif self.op == Operator.DEL:
            if self.d_out: raise Exception('DEL means no output!')
            self.str_rep += ' ' + self.d_in
        elif self.op == Operator.SUB:
            self.str_rep = '%s->%s' % (self.d_in, self.d_out)
        elif self.op == Operator.INS:
            if self.d_in != '': raise Exception('INS means no input!')
            self.str_rep = 'ins %s' % (self.d_out)
    @staticmethod
    def Nop(what : str = '') -> 'Operation':
        return Operation(Operator.NOP, what, what)
    @staticmethod
    def Del(what : str) -> 'Operation':
        return Operation(Operator.DEL, what, '')
    @staticmethod
    def Sub(what : str, withWhat : str) -> 'Operation':
        return Operation(Operator.SUB, what, withWhat)
    @staticmethod
    def Ins(what : str) -> 'Operation':
        return Operation(Operator.INS, '', what)
    def __str__(self):
        return self.str_rep
    def __repr__(self):
        return str(self)
    def __eq__(self, other):
        return self.op == other.op and self.d_in == other.d_in and self.d_out == other.d_out
@dataclasses.dataclass(frozen=True)
class Modification:
    op : Operation = dataclasses.field(default_factory=Operation.Nop)
    cost : float = 0
    from_pos : Position = (-1, -1)
    def __str__(self):
        return '(%s -> %s)' % (str(self.from_pos), str(self.op))
    def __repr__(self):
        return str(self)
    def __eq__(self, other):
        return self.op == other.op and self.cost == other.cost


@dataclasses.dataclass
class MatrixCell:
    modifications : List[Modification] = dataclasses.field(default_factory=list)
    cost : float = 0
    def __post_init__(self):
        if self.modifications:
            if min([m.cost for m in self.modifications]) != max([m.cost for m in self.modifications]):
                raise Exception('Unequal costs in cell!')
            self.cost = self.modifications[0].cost
    def __str__(self):
        return '%d %s' % (self.cost, str(self.modifications))
    def __repr__(self):
        return str(self)

# ModMatrix = List[List[Mod]]
class ModMatrix(List[List[MatrixCell]]):
    def __init__(self, rows : int, cols : int):
        super().__init__([[MatrixCell() for j in range(cols)] for i in range(rows)])
        self.rows = rows
        self.cols = cols
    def at(self, *args): #pos : Tuple[int, int]):
        if len(args) == 1 and len(args[0]) == 2:
            pos = args[0]
        elif len(args) == 2:
            pos = (args[0], args[1])
        else:
            raise Exception('Number of args must be 1 or 2')
        return self[pos[0]][pos[1]]




def add_BOS_EOS(s : str | List[str]) -> str | List[str]:
    if isinstance(s, str):
        return BOS + s + EOS
    return [BOS] + s + [EOS]

def distance(w1 : str|List[str], w2 : str|List[str],
             allowed_fn : Callable[[str, str], bool]|None = None,
             cost_fn : Callable[[str, str], float]|None = None) -> ModMatrix:
    if allowed_fn is None:
        allowed_fn = lambda i, o : True
    if cost_fn is None:
        cost_fn = lambda i, o : 0 if i == o else 1

    w1, w2 = add_BOS_EOS(w1), add_BOS_EOS(w2)
    rows = len(w1)
    cols = len(w2)
    matrix = ModMatrix(rows, cols)
    # init edges of matrix
    for i in range(1, rows):
        # matrix[i][0] = MatrixCell([Modification('del %s' % w1[i], i, (i - 1, 0))])
        matrix[i][0] = MatrixCell([Modification(Operation.Del(w1[i]), i, (i - 1, 0))])
    for j in range(1, cols):
        # matrix[0][j] = MatrixCell([Modification('ins %s' % w2[j], j, (0, j - 1))])
        matrix[0][j] = MatrixCell([Modification(Operation.Ins(w2[j]), j, (0, j - 1))])
    # populate matrix
    for j in range(1, cols):
        for i in range(1, rows):
            d_in = w1[i]
            d_out = w2[j]
            substitution_cost = cost_fn(d_in, d_out) #0 if w1[i] == w2[j] else 1
            del_cost = cost_fn(d_in, '')
            ins_cost = cost_fn('', d_out)
            # substition_string = '%s->%s' % (w1[i], w2[j]) if substitution_cost else 'nop'
            substition_op = Operation.Sub(w1[i], w2[j]) if substitution_cost else Operation.Nop(w1[i])
            options = [
                       Modification(Operation.Del(w1[i]), matrix[i - 1][j].cost + del_cost, (i - 1, j)),  # deletion
                       Modification(Operation.Ins(w2[j]), matrix[i][j - 1].cost + ins_cost, (i, j - 1)),  # insertion
                       Modification(substition_op, matrix[i - 1][j - 1].cost + substitution_cost, (i - 1, j - 1))  # substitution
                       ]
            options = [o for o in options if allowed_fn(o.op.d_in, o.op.d_out)]
            if options:
                min_cost = min([m.cost for m in options])
                options = [m for m in options if m.cost == min_cost]
            matrix[i][j] = MatrixCell(options)
    return matrix

@dataclasses.dataclass
class Transition:
    d_in : str
    d_out : str
    before : 'Transition|None' = None
    after : 'Transition|None' = None
    i_in : int|None = None
    i_out : int|None = None
    cost : int = 0
    @staticmethod
    def _to_output(s : str):
        return s if s else EMPTY
    def __hash__(self):
        return hash((self.d_in, self.d_out))
    def __str__(self):
        return '%s > %s' % (Transition._to_output(self.d_in), Transition._to_output(self.d_out))
    def __repr__(self):
        return str(self)
    def __eq__(self, other):
        return self.d_in == other.d_in and self.d_out == other.d_out
    def __bool__(self):
        return self.d_in != self.d_out


ChangeSequence = List[Transition]

def assign_indices(c_seq : ChangeSequence):
    i_in = 0
    i_out = 0
    for chg in c_seq:
        chg.i_in = i_in
        chg.i_out = i_out
        if chg.d_in:
            i_in += 1
        if chg.d_out:
            i_out += 1

def _change_sequence_cost(c_seq : ChangeSequence) -> int:
    c_seq = [c for c in c_seq if c]
    return len(set(c_seq))


def find_change_sequences(matrix : ModMatrix, pos : Position|None = None) -> List[ChangeSequence]:
    if pos is None:
        pos = (matrix.rows-1, matrix.cols-1)
    if pos == (0, 0):
        return [[Transition(BOS, BOS)]] # beginning of string
    mod_paths : List[ChangeSequence] = []
    for mod in matrix.at(pos).modifications:
        new_paths = find_change_sequences(matrix, mod.from_pos)
        if new_paths is None:
            continue
        for path in new_paths:
            # path.append((pos[0], mod))
            path.append(Transition(d_in=mod.op.d_in, d_out=mod.op.d_out, cost=mod.cost))
        mod_paths.extend(new_paths)
    return mod_paths

class WordTransformation:
    def __init__(self, initial : str, final : str,
                 allowed_fn : Callable[[str, str], bool]|None = None,
                 cost_fn : Callable[[str, str], float]|None = None):
        self.initial = initial
        self.final = final
        self.min_changes = simple_string_distance(self.initial, self.final)
        self.change_sequences = None
        self.allowed_fn = allowed_fn
        self.cost_fn = cost_fn
        

    def compute_change_sequences(self, allowed_fn : Callable[[str, str], bool]|None = None, cost_fn : Callable[[str, str], float]|None = None):
        if allowed_fn is None:
            allowed_fn = self.allowed_fn
        if cost_fn is None:
            cost_fn = self.cost_fn
        matrix = distance(self.initial, self.final, allowed_fn, cost_fn)
        self.change_sequences = find_change_sequences(matrix)
        for change_seq in self.change_sequences: # assign befores and afters
            for i, change in enumerate(change_seq):
                if i != 0:
                    change.before = change_seq[i-1]
                if i != len(change_seq)-1:
                    change.after = change_seq[i+1]
        # self.min_changes = min([_change_sequence_score(cs) for cs in self.change_sequences])
    def __str__(self):
        return '%s->%s' % (str(self.initial), str(self.final))
    def __repr__(self):
        return str(self)
    def __eq__(self, other):
        return (isinstance(other, WordTransformation)
                and self.initial == other.initial and self.final == other.final)
    def __hash__(self):
        return hash(str(self))
