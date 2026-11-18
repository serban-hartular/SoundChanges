
from typing import Any, Callable


class Dummy1:
    pass

class Dummy2:
    pass

BOS = Dummy1()
EOS = Dummy2()
EMPTY = None

def default_cost(i, o) -> float|int:
    return 0 if i == o else 1

def get_bracketed_lists(s, t) -> tuple[list, list]:
    if isinstance(s, str):
        s = list(s)
    if isinstance(t, str):
        t = list(t)
    s = [BOS] + s + [EOS]
    t = [BOS] + t + [EOS]
    return s, t



def cost_matrix(s : list|str, t : list|str,
                cost_fn: Callable[[Any, Any], float|int]|None = None) -> list[list[float|int]]:
    if cost_fn is None:
        cost_fn = default_cost

    matrix = [[-1]*len(t) for _ in range(len(s))]
    for i in range(len(s)):
        matrix[i][0] = i
    for j in range(len(t)):
        matrix[0][j] = j
    for i in range(1, len(s)):
        for j in range(1, len(t)):
            matrix[i][j] = min([
                matrix[i-1][j] + cost_fn(s[i], EMPTY), # deletion
                matrix[i][j-1] + cost_fn(EMPTY, t[j]), #insertion
                matrix[i-1][j-1] + cost_fn(s[i], t[j]), # substitution
            ])
    return matrix

def find_change_sequences(matrix : list[list[int|float]], s: list, t: list,
                          cost_fn: Callable[[Any, Any], float|int]|None = None,
                          position: tuple[int, int]|None = None,
                          greedy: bool = False) -> list[list[tuple]]|None:
    if position is None:
        position = (0, 0)
    if cost_fn is None:
        cost_fn = default_cost

    if position == (len(s)-1, len(t)-1):
        return [[]]
    
    i, j = position
    # targets = {(i+1, j+1): (s[i+1], t[j+1]), # substitution
    #            (i+1, j): (s[i+1], EMPTY), # deletion
    #            (i, j+1): (EMPTY, t[j+1]), # insertion
    # }
    targets = {(i+1, j+1): (), # substitution
               (i+1, j): (), # deletion
               (i, j+1): (), # insertion
    }
    # avoid overflow
    targets = {p:(EMPTY if p[0] == i else s[p[0]], EMPTY if p[1] == j else t[p[1]]) 
                    for p in targets if p[0] < len(s) and p[1] < len(t)}


    targets = {p:chg for p,chg in targets.items()
                    if matrix[p[0]][p[1]] <= matrix[-1][-1] and # do not cost more than minimum
                    matrix[i][j] + cost_fn(*chg) == matrix[p[0]][p[1]]} # sequence cost must match calc cost
    target_items = list(targets.items())

    return_list = []
    for p, chg in target_items:
        paths = find_change_sequences(matrix, s, t, cost_fn=cost_fn, position=p)
        if paths is None:
            continue
        paths = [[chg] + path for path in paths]
        if greedy:
            return_list = [paths[0]]
            break
        return_list.extend(paths)

    return return_list or None

def get_edits(string1: str|list, string2: str|list,
                   cost_fn: Callable[[Any, Any], int|float]|None = None,
                   greedy: bool = False)\
                             -> list[list[tuple[Any, Any]]]:
    if cost_fn is None:
        cost_fn = default_cost
    l1, l2 = get_bracketed_lists(string1, string2)
    matrix = cost_matrix(l1, l2, cost_fn=cost_fn)
    change_seqs = find_change_sequences(matrix, l1, l2, cost_fn = cost_fn, greedy=greedy)
    if change_seqs is None:
        change_seqs = []
    change_seqs = [seq[:-1] for seq in change_seqs] # chop off final $/$
    return change_seqs

def get_edits_greedy(string1: str|list, string2: str|list,
                   cost_fn: Callable[[Any, Any], int|float]|None = None)\
                            -> list[tuple[Any, Any]]|None:
    change_seqs = get_edits(string1, string2, cost_fn=cost_fn, greedy=True)
    return change_seqs[0] if change_seqs else None


if __name__ == "__main__":
    s1 = 'pun'
    s2 = 'pwan'
    seqs = get_edits(s1, s2)