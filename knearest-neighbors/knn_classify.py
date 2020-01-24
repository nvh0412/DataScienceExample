from typing import NamedTuple
from collections import Counter
from linear_algebra import Vector, distance

def marjority_vote(labels: list[str]) -> str:
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count for count in vote_counts.values() if count == winner_count])

    if num_winners == 1:
        return winner
    else:
        return marjority_vote(labels[:-1]) # try again without the farthest

assert marjority_vote(['a', 'b', 'c', 'b', 'a']) == 'b'

class LabeledPoint(NamedTuple):
    point: Vector
    label: str

def knn_classify(k: int, lableled_ponts: List[LabeledPoint], new_point: Vector) -> str:
    # Order the labeled points from nearest to farthest
    by_distance = sorted(lableled_ponts, key=lambda lp: distance(lp.point, new_point))

    # Find the labels for the k closest
    k_nearest_labels = [lp.label for lp in by_distance[:k]]

    return marjority_vote(k_nearest_labels)
