import random
from collections import Counter
from typing import Any

# Like Counter.most_common(1)[0], but ties are broken RANDOMLY instead of "first one seen".
# (Uses the global random, which solve() seeds, so a level that only works with one lucky tie-break fails loudly across seeds)
def _random_most_common(counter:Counter) -> tuple[Any, int]:
    """The (item, count) with the highest count. Ties: a random one of them."""

    # What's the top count, and who's tied for it?
    top_count:int = max(counter.values())
    tied_items:list = [item for item, count in counter.items() if count == top_count]

    # Pick one of the tied items at random (just the one, if there's no tie)
    return random.choice(tied_items), top_count
