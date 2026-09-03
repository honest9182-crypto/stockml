"""The evolutionary search layer on top of step 1.

See CLAUDE.md's "Evolutionary search (step 1.5)" section for the rules this
package must never break: same exam for everyone, the three time zones, the
vault protocol, the two controls, one seeded RNG, full lineage. The
scientific question this package exists to answer is not "can evolution
find a good model" but "does evolution find anything random search and pure
luck don't" -- every design choice here should make that question easier to
answer honestly, not the headline fitness number look better.
"""

from __future__ import annotations
