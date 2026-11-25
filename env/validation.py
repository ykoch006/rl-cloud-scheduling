"""Number checks shared by the simulator, workload generator and metrics.

Python and NumPy numbers are both accepted, so values taken from NumPy arrays
(or a future Gymnasium environment) work. Booleans are rejected even though
Python treats True and False as the integers 1 and 0.
"""

import math
import numbers


def is_integer(value) -> bool:
    """True for Python or NumPy integers, but not booleans or whole-valued floats."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def is_finite_number(value) -> bool:
    """True for finite Python or NumPy real numbers, but not booleans, NaN or infinity."""
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)
