import numpy as np
import numpy.typing as npt

OPENMP_ENABLED: bool

def get_wigner_d_matrix_optimized(
    j_times_2: int, beta: float, num_threads: int = 0
) -> npt.NDArray[np.float64]: ...
