"""Build the Cython extension, with OpenMP if a test program compiles with it.

Environment variables:

    FAST_WIGNER_D_OPENMP=0 / 1   disable / require OpenMP
    LIBOMP_PREFIX                libomp location on macOS (Homebrew and CONDA_PREFIX
                                 are also searched)
    FAST_WIGNER_D_NATIVE=1       add -march=native (local builds only)
"""
import os
import subprocess
import sys
import tempfile

from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

OMP_TEST = r"""
#include <omp.h>
int main(void) {
    int n = 0;
    #pragma omp parallel reduction(+:n)
    n += 1;
    return n > 0 && omp_get_max_threads() > 0 ? 0 : 1;
}
"""


def _env_flag(name):
    """True / False for boolean-like values, else None."""
    value = os.environ.get(name, "").strip().lower()
    if value in ("1", "on", "true", "yes"):
        return True
    if value in ("0", "off", "false", "no"):
        return False
    return None


def _libomp_prefixes():
    prefixes = [os.environ.get("LIBOMP_PREFIX"), os.environ.get("CONDA_PREFIX")]
    try:
        prefixes.append(
            subprocess.check_output(
                ["brew", "--prefix", "libomp"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        pass
    prefixes += ["/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"]
    return [
        p
        for p in dict.fromkeys(filter(None, prefixes))
        if os.path.isfile(os.path.join(p, "include", "omp.h"))
    ]


def _openmp_candidates(compiler_type):
    """(compile_args, link_args) pairs to try, in order."""
    if compiler_type == "msvc":
        return [(["/openmp"], [])]
    candidates = [(["-fopenmp"], ["-fopenmp"])]  # gcc, LLVM clang
    if sys.platform == "darwin":  # Apple clang + libomp
        for prefix in _libomp_prefixes():
            inc, lib = os.path.join(prefix, "include"), os.path.join(prefix, "lib")
            candidates.append(
                (["-Xpreprocessor", "-fopenmp", "-I" + inc],
                 ["-L" + lib, "-lomp", "-Wl,-rpath," + lib])
            )
    return candidates


class BuildExt(build_ext):
    def _links_with(self, compile_args, link_args):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "omp_test.c")
            with open(src, "w") as f:
                f.write(OMP_TEST)
            try:
                objects = self.compiler.compile(
                    [src], output_dir=tmp, extra_postargs=compile_args
                )
                self.compiler.link_executable(
                    objects, "omp_test", output_dir=tmp, extra_postargs=link_args
                )
            except Exception:  # CompileError / LinkError
                return False
        return True

    def build_extensions(self):
        compiler_type = self.compiler.compiler_type
        want_openmp = _env_flag("FAST_WIGNER_D_OPENMP")

        openmp = None
        if want_openmp is not False:
            print("fast_wigner_d: probing OpenMP flags (compiler errors here are expected)")
            for flags in _openmp_candidates(compiler_type):
                if self._links_with(*flags):
                    openmp = flags
                    break
            if openmp is None and want_openmp:
                raise RuntimeError(
                    "FAST_WIGNER_D_OPENMP=1 but the compiler cannot build OpenMP code"
                    " (on macOS install libomp, or set LIBOMP_PREFIX)"
                )
        if openmp is None:
            print("fast_wigner_d: building WITHOUT OpenMP (the code will run serially)")
            openmp = ([], [])
        else:
            print("fast_wigner_d: building with OpenMP:", " ".join(openmp[0] + openmp[1]))

        if compiler_type == "msvc":
            optimize = ["/O2"]
        else:
            optimize = ["-O3"]
            if _env_flag("FAST_WIGNER_D_NATIVE"):
                optimize.append("-march=native")

        for ext in self.extensions:
            ext.extra_compile_args = optimize + openmp[0]
            ext.extra_link_args = openmp[1]
        super().build_extensions()


ext_wigner_d = Extension(
    name="fast_wigner_d.wigner_d_jacobi",
    sources=["src/fast_wigner_d/wigner_d_jacobi.pyx"],
    libraries=[] if os.name == "nt" else ["m"],
)

setup(
    ext_modules=cythonize([ext_wigner_d], language_level=3),
    cmdclass={"build_ext": BuildExt},
)
