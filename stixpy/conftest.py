# Force MPL to use non-gui backends for testing.
try:
    import matplotlib
except ImportError:
    pass
else:
    matplotlib.use("Agg")

# The sunkit-spex spectrum output is an optional extra (stixpy[spectroscopy]), so don't import its module for the
# doctests without it
collect_ignore = []
try:
    import sunkit_spex  # noqa: F401
except ImportError:
    collect_ignore.append("product/sources/_spectrum.py")
