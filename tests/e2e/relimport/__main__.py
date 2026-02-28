import sys
from .subfolder.b import g

if __name__ == "__main__":
    val = float(sys.argv[1])
    print(g(val))
