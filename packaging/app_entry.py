"""PyInstaller entry point for the Mac app: start screen, or CLI if arguments are given."""
from mcbio.launcher import main

if __name__ == "__main__":
    raise SystemExit(main())
