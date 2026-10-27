"""Run the tests without pytest:  python tests/run_tests.py  (pytest also works)."""
import importlib
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
fails = 0
for fn in sorted(f for f in os.listdir(HERE) if f.startswith('test_') and f.endswith('.py')):
    mod = importlib.import_module(fn[:-3])
    for name in sorted(n for n in dir(mod) if n.startswith('test_')):
        try:
            getattr(mod, name)()
            print(f'ok    {fn}::{name}')
        except Exception:
            fails += 1
            print(f'FAIL  {fn}::{name}')
            traceback.print_exc()
print(f'\n{fails} failed')
sys.exit(1 if fails else 0)
