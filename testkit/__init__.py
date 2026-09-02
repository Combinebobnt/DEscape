"""Harness-only helpers shared by tests/ and tools/.

Not part of the shipped `descape` package and not imported by it -- this
exists purely so the verification scripts under tools/ and the pytest
suite under tests/ can share code. tools/ has no __init__.py and its
scripts are loaded by file path, so they cannot `import conftest`; both
trees do put the repo root on sys.path (tools/*.py via their own
sys.path.insert, tests/ via pytest.ini's `pythonpath = .`), which is what
makes a top-level package importable from either side.
"""
