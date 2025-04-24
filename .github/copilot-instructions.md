Follow best practices for Python development.

Use type hints and docstrings for all public functions and classes.

Write clear, concise, and well-commented code.

Prefer standard library modules over third-party packages unless necessary.

Ensure all code is compatible with the project's Python version as specified in pyproject.toml.

Write and maintain unit tests for all new features and bug fixes in the tests/ directory.

Use descriptive commit messages and follow the project's commit conventions.

When asked to generate code, prefer idiomatic and readable Python.

When making changes, update or add relevant tests.

When asked to run commands, use the project's virtual environment if available (e.g., .venv/bin/python).

When editing files, avoid repeating unchanged code; use comments to indicate omitted sections.

When generating documentation, follow the style used in the existing codebase.

The main package is coloredqrcode. Place all core logic in this package.

Tests are located in the tests/ directory and should use pytest.

Dependencies should be managed via requirements.txt and/or pyproject.toml.

When running unit tests, always show the test results output every time.
