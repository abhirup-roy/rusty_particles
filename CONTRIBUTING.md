# Contributing to Rusty Particles

We welcome contributions! Please follow these guidelines to ensure a smooth process.

## Development Setup

1.  Install Rust and Python.
2.  Install `maturin`: `pip install maturin`.
3.  Install development dependencies: `pip install pytest black ruff`.

## Code Style

*   **Rust:** Follow standard Rust formatting. Run `cargo fmt` before committing.
*   **Python:** Follow PEP 8. We use `ruff` for linting and formatting.

## Running Tests

*   **Rust Unit Tests:** `cargo test`
*   **Python Integration Tests:** `pytest pydem/`

## Pull Request Process

1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/amazing-feature`).
3.  Commit your changes (`git commit -m 'Add amazing feature'`).
4.  Push to the branch (`git push origin feature/amazing-feature`).
5.  Open a Pull Request.

## Reporting Issues

Please use the GitHub Issues tracker to report bugs or request features. Provide as much detail as possible, including reproduction steps and environment details.
