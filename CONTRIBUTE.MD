# Contributing Guidelines

Thanks for your interest in contributing! To keep the project stable and manageable, please follow these rules:

## 🔀 Workflow
1. **Fork** the repository (do not create branches directly in this repo).
2. **Create a branch** in your fork using the naming convention below.
3. **Make your changes** and ensure tests pass.
4. **Open a Pull Request (PR)** into the `dev` branch (or `main` if no dev branch exists).
5. Wait for CI to pass and maintainers to review.

## 🌱 Branch Naming
Branches must follow these patterns:
- `feat/<short-description>` → New features
- `fix/<short-description>` → Bug fixes
- `docs/<short-description>` → Documentation changes
- `test/<short-description>` → Testing-related changes

✅ Examples:
- `feat/add-user-auth`
- `fix/navbar-overlap`
- `docs/update-readme`

❌ Bad examples:
- `patch1`
- `update-stuff`

## ✅ Requirements Before PR
- Run all tests locally and ensure they pass.
- Follow code style guidelines (`npm run lint` or `make lint`).
- Write clear commit messages.
- PR title should match branch type (`feat: ...`, `fix: ...`, `docs: ...`).

## 🔒 Protected Branches
- `main` is protected: no direct commits allowed.
- All code must be merged via Pull Requests.
- PRs require review approval + passing checks.

## 📜 Code of Conduct
Be respectful, constructive, and follow the [Code of Conduct](CODE_OF_CONDUCT.md).
