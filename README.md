# CI/CD Group Lab – Customer Churn Prediction

A group project for the **Continuous Integration and Delivery** course at Nackademin (MLOps Engineer 2025).

## About the Project

This repository contains a machine learning project for **Saldes Mobil AB**, a fictional telecom company.

The model predicts customer churn and generates a weekly call list of customers at risk of cancelling their subscriptions.

Our assignment is to improve the existing CI/CD pipeline to ensure reliable testing, model validation, and traceability.

## Project Goals

- Automate testing and validation using GitHub Actions.
- Ensure model quality before generating predictions.
- Track model versions and their origins.
- Improve CI/CD efficiency and reliability.
- Practice collaborative development using Git and GitHub.

## Getting Started

Clone the repository and install the dependencies:

```bash
git clone <repository-url>
cd <repository-name>
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the tests:

```bash
pytest
```

## Collaboration

We work in separate feature branches and use pull requests to review and merge changes into `main`.

Our focus is on workflows, tests, and repository configuration. The existing model, dataset, and feature engineering code should remain unchanged.

---

*All customer data is fictional and used for educational purposes only.*
