"""A tiny rule-based model used for the CI/CD demonstration."""


def predict_result(mark: int) -> str:
    """Predict whether a student passes based on the mark."""
    return "Pass" if mark >= 40 else "Fail"


if __name__ == "__main__":
    print(predict_result(65))
