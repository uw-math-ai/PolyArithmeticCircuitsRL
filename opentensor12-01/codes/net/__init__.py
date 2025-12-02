def __getattr__(name):
    if name == "Net":
        from .network import Net
        return Net
    if name == "PolynomialNet":
        from .polynomial_net import PolynomialNet
        return PolynomialNet
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = ["Net", "PolynomialNet"]
