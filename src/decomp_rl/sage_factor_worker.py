"""JSON-lines worker that factors finite-field polynomials with Sage."""

from __future__ import annotations

import json
import sys

from sage.all import GF, PolynomialRing


def _poly_from_payload(prime: int, variables: list[str], terms: list[list[object]]):
    ring = PolynomialRing(GF(prime), variables, order="degrevlex")
    generators = ring.gens()
    poly = ring.zero()
    for coeff, exponent in terms:
        term = ring(int(coeff) % prime)
        for generator, power in zip(generators, exponent):
            if power:
                term *= generator ** int(power)
        poly += term
    return poly


def _payload_from_poly(poly) -> list[list[object]]:
    payload = []
    for exponent, coeff in poly.dict().items():
        if isinstance(exponent, int):
            exponent_list = [int(exponent)]
        else:
            exponent_list = [int(v) for v in exponent]
        payload.append([int(coeff), exponent_list])
    payload.sort(key=lambda item: (-sum(item[1]), item[1], item[0]))
    return payload


def main() -> None:
    print(json.dumps({"ready": True, "backend": "sage"}), flush=True)
    for line in sys.stdin:
        message = json.loads(line)
        if message.get("shutdown"):
            break
        try:
            poly = _poly_from_payload(
                prime=int(message["prime"]),
                variables=[str(name) for name in message["variables"]],
                terms=message["terms"],
            )
            factorization = poly.factor()
            response = {
                "ok": True,
                "backend": "sage",
                "unit": int(factorization.unit()),
                "factors": [
                    {
                        "terms": _payload_from_poly(factor),
                        "exponent": int(exponent),
                    }
                    for factor, exponent in factorization
                ],
            }
        except Exception as exc:  # pragma: no cover - worker side safeguard
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
