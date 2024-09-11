from functools import partial
from typing import Callable

from pyboolnet.boolean_normal_forms import primes2mindnf  # type: ignore
from pyboolnet.file_exchange import bnet2primes  # type: ignore


def simplify_algebraic(algebraic_string: str, n_inputs: int) -> str:
    central_index = n_inputs // 2
    bnet = f"x_{central_index}_, {algebraic_string}"
    dnf = primes2mindnf(bnet2primes(bnet))
    return dnf[f"x_{central_index}_"]


def convert_to_template(algebraic: str, n_inputs: int) -> str:
    central_index = n_inputs // 2

    for i in range(n_inputs):
        if i < central_index:
            ind = f"l{central_index - i}"
        elif i > central_index:
            ind = f"r{i-central_index}"
        else:
            ind = "c"
        algebraic = algebraic.replace(f"x_{i}_", "x_{" + ind + "}")
    return algebraic


def hex_to_algebraic(
    hex_string: str, n_inputs: int = 7, simplify: bool = True, as_template: bool = True
) -> str:
    dec = int(hex_string, 16)
    if dec == 0:
        return "0"
    bits = bin(dec)[2:].zfill(2**n_inputs)  # [2:] removes the '0b' prefix
    implicants_list: list[str] = []
    for i, b in enumerate(bits):
        if b == "1":
            input_config = bin(i)[2:].zfill(n_inputs)
            literals_list: list[str] = []
            for j, c in enumerate(input_config):
                if c == "1":
                    literals_list.append(f"x_{j}_")
                else:
                    literals_list.append(f"!x_{j}_")
            implicants_list.append("&".join(literals_list))
    algebraic = "|".join(implicants_list)
    if simplify:
        algebraic = simplify_algebraic(algebraic, n_inputs)
    if as_template:
        algebraic = convert_to_template(algebraic, n_inputs)
    return algebraic


def build_rule(rule_template: str, index: int, N: int, noise: float = 0.0):
    if noise < 0 or noise > 1:
        raise ValueError("Noise must be between 0 and 1")
    if index < 0 or index >= N:
        raise ValueError("Index must be between 0 and N-1")
    if N < 7:
        raise ValueError("N must be at least 7")
    c = index
    l1 = (index - 1) % N
    l2 = (index - 2) % N
    l3 = (index - 3) % N
    r1 = (index + 1) % N
    r2 = (index + 2) % N
    r3 = (index + 3) % N
    base_rule = rule_template.format(c=c, l1=l1, l2=l2, l3=l3, r1=r1, r2=r2, r3=r3)

    if noise == 0:
        return f"x_{c}, {base_rule}"
    else:
        return f"x_{c}, ({base_rule})&({noise}<<=1) | !({base_rule})&(0<<={noise})"


def build_network(
    rule_builder: Callable[[int, int, float], str], N: int, noise: float = 0.0
) -> str:
    return "\n".join(rule_builder(i, N, noise) for i in range(N))


def template_to_network(template: str, N: int, noise: float = 0.0) -> str:
    return build_network(partial(build_rule, template), N, noise=noise)
