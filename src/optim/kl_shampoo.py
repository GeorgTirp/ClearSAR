import inspect
from typing import Any, Dict, Iterable, List, Sequence

import torch


class MultiOptimizer:
    """
    Simple wrapper that steps multiple optimizers together.
    """

    def __init__(self, optimizers: Sequence[torch.optim.Optimizer]) -> None:
        self.optimizers = [opt for opt in optimizers if opt is not None]
        if not self.optimizers:
            raise ValueError("MultiOptimizer requires at least one optimizer.")
        self.param_groups = []
        for optimizer in self.optimizers:
            self.param_groups.extend(optimizer.param_groups)

    def zero_grad(self, set_to_none: bool = False) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = None
        for idx, optimizer in enumerate(self.optimizers):
            if closure is not None and idx == 0:
                loss = optimizer.step(closure)
            else:
                optimizer.step()
        return loss

    def state_dict(self) -> Dict[str, Any]:
        return {
            "type": "multi_optimizer",
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        optimizer_states = state_dict.get("optimizers", [])
        if len(optimizer_states) != len(self.optimizers):
            raise ValueError(
                "Mismatched number of optimizers in state_dict: "
                f"expected {len(self.optimizers)}, got {len(optimizer_states)}"
            )
        for optimizer, optimizer_state in zip(self.optimizers, optimizer_states):
            optimizer.load_state_dict(optimizer_state)


def _resolve_soap_class():
    errors = []
    candidates = [
        ("emerging_optimizers.soap.soap", "SOAP"),
        ("emerging_optimizers.soap", "SOAP"),
    ]
    for module_name, class_name in candidates:
        try:
            module = __import__(module_name, fromlist=[class_name])
            return getattr(module, class_name)
        except Exception as exc:  # pragma: no cover
            errors.append(f"{module_name}.{class_name}: {exc}")
    joined = " | ".join(errors)
    raise ImportError(
        "Could not import NVIDIA SOAP optimizer. "
        "Install Emerging-Optimizers first (see NVIDIA docs). "
        f"Import errors: {joined}"
    )


def _normalize_param_groups(
    params: Iterable[torch.nn.Parameter] | Iterable[dict],
    lr: float,
) -> List[Dict[str, Any]]:
    params = list(params)
    if not params:
        return []

    first = params[0]
    if isinstance(first, dict):
        groups: List[Dict[str, Any]] = []
        for group in params:
            copied = dict(group)
            copied["params"] = list(copied.get("params", []))
            groups.append(copied)
        return groups

    return [{"params": list(params), "lr": lr}]


def _split_matrix_param_groups(
    param_groups: Sequence[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    matrix_groups: List[Dict[str, Any]] = []
    other_groups: List[Dict[str, Any]] = []

    for group in param_groups:
        base = {k: v for k, v in group.items() if k != "params"}
        group_params = list(group.get("params", []))
        matrix_params = [p for p in group_params if isinstance(p, torch.nn.Parameter) and p.ndim == 2]
        non_matrix_params = [p for p in group_params if isinstance(p, torch.nn.Parameter) and p.ndim != 2]

        if matrix_params:
            matrix_group = dict(base)
            matrix_group["params"] = matrix_params
            matrix_groups.append(matrix_group)
        if non_matrix_params:
            other_group = dict(base)
            other_group["params"] = non_matrix_params
            other_groups.append(other_group)

    return matrix_groups, other_groups


def build_kl_shampoo_optimizer(
    params: Iterable[torch.nn.Parameter] | Iterable[dict],
    lr: float,
    betas: Sequence[float] = (0.9, 0.95),
    shampoo_beta: float = 0.95,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    weight_decay_method: str = "decoupled",
    nesterov: bool = False,
    precondition_frequency: int = 1,
    adam_warmup_steps: int = 0,
    correct_bias: bool = True,
    fp32_matmul_prec: str = "high",
    use_eigh: bool = False,
    qr_fp32_matmul_prec: str = "high",
    use_adaptive_criteria: bool = False,
    adaptive_update_tolerance: float = 1e-7,
    power_iter_steps: int = 1,
    max_update_rms: float = 0.0,
    use_kl_shampoo: bool = True,
    correct_shampoo_beta_bias: bool | None = None,
    matrix_only: bool = True,
) -> torch.optim.Optimizer | MultiOptimizer:
    """
    Build NVIDIA SOAP optimizer configured for KL-Shampoo.
    """
    param_groups = _normalize_param_groups(params, lr)
    if not param_groups:
        raise ValueError("No parameters provided to build_kl_shampoo_optimizer.")

    matrix_groups = param_groups
    other_groups: List[Dict[str, Any]] = []
    if matrix_only:
        matrix_groups, other_groups = _split_matrix_param_groups(param_groups)

    kwargs = {
        "lr": lr,
        "betas": tuple(betas),
        "shampoo_beta": shampoo_beta,
        "eps": eps,
        "weight_decay": weight_decay,
        "weight_decay_method": weight_decay_method,
        "precondition_frequency": precondition_frequency,
        "adam_warmup_steps": adam_warmup_steps,
        "correct_bias": correct_bias,
        "fp32_matmul_prec": fp32_matmul_prec,
        "use_eigh": use_eigh,
        "qr_fp32_matmul_prec": qr_fp32_matmul_prec,
        "use_adaptive_criteria": use_adaptive_criteria,
        "adaptive_update_tolerance": adaptive_update_tolerance,
        "power_iter_steps": power_iter_steps,
        "max_update_rms": max_update_rms,
        "use_kl_shampoo": use_kl_shampoo,
        "correct_shampoo_beta_bias": correct_shampoo_beta_bias,
    }

    optimizers: List[torch.optim.Optimizer] = []

    if matrix_groups:
        SOAP = _resolve_soap_class()
        signature = inspect.signature(SOAP.__init__)
        supported = set(signature.parameters.keys())

        # API compatibility across package versions.
        if "nesterov" in supported:
            kwargs["nesterov"] = nesterov
        if "use_nesterov" in supported:
            kwargs["use_nesterov"] = nesterov

        filtered = {k: v for k, v in kwargs.items() if k in supported}
        optimizers.append(SOAP(matrix_groups, **filtered))

    if other_groups:
        optimizers.append(
            torch.optim.AdamW(
                other_groups,
                lr=lr,
                betas=tuple(betas),
                eps=eps,
                weight_decay=weight_decay,
            )
        )

    if not optimizers:
        raise ValueError("No optimizer could be constructed from the provided parameter groups.")
    if len(optimizers) == 1:
        return optimizers[0]
    return MultiOptimizer(optimizers)
