import inspect
from typing import Iterable, Sequence

import torch


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
) -> torch.optim.Optimizer:
    """
    Build NVIDIA SOAP optimizer configured for KL-Shampoo.
    """
    SOAP = _resolve_soap_class()
    signature = inspect.signature(SOAP.__init__)
    supported = set(signature.parameters.keys())

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

    # API compatibility across package versions.
    if "nesterov" in supported:
        kwargs["nesterov"] = nesterov
    if "use_nesterov" in supported:
        kwargs["use_nesterov"] = nesterov

    filtered = {k: v for k, v in kwargs.items() if k in supported}
    return SOAP(params, **filtered)
