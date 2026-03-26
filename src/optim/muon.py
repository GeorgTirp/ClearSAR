import importlib
from typing import Any, Dict, Iterable, List, Sequence

import torch


def _resolve_muon_optimizer_class():
    try:
        module = importlib.import_module("muon")
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Could not import Muon optimizer from KellerJordan/Muon. "
            "Install it with: `pip install git+https://github.com/KellerJordan/Muon`"
        ) from exc

    single_device_cls = getattr(module, "SingleDeviceMuonWithAuxAdam", None)
    distributed_cls = getattr(module, "MuonWithAuxAdam", None)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        if world_size > 1 and distributed_cls is not None:
            return distributed_cls

    if single_device_cls is not None:
        return single_device_cls
    if distributed_cls is not None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return distributed_cls
        raise RuntimeError(
            "Only distributed Muon class (`MuonWithAuxAdam`) is available, "
            "but torch.distributed is not initialized. "
            "Install a newer KellerJordan/Muon version with "
            "`SingleDeviceMuonWithAuxAdam`, or initialize distributed training."
        )

    raise ImportError(
        "Muon module is installed but missing expected optimizer classes "
        "`SingleDeviceMuonWithAuxAdam` / `MuonWithAuxAdam`."
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


def build_muon_optimizer(
    params: Iterable[torch.nn.Parameter] | Iterable[dict],
    lr: float,
    weight_decay: float = 0.01,
    muon_lr_mult: float = 20.0,
    muon_momentum: float = 0.95,
    adam_betas: Sequence[float] = (0.9, 0.95),
    adam_eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """
    Build MuonWithAuxAdam / SingleDeviceMuonWithAuxAdam from KellerJordan/Muon.

    Parameters with ndim >= 2 are routed to Muon. Parameters with ndim < 2
    are routed to the internal Adam path.
    """
    base_groups = _normalize_param_groups(params, lr)
    if not base_groups:
        raise ValueError("No parameters provided to build_muon_optimizer.")

    muon_param_groups: List[Dict[str, Any]] = []
    adam_param_groups: List[Dict[str, Any]] = []
    for group in base_groups:
        group_params = [p for p in group.get("params", []) if isinstance(p, torch.nn.Parameter) and p.requires_grad]
        if not group_params:
            continue

        group_lr = float(group.get("lr", lr))
        group_weight_decay = float(group.get("weight_decay", weight_decay))
        matrix_params = [p for p in group_params if p.ndim >= 2]
        other_params = [p for p in group_params if p.ndim < 2]

        if matrix_params:
            muon_param_groups.append(
                {
                    "params": matrix_params,
                    "lr": group_lr * muon_lr_mult,
                    "momentum": muon_momentum,
                    "weight_decay": group_weight_decay,
                    "use_muon": True,
                }
            )
        if other_params:
            adam_param_groups.append(
                {
                    "params": other_params,
                    "lr": group_lr,
                    "betas": tuple(adam_betas),
                    "eps": adam_eps,
                    "weight_decay": group_weight_decay,
                    "use_muon": False,
                }
            )

    combined = [*muon_param_groups, *adam_param_groups]
    if not combined:
        raise ValueError("No optimizer parameter groups left after Muon filtering.")

    muon_optimizer_cls = _resolve_muon_optimizer_class()
    return muon_optimizer_cls(combined)
