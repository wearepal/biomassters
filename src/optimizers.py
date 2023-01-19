import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, cast

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.optim import Optimizer
from typing_extensions import TypeAlias, TypedDict

__all__ = ["Adafactor"]


class ParamGroup(TypedDict):
    params: Iterable[Parameter]
    lr: float
    eps: Tuple[float, float]
    clipping_threshold: float
    decay_rate: float
    beta1: Optional[float]
    weight_decay: float
    multiply_by_parameter_scale: bool
    warmup_init: bool
    relative_step: bool


class ParamState(TypedDict):
    lr: float
    rms: Tensor
    step: int
    exp_avg: Tensor
    exp_avg_sq: Tensor
    exp_avg_sq_row: Tensor
    exp_avg_sq_col: Tensor


LossClosure: TypeAlias = Callable[..., Tensor]


class Adafactor(Optimizer):
    """Implements Adafactor algorithm.
    This implementation is based on:
    `Adafactor: Adaptive Learning Rates with Sublinear Memory Cost
    <see https://arxiv.org/abs/1804.04235>`_.
    Note that this optimizer internally adjusts the learning rate
    depending on the *multiply_by_parameter_scale*, *relative_step* and
    *warmup_init* options. To use a manual (external) learning rate
    schedule you should set `multiply_by_parameter_scale=False` and
    `relative_step=False`.

    :param params: iterable of parameters to optimize or dicts defining
        parameter groups

    :param lr: external learning rate.
    :param eps: regularization constans for square gradient
        and parameter scale respectively.

    :param clipping_threshold: threshold of root mean square of
        final gradient update.

    :param decay_rate: coefficient used to compute running averages of square
        gradient.

    :param beta1: coefficient used for computing running averages of gradient.

    :param weight_decay: weight decay (L2 penalty).
    :param multiply_by_parameter_scale: if True, learning rate is scaled by
        root mean square of parameter.

    :param relative_step: if True, time-dependent learning rate is computed
        instead of external learning rate.

    :param warmup_init: time-dependent learning rate computation depends on
        whether warm-up initialization is being used.
    """

    param_groups: List[ParamGroup]  # type: ignore
    state: Dict[Parameter, ParamState]  # type: ignore

    def __init__(
        self,
        params: Iterable[Parameter],
        *,
        lr: Optional[float] = None,
        eps: Tuple[float, float] = (1e-30, 1e-3),
        clipping_threshold: float = 1.0,
        decay_rate: float = 0.8,
        beta1: Optional[float] = None,
        weight_decay: float = 0.0,
        multiply_by_parameter_scale: bool = False,
        warmup_init: bool = False,
    ) -> None:
        relative_step = lr is None
        defaults = dict(
            lr=lr,
            eps=eps,
            clipping_threshold=clipping_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            multiply_by_parameter_scale=multiply_by_parameter_scale,
            relative_step=relative_step,
            warmup_init=warmup_init,
        )
        super().__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self) -> bool:
        return True

    @property
    def supports_flat_params(self) -> bool:
        return False

    def _get_lr(self, param_group: ParamGroup, param_state) -> float:
        rel_step_sz = param_group["lr"]
        if param_group["relative_step"]:
            min_step = 1e-6 * param_state["step"] if param_group["warmup_init"] else 1e-2
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state["step"]))
        param_scale = 1.0
        if param_group["multiply_by_parameter_scale"]:
            param_scale = max(param_group["eps"][1], param_state["rms"])
        return param_scale * rel_step_sz

    def _get_options(self, param_group: ParamGroup, param_shape: Sequence[int]):
        factored = len(param_shape) >= 2
        use_first_moment = param_group["beta1"] is not None
        return factored, use_first_moment

    def _rms(self, tensor: Tensor) -> Tensor:
        return tensor.norm(p=2) / (tensor.numel() ** 0.5)  # type: ignore

    def _approx_sq_grad(self, *, exp_avg_sq_row: Tensor, exp_avg_sq_col: Tensor):
        r_factor = (
            (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        )
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    def step(self, closure: Optional[LossClosure] = None) -> Optional[Tensor]:
        """Performs a single optimization step.
        :param closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradients.")

                state = self.state[p]
                grad_shape = grad.shape

                factored, use_first_moment = self._get_options(group, grad_shape)
                # State Initialization
                if len(state) == 0:
                    state["step"] = 0

                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(grad)
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).to(grad)
                        state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).to(
                            grad
                        )
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                    state["rms"] = torch.zeros(())
                else:
                    if use_first_moment:
                        state["exp_avg"] = state["exp_avg"].to(grad)
                    if factored:
                        state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
                        state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
                    else:
                        state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

                p_data_fp32 = p.data
                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()

                state["step"] += 1
                state["rms"] = self._rms(p_data_fp32)
                group["lr"] = self._get_lr(group, state)

                beta2t = 1.0 - math.pow(state["step"], -group["decay_rate"])
                update = (grad**2) + group["eps"][0]
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]

                    exp_avg_sq_row.mul_(beta2t).add_(update.mean(dim=-1), alpha=1.0 - beta2t)
                    exp_avg_sq_col.mul_(beta2t).add_(update.mean(dim=-2), alpha=1.0 - beta2t)

                    # Approximation of exponential moving average of square of gradient
                    update = self._approx_sq_grad(
                        exp_avg_sq_row=exp_avg_sq_row, exp_avg_sq_col=exp_avg_sq_col
                    )
                    update.mul_(grad)
                else:
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg_sq.mul_(beta2t).add_(update, alpha=1.0 - beta2t)
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_((self._rms(update) / group["clipping_threshold"]).clamp_(min=1.0))
                update.mul_(group["lr"])

                if use_first_moment:
                    momentum = cast(float, group["beta1"])
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(momentum).add_(update, alpha=1 - momentum)
                    update = exp_avg

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(p_data_fp32, alpha=-group["weight_decay"] * group["lr"])

                p_data_fp32.add_(-update)

                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p.data.copy_(p_data_fp32)

        return loss
