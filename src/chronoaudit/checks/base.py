"""Check registry. Each check is a pure function (df, ctx) -> list[Finding]."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import pandas as pd

from chronoaudit.core.models import DeclaredConvention, Finding, InferredConvention


@dataclass
class AuditContext:
    declared: DeclaredConvention
    inferred: InferredConvention
    interval: pd.Timedelta
    extras: dict = field(default_factory=dict)


class CheckFn(Protocol):
    def __call__(self, df: pd.DataFrame, ctx: AuditContext) -> list[Finding]: ...


REGISTRY: dict[str, CheckFn] = {}


def register(check_id: str) -> Callable[[CheckFn], CheckFn]:
    def deco(fn: CheckFn) -> CheckFn:
        if check_id in REGISTRY:
            raise ValueError(f"duplicate check id {check_id}")
        REGISTRY[check_id] = fn
        return fn

    return deco
