"""Aletheia 澄明 — 推理时计算 Scaling 引擎.

让 1.5B 本地小模型通过推理时计算扩展(inference-time compute scaling)
达到远超单 pass 的推理质量. 全部模块以 protocol 契约隔离, 运行时注入,
离线 mock 零依赖可验证.
"""

__version__ = "1.0.0"
__author__ = "晨星"

__all__ = ["protocol"]
