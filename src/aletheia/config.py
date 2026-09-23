"""Aletheia 配置 (M1).

解析与校验全局配置, 提供模型文件路径辅助函数.
"""
from __future__ import annotations

import os
from dataclasses import replace
from typing import Optional

from .protocol import Config, ConfigError, Strategy

__all__ = ["default_config", "resolve", "embed_path", "rerank_path", "llm_path"]


def default_config(model_dir: Optional[str] = None) -> Config:
    """构造默认配置. model_dir 默认指向仓库内 models/."""
    if model_dir is None:
        model_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "models")
        )
    return Config(model_dir=model_dir)


def resolve(cfg: Config) -> Config:
    """校验模型目录存在并返回规整后的配置."""
    if not os.path.isdir(cfg.model_dir):
        raise ConfigError(f"model_dir not found: {cfg.model_dir}")
    # 规整字符串枚举(允许传字符串)
    if not isinstance(cfg.default_strategy, Strategy):
        try:
            cfg = replace(cfg, default_strategy=Strategy(cfg.default_strategy))
        except ValueError as exc:
            raise ConfigError(f"unknown strategy: {cfg.default_strategy}") from exc
    if cfg.llm_threads < 1:
        raise ConfigError("llm_threads must be >= 1")
    return cfg


def embed_path(cfg: Config) -> str:
    return os.path.join(cfg.model_dir, "embed")


def rerank_path(cfg: Config) -> str:
    return os.path.join(cfg.model_dir, "rerank")


def llm_path(cfg: Config) -> str:
    return os.path.join(cfg.model_dir, "llm", cfg.llm_model + ".gguf")
