# ADR-0003: 本地生成 CPU 线程锁定

- **Status**: Accepted (2026-09-24)
- **Background**: 在纯 CPU（AMD Ryzen 7 H 255, 16GB）上跑 llama.cpp GGUF 生成。
  经验表明小量化模型（1.5B Q4）的瓶颈在内存带宽而非算力；llama.cpp 默认线程数
  `cpu_count-1` 反而因带宽争用使生成慢约 4 倍。
- **Decision**: 锁定 `n_threads = 4`，并作为 `Config.llm_threads` 显式暴露可调。
- **Consequences**:
  - 正面：单问延迟显著下降，吞吐稳定。
  - 负面：未充分利用多核；对更大模型收益递减（将在 v2 按模型规模自适应）。
- **Related**: ADR-0001（双层回退）、ADR-0002（模型供应链）。
