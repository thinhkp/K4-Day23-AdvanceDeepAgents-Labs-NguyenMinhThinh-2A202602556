# LLM Agents and Tool Use: A Survey

## TL;DR
- LLM agents leverage external tools to overcome inherent limitations in reasoning, real-time data access, and task execution [1][2].
- Benchmarking frameworks like API-Bank and ToolBench provide standardized environments for evaluating agentic tool-use accuracy [3][4].
- Modern agentic architectures are shifting toward self-healing mechanisms and structured control planes to mitigate error cascades in complex multi-agent systems [5][6][7].

## Background
Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language understanding and generation. However, they are often constrained by static knowledge bases and an inability to interact directly with external environments. LLM agents bridge this gap by integrating tool-use capabilities, allowing them to perform actions such as querying APIs, executing code, and orchestrating multi-step workflows. This survey explores the evolution of these systems, from basic tool-calling to sophisticated, self-correcting agentic frameworks.

## Tool-Use Benchmarking
Evaluating the efficacy of LLM agents requires robust benchmarks that simulate real-world scenarios. API-Bank and ToolBench are prominent examples that provide standardized environments for testing how well models can select, configure, and execute external tools [3][4]. These benchmarks are widely recognized for their role in assessing the tool-use capabilities of LLMs.

## Agentic Architectures and Orchestration
As agentic systems grow in complexity, the need for structured orchestration becomes paramount. Research into "Control Plane" designs suggests that separating the execution logic from the reasoning process improves scalability and reliability [6]. Furthermore, frameworks like AgentScope 1.0 emphasize developer-centric design, enabling more modular and maintainable agentic applications [8].

## Reliability and Error Mitigation
A significant challenge in agentic AI is the prevalence of error cascades, where a single failure in tool selection or execution propagates through a multi-step process [7]. To address this, recent research has introduced self-healing mechanisms that allow agents to detect and recover from failures autonomously [5]. Additionally, benchmarking tools like OrchestraBench are being developed to measure these failure modes and the quality of task decomposition [9].

## Trends and Open Problems
The field is moving toward more autonomous and resilient agentic systems. Key trends include:
- **Self-Healing:** Implementing feedback loops that allow agents to correct their own errors during execution [5].
- **Standardized Evaluation:** Developing more comprehensive benchmarks that account for real-world API volatility and complex multi-agent interactions [10][11].
- **Bias Mitigation:** Addressing tool-schema bias to ensure agents can generalize across diverse and heterogeneous tool sets [1].
- **Scalability:** Designing architectures that can manage large-scale multi-agent collaborations without sacrificing performance or reliability [6][7].

## References
[1] Action-Space Shaping for LLM Agents: Measuring and Mitigating Tool-Schema Bias. arxiv. https://arxiv.org/abs/2609.34971 (2026-09-28)
[2] ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs. arxiv. https://arxiv.org/abs/2402.01030 (2024-02-01)
[3] API-Bank: A Benchmark for Tool-Augmented LLMs. web. https://github.com/microsoft/API-Bank (n.d.)
[4] ToolBench: Benchmarking LLMs for Tool Use. web. https://github.com/ToolBench/ToolBench (n.d.)
[5] Self-Healing Agentic Orchestrators for Reliable Tool-Augmented Large Language Model Systems. arxiv. https://arxiv.org/abs/2606.01416 (2026-05-31)
[6] Control Plane as a Tool: A Scalable Design Pattern for Agentic AI Systems. hf-search. https://huggingface.co/papers/2505.06817 (2025-05-10)
[7] Modeling and Mitigating Error Cascades in LLM-Based Multi-Agent Systems. arxiv. https://arxiv.org/abs/2603.04474v2 (2026-05-11)
[8] AgentScope 1.0: A Developer-Centric Framework for Building Agentic Applications. hf-search. https://huggingface.co/papers/2508.16279 (2025-08-22)
[9] OrchestraBench: Evaluating Multi-Agent Orchestration Failure Modes, Recovery, and Decomposition Quality. arxiv. https://arxiv.org/abs/2608.05263 (2026-08-05)
[10] CORE: A Deployment-Oriented Framework for Evaluating Tool-Using Agents. arxiv. https://arxiv.org/abs/2509.20998 (2025-09-20)
[11] Evaluation and Benchmarking of LLM Agents: A Survey. arxiv. https://arxiv.org/abs/2507.21504v1 (2025-07-29)
