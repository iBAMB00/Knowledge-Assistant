"""Vector store implementations and abstractions.

具体 Provider/Factory 按需从对应模块导入，避免导入基础接口时立即加载
Qdrant 等可选基础设施依赖。
"""
