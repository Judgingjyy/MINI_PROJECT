在MLP下的model ，horizon 1比4差，因为出现了OOD  -》去改变data..主要这里都是我们已知目标了，  提升巨大，，，，

Action Chunking 可以保持一小段动作的 temporal consistency


你要想指标，你要想问题，你要想解决方法


```text

Action Chunking：一次生成未来 \(H\) 个动作，不代表必须执行完；它提供的是一段局部动作计划。
Receding Horizon：预测 \(H\) 步、只执行前 \(k\) 步，再根据新 observation 重新生成。\(H_{\text{execute}}<H_{\text{predict}}\) 可以提高反馈频率。
效率—反馈 trade-off：\(k\) 越小，纠错越及时，但 policy 调用越频繁；\(k\) 越大，推理便宜，但旧计划更容易失效。
Direct vs Flow：Direct Policy 是 obs → chunk 一次回归；Flow 是 noise → 多次 velocity prediction → chunk，生成成本明显更高。
Flow 的随机性：Flow 每次从新噪声开始，频繁 replanning 不一定更稳定，可能产生 sampling variance；同一个 chunk 内连续执行若干动作反而有 temporal consistency。
Flow sampling steps：更多 Euler steps 理论上可能提高生成质量，但会增加推理成本，是以后值得研究的效率方向。
Horizon error：你已经观察到 Action 0 → Action 3 的 MAE 略微增长，未来动作通常更难预测。
BC distribution shift：模型 rollout 后会进入训练集没覆盖的状态；你通过加入 goal 附近和越过 goal 的数据，亲自验证了“数据分布覆盖”对 policy 的重要性。
Generative policy 不一定在简单任务占优：你的 toy task 是确定性的单峰映射，所以 Direct regression 很容易学好；Flow/Diffusion 真正的优势应该在多模态动作分布、复杂轨迹、不确定性等场景体现。
以后可以形成研究问题：如何用更少 sampling steps 生成高质量 Action Chunk？如何选择 prediction/execution horizon？如何降低 chunk 间不连续？这些都和你未来的“动作生成效率”主线直接相关。
```