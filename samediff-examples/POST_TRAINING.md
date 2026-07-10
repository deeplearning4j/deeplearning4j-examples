# LLM post-training examples

Deeplearning4j's SameDiff training stack supports the common post-training stages used after pretraining:

1. Supervised fine-tuning (SFT) for instruction following.
2. Offline preference optimization with DPO, IPO/RDPO, ORPO, SimPO, or KTO.
3. Reward-model training plus PPO, or online RLVR with GRPO and its DrGRPO, DAPO, and GSPO variants.
4. Parameter-efficient training with LoRA/QLoRA when full-model updates are too expensive.

## Runnable examples

- [LLMInstructionFineTuningExample.java](src/main/java/org/nd4j/examples/samediff/quickstart/training/LLMInstructionFineTuningExample.java) demonstrates response-masked SFT, LoRA, held-out perplexity, and GGUF export.
- [RLAlignmentTrainingExample.java](src/main/java/org/nd4j/examples/samediff/quickstart/training/RLAlignmentTrainingExample.java) is the executable post-training cookbook. It covers reward-model training, DPO/IPO/RDPO, ORPO, SimPO, KTO, PPO, GRPO, DrGRPO, DAPO, GSPO, PEFT wrapping, `RLAlignmentPipeline`, and DSP acceleration.
- [RLAlignmentConfigExample.java](src/main/java/org/nd4j/examples/samediff/quickstart/training/RLAlignmentConfigExample.java) is a compact configuration reference.
- [SFTLoRATrainingConfigExample.java](src/main/java/org/nd4j/examples/samediff/quickstart/training/SFTLoRATrainingConfigExample.java) shows SFT, LoRA, DPO, and GRPO configuration.

Run the cookbook with:

```bash
mvn -q -DskipTests compile exec:java \
  -Dexec.mainClass=org.nd4j.examples.samediff.quickstart.training.RLAlignmentTrainingExample
```

## Choosing a method

- Start with SFT when you have high-quality prompt/completion examples.
- Prefer DPO for paired preference data and a stable reference model.
- Choose ORPO or SimPO when reference-model memory is the limiting constraint.
- Use KTO when feedback is desirable/undesirable labels rather than paired rankings.
- Use GRPO/RLVR when outputs can be scored automatically, such as exact-answer math, code tests, or structured constraints.
- Use reward modeling plus PPO when the reward is learned and full online RLHF control is required.

The high-level `RLAlignmentPipeline` owns the outer training loop, gradient accumulation, checkpointing, logging, and optional PEFT wrapping. The lower-level trainers remain available when an application needs custom batching or rollout control.
