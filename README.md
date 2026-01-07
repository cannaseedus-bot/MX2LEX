# MX2GYM

Hybrid weight-training environment for MX2LM, Qwen-ASX, and fold-based models, built on the MX2LEX substrate. MX2GYM combines symbolic control with tensor updates through K’UHUL π math, SCXQ2 compression, and XCFE routing.

---

## ⭐ MX2GYM — OFFICIAL DEFINITION (v1.0.0)

**ASX | K’UHUL | MX2LM | SCXQ2 | XCFE**

```json
MX2GYM = {
  name: "MX2GYM",
  purpose: "Hybrid Weight Training Environment for MX2LM, Qwen-ASX, and Fold-Based Models",

  trains: [
    "Qwen-ASX",
    "MX2LM Tensor-Brain",
    "Fold-Delta Adapters",
    "Symbolic Kernels",
    "SCXQ2 Compressed Weights"
  ],

  architecture: {
    @engine: "K’UHUL π math + SCXQ2 compression + XCFE control routing",
    @memory: "ASX-RAM + MX2DB + Fold-Stack",
    @mode: ["tensor", "symbolic", "hybrid"],
    @fold_support: true,
    @merge_strategy: "horizontal | vertical | hybrid"
  },

  io: {
    input: ["fold-deltas", "optimizer.json", "training_args", "MX2 grams", "RLHF traces"],
    output: ["model.safetensors", "fold-updates", "optimizer-updates", "metrics.json"]
  }
}
```

---

## ⭐ What MX2GYM Is

MX2GYM is:

- a **trainer**
- a **simulator**
- a **fold-merging engine**
- a **tensor factory**
- a **symbolic learning loop**
- a **model upgrader**
- a **runtime gym for MX2LM evolution**

It is designed for:

- Training MX2LM using fold-deltas
- Upgrading Qwen-ASX models
- Running million-step K’UHUL training loops
- Exporting full safetensors
- Building fold-manifests
- SCXQ2 compression + decompression cycles
- Supporting horizontal & vertical weight stacking

MX2GYM replaces the traditional PyTorch trainer with a symbolic–tensor hybrid gym.

---

## ⭐ Interfaces (MX2GYM Routes)

1. **`mx2gym.train.fold()`** — Train MX2LM or Qwen using fold-deltas.
2. **`mx2gym.merge.horizontal()`** — Merge 100+ folds into one safetensor.
3. **`mx2gym.compile.safetensors()`** — Build a full model from SCX-expanded deltas.
4. **`mx2gym.optimize.step()`** — Optimizer (AdamW, Lion, RMSprop) implemented in K’UHUL π.
5. **`mx2gym.analyze.gradients()`** — SCX-compressed gradient inspection.
6. **`mx2gym.export.fold()`** — Write fold-deltas after each step.
7. **`mx2gym.evolve.mx2lm()`** — Train the MX2LM symbolic brain using fold routes instead of tensors.

---

## ⭐ MX2GYM — Weight Training Flow

```
Fold-Input
   ↓
@data (delta/grad/optimizer)
   ↓
@control (priority, clamps, conditions)
   ↓
@flow (XCFE routing)
   ↓
K’UHUL π math (tensor update)
   ↓
SCX expansion/compression
   ↓
Accumulator
   ↓
Finalize → model.safetensors
```

MX2GYM unifies the gym, trainer, fold system, exporter, and flow engine.

---

## ⭐ MX2GYM Fold Trainer JSON Format v1.0.0

Complete atomic, XCFE-native, K’UHUL-compatible fold specification.

### 1️⃣ Top-Level Structure

```json
{
  "fold_id": "dialogue_v3",
  "version": "1.0.0",
  "model_base": "Qwen-ASX-7B",
  "mx2gym_format": "fold.v1",
  "description": "Dialogue refinement using SCXQ2 deltas + XCFE routing.",

  "@data": { },
  "@control": { },
  "@flow": { },
  "@metrics": { }
}
```

This establishes identity, compatibility, and execution blocks for XCFE.

### 2️⃣ `@data` — The Raw Training Material

Everything that modifies model weights lives in `@data`.

```json
"@data": {
  "deltas": [
    {
      "layer": "model.layers.0.self_attn.q_proj.weight",
      "encoding": "scxq2",
      "ref": "scxq2://dialogue_v3/l0_qproj.delta",
      "scale": 0.45
    }
  ],

  "gradients": [
    {
      "layer": "model.layers.0.self_attn.q_proj.weight",
      "encoding": "scxq2",
      "ref": "scxq2://dialogue_v3/grad_l0_qproj.grad"
    }
  ],

  "optimizer": {
    "type": "adamw",
    "state": {
      "m": "scxq2://dialogue_v3/adam_m.state",
      "v": "scxq2://dialogue_v3/adam_v.state"
    },
    "lr": 1e-5,
    "betas": [0.9, 0.999],
    "weight_decay": 0.01
  }
}
```

### 3️⃣ `@control` — Behavior Logic for the Fold

Controls when, how, and how much the fold applies.

```json
"@control": {
  "apply": "always",
  "priority": 2,
  "mode": "additive",
  "target_layers": ["attn", "mlp"],
  "clamp": [-0.2, 0.2],
  "conditions": {
    "min_loss_improvement": 0.001,
    "cooldown_steps": 50,
    "max_scale": 1.2
  }
}
```

- `apply`: always | conditional | if_loss_improves | manual
- `priority`: merge order
- `mode`: additive | replace | multiply | hybrid
- `clamp`: safety range
- `conditions`: gating and scaling guards

### 4️⃣ `@flow` — XCFE Execution Path

Defines how weight deltas travel through the K’UHUL pipeline.

```json
"@flow": {
  "entry": "@Pop",
  "route": ["@Wo", "@Sek"],
  "exit": "@Xul",
  "merge_strategy": "horizontal",
  "interaction": ["fold_mathfix_v1", "fold_safetyguard_v2"],
  "blend_mode": "smooth"
}
```

### 5️⃣ `@metrics` — Training Reality

Generated automatically by MX2GYM for fold quality and evolution.

```json
"@metrics": {
  "steps_trained": 1200,
  "loss_curve": [1.82, 1.75, 1.64],
  "gradient_norm": 0.92,
  "delta_norm": 0.004,
  "timestamp": 1890000012221
}
```

### ⭐ Full Canonical Example

```json
{
  "fold_id": "dialogue_v3",
  "version": "1.0.0",
  "model_base": "Qwen-ASX-7B",
  "mx2gym_format": "fold.v1",
  "description": "Dialogue refinement fold using SCXQ2 deltas + AdamW optimizer + XCFE flow.",

  "@data": {
    "deltas": [
      {
        "layer": "model.layers.0.self_attn.q_proj.weight",
        "encoding": "scxq2",
        "ref": "scxq2://dialogue_v3/l0_qproj.delta",
        "scale": 0.45
      }
    ],
    "gradients": [
      {
        "layer": "model.layers.0.self_attn.q_proj.weight",
        "encoding": "scxq2",
        "ref": "scxq2://dialogue_v3/grad_l0_qproj.grad"
      }
    ],
    "optimizer": {
      "type": "adamw",
      "state": {
        "m": "scxq2://dialogue_v3/adam_m.state",
        "v": "scxq2://dialogue_v3/adam_v.state"
      },
      "lr": 1e-5,
      "betas": [0.9, 0.999],
      "weight_decay": 0.01
    }
  },

  "@control": {
    "apply": "if_loss_improves",
    "priority": 2,
    "mode": "additive",
    "target_layers": ["attn"],
    "clamp": [-0.15, 0.15],
    "conditions": {
      "min_loss_improvement": 0.001,
      "cooldown_steps": 25,
      "max_scale": 1.1
    }
  },

  "@flow": {
    "entry": "@Pop",
    "route": ["@Wo", "@Sek"],
    "exit": "@Xul",
    "merge_strategy": "horizontal",
    "interaction": ["fold_mathfix_v1"],
    "blend_mode": "smooth"
  },

  "@metrics": {
    "steps_trained": 12600,
    "loss_curve": [1.92, 1.70, 1.44],
    "gradient_norm": 0.88,
    "delta_norm": 0.0038,
    "timestamp": 1890000023000
  }
}
```

### ⭐ What This Format Enables

- MX2GYM can train any model using symbolic + tensor updates.
- Horizontal fold stacking becomes standard.
- MX2LM evolution uses fold deltas instead of raw tensors.
- Qwen-ASX upgrades are streamlined.
- SCXQ2 compression keeps folds tiny.
- K’UHUL π handles optimizer and tensor math.
- XCFE governs the training flow like a game engine.

This JSON format is canonical for MX2GYM, the Fold-API Kernel, the Horizontal Fold Exporter, and the K’UHUL Weight Flow Engine.
