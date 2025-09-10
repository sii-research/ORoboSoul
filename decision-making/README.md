# Commonsense Embodied Decision-making





## Overview 
**Commonsense Embodied Decision-making (CED)** is an open-source research framework for embodied AI. It targets a central challenge: enabling intelligent agents (robots, virtual assistants) to use **commonsense knowledge**—as humans do—to understand complex instructions, interact with the physical world, and make informed decisions.

Traditional robotic planners depend on hand-crafted rules and precise environment models, which limits generalization. CED addresses this by training **AHAT (Any House, Any Task)**, an end-to-end LLM that serves as the agent’s reasoning and decision-making core, using rich scene data paired with human instructions.

**Core capabilities of AHAT**

- **Task generalization**: Understands and plans for a broad spectrum of commands—from simple actions (e.g., heat a cup of milk) to long, multi-step chores (e.g., clean the living room).

- **Environmental generalization**: Not bound to a fixed map; plans effectively in unseen households by adapting to new layouts and object placements.

- **End-to-end planning**: Given a high-level goal, autonomously decomposes it, plans intermediate steps, and produces an executable action sequence.


## Dataset

### Scene Generation
**Real scenes**

From the open-source  [3DSG](https://3dscenegraph.stanford.edu/database.html) and [HSSD](https://3dlg-hcvc.github.io/hssd/), we derive lightweight, text-structured 3D scene graphs of indoor environments. While faithful to reality, these scans carry limitations: detection noise (e.g., misidentified objects, rooms labeled unknownroom), occlusions (e.g., items inside cabinets), and sparse object counts. Such sparsity constrains the length and diversity of tasks we can generate.

Synthetic scenes.
To overcome these issues, we start from the **room layouts of 308 real scenes**, which we first repair and refine. We then use an LLM’s commonsense to densely furnish the layouts with furniture and objects. The result is a large set of realistic, richly populated scenes with commonsense-consistent placements—ideal for long-horizon and complex task generation.
        

### Task Generation

To promote diversity in actions, difficulty, and object interactions, we created 1,600 user personas and generated **86,000 task instructions** across the 308 scenes with varied object distributions. The instructions span multiple styles:

- **Explicit commands**
- **Colloquial requests**
- **Needs-based instructions** (the most challenging), which require inferring underlying user intent and grounding it in the current scene to produce a successful plan.

## Model



## Online demo
https://ahat-planner-app.pages.dev/