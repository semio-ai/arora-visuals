# arora-visuals

The shared layer above Bevy for Semio assets — the code a Vizij face renderer
and a Studio robot scene both need, and that neither should own.

**This is not a renderer.** Bevy is the renderer. This crate holds the asset
semantics, the data modelling, and the interaction primitives that sit above
it. The JavaScript side has the same shape: `packages/vizij` and
`packages/scene` share their data modelling and their routing-to-render, and
diverge only in what they draw.

## The rule that keeps it shared

Its two consumers do not share a rendering model. Vizij's view is a 2D
compositing model — Z-up, assets in the XY plane layered along Z, an
orthographic camera fit to authored bounds, ambient-only light composed into
unlit materials. A robot scene is perspective, lit and shadowed.

**The moment this crate contains an up-axis, a camera convention or a lighting
model, it has excluded one of its two consumers.** The same line applies to
vocabulary: the concepts here are element, controllable, instance and view. A
type named for a face has already chosen a side.

## Two costs, and which to pay

A module that names a Bevy type binds **every** consumer of this crate to one
Bevy minor version. Bevy's derives produce concrete types, so a `Component`
from 0.19 and one from 0.20 are unrelated types that cannot meet; a version
skew is a wall of trait errors. Upgrading Bevy therefore stops being something
any one repo can do alone.

That coupling is worth paying for code that has to agree across products — it
is the same mechanism that stops the two renderers drifting between upgrades.
It is not worth paying for code that needs no Bevy type. **A module that can
stay free of Bevy should stay free of Bevy.** `tone` is the standing proof:
colour science, no imports at all, no lockstep.

Bevy 0.19 is the current line, matching Vizij.

## What is here

| module | what it is |
| -- | -- |
| `tone` | three.js's tone curves (r170) at exposure 1, so an asset authored against a three renderer reads the same under Bevy |

## Consumers

- `arora-scene` — robots and world: the robot reading of `RobotData`, joints
  and their limit-aware controls, and the screen another asset draws into.
- Vizij keeps its own local copies for now. Moving it onto this crate is a
  follow-up, and until then **this crate's copy is the one that takes fixes**:
  a defect fixed only in Vizij's copy is a defect this crate ships.

## Licence

Apache-2.0, matching the source this is extracted from.
