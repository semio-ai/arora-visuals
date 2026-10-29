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
vocabulary: the concepts here are element, controllable, entity instance and
view. A type named for a face has already chosen a side.

**An entity is Model + Behavior running on one engine instance, and entities
nest** — a robot entity contains the entity on its screen. That is the Arora
architecture's definition, and it is why a screen showing another asset is a
modelling relationship rather than something a renderer invents.

**Take care naming anything near it.** "Entity" means something different at
three other places in the stack: Arora's key grammar makes the *joint* the
entity (`robot1/joint1.position`), which is what this crate calls an element;
Studio splits it into `entityId` for the model and `entityInstanceId` for the
occurrence; and Bevy's `Entity` is the ECS id, which appears as a field inside
the very types that would carry the name. "Binding" is spoken for the same way
— at the architecture level it pairs an entity with an engine instance, not an
animatable with the thing it moves.

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

| module | what it is | Bevy |
| -- | -- | -- |
| `meta` | the `RobotData` glTF extension, read out of a GLB's JSON chunk: elements, the animatables bound to them, and the asset's bounds | `bevy_math` |
| `hold` | an authoring lock: rig outputs held against the device that writes them, per instance and per component | `bevy_ecs` |
| `target` | `OffscreenTarget` — where a view draws when it does not draw to a window | `bevy_asset`, `bevy_image` |
| `value` | `Decoded` — a feature value in the two shapes a renderer applies | none |
| `tone` | three.js's tone curves (r170) at exposure 1, so an asset authored against a three renderer reads the same under Bevy | none |
| `snapshot` | headless offscreen render and pixel readback, behind the `snapshot` feature | the whole engine |

`snapshot` is gated because it is the only module here that builds a real
`DefaultPlugins` app, and nothing smaller will do that. Off, the crate resolves
186 dependencies; on, 331. A consumer that wants the asset parse should not pay
for an engine to get it.

It exists because **a frame is the only artifact that answers the question a
renderer is asked.** Every cheaper signal is a proxy that can be right while
the picture is wrong — a mesh that spawns, passes culling and is counted among
the visible still draws nothing if its material has no shader.

`hold` is shared for a subtler reason than most of this. **A hold is the view
declining a write, not the device pausing**, so a release shows the device's
current value at once rather than resuming from where the hold began. Two
implementations of a lock would disagree about that one point first, and the
disagreement would surface only as values jumping on release.

`meta` is shared for a reason that has nothing to do with what an asset is:
**Bevy's glTF loader parses the extensions it knows and drops the rest**, so
every Semio asset read in Rust needs its own pass over the JSON chunk. It
already reads both feature models — the web's whole-feature `translation` and
Studio's per-axis `translation.x` — which is most of why one parse can serve
two products.

A consumer with its own extension alongside `RobotData` calls `glb_json_chunk`
once and hands the same JSON to both parses. It composes rather than extends:
it owns a struct holding an `AssetMeta` and its own parse, so neither can grow
a dependency on the other.

## Consumers

- `arora-scene` — robots and world: the robot reading of `RobotData`, joints
  and their limit-aware controls, and the screen another asset draws into.
- Vizij keeps its own local copies for now. Moving it onto this crate is a
  follow-up, and until then **this crate's copy is the one that takes fixes**:
  a defect fixed only in Vizij's copy is a defect this crate ships.

## Licence

Apache-2.0, matching the source this is extracted from.
