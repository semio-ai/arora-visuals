# The instance-rooted ECS pattern

How the Bevy layer — this crate, arora-scene and vizij — models what it draws,
measured against one pattern, with what blocks it from growing into a full
Studio ECS and the changes that close the gap.

**Words.** *Entity instance* (or *instance*) is one live occurrence of an Arora
entity — Model + Behavior on one engine instance — and is a Bevy entity
carrying `EntityInstance`. *Element* is a RobotData element: a glTF node with
an id. *Animatable* is a RobotData value a device writes, by its id. *Bevy
entity* always means the ECS id. Arora's key grammar calls the last path
segment the "entity"; that reading is written *key entity* here.

**Sources.** Line references are to these states:

| Tag | Repo and state |
|---|---|
| `AV` | arora-visuals `main` |
| `AS` | arora-scene `feat/shared-controls` (moves it onto this crate's control layer; its `main` still carries a local copy) |
| `VZ` | vizij-rs `feat/vizij-arora-visuals` (moves vizij onto this crate) |
| `AK` | arora-sdk `crates/arora-types/src/data/state.rs`, `crates/arora-hal/src/target.rs` |

The `fix/ghost-look` branch of this crate is counted where it changes ghost
material handling.

## 1. The pattern

Three requirements shape it. The scene and vizij must stay usable with no
Studio present. It must meet Arora where Arora already is: devices own store
namespaces, a key is a `/` path whose last segment is the key entity, and a
`.attribute` on that segment is a convention, not structure (`AK` state.rs:104-130).
And each rule must be one a Studio ECS could later be built on rather than
around.

### Rules

| # | Rule | Why |
|---|---|---|
| R1 | **An instance is one Bevy entity, found by an index, not a scan.** The host's id is an opaque string on `EntityInstance`; one shared index maps id → Bevy entity, kept by component hooks. | Every subsystem currently resolves the id by iterating instances. A Studio `entityInstanceId`, a vizij page id and an Arora path must all work unchanged, so the id stays opaque; the lookup must be cheap and agreed. |
| R2 | **Elements are Bevy entities with an `ElementNode { id }` component and a relationship to their instance.** `SceneBindings` maps are an index derived from them. | Element-level queries, change detection and relationships (a nested face mounted on a screen element) are only expressible if the element is in the ECS. A map inside a component is invisible to queries. |
| R3 | **What a host asks for is a component on the thing it is about; systems derive render state from it.** Selection, holds, placement, view, ghost, static feature values, active area: intent components on the instance or element. | Intent kept in render state (a material, a transform) cannot be re-derived, is lost on reload, and cannot be told apart from the effect. Intent in a string-keyed resource outlives its subject and needs bespoke cleanup per table. |
| R4 | **One value path.** Every value source — a host push, a device's pose, an animation — writes the latest value per animatable into one component on the instance; one shared system applies changed values to the scene, in a public set. | Two hosts currently carry parallel copies of index-and-apply. With values as data, a Studio ECS binds to them; with render state as the store, it has nothing to bind to. |
| R5 | **A model is shared; an instance owns only what it changes.** The GLB scene, its parsed `AssetMeta` and joint template are loaded once per content; an instance copies a material only when it writes it. | Studio already splits model (`entityId`) from occurrence (`entityInstanceId`). Per-instance reloads double parse, spawn and GPU cost for every copy, and cannot express "the same model". |
| R6 | **Arora keys are resolved at the host edge.** The shared layer addresses `(instance, animatable)` and `(instance, element, feature)`. A host maps a device namespace to an instance, and a store key under it to an animatable. | Arora has no entity namespace — devices own namespaces, and a nested face becomes a HAL under its robot's device. Which instance a namespace drives is a routing decision only the host can make; baking a key grammar in here would fix one routing for every product. |
| R7 | **This crate defines the frame contract.** Public system sets for intake, index, apply and derive; hosts order against them and export sets for their own chains. | Cross-crate ordering is currently `before(ApplyValues)` in one host and an unexported `.chain()` in the other. A third consumer (Studio) cannot slot in without a contract. |
| R8 | **No process-wide state per host.** Mount state, request queues and asset directories are per-app resources, not statics. | Nesting means two hosts in one app. This crate already designs for that (`ControlsPlugin`, `GhostPlugin` install once, `AV` control.rs:21-28, ghost.rs:154); the hosts' wasm statics do not. |
| R9 | **No product vocabulary or defaults decide behaviour here.** Studio's look may be a default; Studio's key spellings may not be a requirement. | The crate's own rule (`AV` lib.rs:8-19) — extended from axes and lighting to identity and keys. |

### Target shape

```
instance root (Bevy entity)
  EntityInstance { id, meta: Handle<Model> }   intent: Selected, Holds, Placement, ViewFraming, Ghost, GhostStyle
  AnimatableValues (changed → applied)          host data: JointRig, Face, RobotAsset …
  ├─ scene (spawned from the shared model)
  │    ├─ element node: ElementNode { id }, ElementOf(root)  [StaticFeatures]
  │    │     └─ mesh: own material only once written
  │    └─ … nested instance root (a face on a screen element), ChildOf the element
  └─ camera: ViewOf(root)   (Relationship, not two hand-kept fields)
```

Frame contract (R7), all in `Update` unless stated:

| Set | What runs | Owner |
|---|---|---|
| `InstanceSystems::Intake` | host queues → intent components and `AnimatableValues` | hosts |
| `InstanceSystems::Index` | join elements once the scene spawns | shared |
| `ControlSystems::Commands` | control commands | shared (exists) |
| `InstanceSystems::Apply` | changed values → Transform / MorphWeights; rigs (`JointRig`) in a sub-set | shared + host rigs |
| `InstanceSystems::Derive` | materials, ghosts, outline, tone from intent and values | shared + host material policy |
| `FramingSystems` (`PostUpdate`) | camera framing | shared (exists) |

## 2. Where each codebase stands

Verdict per rule: **meets**, **partly**, **misses**.

### arora-visuals (`AV`)

| Rule | Verdict | Evidence |
|---|---|---|
| R1 | partly | `EntityInstance.id: String` is opaque (instance.rs:26-28) and `Picked` carries both Entity and id (pick.rs:62-67). Resolution is a scan: control.rs:1253-1259; outline reads `selections.0.get(&instance.id)` (outline.rs:359). |
| R2 | misses | Elements exist only as `SceneBindings.elements: HashMap<String, ElementTargets>` inside the component (binding.rs:53-66). `select::covered` walks every element and ancestor chain to find a group's meshes (select.rs:74-104). |
| R3 | partly | Components for `Ghost`, `ActiveArea`, `ControlSize`, `ControlView`, `DriverFrame` (on cameras). Resources keyed by instance string for `Selections` (select.rs:41-46) and `Holds` (hold.rs:84-86), deliberately so they survive a reload; each host must clear them on unload itself (both do, by hand). `Controls(HashMap<String, Placed>)` (control.rs:692-693) is a driver-key table, which is legitimate: control keys are the driver's namespace, not an instance's. Ghost on `main` swaps `MeshMaterial3d` and keeps the original in `GhostOf` (ghost.rs:47-54, 189-199), which hides the material from any writer querying `MeshMaterial3d<StandardMaterial>`; `fix/ghost-look` adds `SolidMaterial` so writers reach it either way, and shares ghost materials in a `GhostMaterials` resource. |
| R4 | misses | No apply code here. `binding.rs:12-15` says pose state belongs "with the code that applies it", so each host keeps its own. `Decoded` (value.rs) is the only shared value type. |
| R5 | misses | `EntityInstance.meta: AssetMeta` by value and `asset_path: String` per instance (instance.rs:29-31); no model type. |
| R6 | partly | Addresses are `(instance, animatable)` and `(instance, element)` throughout. Two leaks: `HoldTarget::from_str` accepts a rig path `rig/<instance>/<id>` and takes the last segment (hold.rs:39-61), and `AssetMeta.unbound` lists "the keys Studio writes" (meta.rs:296-299). |
| R7 | partly | Public: `ControlSystems` (control.rs:782-793), `ControlPass`, `FramingSystems` (active_area.rs:76-77), `ViewLayerSystems`, `TransformGizmoSystems`. No instance/value sets. |
| R8 | meets | Plugins that two hosts may both add install once (`installed::first_install`). Controls are explicitly app-wide with one key space (control.rs:21-28). |
| R9 | partly | Studio's ghost look, highlight rule and spring timing are defaults, named as such. Per-axis feature names are parsed as Studio writes them (meta.rs:22-27), which is a RobotData format, not a Studio dependency. |

The relationship `EntityInstance.camera` ↔ `ViewOf(Entity)` (instance.rs:34-35, 66-67) is kept by hand on both ends; a Bevy `Relationship` would keep it.

### arora-scene (`AS`)

| Rule | Verdict | Evidence |
|---|---|---|
| R1 | partly | Host id is any string — "an Arora path, a slot name, whatever the host uses" (robots.rs:4-5) — stored twice (`RobotAsset.instance`, `EntityInstance.id`). Resolved by scans in `apply_values` (values.rs:197-219). |
| R2 | misses | Element join by glTF `Name` over all descendants, first match wins (instance.rs:82-87); no element component. Joints: `JointRig(HashMap<String, Vec<JointBinding>>)` on the root (joints.rs:111-112); bodies carry nothing. |
| R3 | partly | Selection → `Selections` resource; ghost → `Ghost` component (lib.rs:1230-1241); active area → component on the camera. `setPose` inserts `Transform` on the root (robots.rs:307-326), so the authored pose and render state are one value. |
| R4 | misses | `setValue` → wasm `PUSHED` static → `ValueWrites(Vec)` resource → `apply_values` in public `ApplyValues` (values.rs:60-73). Writes go straight into `Transform`, `MorphWeights` and `Assets<StandardMaterial>`; no component records a value. Joint values are read back from `Transform` (controls.rs:136-139). Emissive colour and intensity live in `Emission(HashMap<AssetId<StandardMaterial>, Emitting>)` (values.rs:98-114) — domain values keyed by a render asset id, carried across copies by `copied()`. Material copies are `Local`s keyed by `(root, AssetId)` and mesh `Entity` (`InstanceMaterials`, `OwnMaterials`, values.rs:75-95). |
| R5 | misses | Every wasm load is served at a fresh `robot-{n}.glb` "so the asset server never answers with one it held before" (wasm lib.rs:57-58, 135). `AssetMeta` and joints parsed per load, cloned onto `RobotAsset` and again into `EntityInstance.meta`. |
| R6 | meets | Writes are `(instance, target, feature?)`; Arora keys never parsed. `None` instance means every instance (values.rs:205-206). |
| R7 | partly | Public `ApplyValues`, `RobotRequestSystems`; `index_instances.before(ControlSystems::Commands).before(ApplyValues)` (lib.rs:218-223). After the shared-controls branch, no duplicated control layer remains: joints resolve in `ControlSystems::Joints` (controls.rs:93-154), no adapter key map. |
| R8 | misses | wasm statics `MOUNTED`, `ROBOTS`, `REQUESTS`, `PUSHED`, `CONTROLS`, thread-local `POSES` (wasm lib.rs:56-62, 547, 714-761). One scene per page. |
| R9 | partly | Z-up driver frame, Studio camera bindings, floor, lights, ghost wash are scene defaults (fine). `split_materials` rewrites every mesh's material once when `unique_materials` is on (lib.rs:1595-1624) — a global render mutation, not per instance. |

One camera for every instance: `index_instances` gives each the single `OrbitCamera` and forces `draw_order: 0, ready: true` (instance.rs:71-78, 215-225), so this crate's per-instance readiness model is bypassed.

### vizij (`VZ`)

| Rule | Verdict | Evidence |
|---|---|---|
| R1 | partly | Host id called `face_id` in Rust, `instance` in wasm/TS; resolved by scans `instance.id == face_id` (view/mod.rs:817, 901, 1152, 1341). `bundle.face_id` is a different thing (the GLB's `faceId`) and keys the rig prefix and TF frame. Native uses the constant `"face"`. |
| R2 | misses | Same `SceneBindings` join; plus `PoseState.surfaces: Vec<Entity>` and `assembled: HashMap<(Entity, FeatureKind), [f32; 3]>` on `Face` (view/mod.rs:245-269). |
| R3 | misses | Per-face intent in string-keyed resources: `FaceViews`, `Placements` (never cleared, so they survive reload and outlive the face), `PendingStatics`, `Compositions`, `SafeAreas` (view/mod.rs:366-420). Static features land straight in `Transform`/`Surface`/material; "a reload loses it" (view/mod.rs:330-331). Global `FrameConfig.face_frame_id` overwritten by any load. |
| R4 | partly | One value source per face: the device's `RigHal`, read each frame by `apply_poses` walking the whole pose under its mutex (view/mod.rs:1944-1962); `RigHal::pose_updates` exists and is unused. `PoseState.shown` is the nearest thing to an `AnimatableValues` component anywhere in the three codebases. A host can also push statics (`SetStaticFeature`). `index_faces` + `SceneFeatures` duplicate arora-scene's index and apply, with holds, per-axis assembly and `Surface` composition added. |
| R5 | misses | `FaceAssets::push` serves every load at a fresh path because "the asset server caches a handle by its path" (view/mod.rs:115-150); every mesh gets its own material at index; each load builds its own device and store. |
| R6 | partly | View addresses bare animatable ids; the device side uses `rig/<faceId>/…` paths and `studio.rs` maps Studio's target grammar (`<id>/target.position`, `<id>.target_position`, `AK` target.rs:110-135) onto bare ids. That mapping sits in the right place (a profile, at the edge); the core still matches store paths to animatable ids verbatim (view/mod.rs:1958-1962). |
| R7 | misses | One long unexported `.chain()` in `ViewPlugin` (view/mod.rs:697-715); answering systems ordered `.before(ControlSystems::Commands)`. No local `SystemSet`. |
| R8 | misses | wasm thread-locals `PAGE`, `DRAWN`, `HANDLER`, `KEYS`, `AREAS`, static `READY` (web.rs:85-103); one global `OffscreenTarget` all face cameras draw to (view/mod.rs:606); `clear_target` assumes one `ClearCamera`. |
| R9 | meets | Studio's key spellings are confined to `studio.rs` behind a feature. |

Faces are laid out `SLOT_STRIDE = 1000` apart on X and kept apart only by distance (view/mod.rs:609-642); cameras are placed from the slot, not the face's `GlobalTransform`.

## 3. Host API parity

Both handles take an opaque `instance` string first and emit the same core
events, because both serialise this crate's types. The divergences:

| Surface | `@semio/scene` | `@vizij/runtime` | Note |
|---|---|---|---|
| Collection | `robots: ReadonlyMap<string, ShownRobot>` | `faces: ReadonlyMap<string, ShownFace>` | Same concept; product nouns. `loading` matches. |
| Load | `load(instance, glb, {key?, pose?}) → Promise<void>` | `load(instance, glb, {…VizijOptions, key?}) → Promise<Runtime>` | vizij returns the device handle. Both document `key` as the caller's asset id. |
| Position | `setPose(instance, Pose)` — world pose | `place(instance, Rect \| "fill")`, `placeIn`, `setView` — screen rectangle and framing | Different concepts sharing no name; `setPose` is also a control op name. |
| Values | `setValue(instance, target, value, feature?)` | `setValue(instance, target, value, feature)`; wasm `setValue(instance, target, feature, values)` | TS order matches; vizij refuses a whole-animatable write (statics only) — the same name means "write an animatable" in one and "set a static feature" in the other. |
| Selection, control, pick, overlay, ready, loaded, loadFailed | same names and shapes | same | Shared serde types. |
| Control target | `instance`, deprecated `robot`, `element`, `goal{pose}` | `instance`, `element`, `goal{pose, view?}` | scene TS omits `goal.view`. |
| `safeArea` | — | event + map | vizij only; scene uses `setActiveArea`. |
| Overlay `GizmoKind` | `"Transform" \| "Joint"` | same | Capitalised, unlike every other `kind`. |
| Joint controls | resolved (`JointRig`) | refused at the API with an error | Faces have no scene-style joints (§7). |

Rule going forward: a concept both hosts have takes one name and one shape,
defined once (serde types here, TS types generated or shared), and a
product-specific concept gets a name that cannot be mistaken for a shared one.

## 4. Nesting: a face on a robot's screen

What blocks it today, in the order it would be hit:

1. **Two hosts cannot share an app** — wasm statics in both (R8).
2. **No per-instance render target.** vizij draws every face to one `OffscreenTarget`; arora-scene's screen mount exists natively only, is a singleton resource, and finds its node by a world-wide name scan with `Local<bool>` (lib.rs:869-886).
3. **Instance-relative layout is missing.** Face cameras are placed from the slot, not the face's transform; all faces share render layer 0 and are isolated by distance, so scene cameras would see them.
4. **Indexing walks into children.** `index_instances` joins by name over all descendants (first wins) and ready counts every descendant mesh, so a nested face's nodes would join into the robot's element table.
5. **Ghost and global material passes reach nested instances**: `Ghost` covers everything beneath it (by design — likely right), `split_materials` covers every mesh.
6. **Flat id space with no parent.** Nothing says an instance is inside another; `Selections`, `Holds`, overlay anchors and picks report a flat instance id.
7. **The device lives outside the world.** A face's `RigHal` is a handle held on `Face`; in Arora's model the face becomes a HAL under the robot device's namespace, so the host must route that namespace prefix to the nested instance (R6).

R2 (element entities) plus R1 (instance index) plus a `ChildOf` from the
nested root to the screen element is the ECS shape; render-to-texture per
instance and instance-relative cameras are the rendering work.

## 5. Roadblocks to a Studio ECS, and what to do as we go

| Roadblock | Why it blocks | Mitigation, taken incrementally |
|---|---|---|
| Render state is the value store (materials, `Transform`, `Emission` by `AssetId`) | A Studio ECS binds editors, undo and recording to values; there are none to bind to | Introduce `AnimatableValues` first as a write-through record beside the existing apply; switch readers (joint read-back, hold resolution) to it; then make apply consume only changes |
| String-keyed per-instance resources | No change detection, no despawn cleanup, scans per lookup | Add the instance index (R1) now; move each table to a component when its code is next touched; survive-reload needs a stable root (below) |
| Reload spawns a new root (`Superseded`) | Anything per instance either dies on reload or must live outside the entity — which is why `Selections`, `Holds`, `FaceViews`, `Placements` are resources | The instance root is stable for the instance's life and a reload swaps its scene child (§7); readiness gates the new child, not a new root |
| Elements are map entries | Element-level selection, inspection, relationships (mounting, attachment) need queries | Tag nodes with `ElementNode { id }` + `ElementOf` from the bindings after the index pass; keep `SceneBindings` as the index values apply through |
| Per-instance asset loads | Model/instance split is Studio's core identity split; per-instance loads double cost and hide "same model" | Content-addressed asset paths (hash of bytes) so equal GLBs share a handle and different ones cannot collide; `Model` asset holding `AssetMeta` and joint template |
| Duplicated index/apply in two hosts | A third consumer would be a third copy; fixes land in one | Move element join, transform/morph apply and material writes here behind `InstanceSystems`, through one material API (§7); rig forms stay with hosts |
| No change detection on value intake | Cost grows with animatable count, not with changes; hard to record | vizij: consume `pose_updates`; scene: already event-driven, keep it. The shared apply runs on `Changed<AnimatableValues>` and on two triggers with no source change: a hold released, which shows the device's current value at once (`hold.rs`), and a scene child newly indexed under a stable root, which replays the instance's values. `AnimatableValues` keeps the latest source value of a held animatable, so a release has it to show |
| Process-wide statics | Blocks nesting and any multi-view Studio page | Move into per-app resources behind the existing wasm entry points |
| Ids | Already opaque strings, so Studio ids fit | Keep them opaque; never parse an instance id; add parent relation rather than path-shaped ids |
| Hand-kept relationships (`camera`/`ViewOf`, `PoseState.surfaces`, caption `face`) | Drift on despawn | Bevy `Relationship` components |

## 6. Ranked changes

Ranked by what each unblocks per unit of risk. Size: S < 1 day, M a few days,
L a week or more.

| # | Change | Repo | Size | Risk |
|---|---|---|---|---|
| 1 | Instance index: `id → Entity` maintained by hooks on `EntityInstance` (and `Loading`), honouring `Superseded`; replace the scans | arora-visuals, then scene and vizij | S | Low |
| 2 | Export frame sets: `InstanceSystems::{Intake, Index, Apply, Derive}` here; vizij wraps its chain in an exported set | arora-visuals, vizij | S | Low |
| 3 | `ElementNode { id }` + `ElementOf` on joined nodes, reconciled with the bindings after the index pass; `select::covered` uses relationships | arora-visuals, scene, vizij | S–M | Low |
| 4 | `AnimatableValues` component written through by both apply paths; arora-scene's `Emission` and joint read-back move onto it | arora-visuals, scene, vizij | M | Medium — material copy rules must not change |
| 5 | Host API alignment, additive: one noun for the instance map (alias `robots`/`faces` to `instances`), expose `goal.view` in scene TS, rename vizij's static-feature `setValue` with the old name kept as a deprecated alias, shared TS types for control/overlay | scene, vizij | S–M | Low — additive with deprecations |
| 5b | Lowercase `GizmoKind` in the overlay and control payloads. Not additive: it changes what every host emits today, so it ships behind an opt-in or a versioned payload that keeps the current spelling as the default until both hosts' consumers move | arora-visuals, scene, vizij | S | Medium — a breaking wire change if not opted into |
| 6 | Shared index, transform/morph apply and one material API moved here; host rigs plug into `Apply` | arora-visuals, then both hosts | L | Medium–High — vizij holds, per-axis assembly, contended keys, vizij's surfaces moving onto the common material API |
| 7 | Content-addressed model loading and a shared `Model`; per-instance material copy-on-write only | arora-visuals, both hosts | M | Medium — asset lifetime in wasm memory |
| 8 | Stable instance root across reload; move `Selections`, `Holds`, `FaceViews`, `Placements`, statics into components | arora-visuals, both hosts | M–L | Medium — readiness and swap ordering |
| 9 | Per-app resources replace wasm statics | scene-wasm, vizij web | M | Medium |
| 10 | vizij consumes `pose_updates` deltas | vizij | S–M | Low |
| 11 | Nesting: per-instance render target, instance-relative cameras and layers, indexer stops at nested roots | all three | L | High |

Changes 1–3 are independent and can land in any order; 4 precedes 6; 8
precedes moving survive-reload intent onto components; 9 and 11 follow 1, 2
and 8.

## 7. Settled design

- **An instance is one long-lived Bevy entity.** It stands for the instance
  from its first load to its unload. The loaded file's content is its child,
  and a reload replaces that child: the old child keeps drawing until the new
  one is ready, then goes. Per-instance state — selections, holds, views,
  placements, values — is components on the instance entity, so it survives
  every reload with no table keyed by the host's id. Built by change 8.
- **One material API, in this crate.** Colour, opacity, emissive and the other
  material features are written through one shared material path for every
  host. Nothing assumes a host's surfaces are unlit: vizij draws unlit
  surfaces today, and may draw lit, PBR ones later, so the shared path must
  serve both. Change 6 carries this.
- **Arora keys are routed by the hosts.** The shared layer addresses
  `(instance, animatable)` and `(instance, element, feature)` only (R6); a host
  maps a device namespace to an instance. A shared routing helper waits until
  nesting needs one.
- **Faces take no joint controls.** vizij refuses a joint control add at the
  API with an error the driver hears. A face is more likely to gain a skeleton
  with bones than scene-style joints, so joint controls are not the form a
  face's rig control will take.
