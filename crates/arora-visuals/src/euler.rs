//! Euler rotations written one component at a time.
//!
//! A RobotData rotation is an euler triple, and a device or an inspector may
//! write it per axis: `rotation.r = 0.1` changes the roll and must leave the
//! pitch and yaw as they were. The node holds a quaternion, though, and a
//! quaternion does not remember the angles it was made from: read back past
//! ±π/2 of pitch, it returns a different triple for the same rotation, so
//! replacing one component of the read-back triple changes the other two as
//! well.
//!
//! So [`EulerWrites`] keeps, per node, the angles last written together with
//! the rotation they produced, and a per-axis write starts from those angles
//! **only while the node still holds that rotation**. Anything else that
//! rotates the node — a gizmo drag, an animation, a rig, a host's own write —
//! leaves a different rotation there, and the write then starts from the
//! angles read back off it. The comparison is what notices those writers, so
//! a host need not list them or tell this module about them.
//!
//! # Equal means bit-for-bit, either sign
//!
//! The remembered rotation is compared exactly, and as `q` or `-q`.
//!
//! Exact, because the remembered rotation is the very value the host stored
//! on the node, and nothing in Bevy rewrites a local rotation on its own —
//! transform propagation reads it, and every method that changes it is a
//! writer called by someone. So an untouched node compares equal bit for bit.
//! A tolerance would be wrong in a way that shows: another writer moving the
//! node by less than it — a slow animation, a fine gizmo drag — would be taken
//! for no change, and the next per-axis write would snap the node back to the
//! remembered angles, losing that motion every frame it happened. Exact
//! comparison errs only the other way, reading back from a rotation that
//! really did change, which is what a changed rotation calls for.
//!
//! Either sign, because `q` and `-q` are one rotation, so the remembered
//! angles describe a node holding `-q` exactly as well as one holding `q`. A
//! writer that hands back the same rotation with its sign flipped — an
//! interpolation choosing the short way round does — has not changed what the
//! node shows, and must not cost it its angles.
//!
//! # The order
//!
//! A triple is `[x, y, z]` — roll, pitch, yaw, the order [`FeatureKind::axes`]
//! names them in — composed as `R = Rz(z)·Ry(y)·Rx(x)`, which is glam's
//! `EulerRot::ZYX` and three.js's `"ZYX"` order. It is not three.js's
//! default `"XYZ"`, which composes `Rx·Ry·Rz`, so a three.js page has to
//! name `"ZYX"` explicitly to agree with it. [`rotation_zyx`] and [`angles_zyx`]
//! are that convention in each direction, for a host to use wherever it
//! converts a rotation outside a write.
//!
//! [`FeatureKind::axes`]: crate::meta::FeatureKind::axes

use std::collections::HashMap;

use bevy_ecs::entity::Entity;
use bevy_math::{EulerRot, Quat};

/// The rotation an euler triple `[x, y, z]` gives: `Rz(z)·Ry(y)·Rx(x)`.
///
/// Composed axis by axis rather than through `EulerRot`, so the product it
/// stands for is written here and cannot move with how a glam version names
/// its orders.
pub fn rotation_zyx([x, y, z]: [f32; 3]) -> Quat {
    Quat::from_rotation_z(z) * Quat::from_rotation_y(y) * Quat::from_rotation_x(x)
}

/// The euler triple `[x, y, z]` read back off `rotation`, in the order
/// [`rotation_zyx`] composes. The pitch it returns is within ±π/2, so a
/// rotation made from a pitch outside that reads back as other angles.
pub fn angles_zyx(rotation: Quat) -> [f32; 3] {
    let (z, y, x) = rotation.to_euler(EulerRot::ZYX);
    [x, y, z]
}

/// The angles last written to one node, and the rotation they produced.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Written {
    angles: [f32; 3],
    rotation: Quat,
}

impl Written {
    fn of(angles: [f32; 3]) -> Self {
        Self {
            angles,
            rotation: rotation_zyx(angles),
        }
    }

    /// Whether `rotation` is still the one these angles produced.
    fn holds(&self, rotation: Quat) -> bool {
        rotation == self.rotation || rotation == -self.rotation
    }
}

/// What each node's rotation was last written from, by the node's entity.
///
/// A host keeps one wherever it keeps the rest of its write state, and
/// routes every rotation write it applies through it: whole writes through
/// [`write`](Self::write), per-axis ones through
/// [`write_axis`](Self::write_axis). Writes it does not route here are other
/// writers, and are noticed as such.
///
/// An entry for a node that is gone is never wrong — an `Entity` is not
/// reused with the same generation, and the rotation comparison would refuse
/// it anyway — only kept. [`retain`](Self::retain) drops those, and
/// [`clear`](Self::clear) drops everything, for a host whose scene is
/// replaced wholesale.
#[derive(Debug, Clone, Default)]
pub struct EulerWrites(HashMap<Entity, Written>);

impl EulerWrites {
    /// Write a whole euler triple to `entity`'s `rotation`, and remember it in
    /// place of whatever was remembered before.
    pub fn write(&mut self, entity: Entity, rotation: &mut Quat, angles: [f32; 3]) {
        let written = Written::of(angles);
        *rotation = written.rotation;
        self.0.insert(entity, written);
    }

    /// Write one component of `entity`'s euler triple — `axis` 0, 1 or 2 for
    /// `x`, `y` or `z` — keeping the other two as the node shows them
    /// ([`angles`](Self::angles)), and remember the result.
    ///
    /// # Panics
    ///
    /// If `axis` is not 0, 1 or 2.
    pub fn write_axis(&mut self, entity: Entity, rotation: &mut Quat, axis: usize, value: f32) {
        let mut angles = self.angles(entity, *rotation);
        angles[axis] = value;
        self.write(entity, rotation, angles);
    }

    /// The euler triple `entity` shows with `rotation` on it: the angles last
    /// written to it while it still holds what they produced, and otherwise
    /// the angles read back off `rotation`.
    pub fn angles(&self, entity: Entity, rotation: Quat) -> [f32; 3] {
        match self.0.get(&entity) {
            Some(written) if written.holds(rotation) => written.angles,
            _ => angles_zyx(rotation),
        }
    }

    /// Forget what `entity` was written, so its next per-axis write starts
    /// from its rotation alone.
    pub fn forget(&mut self, entity: Entity) {
        self.0.remove(&entity);
    }

    /// Keep only the entities `live` answers true for: how a host drops the
    /// nodes that have been despawned.
    pub fn retain(&mut self, mut live: impl FnMut(Entity) -> bool) {
        self.0.retain(|&entity, _| live(entity));
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::world::World;

    /// A pitch past π/2, which a quaternion reads back as other angles.
    const STEEP_PITCH: f32 = 2.0;

    fn close(a: Quat, b: Quat) -> bool {
        a.abs_diff_eq(b, 1e-6) || a.abs_diff_eq(-b, 1e-6)
    }

    fn node() -> (World, Entity) {
        let mut world = World::new();
        let entity = world.spawn_empty().id();
        (world, entity)
    }

    /// The order both hosts compose in: `R = Rz·Ry·Rx`, which is glam's
    /// `EulerRot::ZYX` taking `(z, y, x)`, and which reads back as the triple
    /// it was made from while the pitch is within ±π/2.
    #[test]
    fn the_triple_is_roll_pitch_yaw_composed_z_y_x() {
        for angles in [[0.3, -0.2, 1.1], [0.0, 0.0, 0.0], [-1.2, 0.4, -2.5]] {
            let [x, y, z] = angles;
            assert!(close(
                rotation_zyx(angles),
                Quat::from_euler(EulerRot::ZYX, z, y, x)
            ));
            let back = angles_zyx(rotation_zyx(angles));
            for (got, want) in back.iter().zip(angles) {
                assert!((got - want).abs() < 1e-5, "{angles:?} read back {back:?}");
            }
        }
        // Each component turns about its own axis.
        assert!(close(
            rotation_zyx([0.5, 0.0, 0.0]),
            Quat::from_rotation_x(0.5)
        ));
        assert!(close(
            rotation_zyx([0.0, 0.5, 0.0]),
            Quat::from_rotation_y(0.5)
        ));
        assert!(close(
            rotation_zyx([0.0, 0.0, 0.5]),
            Quat::from_rotation_z(0.5)
        ));
    }

    #[test]
    fn a_per_axis_write_keeps_a_steep_pitch() {
        let (_world, entity) = node();
        let mut writes = EulerWrites::default();
        let mut rotation = Quat::IDENTITY;
        writes.write_axis(entity, &mut rotation, 1, STEEP_PITCH);
        writes.write_axis(entity, &mut rotation, 0, 0.1);
        assert_eq!(rotation, rotation_zyx([0.1, STEEP_PITCH, 0.0]));
        assert_eq!(
            writes.angles(entity, rotation),
            [0.1, STEEP_PITCH, 0.0],
            "the angles shown are the ones written"
        );
        // What reading the angles back would have done instead.
        let read_back = {
            let mut angles = angles_zyx(rotation_zyx([0.0, STEEP_PITCH, 0.0]));
            angles[0] = 0.1;
            rotation_zyx(angles)
        };
        assert!(
            !close(rotation, read_back),
            "the case is one read-back fails"
        );
    }

    #[test]
    fn a_rotation_another_writer_left_is_where_a_per_axis_write_starts() {
        let (_world, entity) = node();
        let mut writes = EulerWrites::default();
        let mut rotation = Quat::IDENTITY;
        writes.write(entity, &mut rotation, [0.0, STEEP_PITCH, 0.0]);
        // A gizmo, an animation or a rig turns the node about z.
        let dragged = Quat::from_rotation_z(0.7) * rotation;
        rotation = dragged;
        assert_eq!(writes.angles(entity, rotation), angles_zyx(dragged));
        writes.write_axis(entity, &mut rotation, 0, 0.1);
        let mut expected = angles_zyx(dragged);
        expected[0] = 0.1;
        assert_eq!(rotation, rotation_zyx(expected));
        assert!(
            !close(rotation, rotation_zyx([0.1, STEEP_PITCH, 0.0])),
            "the drag is not snapped back"
        );
    }

    /// A change smaller than any tolerance worth having is still a change:
    /// taken for none, the write would snap it back every time it happened.
    #[test]
    fn a_tiny_change_by_another_writer_is_still_a_change() {
        let (_world, entity) = node();
        let mut writes = EulerWrites::default();
        let mut rotation = Quat::IDENTITY;
        writes.write(entity, &mut rotation, [0.0, 0.5, 0.0]);
        let nudged = Quat::from_rotation_z(1e-4) * rotation;
        rotation = nudged;
        writes.write_axis(entity, &mut rotation, 0, 0.1);
        let mut expected = angles_zyx(nudged);
        expected[0] = 0.1;
        assert_eq!(rotation, rotation_zyx(expected));
        assert!(!close(rotation, rotation_zyx([0.1, 0.5, 0.0])));
    }

    /// `-q` is the rotation `q` is, so the angles that made `q` still
    /// describe it.
    #[test]
    fn the_same_rotation_with_its_sign_flipped_keeps_the_angles() {
        let (_world, entity) = node();
        let mut writes = EulerWrites::default();
        let mut rotation = Quat::IDENTITY;
        writes.write(entity, &mut rotation, [0.0, STEEP_PITCH, 0.0]);
        rotation = -rotation;
        writes.write_axis(entity, &mut rotation, 0, 0.1);
        assert_eq!(rotation, rotation_zyx([0.1, STEEP_PITCH, 0.0]));
    }

    #[test]
    fn a_whole_write_replaces_what_is_remembered() {
        let (_world, entity) = node();
        let mut writes = EulerWrites::default();
        let mut rotation = Quat::IDENTITY;
        writes.write_axis(entity, &mut rotation, 1, 0.3);
        writes.write(entity, &mut rotation, [0.2, STEEP_PITCH, -0.4]);
        assert_eq!(rotation, rotation_zyx([0.2, STEEP_PITCH, -0.4]));
        writes.write_axis(entity, &mut rotation, 2, 0.6);
        assert_eq!(rotation, rotation_zyx([0.2, STEEP_PITCH, 0.6]));
    }

    /// A node nothing was written to starts from the rotation it has — its
    /// GLB's, or whatever placed it — not from zero.
    #[test]
    fn a_node_never_written_starts_from_its_own_rotation() {
        let (_world, entity) = node();
        let mut writes = EulerWrites::default();
        let mut rotation = rotation_zyx([0.3, 0.2, -0.1]);
        writes.write_axis(entity, &mut rotation, 0, 0.5);
        let expected = rotation_zyx([0.5, 0.2, -0.1]);
        assert!(
            close(rotation, expected),
            "{rotation:?} is not {expected:?}"
        );
    }

    /// Remembering one node says nothing about another, a node spawned
    /// after it was despawned included.
    #[test]
    fn a_despawned_node_is_dropped_and_its_successor_starts_fresh() {
        let (mut world, gone) = node();
        let mut writes = EulerWrites::default();
        let mut rotation = Quat::IDENTITY;
        writes.write(gone, &mut rotation, [0.0, STEEP_PITCH, 0.0]);
        world.despawn(gone);
        let next = world.spawn_empty().id();
        assert_ne!(next, gone);
        let mut next_rotation = rotation_zyx([0.3, 0.2, -0.1]);
        writes.write_axis(next, &mut next_rotation, 0, 0.5);
        assert!(close(next_rotation, rotation_zyx([0.5, 0.2, -0.1])));

        writes.retain(|entity| world.get_entity(entity).is_ok());
        assert!(
            !writes.0.contains_key(&gone),
            "the despawned node is dropped"
        );
        assert!(writes.0.contains_key(&next), "a live node is kept");
        writes.clear();
        assert!(writes.is_empty());
    }

    #[test]
    fn a_forgotten_node_reads_its_rotation_back() {
        let (_world, entity) = node();
        let mut writes = EulerWrites::default();
        let mut rotation = Quat::IDENTITY;
        writes.write(entity, &mut rotation, [0.0, STEEP_PITCH, 0.0]);
        writes.forget(entity);
        assert_eq!(writes.angles(entity, rotation), angles_zyx(rotation));
    }
}
