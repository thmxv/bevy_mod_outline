use bevy::{
    color::palettes::tailwind::*, prelude::*, render::mesh::VertexAttributeValues,
    scene::SceneInstance,
};

use bevy_mod_outline::*;

#[bevy_main]
fn main() {
    App::new()
        .insert_resource(Msaa::Sample4)
        .insert_resource(ClearColor(Color::BLACK))
        .add_plugins((
            DefaultPlugins,
            OutlinePlugin,
            //AutoGenerateOutlineNormalsPlugin,
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, (setup_scene_once_loaded, draw_normals))
        .run();
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut config_store: ResMut<GizmoConfigStore>,
) {
    commands
        .spawn(SceneBundle {
            scene: asset_server.load("non_manifold.glb#Scene0"),
            ..default()
        })
        .insert(OutlineBundle {
            outline: OutlineVolume {
                visible: true,
                width: 3.0,
                colour: Color::srgb(1.0, 0.0, 0.0),
            },
            ..default()
        })
        .insert(AsyncSceneInheritOutline);

    // Add light source, and camera
    commands.spawn(PointLightBundle {
        point_light: PointLight {
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(4.0, 8.0, 4.0),
        ..default()
    });
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(0.0, 5.0, 14.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });

    // Config gizmos
    let (config, _) = config_store.config_mut::<DefaultGizmoConfigGroup>();
    config.line_width = 1.0;
    //config.depth_bias = -1.0;
}

fn setup_scene_once_loaded(
    mut commands: Commands,
    scene_query: Query<&SceneInstance>,
    scene_manager: Res<SceneSpawner>,
    object_query: Query<(Entity, &Name, &Handle<Mesh>)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut done: Local<bool>,
) {
    if !*done {
        if let Ok(scene) = scene_query.get_single() {
            if scene_manager.instance_is_ready(**scene) {
                for (entity, _name, mesh_handle) in &object_query {
                    let mesh = meshes.get_mut(mesh_handle).unwrap();
                    //let _ = mesh.generate_outline_normals();
                    let _ = mesh.modify_for_non_manifold_outlines();
                    commands.entity(entity).insert(OutlineBundle {
                        outline: OutlineVolume {
                            visible: true,
                            colour: Color::WHITE,
                            width: 16.0,
                        },
                        //mode: OutlineMode::RealVertex,
                        ..default()
                    });
                }
                *done = true;
            }
        }
    }
}

fn draw_normals(
    object_query: Query<(&GlobalTransform, &Handle<Mesh>), With<OutlineVolume>>,
    meshes: ResMut<Assets<Mesh>>,
    mut gizmos: Gizmos,
) {
    for (transfrom, mesh_handle) in &object_query {
        let mesh = meshes.get(mesh_handle).unwrap();
        let positions = mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap();
        let VertexAttributeValues::Float32x3(positions) = positions else {
            continue;
        };
        let normals = mesh.attribute(ATTRIBUTE_OUTLINE_NORMAL).unwrap();
        let VertexAttributeValues::Float32x3(normals) = normals else {
            continue;
        };
        for (position, normal) in positions.iter().zip(normals) {
            let position = Vec3::from_array(*position);
            let position = transfrom.transform_point(position);
            let (_, rotation, _) = transfrom.to_scale_rotation_translation();
            let normal = 0.2 * Vec3::from_array(*normal);
            let normal = rotation * normal;
            gizmos.rect(position, rotation, Vec2::splat(0.05), RED_600);
            gizmos.ray(position, normal, CYAN_600);
        }
    }
}
