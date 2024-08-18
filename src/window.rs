// window.rs

use bevy::{
    prelude::*,
    window::{PrimaryWindow, WindowMode, WindowResolution},
};

const WINDOW_WIDTH: f32 = 800.0;
const WINDOW_HEIGHT: f32 = 600.0;

pub fn window_setup(
    mut commands: Commands,
    mut window_query: Query<&mut Window, With<PrimaryWindow>>,
) {
    let mut window = window_query.single_mut();
    window.resizable = false;
    window.resolution = WindowResolution::new(WINDOW_WIDTH, WINDOW_HEIGHT);
    window.mode = WindowMode::Windowed;
    window.title = "Plotwar".to_string();

    commands.spawn(Camera2dBundle {
        transform: Transform::from_xyz(window.width() / 2.0, window.height() / 2.0, 0.0),
        ..default()
    });
}
