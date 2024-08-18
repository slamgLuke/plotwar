// main.rs
// project: plotwar
// author: slamgLuke
// A Bevy game.

#[allow(dead_code)]
mod parser;
mod window;

use crate::window::window_setup;
use bevy::prelude::*;

fn main() {
    println!("Running Bevy!");
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, window_setup)
        .run();
}
