//! Command-line interface for icon-to-image.
//!
//! Renders Font Awesome icons to PNG or WebP image files.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

use icon_to_image::{encode, Color, IconRenderer, ImageFormat, RenderConfig};

/// Render Font Awesome icons to image files.
#[derive(Parser)]
#[command(name = "icon-to-image")]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Icon name (e.g., 'heart', 'github', 'star')
    #[arg(value_name = "ICON")]
    icon: Option<String>,

    /// Output file path (e.g., 'icon.png' or 'icon.webp')
    #[arg(value_name = "OUTPUT")]
    output: Option<PathBuf>,

    /// Icon color as hex (e.g., '#FF0000' or 'FF0000')
    #[arg(short = 'c', long, default_value = "#000000")]
    color: String,

    /// Background color as hex, or 'transparent'
    #[arg(short = 'b', long, default_value = "#FFFFFF")]
    background: String,

    /// Canvas size in pixels
    #[arg(short = 's', long, default_value = "512")]
    size: u32,

    /// Icon size in pixels (default: 95% of canvas size)
    #[arg(long)]
    icon_size: Option<u32>,

    /// Supersampling factor for antialiasing (1, 2, or 4)
    #[arg(long, default_value = "2", value_parser = clap::value_parser!(u32).range(1..=4))]
    supersample: u32,

    /// Rotation angle in degrees (positive = clockwise, negative = counter-clockwise)
    #[arg(short = 'r', long, default_value = "0")]
    rotate: f64,

    /// Path to custom Font Awesome assets directory
    #[arg(long, value_name = "DIR")]
    assets: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Commands {
    /// List all available icon names
    List,
    /// Search for icons matching a pattern
    Search {
        /// Pattern to search for
        pattern: String,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    // Initialize renderer
    let renderer = match &cli.assets {
        Some(path) => match IconRenderer::from_path(path) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Error loading assets from {:?}: {}", path, e);
                return ExitCode::FAILURE;
            }
        },
        None => match IconRenderer::new() {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Error initializing renderer: {}", e);
                return ExitCode::FAILURE;
            }
        },
    };

    // Handle subcommands
    match &cli.command {
        Some(Commands::List) => {
            let icons = renderer.list_icons();
            println!("Available icons ({}):\n", icons.len());

            // Print in columns
            let mut icons_sorted: Vec<_> = icons.into_iter().collect();
            icons_sorted.sort();

            let col_width = icons_sorted.iter().map(|s| s.len()).max().unwrap_or(0) + 2;
            let cols = 80 / col_width;

            for chunk in icons_sorted.chunks(cols) {
                for name in chunk {
                    print!("{:width$}", name, width = col_width);
                }
                println!();
            }
            return ExitCode::SUCCESS;
        }
        Some(Commands::Search { pattern }) => {
            let pattern_lower = pattern.to_lowercase();
            let icons = renderer.list_icons();
            let mut matches: Vec<_> = icons
                .into_iter()
                .filter(|name| name.to_lowercase().contains(&pattern_lower))
                .collect();
            matches.sort();

            if matches.is_empty() {
                println!("No icons found matching '{}'", pattern);
            } else {
                println!("Icons matching '{}' ({}):\n", pattern, matches.len());
                for name in matches {
                    println!("  {}", name);
                }
            }
            return ExitCode::SUCCESS;
        }
        None => {}
    }

    // Validate required arguments for rendering
    let icon = match &cli.icon {
        Some(i) => i,
        None => {
            eprintln!("Error: ICON argument is required for rendering");
            eprintln!("Usage: icon-to-image <ICON> <OUTPUT> [OPTIONS]");
            eprintln!("       icon-to-image list");
            eprintln!("       icon-to-image search <PATTERN>");
            return ExitCode::FAILURE;
        }
    };

    let output = match &cli.output {
        Some(o) => o,
        None => {
            eprintln!("Error: OUTPUT argument is required for rendering");
            eprintln!("Usage: icon-to-image <ICON> <OUTPUT> [OPTIONS]");
            return ExitCode::FAILURE;
        }
    };

    // Check if icon exists
    if !renderer.has_icon(icon) {
        eprintln!("Error: Icon '{}' not found", icon);

        // Suggest similar icons
        let icons = renderer.list_icons();
        let icon_lower = icon.to_lowercase();
        let suggestions: Vec<_> = icons
            .into_iter()
            .filter(|name| name.to_lowercase().contains(&icon_lower))
            .take(5)
            .collect();

        if !suggestions.is_empty() {
            eprintln!("Did you mean: {}?", suggestions.join(", "));
        }
        return ExitCode::FAILURE;
    }

    // Parse colors
    let icon_color = match Color::from_hex(&cli.color) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: Invalid icon color '{}': {}", cli.color, e);
            return ExitCode::FAILURE;
        }
    };

    let background_color = if cli.background.to_lowercase() == "transparent" {
        Color::transparent()
    } else {
        match Color::from_hex(&cli.background) {
            Ok(c) => c,
            Err(e) => {
                eprintln!(
                    "Error: Invalid background color '{}': {}",
                    cli.background, e
                );
                return ExitCode::FAILURE;
            }
        }
    };

    // Calculate icon size (default: 95% of canvas)
    let icon_size = cli.icon_size.unwrap_or(((cli.size as f64) * 0.95) as u32);

    // Build render config
    let config = RenderConfig {
        canvas_width: cli.size,
        canvas_height: cli.size,
        icon_size,
        supersample_factor: cli.supersample,
        icon_color,
        background_color,
        rotate: cli.rotate,
        ..Default::default()
    };

    // Render the icon
    let (width, height, pixels) = match renderer.render(icon, &config) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("Error rendering icon: {}", e);
            return ExitCode::FAILURE;
        }
    };

    // Determine output format from extension
    let format = match output.extension().and_then(|e| e.to_str()) {
        Some("webp") => ImageFormat::WebP,
        Some("png") => ImageFormat::Png,
        _ => ImageFormat::Png,
    };

    // Encode the image
    let encoded = match encode(&pixels, width, height, format) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error encoding image: {}", e);
            return ExitCode::FAILURE;
        }
    };

    // Write to file
    if let Err(e) = std::fs::write(output, encoded) {
        eprintln!("Error writing file: {}", e);
        return ExitCode::FAILURE;
    }

    println!("Saved {} to {}", icon, output.display());
    ExitCode::SUCCESS
}
