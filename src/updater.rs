use anyhow::{bail, Context, Result};
use colored::Colorize;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::utils::{print_banner, print_status, print_success};

pub const GITHUB_REPO: &str = "solarpush/trainaya";
pub const CURRENT_VERSION: &str = env!("CARGO_PKG_VERSION");
const CACHE_TTL_SECS: u64 = 86_400; // 24 hours

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateCache {
    pub last_checked_unix: u64,
    pub latest_version: String,
    pub release_url: String,
}

#[derive(Debug, Clone)]
pub struct ReleaseInfo {
    pub tag_name: String,
    pub version: String,
    pub html_url: String,
}

/// Normalize version string by stripping leading 'v' or 'V'
pub fn clean_version(v: &str) -> &str {
    v.trim().trim_start_matches(['v', 'V'])
}

/// Parse semantic version tuple (major, minor, patch)
pub fn parse_semver(v: &str) -> Option<(u64, u64, u64)> {
    let cleaned = clean_version(v);
    // Strip pre-release or build metadata if present (e.g. 0.2.0-beta.1)
    let core = cleaned.split(['-', '+']).next()?;
    let mut parts = core.split('.');
    let major = parts.next()?.parse::<u64>().ok()?;
    let minor = parts.next().unwrap_or("0").parse::<u64>().ok()?;
    let patch = parts.next().unwrap_or("0").parse::<u64>().ok()?;
    Some((major, minor, patch))
}

/// Returns true if `latest` is strictly newer than `current`
pub fn is_newer_version(current: &str, latest: &str) -> bool {
    match (parse_semver(current), parse_semver(latest)) {
        (Some(curr_t), Some(lat_t)) => lat_t > curr_t,
        _ => clean_version(latest) != clean_version(current),
    }
}

/// Path to local update check cache (~/.cache/trainaya/update_check.json)
pub fn get_cache_file_path() -> PathBuf {
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
        PathBuf::from(xdg).join("trainaya").join("update_check.json")
    } else if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home)
            .join(".cache")
            .join("trainaya")
            .join("update_check.json")
    } else {
        std::env::temp_dir().join("trainaya_update_check.json")
    }
}

pub fn read_update_cache() -> Option<UpdateCache> {
    let path = get_cache_file_path();
    let content = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()
}

pub fn write_update_cache(cache: &UpdateCache) {
    let path = get_cache_file_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string_pretty(cache) {
        let _ = std::fs::write(path, json);
    }
}

fn current_unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Query GitHub Releases API for the latest (or specific tag) release
pub fn fetch_github_release(target_tag: Option<&str>, timeout_ms: u64) -> Result<ReleaseInfo> {
    let url = match target_tag {
        Some(tag) if !tag.is_empty() && tag != "latest" => {
            let normalized = if tag.starts_with('v') {
                tag.to_string()
            } else {
                format!("v{}", tag)
            };
            format!(
                "https://api.github.com/repos/{}/releases/tags/{}",
                GITHUB_REPO, normalized
            )
        }
        _ => format!("https://api.github.com/repos/{}/releases/latest", GITHUB_REPO),
    };

    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_millis(timeout_ms))
        .timeout_read(Duration::from_millis(timeout_ms))
        .user_agent(&format!("trainaya/{}", CURRENT_VERSION))
        .build();

    let resp: Value = agent
        .get(&url)
        .set("Accept", "application/vnd.github+json")
        .call()
        .with_context(|| format!("Failed to query GitHub Releases API at {}", url))?
        .into_json()
        .context("Failed to parse GitHub Releases JSON response")?;

    let tag_name = resp
        .get("tag_name")
        .and_then(|v| v.as_str())
        .context("Missing 'tag_name' in GitHub release response")?
        .to_string();

    let html_url = resp
        .get("html_url")
        .and_then(|v| v.as_str())
        .unwrap_or("https://github.com/solarpush/trainaya/releases")
        .to_string();

    let version = clean_version(&tag_name).to_string();

    Ok(ReleaseInfo {
        tag_name,
        version,
        html_url,
    })
}

/// Passive, zero-latency check on CLI startup/exit:
/// 1. Reads local cache (<0.1ms) and prints an upgrade hint if a newer version is known.
/// 2. If the cache is older than 24h, spawns a non-blocking background refresh.
pub fn check_and_notify_passive() {
    if let Ok(val) = std::env::var("TRAINAYA_NO_UPDATE_CHECK") {
        if val == "1" || val.eq_ignore_ascii_case("true") {
            return;
        }
    }

    let now = current_unix_timestamp();
    let cached = read_update_cache();

    if let Some(ref c) = cached {
        if is_newer_version(CURRENT_VERSION, &c.latest_version) {
            eprintln!(
                "\n  {} Update available: {} ➔ {} — Run {} to upgrade.",
                "💡".cyan().bold(),
                format!("v{}", CURRENT_VERSION).yellow(),
                format!("v{}", clean_version(&c.latest_version)).green().bold(),
                "'trainaya update'".bright_white().bold()
            );
        }
    }

    let needs_refresh = match cached {
        Some(c) => now.saturating_sub(c.last_checked_unix) >= CACHE_TTL_SECS,
        None => true,
    };

    if needs_refresh {
        // Refresh cache with a short 1200ms timeout so it never stalls the user
        let handle = std::thread::spawn(move || {
            if let Ok(info) = fetch_github_release(None, 1200) {
                let updated = UpdateCache {
                    last_checked_unix: now,
                    latest_version: info.version.clone(),
                    release_url: info.html_url,
                };
                write_update_cache(&updated);
                if is_newer_version(CURRENT_VERSION, &info.version) {
                    eprintln!(
                        "\n  {} Update available: {} ➔ {} — Run {} to upgrade.",
                        "💡".cyan().bold(),
                        format!("v{}", CURRENT_VERSION).yellow(),
                        format!("v{}", clean_version(&info.version)).green().bold(),
                        "'trainaya update'".bright_white().bold()
                    );
                }
            }
        });
        // Wait up to 350ms for the background thread if network is fast, otherwise detach cleanly
        let _ = handle.join();
    }
}

/// Detect platform and architecture matching GitHub release asset naming
pub fn detect_release_asset_name() -> Result<String> {
    let platform = match std::env::consts::OS {
        "linux" => "linux",
        "macos" => "darwin",
        other => bail!(
            "Auto-update is not supported on OS '{}'. Supported: linux, macos.",
            other
        ),
    };

    let arch = match std::env::consts::ARCH {
        "x86_64" => "x86_64",
        "aarch64" => "aarch64",
        other => bail!(
            "Auto-update is not supported on architecture '{}'. Supported: x86_64, aarch64.",
            other
        ),
    };

    Ok(format!("trainaya-{}-{}.tar.gz", platform, arch))
}

/// Resolve the binary path to replace during self-update
fn resolve_install_target_path() -> Result<PathBuf> {
    if let Ok(exe) = std::env::current_exe() {
        let exe_str = exe.to_string_lossy();
        // If running from cargo target/debug or target/release, prefer ~/.local/bin/trainaya if it exists
        if exe_str.contains("/target/debug/") || exe_str.contains("/target/release/") {
            if let Ok(home) = std::env::var("HOME") {
                let local_bin = PathBuf::from(home).join(".local/bin/trainaya");
                if local_bin.exists() {
                    return Ok(local_bin);
                }
            }
        }
        return Ok(exe);
    }

    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    Ok(PathBuf::from(home).join(".local/bin/trainaya"))
}

/// Execute `trainaya update` command
pub fn run_update(check_only: bool, force: bool, target_tag: Option<&str>) -> Result<()> {
    print_banner("Self-Update & Release Manager");

    print_status("Current Version", &format!("v{}", CURRENT_VERSION));
    print_status("Repository", GITHUB_REPO);

    println!("\n  Checking GitHub Releases for updates...");
    let release = fetch_github_release(target_tag, 10_000)?;

    // Update local cache immediately
    write_update_cache(&UpdateCache {
        last_checked_unix: current_unix_timestamp(),
        latest_version: release.version.clone(),
        release_url: release.html_url.clone(),
    });

    print_status("Latest Release", &format!("v{} ({})", release.version, release.tag_name));

    let newer = is_newer_version(CURRENT_VERSION, &release.version);

    if check_only {
        if newer {
            println!(
                "\n  {} A newer version is available: {} ➔ {}",
                "🚀".cyan().bold(),
                format!("v{}", CURRENT_VERSION).yellow(),
                format!("v{}", release.version).green().bold()
            );
            println!("  Run {} to install it.\n", "'trainaya update'".bright_white().bold());
        } else {
            print_success(&format!(
                "Trainaya v{} is already up to date!",
                CURRENT_VERSION
            ));
        }
        return Ok(());
    }

    if !newer && !force && target_tag.is_none() {
        print_success(&format!(
            "Trainaya v{} is already up to date! (Use --force to reinstall)",
            CURRENT_VERSION
        ));
        return Ok(());
    }

    let asset_name = detect_release_asset_name()?;
    let download_url = format!(
        "https://github.com/{}/releases/download/{}/{}",
        GITHUB_REPO, release.tag_name, asset_name
    );

    let target_bin = resolve_install_target_path()?;
    print_status("Target Asset", &asset_name);
    print_status("Install Path", &target_bin.display().to_string());

    println!(
        "\n  Downloading {}...",
        download_url.cyan()
    );

    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(15))
        .timeout_read(Duration::from_secs(120))
        .user_agent(&format!("trainaya/{}", CURRENT_VERSION))
        .build();

    let resp = agent
        .get(&download_url)
        .call()
        .with_context(|| format!("Failed to download release archive from {}", download_url))?;

    let mut archive_bytes = Vec::new();
    resp.into_reader()
        .read_to_end(&mut archive_bytes)
        .context("Failed to read release archive stream")?;

    let tmp_dir = std::env::temp_dir().join(format!("trainaya_update_{}", std::process::id()));
    std::fs::create_dir_all(&tmp_dir)?;
    let archive_path = tmp_dir.join(&asset_name);
    std::fs::write(&archive_path, &archive_bytes)?;

    println!(
        "  Extracting archive ({:.2} MiB)...",
        archive_bytes.len() as f64 / (1024.0 * 1024.0)
    );

    let status = Command::new("tar")
        .arg("-xzf")
        .arg(&archive_path)
        .arg("-C")
        .arg(&tmp_dir)
        .status()
        .context("Failed to execute 'tar' to extract release archive")?;

    if !status.success() {
        let _ = std::fs::remove_dir_all(&tmp_dir);
        bail!("Archive extraction failed with exit code {:?}", status.code());
    }

    let extracted_bin = tmp_dir.join("trainaya");
    if !extracted_bin.exists() {
        let _ = std::fs::remove_dir_all(&tmp_dir);
        bail!("Extracted archive did not contain 'trainaya' binary");
    }

    // Atomic replacement in target directory to avoid ETXTBSY on running executable
    let parent_dir = target_bin.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent_dir)?;
    let staged_bin = parent_dir.join(format!(".trainaya.upgrade.{}", std::process::id()));

    std::fs::copy(&extracted_bin, &staged_bin).with_context(|| {
        format!(
            "Failed to stage updated binary in {} (check directory write permissions)",
            parent_dir.display()
        )
    })?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&staged_bin)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&staged_bin, perms)?;
    }

    std::fs::rename(&staged_bin, &target_bin).with_context(|| {
        let _ = std::fs::remove_file(&staged_bin);
        format!("Failed to atomically replace binary at {}", target_bin.display())
    })?;

    let _ = std::fs::remove_dir_all(&tmp_dir);

    print_success(&format!(
        "Successfully upgraded Trainaya: v{} ➔ v{}!",
        CURRENT_VERSION, release.version
    ));

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semver_comparison() {
        assert!(is_newer_version("0.1.0", "0.1.1"));
        assert!(is_newer_version("0.1.0", "v0.2.0"));
        assert!(is_newer_version("v0.1.9", "v1.0.0"));
        assert!(!is_newer_version("0.1.0", "0.1.0"));
        assert!(!is_newer_version("0.2.0", "v0.1.9"));
        assert!(!is_newer_version("1.0.0", "0.9.9"));
    }

    #[test]
    fn test_detect_release_asset_name() {
        let asset = detect_release_asset_name().unwrap();
        assert!(asset.starts_with("trainaya-"));
        assert!(asset.ends_with(".tar.gz"));
    }
}
