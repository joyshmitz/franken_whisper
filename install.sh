#!/usr/bin/env bash
#
# franken_whisper installer — cross-platform binary installer
#
# Agent-first Rust ASR orchestrator with ffmpeg normalization and
# frankensqlite-backed persistence.
#
# One-liner install (with cache buster):
#   curl -fsSL "https://raw.githubusercontent.com/Dicklesworthstone/franken_whisper/main/install.sh?$(date +%s)" | bash
#
# Or without cache buster:
#   curl -fsSL https://raw.githubusercontent.com/Dicklesworthstone/franken_whisper/main/install.sh | bash
#
# Options:
#   --version vX.Y.Z   Install specific version (default: latest)
#   --dest DIR         Install to DIR (default: ~/.local/bin)
#   --system           Install to /usr/local/bin (requires sudo)
#   --easy-mode        Auto-update PATH in shell rc files
#   --verify           Run self-test after install
#   --force            Reinstall even if the same version is already present
#   --artifact-url URL Use a custom release artifact URL
#   --checksum SHA     Provide expected SHA256 checksum
#   --offline TARBALL  Install from a local archive (airgap); verifies a
#                      sibling .sha256 if present
#   --from-source      Build from source instead of downloading a binary
#   --no-verify        Skip checksum verification (testing only)
#   --quiet            Suppress non-error output
#   --no-gum           Disable gum formatting even if available
#   --uninstall        Remove franken_whisper and clean up
#   --help             Show this help
#
# Environment:
#   HTTP_PROXY / HTTPS_PROXY   Honored on every download
#   FW_INSTALL_DIR             Override default install directory
#   VERSION                    Override version to install
#
# Platforms (prebuilt binaries — 5 targets as of v0.2.0):
#   Linux x86_64          franken_whisper-X.Y.Z-linux_amd64.tar.gz
#   Linux aarch64         franken_whisper-X.Y.Z-linux_arm64.tar.gz
#   macOS x86_64 (Intel)  franken_whisper-X.Y.Z-darwin_amd64.tar.gz
#   macOS aarch64 (M-series) franken_whisper-X.Y.Z-darwin_arm64.tar.gz
#   Windows x64           franken_whisper-X.Y.Z-windows_amd64.zip
#
# Windows users: this bash installer covers linux + darwin (and WSL). On
# native Windows, download the windows_amd64.zip from the releases page and
# unzip franken_whisper.exe onto your PATH manually.
#
set -euo pipefail
umask 022
shopt -s lastpipe 2>/dev/null || true

# ============================================================================
# Configuration
# ============================================================================
VERSION="${VERSION:-}"
OWNER="${OWNER:-Dicklesworthstone}"
REPO="${REPO:-franken_whisper}"
BINARY_NAME="franken_whisper"
DEST_DEFAULT="$HOME/.local/bin"
DEST="${DEST:-$DEST_DEFAULT}"
EASY=0
QUIET=0
VERIFY=0
FROM_SOURCE=0
UNINSTALL=0
FORCE_INSTALL=0
NO_CHECKSUM=0
CHECKSUM="${CHECKSUM:-}"
ARTIFACT_URL="${ARTIFACT_URL:-}"
OFFLINE_TARBALL=""
LOCK_FILE="/tmp/franken-whisper-install.lock"
SYSTEM=0
NO_GUM=0
MAX_RETRIES=3
DOWNLOAD_TIMEOUT=120
# shellcheck disable=SC2034  # informational metadata, not read by the script
INSTALLER_VERSION="2.0.0"

# Where we record the installed version. The v0.2.0 binary has NO --version
# flag (running `franken_whisper --version` ERRORS), so we cannot ask the
# binary what version it is. Instead we write a marker file on every install
# and read it back for the already-installed short-circuit. Falls back to
# "unknown" for binaries placed by something other than this installer.
VERSION_MARKER_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/franken_whisper"
VERSION_MARKER="$VERSION_MARKER_DIR/.installed-version"

# Proxy args applied to EVERY curl invocation. Empty array expands to nothing.
PROXY_ARGS=()

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

GUM_AVAILABLE=false

# ============================================================================
# Gum detection (no auto-install — keep installer lean)
# ============================================================================
check_gum() {
    [[ "$NO_GUM" -eq 1 ]] && return 1
    if command -v gum &>/dev/null && [ -t 1 ]; then
        GUM_AVAILABLE=true
        return 0
    fi
    return 1
}

# ============================================================================
# Proxy
# ============================================================================
setup_proxy() {
    PROXY_ARGS=()
    if [[ -n "${HTTPS_PROXY:-${https_proxy:-}}" ]]; then
        PROXY_ARGS=(--proxy "${HTTPS_PROXY:-$https_proxy}")
    elif [[ -n "${HTTP_PROXY:-${http_proxy:-}}" ]]; then
        PROXY_ARGS=(--proxy "${HTTP_PROXY:-$http_proxy}")
    fi
}

# ============================================================================
# Styled output
# ============================================================================
print_banner() {
    [ "$QUIET" -eq 1 ] && return 0
    if [[ "$GUM_AVAILABLE" == "true" ]]; then
        gum style \
            --border double \
            --border-foreground 208 \
            --padding "0 2" \
            --margin "1 0" \
            --bold \
            "$(gum style --foreground 208 'franken_whisper installer')" \
            "$(gum style --foreground 245 'Agent-first Rust ASR orchestration')"
    else
        echo ""
        echo -e "${BOLD}${BLUE}╔══════════════════════════════════════════════════════╗${NC}"
        echo -e "${BOLD}${BLUE}║${NC}  ${BOLD}${GREEN}franken_whisper installer${NC}                           ${BOLD}${BLUE}║${NC}"
        echo -e "${BOLD}${BLUE}║${NC}  ${DIM}Agent-first Rust ASR orchestration${NC}                  ${BOLD}${BLUE}║${NC}"
        echo -e "${BOLD}${BLUE}╚══════════════════════════════════════════════════════╝${NC}"
        echo ""
    fi
}

# Draw a box around text with automatic width calculation.
# Usage: draw_box "ansi_color" "line1" "line2" ...
draw_box() {
    local color="$1"; shift
    local lines=("$@")
    local max_width=0 esc stripped len
    esc=$(printf '\033')
    local strip_ansi_sed="s/${esc}\\[[0-9;]*m//g"

    for line in "${lines[@]}"; do
        stripped=$(printf '%b' "$line" | LC_ALL=C sed "$strip_ansi_sed")
        len=${#stripped}
        [ "$len" -gt "$max_width" ] && max_width=$len
    done

    local inner_width=$((max_width + 4))
    local border=""
    for ((i=0; i<inner_width; i++)); do border+="═"; done

    printf "\033[%sm╔%s╗\033[0m\n" "$color" "$border"
    for line in "${lines[@]}"; do
        stripped=$(printf '%b' "$line" | LC_ALL=C sed "$strip_ansi_sed")
        len=${#stripped}
        local padding=$((max_width - len)) pad_str=""
        for ((i=0; i<padding; i++)); do pad_str+=" "; done
        printf "\033[%sm║\033[0m  %b%s  \033[%sm║\033[0m\n" "$color" "$line" "$pad_str" "$color"
    done
    printf "\033[%sm╚%s╝\033[0m\n" "$color" "$border"
}

log_info() {
    [ "$QUIET" -eq 1 ] && return 0
    if [[ "$GUM_AVAILABLE" == "true" ]]; then
        gum style --foreground 39 "→ $1" >&2
    else
        echo -e "${BLUE}→${NC} $1" >&2
    fi
}

log_warn() {
    [ "$QUIET" -eq 1 ] && return 0
    if [[ "$GUM_AVAILABLE" == "true" ]]; then
        gum style --foreground 214 "⚠ $1" >&2
    else
        echo -e "${YELLOW}⚠${NC} $1" >&2
    fi
}

log_error() {
    if [[ "$GUM_AVAILABLE" == "true" ]]; then
        gum style --foreground 196 "✗ $1" >&2
    else
        echo -e "${RED}✗${NC} $1" >&2
    fi
}

log_step() {
    [ "$QUIET" -eq 1 ] && return 0
    if [[ "$GUM_AVAILABLE" == "true" ]]; then
        gum style --foreground 208 "→ $1" >&2
    else
        echo -e "${BLUE}→${NC} $1" >&2
    fi
}

log_success() {
    [ "$QUIET" -eq 1 ] && return 0
    if [[ "$GUM_AVAILABLE" == "true" ]]; then
        gum style --foreground 82 "✓ $1" >&2
    else
        echo -e "${GREEN}✓${NC} $1" >&2
    fi
}

log_debug() {
    [[ "${DEBUG:-0}" -eq 1 ]] || return 0
    echo -e "${CYAN}[fw:debug]${NC} $1" >&2
}

# Run a command behind a gum spinner (or a plain step line as fallback).
run_with_spinner() {
    local title="$1"; shift
    if [[ "$GUM_AVAILABLE" == "true" ]] && [ "$QUIET" -eq 0 ]; then
        gum spin --spinner dot --title "$title" -- "$@"
    else
        log_step "$title"
        "$@"
    fi
}

die() {
    log_error "$@"
    exit 1
}

# ============================================================================
# Usage
# ============================================================================
usage() {
    cat <<'EOF'
franken_whisper installer — install the agent-first ASR orchestration CLI

Usage:
  curl -fsSL https://raw.githubusercontent.com/Dicklesworthstone/franken_whisper/main/install.sh | bash
  curl -fsSL .../install.sh | bash -s -- [OPTIONS]

Options:
  --version vX.Y.Z   Install specific version (default: latest)
  --dest DIR         Install to DIR (default: ~/.local/bin)
  --system           Install to /usr/local/bin (requires sudo)
  --easy-mode        Auto-update PATH in shell rc files
  --verify           Run self-test after install
  --force            Reinstall even if the same version is already present
  --artifact-url URL Use a custom release artifact URL
  --checksum SHA     Provide expected SHA256 checksum
  --offline TARBALL  Install from a local archive (airgap); verifies a
                     sibling <TARBALL>.sha256 if present
  --from-source      Build from source instead of downloading a binary
  --no-verify        Skip checksum verification (testing only)
  --quiet            Suppress non-error output
  --no-gum           Disable gum formatting even if available
  --uninstall        Remove franken_whisper and clean up
  --help             Show this help

Environment Variables:
  HTTP_PROXY / HTTPS_PROXY   Honored on every download
  FW_INSTALL_DIR             Override default install directory
  VERSION                    Override version to install

Platforms (prebuilt binaries):
  Linux x86_64             franken_whisper-X.Y.Z-linux_amd64.tar.gz
  Linux aarch64            franken_whisper-X.Y.Z-linux_arm64.tar.gz
  macOS x86_64 (Intel)     franken_whisper-X.Y.Z-darwin_amd64.tar.gz
  macOS aarch64 (M-series) franken_whisper-X.Y.Z-darwin_arm64.tar.gz
  Windows x64              franken_whisper-X.Y.Z-windows_amd64.zip

Windows note:
  This bash installer covers linux + darwin (and WSL). On native Windows,
  download the windows_amd64.zip from the releases page and unzip
  franken_whisper.exe onto your PATH manually.

Examples:
  # Default install (latest release)
  curl -fsSL .../install.sh | bash

  # System install with PATH auto-update
  curl -fsSL .../install.sh | sudo bash -s -- --system --easy-mode

  # Specific version
  curl -fsSL .../install.sh | bash -s -- --version v0.2.0

  # Airgapped install from a local archive
  bash install.sh --offline ./franken_whisper-0.2.0-linux_amd64.tar.gz

  # Uninstall
  curl -fsSL .../install.sh | bash -s -- --uninstall
EOF
    exit 0
}

# ============================================================================
# Argument Parsing
# ============================================================================
# shellcheck disable=SC2034  # SYSTEM records --system intent for clarity; DEST is what's actually used
while [ $# -gt 0 ]; do
    case "$1" in
        --version) VERSION="$2"; shift 2;;
        --version=*) VERSION="${1#*=}"; shift;;
        --dest) DEST="$2"; shift 2;;
        --dest=*) DEST="${1#*=}"; shift;;
        --system) SYSTEM=1; DEST="/usr/local/bin"; shift;;
        --easy-mode) EASY=1; shift;;
        --verify) VERIFY=1; shift;;
        --force) FORCE_INSTALL=1; shift;;
        --artifact-url) ARTIFACT_URL="$2"; shift 2;;
        --artifact-url=*) ARTIFACT_URL="${1#*=}"; shift;;
        --checksum) CHECKSUM="$2"; shift 2;;
        --checksum=*) CHECKSUM="${1#*=}"; shift;;
        --offline) OFFLINE_TARBALL="$2"; shift 2;;
        --offline=*) OFFLINE_TARBALL="${1#*=}"; shift;;
        --from-source) FROM_SOURCE=1; shift;;
        --no-verify) NO_CHECKSUM=1; shift;;
        --quiet|-q) QUIET=1; shift;;
        --no-gum) NO_GUM=1; shift;;
        --uninstall) UNINSTALL=1; shift;;
        -h|--help) usage;;
        *) shift;;
    esac
done

# Environment variable overrides
[ -n "${FW_INSTALL_DIR:-}" ] && DEST="$FW_INSTALL_DIR"

check_gum || true
setup_proxy

# ============================================================================
# Uninstall
# ============================================================================
do_uninstall() {
    print_banner
    log_step "Uninstalling franken_whisper..."

    if [ -f "$DEST/$BINARY_NAME" ]; then
        rm -f "$DEST/$BINARY_NAME"
        log_success "Removed $DEST/$BINARY_NAME"
    else
        log_warn "Binary not found at $DEST/$BINARY_NAME"
    fi

    if [ -f "$VERSION_MARKER" ]; then
        rm -f "$VERSION_MARKER"
        rmdir "$VERSION_MARKER_DIR" 2>/dev/null || true
        log_step "Removed version marker"
    fi

    for rc in "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.profile"; do
        if [ -f "$rc" ] && grep -q "# franken_whisper installer" "$rc" 2>/dev/null; then
            if [[ "$OSTYPE" == "darwin"* ]]; then
                sed -i '' '/# franken_whisper installer/d' "$rc" 2>/dev/null || true
            else
                sed -i '/# franken_whisper installer/d' "$rc" 2>/dev/null || true
            fi
            log_step "Cleaned $rc"
        fi
    done

    log_success "franken_whisper uninstalled"
    exit 0
}

[ "$UNINSTALL" -eq 1 ] && do_uninstall

# ============================================================================
# Platform Detection
# ============================================================================
# Sets PLATFORM (release asset suffix) and IS_WSL. Falls back to source build
# if no prebuilt artifact maps to this OS/arch.
PLATFORM=""
IS_WSL=0
detect_platform() {
    local os arch
    case "$(uname -s)" in
        Linux*)  os="linux" ;;
        Darwin*) os="darwin" ;;
        MINGW*|MSYS*|CYGWIN*) os="windows" ;;
        *) die "Unsupported OS: $(uname -s)" ;;
    esac

    case "$(uname -m)" in
        x86_64|amd64) arch="amd64" ;;
        aarch64|arm64) arch="arm64" ;;
        *)
            log_warn "Unsupported architecture $(uname -m); falling back to source build"
            FROM_SOURCE=1
            PLATFORM="${os}_unknown"
            return 0
            ;;
    esac

    # WSL: behaves like linux for our purposes, but warn the user.
    if [ "$os" = "linux" ] && grep -qi microsoft /proc/version 2>/dev/null; then
        # shellcheck disable=SC2034  # IS_WSL records detection state for future use/debugging
        IS_WSL=1
        log_warn "WSL detected — installing the linux_${arch} binary (some audio/tty features may need extra setup)"
    fi

    # All four POSIX targets ship a prebuilt binary as of v0.2.0:
    #   linux_amd64, linux_arm64, darwin_amd64, darwin_arm64
    # (windows_amd64 ships a .zip but native Windows isn't covered by this
    # bash installer — see the --help Windows note.)
    PLATFORM="${os}_${arch}"
    if [ "$os" = "windows" ]; then
        die "Native Windows is not supported by this bash installer. Download franken_whisper-<ver>-windows_amd64.zip from the releases page instead."
    fi
}

# ============================================================================
# Version Resolution
# ============================================================================
resolve_version() {
    if [ -n "$VERSION" ]; then return 0; fi

    log_step "Resolving latest version..."
    local latest_url="https://api.github.com/repos/${OWNER}/${REPO}/releases/latest"
    local tag="" attempts=0

    while [ $attempts -lt $MAX_RETRIES ] && [ -z "$tag" ]; do
        attempts=$((attempts + 1))
        if command -v curl &>/dev/null; then
            tag=$(curl -fsSL "${PROXY_ARGS[@]}" \
                --connect-timeout 10 --max-time 30 \
                -H "Accept: application/vnd.github.v3+json" \
                "$latest_url" 2>/dev/null | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/' || echo "")
        elif command -v wget &>/dev/null; then
            tag=$(wget -qO- --timeout=30 "$latest_url" 2>/dev/null | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/' || echo "")
        fi
        [ -z "$tag" ] && [ $attempts -lt $MAX_RETRIES ] && sleep 2
    done

    if [ -n "$tag" ] && [[ "$tag" =~ ^v[0-9] ]]; then
        VERSION="$tag"
        log_success "Latest version: $VERSION"
        return 0
    fi

    # Fallback: parse the latest-release redirect.
    log_step "Trying redirect-based version resolution..."
    local redirect_url="https://github.com/${OWNER}/${REPO}/releases/latest"
    if command -v curl &>/dev/null; then
        tag=$(curl -fsSL "${PROXY_ARGS[@]}" -o /dev/null -w '%{url_effective}' "$redirect_url" 2>/dev/null | sed -E 's|.*/tag/||' || echo "")
    fi

    if [ -n "$tag" ] && [[ "$tag" =~ ^v[0-9] ]] && [[ "$tag" != *"/"* ]]; then
        VERSION="$tag"
        log_success "Latest version (via redirect): $VERSION"
        return 0
    fi

    log_warn "Could not resolve latest version; will try building from source"
    VERSION=""
}

# ============================================================================
# Locking
# ============================================================================
LOCK_DIR="${LOCK_FILE}.d"
LOCKED=0

acquire_lock() {
    if mkdir "$LOCK_DIR" 2>/dev/null; then
        LOCKED=1
        echo $$ > "$LOCK_DIR/pid"
        return 0
    fi

    if [ -f "$LOCK_DIR/pid" ]; then
        local old_pid
        old_pid=$(cat "$LOCK_DIR/pid" 2>/dev/null || echo "")
        if [ -n "$old_pid" ] && ! kill -0 "$old_pid" 2>/dev/null; then
            log_warn "Removing stale lock (PID $old_pid not running)"
            rm -rf "$LOCK_DIR"
            if mkdir "$LOCK_DIR" 2>/dev/null; then
                LOCKED=1; echo $$ > "$LOCK_DIR/pid"; return 0
            fi
        fi

        local lock_age=0
        if [[ "$OSTYPE" == "darwin"* ]]; then
            lock_age=$(( $(date +%s) - $(stat -f %m "$LOCK_DIR/pid" 2>/dev/null || echo 0) ))
        else
            lock_age=$(( $(date +%s) - $(stat -c %Y "$LOCK_DIR/pid" 2>/dev/null || echo 0) ))
        fi
        if [ "$lock_age" -gt 300 ]; then
            log_warn "Removing stale lock (age: ${lock_age}s)"
            rm -rf "$LOCK_DIR"
            if mkdir "$LOCK_DIR" 2>/dev/null; then
                LOCKED=1; echo $$ > "$LOCK_DIR/pid"; return 0
            fi
        fi
    fi

    if [ "$LOCKED" -eq 0 ]; then
        die "Another installation is running. If incorrect, run: rm -rf $LOCK_DIR"
    fi
}

# ============================================================================
# Cleanup
# ============================================================================
TMP=""
cleanup() {
    [ -n "$TMP" ] && rm -rf "$TMP"
    [ "$LOCKED" -eq 1 ] && rm -rf "$LOCK_DIR"
}
trap cleanup EXIT

# ============================================================================
# Preflight checks
# ============================================================================
check_disk_space() {
    local min_kb=51200   # ~50MB headroom for the archive + extracted binary
    # Walk up to the nearest existing ancestor so df has a real path to stat.
    local path="$DEST"
    while [ -n "$path" ] && [ ! -d "$path" ]; do
        local parent; parent=$(dirname "$path")
        [ "$parent" = "$path" ] && break
        path="$parent"
    done
    [ -d "$path" ] || path="/"
    if command -v df >/dev/null 2>&1; then
        local avail_kb
        avail_kb=$(df -Pk "$path" 2>/dev/null | awk 'NR==2 {print $4}' || true)
        if [ -n "$avail_kb" ] && [ "$avail_kb" -lt "$min_kb" ]; then
            die "Insufficient disk space in $path (need at least 50MB)"
        fi
    fi
}

check_write_permissions() {
    if [ ! -d "$DEST" ]; then
        if ! mkdir -p "$DEST" 2>/dev/null; then
            log_error "Cannot create $DEST (insufficient permissions)"
            die "Try running with sudo or choose a writable --dest"
        fi
    fi
    if [ ! -w "$DEST" ]; then
        log_error "No write permission to $DEST"
        die "Try running with sudo or choose a writable --dest"
    fi
}

check_existing_install() {
    if [ -x "$DEST/$BINARY_NAME" ]; then
        local current
        current=$(read_installed_version)
        log_info "Existing franken_whisper detected (version: ${current})"
    fi
}

check_network() {
    # Offline / source / custom-artifact paths don't need github reachable.
    [ -n "$OFFLINE_TARBALL" ] && return 0
    [ "$FROM_SOURCE" -eq 1 ] && return 0
    command -v curl >/dev/null 2>&1 || { log_warn "curl not found; skipping network check"; return 0; }
    if ! curl -fsSL "${PROXY_ARGS[@]}" --connect-timeout 3 --max-time 5 -o /dev/null "https://github.com" 2>/dev/null; then
        log_warn "Could not reach github.com; download may fail"
    fi
}

preflight_checks() {
    log_info "Running preflight checks"
    check_disk_space
    check_write_permissions
    check_existing_install
    check_network
}

# ============================================================================
# Installed-version marker (binary has no --version flag)
# ============================================================================
read_installed_version() {
    if [ -f "$VERSION_MARKER" ]; then
        cat "$VERSION_MARKER" 2>/dev/null || echo "unknown"
    else
        echo "unknown"
    fi
}

write_installed_version() {
    mkdir -p "$VERSION_MARKER_DIR" 2>/dev/null || true
    printf '%s\n' "${1:-unknown}" > "$VERSION_MARKER" 2>/dev/null || true
}

# Already-installed short-circuit. The binary cannot self-report its version,
# so we compare the requested VERSION against the marker file we wrote on a
# prior install. Honors --force.
already_installed() {
    [ "$FORCE_INSTALL" -eq 1 ] && return 1
    [ -n "$OFFLINE_TARBALL" ] && return 1   # offline: caller knows what they want
    [ -z "$VERSION" ] && return 1
    [ -x "$DEST/$BINARY_NAME" ] || return 1
    local installed
    installed=$(read_installed_version)
    [ "$installed" = "$VERSION" ]
}

# ============================================================================
# PATH modification
# ============================================================================
maybe_add_path() {
    case ":$PATH:" in
        *:"$DEST":*) return 0;;
        *)
            if [ "$EASY" -eq 1 ]; then
                local updated=0
                for rc in "$HOME/.zshrc" "$HOME/.bashrc"; do
                    if [ -f "$rc" ] && [ -w "$rc" ]; then
                        if ! grep -qF "$DEST" "$rc" 2>/dev/null; then
                            echo "" >> "$rc"
                            echo "export PATH=\"$DEST:\$PATH\"  # franken_whisper installer" >> "$rc"
                        fi
                        updated=1
                    fi
                done
                if [ "$updated" -eq 1 ]; then
                    log_warn "PATH updated; restart shell or run: export PATH=\"$DEST:\$PATH\""
                else
                    log_warn "Add $DEST to PATH to use franken_whisper"
                fi
            else
                log_warn "Add $DEST to PATH to use franken_whisper"
            fi
        ;;
    esac
}

# ============================================================================
# Download with retry (proxy-aware)
# ============================================================================
download_file() {
    local url="$1" dest="$2" attempt=0
    local partial="${dest}.part"

    while [ $attempt -lt $MAX_RETRIES ]; do
        attempt=$((attempt + 1))
        log_debug "Download attempt $attempt for $url"
        if command -v curl &>/dev/null; then
            if curl -fsSL "${PROXY_ARGS[@]}" --connect-timeout 30 --max-time "$DOWNLOAD_TIMEOUT" \
                --retry 2 -o "$partial" "$url"; then
                mv -f "$partial" "$dest"; return 0
            fi
        elif command -v wget &>/dev/null; then
            if wget --quiet --timeout="$DOWNLOAD_TIMEOUT" -O "$partial" "$url"; then
                mv -f "$partial" "$dest"; return 0
            fi
        else
            die "Neither curl nor wget found"
        fi
        [ $attempt -lt $MAX_RETRIES ] && { log_warn "Download failed, retrying in 3s..."; sleep 3; }
    done
    rm -f "$partial" 2>/dev/null || true
    return 1
}

# ============================================================================
# Checksum verification (dual tool: sha256sum / shasum)
# ============================================================================
sha256_of() {
    if command -v sha256sum &>/dev/null; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum &>/dev/null; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        echo ""
    fi
}

# ============================================================================
# Atomic binary install
# ============================================================================
install_binary_atomic() {
    local src="$1" dest="$2"
    local tmp_dest="${dest}.tmp.$$"
    install -m 0755 "$src" "$tmp_dest"
    if ! mv -f "$tmp_dest" "$dest"; then
        rm -f "$tmp_dest" 2>/dev/null || true
        die "Failed to move binary into place"
    fi
}

# Extract an archive into TMP and install the franken_whisper binary it
# contains. The release tarballs hold a single bare `franken_whisper` binary.
extract_and_install() {
    local archive="$1" archive_ext="$2"

    log_step "Extracting..."
    if [[ "$archive_ext" == "zip" ]]; then
        command -v unzip &>/dev/null || die "unzip required for .zip archives"
        unzip -o "$archive" -d "$TMP/extract" >/dev/null 2>&1 || return 1
    else
        mkdir -p "$TMP/extract"
        tar -xzf "$archive" -C "$TMP/extract" 2>/dev/null || return 1
    fi

    local bin=""
    if [ -f "$TMP/extract/$BINARY_NAME" ]; then
        bin="$TMP/extract/$BINARY_NAME"
    else
        bin=$(find "$TMP/extract" -name "${BINARY_NAME}*" -type f \
            ! -name "*.txt" ! -name "*.md" ! -name "*.tar.*" ! -name "*.zip" ! -name "*.part" 2>/dev/null | head -1)
    fi
    if [ -z "$bin" ] || [ ! -f "$bin" ]; then
        log_error "Binary not found after extraction"
        return 1
    fi

    chmod +x "$bin"
    install_binary_atomic "$bin" "$DEST/$BINARY_NAME"
    return 0
}

# ============================================================================
# Offline install (airgap) — install from a local archive
# ============================================================================
install_offline() {
    [ -f "$OFFLINE_TARBALL" ] || die "Offline archive not found: $OFFLINE_TARBALL"
    log_step "Installing from local archive: $OFFLINE_TARBALL"

    local archive_ext="tar.gz"
    [[ "$OFFLINE_TARBALL" == *.zip ]] && archive_ext="zip"

    # Verify against a sibling .sha256 if one exists (or an explicit --checksum).
    local expected=""
    if [ "$NO_CHECKSUM" -eq 0 ]; then
        if [ -n "$CHECKSUM" ]; then
            expected="${CHECKSUM%% *}"
        elif [ -f "${OFFLINE_TARBALL}.sha256" ]; then
            expected=$(awk '{print $1}' "${OFFLINE_TARBALL}.sha256" 2>/dev/null | head -1)
        fi
        if [ -n "$expected" ]; then
            log_step "Verifying checksum..."
            local actual; actual=$(sha256_of "$OFFLINE_TARBALL")
            if [ -z "$actual" ]; then
                log_warn "No SHA256 tool found; skipping verification"
            elif [ "$expected" != "$actual" ]; then
                die "Checksum mismatch! expected=$expected got=$actual"
            else
                log_success "Checksum verified"
            fi
        else
            log_warn "No checksum available for offline archive; skipping verification"
        fi
    else
        log_warn "Checksum verification disabled (--no-verify)"
    fi

    extract_and_install "$OFFLINE_TARBALL" "$archive_ext" || die "Failed to install from offline archive"
    write_installed_version "${VERSION:-offline}"
    log_success "Installed to $DEST/$BINARY_NAME (offline)"
}

# ============================================================================
# Build from source
# ============================================================================
# HONESTY NOTE on source builds:
#   franken_whisper depends on SIBLING path crates that live OUTSIDE its repo:
#     - ../frankensqlite/crates/{fsqlite,fsqlite-types}   (required)
#     - ../frankentorch/crates/{ft-kernel-cpu,ft-core}    (required)
#     - ../frankentui/crates/ftui                         (only for --features tui)
#   These are `path = "..."` dependencies, so Cargo.lock records NO version or
#   commit for them — there is no pinned sibling revision to check out. That
#   means a source build necessarily tracks the siblings' default branch
#   (main) and CAN FAIL if their APIs have drifted past the franken_whisper
#   tag you asked for. (asupersync / franken-kernel / franken-evidence /
#   franken-decision are real crates.io deps and resolve normally — we do NOT
#   clone them.)
#   We clone the siblings honestly and attempt the build with default
#   (no-tui) features. If it fails, we surface the real cargo error and exit
#   non-zero rather than pretending it worked.
ensure_rust() {
    command -v cargo >/dev/null 2>&1 && return 0
    log_step "Installing Rust via rustup..."
    curl -fsSL "${PROXY_ARGS[@]}" https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
    export PATH="$HOME/.cargo/bin:$PATH"
    # shellcheck disable=SC1091  # rustup-generated env file not present at lint time
    [ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
    command -v cargo >/dev/null 2>&1
}

build_from_source() {
    log_step "Building from source..."
    ensure_rust || die "Rust (cargo) is required for source builds and could not be installed"

    local build_root="$TMP/src"
    mkdir -p "$build_root"
    local fw_dir="$build_root/franken_whisper"

    log_step "Cloning franken_whisper..."
    git clone --quiet "https://github.com/${OWNER}/${REPO}.git" "$fw_dir" || die "Failed to clone franken_whisper"

    # Check out the requested tag (default branch if none / resolution failed).
    if [ -n "$VERSION" ]; then
        if ! (cd "$fw_dir" && git checkout --quiet "$VERSION" 2>/dev/null); then
            log_warn "Could not check out tag $VERSION; building default branch"
        fi
    fi

    # Clone REQUIRED sibling path-dependency repos next to franken_whisper.
    # No pin is available (path deps) — we track each sibling's main branch.
    log_step "Cloning sibling path dependencies (frankensqlite, frankentorch)..."
    log_warn "Source builds track sibling main branches (no commit pin in Cargo.lock); this may fail if APIs have drifted"
    for dep in frankensqlite frankentorch; do
        if ! git clone --quiet --depth 1 "https://github.com/${OWNER}/${dep}.git" "$build_root/$dep"; then
            die "Failed to clone required sibling dependency: $dep"
        fi
    done

    log_step "Building with cargo (default features; this may take several minutes)..."
    local target_dir="$TMP/target"
    if ! (cd "$fw_dir" && CARGO_TARGET_DIR="$target_dir" cargo build --release -p franken_whisper); then
        log_error "Source build failed."
        log_error "This is most likely sibling API drift (frankensqlite/frankentorch main moved past this tag)."
        log_error "Prefer a prebuilt release binary, or build manually with matching sibling checkouts:"
        log_error "  git clone https://github.com/${OWNER}/${REPO}.git && cd ${REPO} && git checkout ${VERSION:-main}"
        log_error "  (clone frankensqlite + frankentorch as siblings, then: cargo build --release)"
        die "Build from source failed"
    fi

    local bin="$target_dir/release/$BINARY_NAME"
    if [ ! -x "$bin" ]; then
        bin=$(find "$target_dir" -name "$BINARY_NAME" -type f -perm -111 2>/dev/null | head -1)
    fi
    [ -x "$bin" ] || die "Binary not found after build"

    install_binary_atomic "$bin" "$DEST/$BINARY_NAME"
    write_installed_version "${VERSION:-source}"
    log_success "Installed to $DEST/$BINARY_NAME (source build)"
}

# ============================================================================
# Download release binary
# ============================================================================
download_release() {
    local platform="$1"
    local archive_ext="tar.gz"
    [[ "$platform" == windows_* ]] && archive_ext="zip"

    local archive_name url
    if [ -n "$ARTIFACT_URL" ]; then
        url="$ARTIFACT_URL"
        archive_name="$(basename "$ARTIFACT_URL")"
        [[ "$archive_name" == *.zip ]] && archive_ext="zip"
    else
        # Assets use the version WITHOUT the leading 'v' (e.g. 0.2.0).
        local ver_no_v="${VERSION#v}"
        archive_name="${BINARY_NAME}-${ver_no_v}-${platform}.${archive_ext}"
        url="https://github.com/${OWNER}/${REPO}/releases/download/${VERSION}/${archive_name}"
    fi

    log_step "Downloading $archive_name..."
    download_file "$url" "$TMP/$archive_name" || return 1
    [ -f "$TMP/$archive_name" ] || return 1

    # Checksum verification against the combined checksums-sha256.txt.
    if [ "$NO_CHECKSUM" -eq 1 ]; then
        log_warn "Checksum verification disabled (--no-verify)"
    else
        local expected=""
        if [ -n "$CHECKSUM" ]; then
            expected="${CHECKSUM%% *}"
        else
            local checksums_url="https://github.com/${OWNER}/${REPO}/releases/download/${VERSION}/checksums-sha256.txt"
            if download_file "$checksums_url" "$TMP/checksums-sha256.txt"; then
                expected=$(grep -F "$archive_name" "$TMP/checksums-sha256.txt" 2>/dev/null | awk '{print $1}' | head -1)
            fi
        fi

        if [ -n "$expected" ]; then
            log_step "Verifying checksum..."
            local actual; actual=$(sha256_of "$TMP/$archive_name")
            if [ -z "$actual" ]; then
                log_warn "No SHA256 tool found (sha256sum/shasum); skipping verification"
            elif [ "$expected" != "$actual" ]; then
                log_error "Checksum mismatch!"
                log_error "  Expected: $expected"
                log_error "  Got:      $actual"
                rm -f "$TMP/$archive_name"
                return 1
            else
                log_success "Checksum verified: ${actual:0:16}..."
            fi
        else
            log_warn "Checksum not available; skipping verification"
        fi
    fi

    extract_and_install "$TMP/$archive_name" "$archive_ext" || return 1
    write_installed_version "$VERSION"
    log_success "Installed to $DEST/$BINARY_NAME"
    return 0
}

# ============================================================================
# Self-test
# ============================================================================
# NOTE: the binary has NO --version flag. We exercise `robot health`, which is
# a fast, side-effect-free JSON probe that proves the binary actually runs.
run_self_test() {
    log_step "Running self-test (robot health)..."
    if "$DEST/$BINARY_NAME" robot health >/dev/null 2>&1; then
        log_success "Self-test passed"
    else
        log_warn "Binary installed but 'robot health' returned non-zero (backends may be missing)"
    fi
}

# ============================================================================
# Shell completions
# ============================================================================
# NOTE: franken_whisper v0.2.0 has NO `completions` subcommand (verified:
# `franken_whisper completions` => "unrecognized subcommand"). There is
# nothing to generate, so completion installation is deliberately skipped.

# ============================================================================
# AI agent hooks / skills
# ============================================================================
# NOTE: franken_whisper is a plain ASR CLI, not a guardrail/hook tool. It has
# no PreToolUse/BeforeTool semantics and ships no agent skill, so agent
# auto-configuration is deliberately omitted (unlike dcg/rch which gate or
# offload agent tool calls).

# ============================================================================
# Summary
# ============================================================================
print_summary() {
    [ "$QUIET" -eq 1 ] && return 0
    local installed_version path_status
    installed_version=$(read_installed_version)

    if [[ ":$PATH:" == *":$DEST:"* ]]; then
        path_status="on PATH"
    else
        path_status="NOT on PATH"
    fi

    echo ""
    local model_hint="scripts/fetch_test_models.sh (in the repo)"
    if [[ "$GUM_AVAILABLE" == "true" ]]; then
        gum style \
            --border rounded --border-foreground 82 --padding "1 2" --margin "1 0" \
            "$(gum style --foreground 82 --bold '✓ franken_whisper installed!')" \
            "" \
            "$(gum style --foreground 245 "Version:  $installed_version")" \
            "$(gum style --foreground 245 "Location: $DEST/$BINARY_NAME")" \
            "$(gum style --foreground 245 "PATH:     $path_status")"
    else
        draw_box "0;32" \
            "${GREEN}✓ franken_whisper installed!${NC}" \
            "" \
            "Version:  $installed_version" \
            "Location: $DEST/$BINARY_NAME" \
            "PATH:     $path_status"
    fi
    echo ""

    if [[ ":$PATH:" != *":$DEST:"* ]]; then
        log_warn "Add to PATH: export PATH=\"$DEST:\$PATH\"   (or re-run with --easy-mode)"
        echo ""
    fi

    echo "  Quick start:"
    echo "    franken_whisper transcribe --input audio.mp3 --json"
    echo "    franken_whisper robot health"
    echo "    franken_whisper robot backends"
    echo "    franken_whisper robot run --input audio.mp3 --backend auto"
    echo "    franken_whisper --help"
    echo ""
    echo "  Fetch test models:"
    echo "    $model_hint"
    echo ""
    echo "  Uninstall:"
    echo "    curl -fsSL https://raw.githubusercontent.com/${OWNER}/${REPO}/main/install.sh | bash -s -- --uninstall"
    echo "    (or: rm -f $DEST/$BINARY_NAME)"
    echo ""
}

# ============================================================================
# Main
# ============================================================================
main() {
    acquire_lock
    print_banner

    TMP=$(mktemp -d)

    # Offline / airgap path: no platform/version resolution needed.
    if [ -n "$OFFLINE_TARBALL" ]; then
        log_step "Install directory: $DEST"
        check_write_permissions
        install_offline
        maybe_add_path
        [ "$VERIFY" -eq 1 ] && run_self_test
        print_summary
        return 0
    fi

    detect_platform
    log_step "Platform: $PLATFORM"
    log_step "Install directory: $DEST"

    if [ "$FROM_SOURCE" -eq 0 ]; then
        resolve_version
    fi

    preflight_checks

    # Already-installed short-circuit (marker-file based; binary has no --version).
    if already_installed; then
        log_success "franken_whisper $VERSION is already installed at $DEST/$BINARY_NAME"
        log_info "Use --force to reinstall"
        maybe_add_path
        print_summary
        return 0
    fi

    if [ "$FROM_SOURCE" -eq 0 ] && [ -n "$VERSION" ]; then
        if download_release "$PLATFORM"; then
            :
        else
            log_warn "Binary download failed; falling back to build-from-source"
            build_from_source
        fi
    elif [ "$FROM_SOURCE" -eq 0 ]; then
        log_warn "No release version found; building from source"
        build_from_source
    else
        build_from_source
    fi

    maybe_add_path
    [ "$VERIFY" -eq 1 ] && run_self_test
    print_summary
}

if [[ "${BASH_SOURCE[0]:-}" == "${0:-}" ]] || [[ -z "${BASH_SOURCE[0]:-}" ]]; then
    main "$@"
fi
