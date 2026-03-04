#!/usr/bin/env bash
#
# franken_whisper installer - Cross-platform binary installer
#
# One-liner install:
#   curl -fsSL "https://raw.githubusercontent.com/Dicklesworthstone/franken_whisper/main/install.sh?$(date +%s)" | bash
#
# Options:
#   --version vX.Y.Z   Install specific version (default: latest)
#   --dest DIR         Install to DIR (default: ~/.local/bin)
#   --system           Install to /usr/local/bin (requires sudo)
#   --easy-mode        Auto-update PATH in shell rc files
#   --verify           Run self-test after install
#   --artifact-url URL Use a custom release artifact URL
#   --checksum SHA     Provide expected SHA256 checksum
#   --from-source      Build from source instead of downloading binary
#   --quiet            Suppress non-error output
#   --no-gum           Disable gum formatting even if available
#   --uninstall        Remove franken_whisper and clean up
#   --help             Show this help
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
CHECKSUM="${CHECKSUM:-}"
ARTIFACT_URL="${ARTIFACT_URL:-}"
LOCK_FILE="/tmp/franken-whisper-install.lock"
SYSTEM=0
NO_GUM=0
MAX_RETRIES=3
DOWNLOAD_TIMEOUT=120
INSTALLER_VERSION="1.0.0"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
ITALIC='\033[3m'
NC='\033[0m'

GUM_AVAILABLE=false

# ============================================================================
# Gum detection (no auto-install — keep installer lean)
# ============================================================================
check_gum() {
    [[ "$NO_GUM" -eq 1 ]] && return 1
    if command -v gum &>/dev/null; then
        GUM_AVAILABLE=true
        return 0
    fi
    return 1
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

log_info() {
    [ "$QUIET" -eq 1 ] && return 0
    if [[ "$GUM_AVAILABLE" == "true" ]]; then
        gum log --level info "$1" >&2
    else
        echo -e "${GREEN}[fw]${NC} $1" >&2
    fi
}

log_warn() {
    if [[ "$GUM_AVAILABLE" == "true" ]]; then
        gum log --level warn "$1" >&2
    else
        echo -e "${YELLOW}[fw]${NC} $1" >&2
    fi
}

log_error() {
    if [[ "$GUM_AVAILABLE" == "true" ]]; then
        gum log --level error "$1" >&2
    else
        echo -e "${RED}[fw]${NC} $1" >&2
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

die() {
    log_error "$@"
    exit 1
}

# ============================================================================
# Usage
# ============================================================================
usage() {
    check_gum || true
    cat <<'EOF'
franken_whisper installer - Install the ASR orchestration CLI

Usage:
  curl -fsSL https://raw.githubusercontent.com/Dicklesworthstone/franken_whisper/main/install.sh | bash
  curl -fsSL .../install.sh | bash -s -- [OPTIONS]

Options:
  --version vX.Y.Z   Install specific version (default: latest)
  --dest DIR         Install to DIR (default: ~/.local/bin)
  --system           Install to /usr/local/bin (requires sudo)
  --artifact-url URL Use a custom release artifact URL
  --checksum SHA     Provide expected SHA256 checksum
  --from-source      Build from source instead of downloading binary
  --easy-mode        Auto-update PATH in shell rc files
  --verify           Run self-test after install
  --quiet            Suppress non-error output
  --no-gum           Disable gum formatting even if available
  --uninstall        Remove franken_whisper and clean up
  --help             Show this help

Platforms:
  Linux x86_64         franken_whisper-vX.Y.Z-linux_amd64.tar.gz
  macOS Apple Silicon   franken_whisper-vX.Y.Z-darwin_arm64.tar.gz
  Windows x64          franken_whisper-vX.Y.Z-windows_amd64.zip

Environment Variables:
  FW_INSTALL_DIR     Override default install directory
  VERSION            Override version to install

Examples:
  # Default install (latest release)
  curl -fsSL .../install.sh | bash

  # System install with PATH auto-update
  curl -fsSL .../install.sh | sudo bash -s -- --system --easy-mode

  # Specific version
  curl -fsSL .../install.sh | bash -s -- --version v0.1.0

  # Uninstall
  curl -fsSL .../install.sh | bash -s -- --uninstall
EOF
    exit 0
}

# ============================================================================
# Argument Parsing
# ============================================================================
while [ $# -gt 0 ]; do
    case "$1" in
        --version) VERSION="$2"; shift 2;;
        --version=*) VERSION="${1#*=}"; shift;;
        --dest) DEST="$2"; shift 2;;
        --dest=*) DEST="${1#*=}"; shift;;
        --system) SYSTEM=1; DEST="/usr/local/bin"; shift;;
        --easy-mode) EASY=1; shift;;
        --verify) VERIFY=1; shift;;
        --artifact-url) ARTIFACT_URL="$2"; shift 2;;
        --checksum) CHECKSUM="$2"; shift 2;;
        --from-source) FROM_SOURCE=1; shift;;
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
        *) die "Unsupported architecture: $(uname -m)" ;;
    esac

    echo "${os}_${arch}"
}

# ============================================================================
# Version Resolution
# ============================================================================
resolve_version() {
    if [ -n "$VERSION" ]; then return 0; fi

    log_step "Resolving latest version..."
    local latest_url="https://api.github.com/repos/${OWNER}/${REPO}/releases/latest"
    local tag=""
    local attempts=0

    while [ $attempts -lt $MAX_RETRIES ] && [ -z "$tag" ]; do
        attempts=$((attempts + 1))

        if command -v curl &>/dev/null; then
            tag=$(curl -fsSL \
                --connect-timeout 10 \
                --max-time 30 \
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

    # Fallback: redirect-based
    log_step "Trying redirect-based version resolution..."
    local redirect_url="https://github.com/${OWNER}/${REPO}/releases/latest"
    if command -v curl &>/dev/null; then
        tag=$(curl -fsSL -o /dev/null -w '%{url_effective}' "$redirect_url" 2>/dev/null | sed -E 's|.*/tag/||' || echo "")
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
                LOCKED=1
                echo $$ > "$LOCK_DIR/pid"
                return 0
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
                LOCKED=1
                echo $$ > "$LOCK_DIR/pid"
                return 0
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
# Download with retry
# ============================================================================
download_file() {
    local url="$1"
    local dest="$2"
    local attempt=0
    local partial="${dest}.part"

    local proxy_env=()
    local proxy_http="${HTTP_PROXY:-${http_proxy:-}}"
    local proxy_https="${HTTPS_PROXY:-${https_proxy:-}}"
    [ -n "$proxy_http" ] && proxy_env+=(HTTP_PROXY="$proxy_http" http_proxy="$proxy_http")
    [ -n "$proxy_https" ] && proxy_env+=(HTTPS_PROXY="$proxy_https" https_proxy="$proxy_https")

    while [ $attempt -lt $MAX_RETRIES ]; do
        attempt=$((attempt + 1))
        log_debug "Download attempt $attempt for $url"

        if command -v curl &>/dev/null; then
            if env ${proxy_env[@]+"${proxy_env[@]}"} \
                curl -fsSL --connect-timeout 30 --max-time "$DOWNLOAD_TIMEOUT" \
                --retry 2 -o "$partial" "$url"; then
                mv -f "$partial" "$dest"
                return 0
            fi
        elif command -v wget &>/dev/null; then
            if env ${proxy_env[@]+"${proxy_env[@]}"} \
                wget --quiet --timeout="$DOWNLOAD_TIMEOUT" -O "$partial" "$url"; then
                mv -f "$partial" "$dest"
                return 0
            fi
        else
            die "Neither curl nor wget found"
        fi

        [ $attempt -lt $MAX_RETRIES ] && {
            log_warn "Download failed, retrying in 3s..."
            sleep 3
        }
    done

    return 1
}

# ============================================================================
# Atomic binary install
# ============================================================================
install_binary_atomic() {
    local src="$1"
    local dest="$2"
    local tmp_dest="${dest}.tmp.$$"

    install -m 0755 "$src" "$tmp_dest"
    if ! mv -f "$tmp_dest" "$dest"; then
        rm -f "$tmp_dest" 2>/dev/null || true
        die "Failed to move binary into place"
    fi
}

# ============================================================================
# Build from source
# ============================================================================
ensure_rust() {
    if command -v cargo >/dev/null 2>&1; then return 0; fi

    log_step "Installing Rust via rustup..."
    curl -fsSL https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly --profile minimal
    export PATH="$HOME/.cargo/bin:$PATH"
    [ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
}

build_from_source() {
    log_step "Building from source..."

    if ! ensure_rust; then
        die "Rust is required for source builds"
    fi

    local build_dir="$TMP/src"

    log_step "Cloning repository..."
    git clone --depth 1 "https://github.com/${OWNER}/${REPO}.git" "$build_dir" || die "Failed to clone"

    # Clone workspace path dependencies
    log_step "Cloning workspace dependencies..."
    local parent_dir
    parent_dir=$(dirname "$build_dir")
    for dep in asupersync frankensqlite; do
        git clone --depth 1 "https://github.com/${OWNER}/${dep}.git" "$parent_dir/$dep" 2>/dev/null || true
    done
    # Create stubs for optional deps
    for stub in frankentui frankentorch frankenjax; do
        if [ ! -d "$parent_dir/$stub" ]; then
            mkdir -p "$parent_dir/$stub"
            echo -e "[package]\nname = \"$stub\"\nversion = \"0.0.1\"\nedition = \"2021\"\n\n[lib]\npath = \"lib.rs\"" \
                > "$parent_dir/$stub/Cargo.toml"
            echo "" > "$parent_dir/$stub/lib.rs"
        fi
    done

    log_step "Building with Cargo (this may take several minutes)..."
    local target_dir="$TMP/target"
    (cd "$build_dir" && CARGO_TARGET_DIR="$target_dir" cargo build --release -p franken_whisper) || die "Build failed"

    local bin="$target_dir/release/$BINARY_NAME"
    if [ ! -x "$bin" ]; then
        bin=$(find "$target_dir" -name "$BINARY_NAME" -type f -perm -111 2>/dev/null | head -1)
    fi
    [ -x "$bin" ] || die "Binary not found after build"

    install_binary_atomic "$bin" "$DEST/$BINARY_NAME"
    log_success "Installed to $DEST/$BINARY_NAME (source build)"
}

# ============================================================================
# Download release binary
# ============================================================================
download_release() {
    local platform="$1"

    local archive_name=""
    local archive_ext="tar.gz"
    local url=""

    # Windows uses .zip
    if [[ "$platform" == windows_* ]]; then
        archive_ext="zip"
    fi

    if [ -n "$ARTIFACT_URL" ]; then
        url="$ARTIFACT_URL"
        archive_name="$(basename "$ARTIFACT_URL")"
    else
        archive_name="${BINARY_NAME}-${VERSION}-${platform}.${archive_ext}"
        url="https://github.com/${OWNER}/${REPO}/releases/download/${VERSION}/${archive_name}"
    fi

    log_step "Downloading $archive_name..."
    download_file "$url" "$TMP/$archive_name" || return 1

    if [ ! -f "$TMP/$archive_name" ]; then
        return 1
    fi

    # Verify checksum from combined checksums file
    local expected=""
    if [ -n "$CHECKSUM" ]; then
        expected="${CHECKSUM%% *}"
    else
        local checksums_url="https://github.com/${OWNER}/${REPO}/releases/download/${VERSION}/checksums-sha256.txt"
        if download_file "$checksums_url" "$TMP/checksums-sha256.txt"; then
            expected=$(grep "$archive_name" "$TMP/checksums-sha256.txt" 2>/dev/null | awk '{print $1}')
        fi
    fi

    if [ -n "$expected" ]; then
        log_step "Verifying checksum..."
        local actual
        if command -v sha256sum &>/dev/null; then
            actual=$(sha256sum "$TMP/$archive_name" | awk '{print $1}')
        elif command -v shasum &>/dev/null; then
            actual=$(shasum -a 256 "$TMP/$archive_name" | awk '{print $1}')
        else
            log_warn "No SHA256 tool found, skipping verification"
            actual="$expected"
        fi

        if [ "$expected" != "$actual" ]; then
            log_error "Checksum mismatch!"
            log_error "  Expected: $expected"
            log_error "  Got:      $actual"
            return 1
        fi
        log_success "Checksum verified"
    else
        log_warn "Checksum not available, skipping verification"
    fi

    # Extract
    log_step "Extracting..."
    if [[ "$archive_ext" == "zip" ]]; then
        if command -v unzip &>/dev/null; then
            unzip -o "$TMP/$archive_name" -d "$TMP" 2>/dev/null || return 1
        else
            die "unzip required for Windows archives"
        fi
    else
        tar -xzf "$TMP/$archive_name" -C "$TMP" 2>/dev/null || return 1
    fi

    # Find binary (may be at top level or in a subdirectory)
    local bin=""
    if [ -x "$TMP/$BINARY_NAME" ]; then
        bin="$TMP/$BINARY_NAME"
    else
        bin=$(find "$TMP" -name "$BINARY_NAME" -type f 2>/dev/null | head -1)
    fi

    if [ -z "$bin" ] || [ ! -f "$bin" ]; then
        return 1
    fi

    chmod +x "$bin"
    install_binary_atomic "$bin" "$DEST/$BINARY_NAME"
    log_success "Installed to $DEST/$BINARY_NAME"
    return 0
}

# ============================================================================
# Print summary
# ============================================================================
print_summary() {
    local installed_version
    installed_version=$("$DEST/$BINARY_NAME" --version 2>/dev/null || echo "unknown")

    echo ""
    if [[ "$GUM_AVAILABLE" == "true" ]]; then
        gum style \
            --border rounded \
            --border-foreground 82 \
            --padding "1 2" \
            --margin "1 0" \
            "$(gum style --foreground 82 --bold '✓ franken_whisper installed!')" \
            "" \
            "$(gum style --foreground 245 "Version:  $installed_version")" \
            "$(gum style --foreground 245 "Location: $DEST/$BINARY_NAME")"
    else
        log_success "franken_whisper installed!"
        echo ""
        echo "  Version:  $installed_version"
        echo "  Location: $DEST/$BINARY_NAME"
    fi
    echo ""

    if [[ ":$PATH:" != *":$DEST:"* ]]; then
        log_warn "Add to PATH: export PATH=\"$DEST:\$PATH\""
        echo ""
    fi

    echo "  Quick Start:"
    echo "    franken_whisper transcribe --input audio.mp3 --json"
    echo "    franken_whisper robot run --input audio.mp3 --backend auto"
    echo "    franken_whisper robot backends"
    echo "    franken_whisper robot health"
    echo "    franken_whisper --help"
    echo ""
}

# ============================================================================
# Main
# ============================================================================
main() {
    acquire_lock
    print_banner

    TMP=$(mktemp -d)

    local platform
    platform=$(detect_platform)
    log_step "Platform: $platform"
    log_step "Install directory: $DEST"

    mkdir -p "$DEST"

    if [ "$FROM_SOURCE" -eq 0 ]; then
        resolve_version

        if [ -n "$VERSION" ]; then
            if download_release "$platform"; then
                :
            else
                log_warn "Binary download failed, building from source..."
                build_from_source
            fi
        else
            log_warn "No release version found, building from source..."
            build_from_source
        fi
    else
        build_from_source
    fi

    maybe_add_path

    if [ "$VERIFY" -eq 1 ]; then
        log_step "Running self-test..."
        if "$DEST/$BINARY_NAME" --version 2>/dev/null; then
            log_success "Self-test passed"
        else
            log_warn "Binary runs but --version returned non-zero (may need backends)"
        fi
    fi

    print_summary
}

if [[ "${BASH_SOURCE[0]:-}" == "${0:-}" ]] || [[ -z "${BASH_SOURCE[0]:-}" ]]; then
    main "$@"
fi
