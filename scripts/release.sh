#!/usr/bin/env bash
# Raylytics VC Standard, scripts/release.sh (STANDARD.md §7).
# Usage: scripts/release.sh X.Y.Z   (run from the repo root, on main)
# 1. Refuses unless on main, clean, in sync with origin, and the last ci run on
#    main is green. 2. Bumps the stack's source-of-truth version file (and the
#    React Native derived copies). 3. Prepends the release to CHANGELOG.md with
#    git-cliff. 4. Commits chore(release): vX.Y.Z, creates a signed annotated
#    tag, pushes both. release.yml then creates the GitHub Release.
# Stacks: python (pyproject.toml), extension (manifest.json), react-native
# (package.json + android/ + ios/), js (package.json), android
# (app/build.gradle(.kts)), go (tag only; goreleaser builds).
set -euo pipefail

VERSION="${1:?usage: $0 X.Y.Z}"
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-rc\.[0-9]+)?$ ]] || { echo "not a SemVer version: $VERSION" >&2; exit 2; }
TAG="v$VERSION"

die() { echo "release.sh: $*" >&2; exit 1; }

# ---- 1. preconditions -------------------------------------------------------
[ "$(git rev-parse --abbrev-ref HEAD)" = main ] || die "not on main"
[ -z "$(git status --porcelain)" ] || die "working tree not clean"
git fetch -q origin main
[ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ] || die "main is not in sync with origin/main"
git rev-parse -q --verify "refs/tags/$TAG" >/dev/null && die "tag $TAG already exists"
if [ -f .github/workflows/ci.yml ]; then
  conclusion="$(gh run list --branch main --workflow ci.yml --limit 1 --json conclusion --jq '.[0].conclusion' 2>/dev/null || echo unknown)"
  [ "$conclusion" = success ] || die "last ci run on main is '$conclusion', not success"
fi
if [ -f CHANGELOG.md ] && grep -q "^## \[$VERSION\]" CHANGELOG.md; then die "CHANGELOG.md already has an entry for $VERSION"; fi

# ---- 2. bump ----------------------------------------------------------------
detect_stack() {
  if [ -f go.mod ]; then echo go
  elif [ -f pyproject.toml ]; then echo python
  elif [ -f manifest.json ] && [ -f package.json ]; then echo extension
  elif [ -f manifest.json ]; then echo extension
  elif [ -f package.json ] && { [ -d android ] || [ -d ios ]; }; then echo react-native
  elif [ -f package.json ]; then echo js
  elif [ -f app/build.gradle.kts ] || [ -f app/build.gradle ]; then echo android
  else echo none; fi
}

sedi() { if sed --version >/dev/null 2>&1; then sed -i "$@"; else sed -i '' "$@"; fi; }

current_json_version() { python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["version"])' "$1"; }
set_json_version() {
  python3 - "$1" "$2" <<'PY'
import json, sys, re
path, version = sys.argv[1], sys.argv[2]
raw = open(path).read()
new, n = re.subn(r'("version"\s*:\s*")[^"]*(")', r'\g<1>' + version + r'\2', raw, count=1)
if n != 1: sys.exit(f"{path}: no \"version\" field")
open(path, "w").write(new)
PY
}

gradle_file() { if [ -f app/build.gradle.kts ]; then echo app/build.gradle.kts; else echo app/build.gradle; fi; }
gradle_version_name() { sed -nE 's/^[[:space:]]*versionName[[:space:]]*=?[[:space:]]*"([^"]+)".*/\1/p' "$1" | head -1; }
gradle_version_code() { sed -nE 's/^[[:space:]]*versionCode[[:space:]]*=?[[:space:]]*([0-9]+).*/\1/p' "$1" | head -1; }
bump_gradle() { # file version
  local f="$1" v="$2" code
  code="$(gradle_version_code "$f")"; [ -n "$code" ] || die "$f: versionCode not found"
  sedi -E "s/^([[:space:]]*versionName[[:space:]]*=?[[:space:]]*\")[^\"]+(\".*)/\1$v\2/" "$f"
  sedi -E "s/^([[:space:]]*versionCode[[:space:]]*=?[[:space:]]*)[0-9]+/\1$((code + 1))/" "$f"
  echo "$f: versionName $v, versionCode $((code + 1))"
}

bump_python() {
  grep -Eq '^version *= *"' pyproject.toml || die "pyproject.toml: project.version not found"
  sedi -E "s/^(version *= *\")[^\"]+(\")/\1$VERSION\2/" pyproject.toml
  echo "pyproject.toml: version $VERSION"
}
bump_extension() {
  set_json_version manifest.json "$VERSION"; echo "manifest.json: version $VERSION"
  [ -f package.json ] && { set_json_version package.json "$VERSION"; echo "package.json: version $VERSION"; }
}
bump_js() { set_json_version package.json "$VERSION"; echo "package.json: version $VERSION"; [ -f package-lock.json ] && npm install --package-lock-only --ignore-scripts >/dev/null && echo "package-lock.json: synced"; }
bump_react_native() {
  local pkg gf
  pkg="$(current_json_version package.json)"
  if [ -d android ]; then
    gf="android/app/build.gradle"; [ -f "$gf" ] || gf="android/app/build.gradle.kts"
    [ "$(gradle_version_name "$gf")" = "$pkg" ] || die "drift: $gf versionName $(gradle_version_name "$gf") != package.json $pkg"
  fi
  if [ -d ios ]; then
    local mv; mv="$(grep -ho 'MARKETING_VERSION = [^;]*' ios/*.xcodeproj/project.pbxproj | head -1 | awk '{print $3}')"
    [ "$mv" = "$pkg" ] || die "drift: ios MARKETING_VERSION $mv != package.json $pkg"
  fi
  bump_js
  [ -d android ] && bump_gradle "$gf" "$VERSION"
  if [ -d ios ]; then
    sedi -E "s/(MARKETING_VERSION = )[^;]*;/\1$VERSION;/" ios/*.xcodeproj/project.pbxproj
    local cur; cur="$(grep -ho 'CURRENT_PROJECT_VERSION = [0-9]*' ios/*.xcodeproj/project.pbxproj | head -1 | awk '{print $3}')"
    sedi -E "s/(CURRENT_PROJECT_VERSION = )[0-9]*;/\1$((cur + 1));/" ios/*.xcodeproj/project.pbxproj
    echo "ios: MARKETING_VERSION $VERSION, CURRENT_PROJECT_VERSION $((cur + 1))"
  fi
}
bump_android() {
  local gf; gf="$(gradle_file)"
  if [ -f VERSION ] && [ -x scripts/sync-version.sh ]; then
    # Repos with a second app (just-my-weather's iOS target) keep one semver in
    # VERSION and copy it into versionName and MARKETING_VERSION with their own
    # sync script; versionCode is still bumped here (§7: +1 per upload).
    [ "$(cat VERSION)" = "$(gradle_version_name "$gf")" ] || die "drift: VERSION $(cat VERSION) != $gf versionName $(gradle_version_name "$gf")"
    printf '%s\n' "$VERSION" > VERSION; scripts/sync-version.sh; echo "VERSION: $VERSION (synced by scripts/sync-version.sh)"
    local code; code="$(gradle_version_code "$gf")"
    sedi -E "s/^([[:space:]]*versionCode[[:space:]]*=?[[:space:]]*)[0-9]+/\1$((code + 1))/" "$gf"; echo "$gf: versionCode $((code + 1))"
  else
    bump_gradle "$gf" "$VERSION"
    [ -f VERSION ] && { printf '%s\n' "$VERSION" > VERSION; echo "VERSION: $VERSION"; }
  fi
}
bump_go() { echo "go: the tag is the version; nothing to bump"; }

STACK="$(detect_stack)"
echo "== release $TAG ($STACK)"
case "$STACK" in
  python) bump_python ;; extension) bump_extension ;; react-native) bump_react_native ;;
  js) bump_js ;; android) bump_android ;; go) bump_go ;;
  *) die "no version source-of-truth file found (§7)" ;;
esac

# ---- 3. changelog -----------------------------------------------------------
command -v git-cliff >/dev/null || die "git-cliff is not on PATH"
if [ -f CHANGELOG.md ]; then
  git cliff --tag "$TAG" --unreleased --prepend CHANGELOG.md
else
  git cliff --tag "$TAG" > CHANGELOG.md
fi
echo "CHANGELOG.md: $TAG entry written"

# ---- 4. commit, tag, push ---------------------------------------------------
git add -A
git commit -q -m "chore(release): $TAG" -m "Source: VC Standard §7"
git tag -s -a "$TAG" -m "$TAG"
git push origin main
git push origin "$TAG"
echo "== pushed main and $TAG; release.yml creates the GitHub Release"
