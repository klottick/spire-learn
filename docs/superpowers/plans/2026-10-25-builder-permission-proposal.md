# Builder permission repair proposal

**Status: user-approved and completed 2026-10-25; protected preflight passed.**

Approval: the user replied **“I approve”** to the explicit request to create
this local derivative (one image-root chmod, default user `1000:1000`,
every-UID traversal disclosure) and repeat the non-root preflight. Approval
recorded at `2026-10-25T11:52:20Z`. No broader root/group workaround is covered.

Verified result: local linux/amd64 builder ID
`sha256:470a1d6751fa197eaeb99319d9a2f2f24f3d7da11347659c7b24b4ff5527c738`.
The original 26 filesystem layers are retained with one added layer containing
only directory `root`, mode `0551`, owner `0:0`, no payload. `User=1000:1000`
is the only image-config difference; environment and entrypoint are unchanged.
Direct Rust/Cargo **1.98.0** and builder Python **3.12.14** passed protected
non-root preflight. The approved source projection and empty read-only fixture
overlay passed; Gym source/lock/fixture hashes and existing learner changes
remain unchanged. Both disposable task containers were removed.

Evidence is retained in
`artifacts/development/foundation/builder-repair-report-2026-10-25.md`,
`builder-repair-review-2026-10-25.md`, and
`builder-repair-parent-review-2026-10-25.md` under that same foundation directory.
Independent task review passed specification and quality, with a minor audit-log
completeness limitation retained in its report. Parent reconciled the history
count question: 50/52 JSON records contain 63/65 text lines because some
instructions contain newlines. Parent accepted the actual repair evidence.
This result is builder preflight, not native-wheel/runtime conformance.

This is a proposed narrow amendment to the
[Stage A setup addendum](2026-10-24-native-setup-addendum.md). It addresses only
the builder permission blocker; it does not establish native conformance or
Task 6 completion. This approval request covers only the image repair and
repeated preflight; it makes no decision about downstream setup authority.

## Evidence

Reconciliation on 2026-10-25 confirms both pinned images remain local. Gym is
clean at `4874fb398128c142bd56f10a12e6fca939a88f93`; its Cargo.lock hash matches
the addendum. No setup/build process or preflight container is running.

A read-only `docker image save` stream of the exact pinned builder was inspected
with Python 3.12 `tarfile`, without extracting payloads, creating a container,
or executing image programs. Layer metadata was applied in manifest order.
For `/root` and the Cargo/rustup trees, the audit observed 67,447 entries and
no whiteouts or links outside those trees. Every regular file below those
trees has other-read permission; every directory below `/root` has other-read
and other-search permissions. Specifically:

| Path | Mode | Owner | Kind |
| --- | --- | --- | --- |
| `/root` | `0550` | `0:0` | directory |
| `/root/.cargo`, `/root/.cargo/bin` | `0755` | `0:0` | directories |
| `/root/.cargo/bin/rustup` | `0755` | `0:0` | executable |
| `/root/.cargo/bin/rustc`, `/root/.cargo/bin/cargo` | `0777` | `0:0` | symlinks to `rustup` |
| `/root/.rustup`, `/root/.rustup/toolchains` | `0755` | `0:0` | directories |
| `/root/.rustup/settings.toml` | `0644` | `0:0` | file; contents not read |

The only toolchain directory observed is `stable-x86_64-unknown-linux-gnu`.
Its name does not establish the compiler version. Existing non-root diagnostic
logs independently establish the `/root` traversal failure. File execution and
Rust >=1.89 still require non-root preflight.

A supplementary archive check confirmed toolchain-bin `rustc` and `cargo`
are regular executable files with mode `0755`. `/bin` links to `/usr/bin`,
where `chmod` is a regular executable with mode `0755`. It found no relevant
ancestor whiteouts either. The exact selected metadata is retained in the owned
runtime's `logs/handoff-executable-metadata-2026-10-25.json`.

## Exact proposed image change

Prepare a local derivative from the already-pulled digest using this recipe:

```dockerfile
FROM ghcr.io/pyo3/maturin@sha256:0d1711efd2d3aab276f738bdcda48d378bceef2eeb7f8a8caf59451defddee72 AS builder
RUN ["/bin/chmod", "o+x", "/root"]
USER 1000:1000
```

The filesystem change is exactly `/root` mode `0550` -> `0551`: add search
permission for known paths, without granting directory listing or write access.
This applies to **every container UID**, not just UID 1000: processes knowing
paths can reach existing other-readable files anywhere below the image's
`/root`, subject to descendant permissions. Host home directories are never
mounted there.
Ownership and child file permissions stay unchanged. No recursive chmod,
toolchain copy, new toolchain, package installation, or source input is needed.
The intentional second change is image configuration: the derivative defaults
to the existing non-root UID:GID. Every run still specifies `--user 1000:1000`.

**The one `chmod` image-build instruction executes as image root. The approval
above covers only this root operation.** It is confined to image preparation with no host
source/work/home/credential/socket mounts, no secrets or SSH forwarding, and
network disabled. Wheel compilation and all diagnostic containers still run
as `1000:1000`. No root shell or supplemental group is proposed.

Use an empty build context (Dockerfile supplied on stdin), `--network=none`
and `--pull=false`, with the existing isolated Docker client config. No external
Dockerfile frontend directive is needed for this basic recipe. Record the exact
recipe, command, resulting image/config/layer identities, storage measurements
and history. Retain the pinned original and resulting local derivative; do not
push or prune. Stop on unexpected base resolution, filesystem changes, storage
growth, or failure.

## Validation after approval

1. Check >=30 GiB free and the original 20 GiB soft storage budget, including
   retained image/cache growth. Inspect the derivative's layer metadata: only
   `/root` may differ in filesystem metadata; compare any included file payloads
   before accepting the derivative. Verify the intentional `Config.User` change
   to `1000:1000` and inherited entrypoint/environment. Record its immutable
   image ID. Reconfirm the local base identity immediately before building.
2. Repeat the existing hardened non-root/no-network preflight, using that exact
   local image ID. Verify UID:GID, read-only root, caps, security/resource limits,
   and `/root` mode. First use only the owned `/work` bind, without Gym inputs.
3. Point PATH at
   `/root/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin` before the
   inherited PATH. Invoke bundled `rustc --version` and `cargo --version`
   directly, avoiding rustup initialization or updates. Keep `CARGO_HOME`,
   `HOME`, `TMPDIR` and build output under the owned runtime root. Verify compiler
   executable access and Rust >=1.89; stop on any inaccessible descendant.
4. Recheck the approved source mounts/empty read-only fixture overlay and
   resource controls before any wheel build under the applicable phase
   authority. The pinned runtime image and all Stage A dependency and
   conformance bounds stay as specified in the original addendum.

Approval of this repair permits creating this derivative and repeating preflight;
it does not by itself authorize broader setup, native/vector experiments,
TorchRL, GPU work, training, service changes or Gym edits. The original addendum
records a Task 1-only approval and leaves wheel build/install/conformance
unchecked. The current handoff states CPU Stage A setup authority. This repair
request neither expands nor revokes that authority; subsequent phases must be
matched to the applicable user instructions before execution. The repair itself
does not complete those phases.
