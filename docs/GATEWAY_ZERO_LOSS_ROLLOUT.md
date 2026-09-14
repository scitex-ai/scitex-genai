# Gateway zero-loss rollout

The rollout frontend is a permanent systemd socket and
`systemd-socket-proxyd`. It owns the fleet-facing TCP port. Each gateway build
listens on a generation-specific Unix socket:

```text
clients -> frontend.socket:18772 -> current.sock -> <generation>.sock
                                                   old.sock (draining)
```

The proxy does not retry HTTP requests. Replacing the `current.sock` symlink is
one atomic `rename(2)`: existing connections stay with their original process,
while new connections resolve the promoted socket. After the promoted process
is verified through the public port, SIGTERM stops the old listener and lets
all of its admitted and queued ASGI tasks complete. Rollout backends do not use
the legacy shutdown hook that closes admission and wakes queued requests with
503.

`/health` and authenticated `/admin/status` expose:

```json
{"gateway":{"build":"<commit>","incarnation":"<systemd invocation>","frontend_generation":"<generation>","code_fingerprint":"<sha256>"}}
```

The controller verifies the identity directly on the candidate Unix socket and
again through the frontend. The source fingerprint must match the installed
gateway package executing the rollout command, so a caller-supplied build label
cannot verify the wrong code. Rollback uses the fingerprint retained when the
prior generation was promoted. The controller also requires gateway readiness,
every configured inference member's readiness, exact label set, request
capacity, and token capacity. A failed public verification restores the prior
symlink before stopping the candidate. `rollback-generation` starts and
verifies the retained prior unit before reversing the same switch. A
nonblocking file lock serializes promotion and rollback controllers for the
shared frontend.

Sticky routes and continuation-success classification are stored as bounded,
hashed, payload-free metadata in
`~/.scitex/genai/runtime/gateway-session-state.json`. Updates from overlapping
old and new processes use a file lock and atomic replacement. Raw session IDs
and request bodies are never written. Admission-prediction history retains its
separate engine-generation guard; queues remain owned by and drain in the
process that accepted them.

## One-time coordinated bootstrap

This is the only rollout requiring a client pause. The legacy gateway itself
owns port 18772, so the permanent frontend cannot bind it until that process is
stopped. Do not use `restart-unit` or `/admin/drain`: global drain deliberately
wakes queue waiters with 503.

Prepare an immutable candidate first. Use the exact merged commit, copy the
currently effective deployment config into the artifact, install its gateway
extra in an artifact-local virtual environment, and retain a dependency freeze.
Before the pause, run it on a spare loopback port and verify `/health`, expected
member labels and capacities, and the installed module/source identity. Stop
that candidate after the probe.

Save rollback material before writing any unit:

```console
$ systemctl --user cat scitex-genai-gateway.service
$ cp ~/.config/systemd/user/scitex-genai-gateway.service /safe/rollback/
$ cp -a ~/.config/systemd/user/scitex-genai-gateway.service.d /safe/rollback/
```

Write, but do not start, the permanent frontend:

```console
$ /candidate/venv/bin/scitex-genai-gateway install-rollout-units \
    --config /candidate/config.yaml
```

Coordinate every client/agent to finish its current atomic turn and hold before
submitting another. This is an external admission pause; stopping agent sessions
is neither required nor desired. Poll the legacy JSON without `curl -f` because
degraded health is HTTP 503:

```console
$ curl -sS http://127.0.0.1:18772/health | jq \
    '{ready,in_flight,queued,held,members}'
```

Proceed only after `in_flight`, `queued`, and `held` are zero on two observations
several seconds apart and the client hold is independently confirmed. Then run:

```console
$ /candidate/venv/bin/scitex-genai-gateway rollout-generation \
    --generation <short-commit> --build <full-commit> \
    --config /candidate/config.yaml --bootstrap-coordinated
```

The command starts and validates the private Unix-socket candidate, rechecks the
legacy empty boundary, enables the candidate for reboot, disables and stops the
legacy direct unit, switches `current.sock`, enables the stable frontend socket,
and validates the exact candidate through port 18772. If any post-stop step
fails, it disables the frontend and candidate and re-enables the saved legacy
unit.

Keep clients held while checking `MainPID`, unit status, the public health
identity, expected member labels/capacities, one authenticated metadata request,
and one deliberately selected inference smoke request. Release clients only
after those checks pass. If validation fails, restore the saved legacy unit and
drop-in, reload systemd, start it, and verify its old artifact path before
releasing clients.

The first bootstrap cannot recover sticky/QoS mappings already trapped inside
the legacy process because that version exposes no state export. Engine KV and
agent sessions remain intact. All subsequent generations share the new hashed
metadata store.

## Routine rollout and rollback

After bootstrap, no client pause is needed:

```console
$ /new/venv/bin/scitex-genai-gateway rollout-generation \
    --generation <short-commit> --build <full-commit> \
    --config /new/config.yaml
$ /new/venv/bin/scitex-genai-gateway rollback-generation \
    --config /new/config.yaml
```

Successful promotion leaves the active backend enabled and disables the old
backend only after its admitted and queued work drains. Rollback performs the
same lifecycle in reverse.

Generation labels must be unique and contain only letters, digits, dots,
underscores, or hyphens. Generation sockets live under
`$XDG_RUNTIME_DIR/scitex-genai-gateway`. The durable frontend selector and
active/previous generation metadata live under `~/.scitex/genai/runtime/`, so
the enabled active backend and frontend retain the same route after reboot.
