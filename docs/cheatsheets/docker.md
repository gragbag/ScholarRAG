# Docker — Study Cheat Sheet

Grounded in the ScholarRAG build (`Dockerfile`, `docker-compose.yml`).

## The one mental model

- **Image** = a read-only *template* (a frozen filesystem + a default command). Inert. Like a class.
- **Container** = a *running instance* of an image. Like an object. One image → many containers.
- **Layers** = each Dockerfile instruction adds a cached layer; images share layers (efficient).
- **Registry** = where images are stored/shared (Docker Hub, GHCR, AWS ECR). `push`/`pull`.

> Image size ≠ runtime RAM. Size affects disk + pull time; RAM is set by the *running process*.
> One image, three roles: the same image runs uvicorn / celery / streamlit via different commands.

## Dockerfile instructions

| Instruction | Does | Note |
|---|---|---|
| `FROM img AS stage` | Base image / names a build stage | Multi-stage = build in one, copy the artifact into a slim final |
| `WORKDIR /app` | Sets the working dir | |
| `COPY src dst` | Copies build-context files in | Only what you COPY is in the image |
| `RUN cmd` | Runs a command at *build* time | Each RUN = a layer; use `--mount=type=cache` for pkg caches |
| `ENV K=V` | Env var baked into the image | |
| `ARG K=V` | Build-time variable | Not present at runtime (unlike ENV) |
| `EXPOSE 8000` | Documents a port | Doesn't publish it |
| `CMD ["a","b"]` | Default command | Overridable at `run`/compose `command:` |
| `ENTRYPOINT` | Fixed command prefix | CMD becomes its args |
| `HEALTHCHECK` | How Docker probes health | |

**Order for cache hits:** copy dependency manifests → install deps → *then* copy source.
Deps rarely change, so that layer stays cached across code edits.

## Essential commands

```bash
docker build -t name:tag .          # build an image from ./Dockerfile
docker images                       # list images (+ sizes)
docker run --rm -p 8000:8000 name   # run; --rm auto-cleans; -p host:container
docker run --rm --entrypoint sh img -c "..."   # override entrypoint (debug)
docker ps            /  docker ps -a            # running / all containers
docker exec -it <c> sh              # shell into a running container
docker logs -f <c>                  # tail its logs
docker system df     /  docker system prune     # disk usage / reclaim space
```

### Compose (multi-service local stacks)

```bash
docker compose up -d --build        # build + start all services, detached
docker compose up -d api ui         # just these services
docker compose ps        /  logs -f [svc]        # status / tail logs
docker compose config               # validate + print the merged config
docker compose down [-v]            # stop; -v also removes named volumes
docker compose build api            # (re)build one service's image
```

**Compose networking (the recurring lesson):** services reach each other by
**service name + container port** (`http://api:8000`), NOT `localhost` and NOT the
host-mapped port. `localhost` inside a container is *that* container.

## Gotchas we hit

- **Deps/extras missing at runtime** → the image installed core deps only. Install the
  runtime extras explicitly in the Dockerfile (an `ARG EXTRAS` used on the `uv sync` lines).
- **6 GB image** → PyTorch's default wheel bundles CUDA (`nvidia*`, `triton` ≈ 4 GB).
  We run on CPU → pin torch to the **CPU wheel index** → ~2 GB.
- **`.dockerignore`** keeps `.env` (secrets!), `data/`, `.venv`, caches OUT of the image/context.
- **`--rm`** on throwaway runs so stopped containers don't pile up.

## Slimming an image (in order of impact)

1. Multi-stage build (build deps stay out of the final image).
2. Avoid GPU libs you don't use (CPU-only torch).
3. Don't install dev/test/eval deps (`--no-dev`, curated extras).
4. `.dockerignore` a small build context.
5. Slim base (`python:3.12-slim`); distroless for the last mile.
