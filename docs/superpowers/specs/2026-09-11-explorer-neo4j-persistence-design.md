# Spec: Explorer Neo4j persistence

**Date:** 2026-09-11
**Status:** Proposed — awaiting approval

## Objective

Allow `semantica server` to opt into Neo4j-backed persistence while preserving
the existing Explorer behavior and its in-memory default. In Neo4j mode, data
imported or mutated through Explorer must survive both API-process and Neo4j
container restarts.

The user-facing workflow is:

```bash
export SEMANTICA_EXPLORER_BACKEND=neo4j
export GRAPH_STORE_NEO4J_URI=bolt://127.0.0.1:7687
export GRAPH_STORE_NEO4J_USER=neo4j
export GRAPH_STORE_NEO4J_PASSWORD=password
export GRAPH_STORE_NEO4J_DATABASE=neo4j
export SEMANTICA_EXPLORER_GRAPH_ID=default
semantica server start --reload
```

## Acceptance criteria

1. With no new environment variables, `semantica server` continues to use an
   in-memory `ContextGraph` exactly as it does today.
2. With `SEMANTICA_EXPLORER_BACKEND=neo4j`, startup connects through the existing
   `GraphStore(backend="neo4j")` facade and hydrates the selected Explorer graph.
3. Nodes, relationships, properties, weights, temporal bounds, edge identities,
   and graph identity round-trip without loss.
4. Explorer imports and graph mutations are visible immediately and survive a
   Python server restart and a Neo4j container restart.
5. Multiple Explorer graphs can share one Neo4j database without collisions by
   using `SEMANTICA_EXPLORER_GRAPH_ID`.
6. Neo4j connection or persistence failures are explicit. Startup fails closed;
   mutations must not be reported as durable when the Neo4j write failed.
7. On shutdown, the Neo4j driver is closed cleanly.
8. Existing REST, analytics, search, WebSocket, ontology, and memory-mode tests
   remain compatible.

## Architecture

`GraphSession` currently depends on `ContextGraph` internals (`nodes`, `edges`,
type indexes, adjacency, temporal helpers). Passing `GraphStore` directly would
break these features. The implementation will therefore introduce a small
Neo4j persistence component around the existing `ContextGraph`:

```text
Explorer REST / WebSocket
          |
      GraphSession
          |
     ContextGraph  <---- startup hydration ---- Neo4jExplorerPersistence
          |                                      |
          +------- synchronous mutations ------> GraphStore(neo4j)
                                                         |
                                                        Bolt
```

The in-memory graph remains the analytical working set. Neo4j is the durable
source of truth in Neo4j mode.

### Neo4j representation

Explorer-owned data uses stable, namespaced records so it does not consume or
delete unrelated GraphStore data:

```cypher
(:SemanticaExplorerNode {
  graph_id, id, node_type, content, properties_json, valid_from, valid_until
})

(:SemanticaExplorerNode)-[:SEMANTICA_EXPLORER_EDGE {
  graph_id, id, family_id, edge_type, weight, properties_json,
  valid_from, valid_until
}]->(:SemanticaExplorerNode)
```

`properties_json` is canonical JSON. This avoids Neo4j property restrictions on
nested maps while retaining the complete ContextGraph payload. Static labels and
relationship types keep Cypher construction parameterized and avoid identifier
injection.

### Consistency rules

- Startup hydration suppresses mutation callbacks so restored data is not
  written back as new mutations.
- A mutation is persisted synchronously before the API reports success.
- Persistence errors raise a typed error and mark the Explorer session unhealthy;
  there is no silent fallback to memory mode.
- External Neo4j writes are not live-synchronized into a running Explorer
  process in this first version. They become visible after API restart.
- One running Explorer writer per `graph_id` is supported. Multi-writer conflict
  resolution is out of scope.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `SEMANTICA_EXPLORER_BACKEND` | `memory` | `memory` or `neo4j` |
| `SEMANTICA_EXPLORER_GRAPH_ID` | `default` | Namespace within the Neo4j database |
| `GRAPH_STORE_NEO4J_URI` | existing GraphStore default | Bolt URI |
| `GRAPH_STORE_NEO4J_USER` | existing GraphStore default | Neo4j user |
| `GRAPH_STORE_NEO4J_PASSWORD` | existing GraphStore default | Neo4j password |
| `GRAPH_STORE_NEO4J_DATABASE` | existing GraphStore default | Neo4j database |

Unknown backend values fail startup with a clear configuration error.

## Tech stack

- Python 3.9+
- FastAPI lifespan integration
- Existing `ContextGraph`, `GraphSession`, and `GraphStore`
- Existing Neo4j Python driver; no new dependency
- pytest for unit and integration tests

## Commands

```bash
# Focused tests
.senmantica/bin/python -m pytest tests/explorer/test_neo4j_persistence.py -q
.senmantica/bin/python -m pytest tests/explorer/test_explorer_entrypoints.py -q

# Explorer regression suite
.senmantica/bin/python -m pytest tests/explorer -q

# Manual development startup
SEMANTICA_EXPLORER_BACKEND=neo4j semantica server start --reload

# End-to-end health check
curl http://127.0.0.1:8000/api/graph/stats
```

## Project structure

- `semantica/explorer/persistence.py` — Neo4j hydration and durable mutation API.
- `semantica/explorer/runtime.py` — construct the selected Explorer graph/session.
- `semantica/server.py` — use the runtime factory and close it in lifespan cleanup.
- `tests/explorer/test_neo4j_persistence.py` — fake-backed contract and failure tests.
- `docs/reference/explorer.md` — configuration and operational behavior.

## Code style

Configuration parsing is explicit and fail-fast:

```python
backend = os.environ.get("SEMANTICA_EXPLORER_BACKEND", "memory").strip().lower()
if backend not in {"memory", "neo4j"}:
    raise ValueError(f"Unsupported Explorer backend: {backend}")
```

All Cypher values are parameters. Dynamic user values are never interpolated
into labels, relationship types, or query strings.

## Testing strategy

### Unit tests

- Memory remains the default and does not construct GraphStore.
- Neo4j configuration constructs and connects the existing facade.
- Hydration preserves nodes, edges, nested properties, and temporal metadata.
- Mutation operations emit idempotent parameterized writes.
- Hydration suppresses persistence callbacks.
- Unsupported backend and failed connections fail closed.
- Persistence errors propagate instead of being logged as successful writes.
- Shutdown closes GraphStore.

### Integration test

Against the local `semantica-neo4j` container:

1. Use an isolated random `graph_id`.
2. Import a four-node supply-chain graph through the Explorer API.
3. Restart the Python API and verify stats/search.
4. Restart the Neo4j container and verify again.
5. Delete only records with the random `graph_id`.

## Boundaries

### Always

- Preserve memory-mode behavior and API response shapes.
- Parameterize Cypher and isolate records by `graph_id`.
- Clean up integration-test records precisely.
- Add tests before implementation.

### Ask first

- Add database constraints or indexes.
- Change the public GraphStore API.
- Introduce background synchronization or multi-writer coordination.

### Never

- Store credentials in source or committed configuration.
- Delete or rewrite unrelated Neo4j nodes.
- Silently fall back to memory after Neo4j was explicitly selected.
- Couple Explorer to Neo4j driver internals instead of `GraphStore`.

## Out of scope

- Migrating the current process's in-memory graph automatically.
- Treating arbitrary pre-existing Neo4j nodes as Explorer-owned data.
- Live ingestion of external Cypher changes.
- Multi-process/multi-writer conflict resolution.
- Fixing the independently discovered `/api/graph/path` dictionary-adapter bug.

## Open questions

1. Should the first version guarantee per-mutation atomic rollback of the local
   ContextGraph when Neo4j rejects a write, or is fail-closed API behavior plus
   rehydration on restart sufficient? Recommendation: guarantee rollback for the
   four Explorer mutation entry points, then broaden coverage incrementally.
2. Should production require an explicit non-default `graph_id`? Recommendation:
   keep `default` for local compatibility and document explicit IDs for deployed
   environments.
