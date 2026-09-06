# CEREBRO TIDE-X

CEREBRO TIDE-X es un motor Rust para adquisición de evidencia, reconstrucción de subespacios de habilidad, consolidación de memoria, aprendizaje adaptativo y ejecución gobernada. Su diseño prioriza autoridad explícita, persistencia direccionada por contenido, operaciones idempotentes, recuperación fail-closed y evidencia reproducible de calidad.

Este repositorio no declara por sí solo conformidad DO-178C, ISO 26262, ASIL D ni otra certificación normativa externa. Las puertas `P0`, `P1` y `P2` son controles internos reproducibles del proyecto. `P3` añade aseguramiento acotado de concurrencia y recuperación, pero tampoco constituye una prueba universal del kernel, filesystem, hardware o entorno de despliegue.

## Estado de calidad

El commit `c8bbb6e8694ecf41df7b82c4962ddfedeeed3dda` cerró Puerta 2 con:

- P0 acumulativa: formato, Clippy con `-D warnings`, pruebas `--all-targets`, auditoría de dependencias y arranque vacío fail-closed.
- P1 acumulativa: cobertura, Miri, ASan, LSan, TSan y fuzzing de 100.000 ejecuciones por target configurado.
- P2: cobertura global 86,91% líneas / 80,14% funciones / 87,57% regiones; Miri ampliado a `low_rank_math`, `linalg`, `trust_region` y `transport`; TSan sobre todos los targets; snapshot SHA-256 pre/post sin mutación del checkout.

Pisos críticos P2 medidos:

| Componente | Líneas |
| --- | ---: |
| `src/engine/runtime.rs` | 80,18% |
| `src/engine/transition.rs` | 75,58% |
| `src/engine/support.rs` | 83,77% |
| `src/engine/analysis.rs` | 85,30% |
| `src/isolated_execution.rs` | 87,33% |
| `src/digest.rs` | 98,97% |

La definición detallada de las puertas está en [`quality/README.md`](quality/README.md).

## Toolchain

`rust-toolchain.toml` fija el toolchain estable `1.96.0` con `clippy` y `rustfmt`. Las puertas que necesitan Miri o sanitizadores usan además el nightly fijado por sus propios scripts de calidad y verifican su commit de toolchain.

`Cargo.toml` declara `rust-version = "1.85"` como versión mínima de lenguaje/compilador admitida por el paquete; no es la versión con la que se certifican las puertas actuales.

El crate aplica:

```toml
[lints.rust]
unsafe_code = "forbid"
```

Actualmente no hay bloques `unsafe` en `src/` ni `tests/`. Esta propiedad reduce clases de errores de memoria en código Rust propio, pero no equivale a demostrar ausencia universal de fugas, carreras, fallos lógicos o defectos en dependencias/sistema operativo; por eso existen Miri y los sanitizadores.

## Raíz privada de autoridad

Las rutas de producción que usan estado TIDE-X obtienen la raíz desde:

```bash
export TIDEX_PRIVATE_ROOT=/var/lib/tidex-brain
```

La ruta debe:

- ser absoluta;
- existir antes de iniciar el programa;
- ser un directorio real, no un symlink;
- no conceder permisos a grupo u otros.

Configuración recomendada:

```bash
sudo install -d -m 0700 /var/lib/tidex-brain
export TIDEX_PRIVATE_ROOT=/var/lib/tidex-brain
```

Los helpers internos `secure_dir` y `secure_file` fijan respectivamente `0700` y `0600` cuando crean/protegen artefactos administrados por el motor.

## Compilación

Compilación reproducible sin resolución de red:

```bash
cargo build --release --bins --offline --locked
```

El perfil release usa `lto = "thin"`, `codegen-units = 1`, `panic = "abort"` y `strip = "symbols"`.

`Cargo.toml` declara ocho binarios:

1. `cerebro-tidex`
2. `acquire-system`
3. `adaptive-learning-cycle`
4. `autonomous-learning-plan`
5. `ledger-diagnose`
6. `pure-linear-runner`
7. `record-representation-evidence`
8. `tidex-finalize`

## Interfaces reales de los binarios

### `cerebro-tidex`

El CLI principal obtiene la raíz de `TIDEX_PRIVATE_ROOT`. No acepta `--root`.

```bash
cerebro-tidex status
cerebro-tidex analyze /var/lib/tidex-brain/observations.json
cerebro-tidex sleep
```

`analyze` sólo acepta un JSON confinado bajo la raíz privada. `commit`, `artifact-import-f32` y `artifact-ties` están retirados; las mutaciones de aprendizaje/finalización usan binarios receipt-bound separados. El CLI principal actual no expone un comando `search`.

### `acquire-system`

Captura una raíz fuente externa y sella una autoridad de adquisición bajo `TIDEX_PRIVATE_ROOT`.

Ejemplo de proyecto completo:

```bash
acquire-system \
  --source-root /srv/project \
  --acquisition-id capture-001 \
  --scope whole-project
```

Para alcance declarado se usa `--scope declared-paths` junto con uno o más `--path`, o la forma de compatibilidad `--scope paths:a,b`. También existen `--residency`, `--max-files`, `--max-bytes`, `--exclude` y `--noise-policy`.

### `adaptive-learning-cycle`

La sintaxis implementada es:

```text
adaptive_learning_cycle start <session-id> <learning-target.json> <policy.json>
adaptive_learning_cycle next <session-id>
adaptive_learning_cycle assimilate <session-id> <experiment-evidence.json>
adaptive_learning_cycle show <session-id>
adaptive_learning_cycle controller-train <controller-training-dataset.json> <controller-policy.json> <controller-binding.json>
adaptive_learning_cycle controller-show <session-id>
adaptive_learning_cycle controller-compose <invocation.json>
```

Las operaciones persistentes usan `TIDEX_PRIVATE_ROOT`. `controller-compose` exige una invocación confinada en la raíz privada.

### `autonomous-learning-plan`

No necesita la raíz privada porque calcula un plan puro desde un fichero de entrada:

```bash
autonomous-learning-plan learning-target.json
```

### `ledger-diagnose`

No acepta argumentos. Verifica el ledger de la raíz configurada y emite sus eventos y cabeza verificada:

```bash
export TIDEX_PRIVATE_ROOT=/var/lib/tidex-brain
ledger-diagnose
```

### `pure-linear-runner`

Es una frontera de ejecución aislada para el protocolo `pure_capability_e2e`. No es un CLI interactivo de propósito general y falla cerrado cuando no recibe el contrato de entrada que espera el entorno aislado.

### `record-representation-evidence`

```bash
record-representation-evidence <sealed-install-request.json>
```

La autoridad destino se toma de `TIDEX_PRIVATE_ROOT`.

### `tidex-finalize`

La interfaz es posicional, no usa flags `--root` ni `--session-id`:

```bash
tidex-finalize <session-id> <representation-evidence-receipt.json>
```

La raíz se obtiene de `TIDEX_PRIVATE_ROOT`. La invocación se valida antes de abrir el lifecycle o el engine.

## Puertas de calidad

```bash
bash quality/gate0-release.sh
bash quality/gate1-tooling.sh
bash quality/gate2-verification.sh
bash quality/gate3-assurance.sh
```

P3 es acumulativa sobre P2 y añade model checking determinista acotado, pruebas concurrentes/recovery obligatorias y un recibo de aseguramiento SHA-256 externo al checkout. No debe describirse como “verificación formal universal”.

## Dependencias y licencias

`deny.toml` controla las licencias y fuentes permitidas de dependencias. La allowlist actual incluye Apache-2.0, Apache-2.0 WITH LLVM-exception, MIT, Unicode-3.0 y Unlicense. El paquete raíz no declara actualmente un campo `license` en `Cargo.toml`; por tanto no debe inferirse una licencia del propio producto a partir de la política de dependencias.

La base RustSec usada por las puertas es una snapshot local. Una ejecución reproducible offline demuestra ausencia de avisos respecto de esa snapshot concreta, no respecto de vulnerabilidades publicadas con posterioridad.
