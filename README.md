# CEREBRO TIDE-X

CEREBRO TIDE-X es un motor Rust para adquisición de evidencia, reconstrucción de subespacios de habilidad, consolidación de memoria, aprendizaje adaptativo y ejecución gobernada. Su diseño prioriza autoridad explícita, persistencia direccionada por contenido, idempotencia en rutas críticas, recuperación fail-closed y evidencia de calidad vinculada por hashes.

Este repositorio no declara por sí solo conformidad DO-178C, ISO 26262, ASIL D ni otra certificación normativa externa. Las puertas `P0` a `P3` son controles internos automatizados del proyecto y fijan toolchains, dependencias y snapshots para hacer sus ejecuciones repetibles bajo un entorno compatible. `P3` añade aseguramiento acotado de concurrencia y recuperación, pero tampoco constituye una prueba universal del kernel, filesystem, hardware o entorno de despliegue.

## Estado de calidad

Puerta 2 quedó congelada inicialmente en el commit `c8bbb6e8694ecf41df7b82c4962ddfedeeed3dda`. Puerta 3 se ejecutó después sobre ese commit más el candidato P3 y el snapshot exacto que superó la puerta se congeló finalmente en:

```text
43588d43d76269258efd6928b098369030f049cb
```

La ejecución P3 fue acumulativa: volvió a ejecutar P0, P1 y P2 antes de sus propios controles. En esa corrida:

- P0 pasó formato, Clippy con `-D warnings`, pruebas `--all-targets`, auditoría/políticas de dependencias y arranque vacío fail-closed.
- P1 pasó cobertura, Miri, ASan, LSan obligatorio por herencia de P2, TSan y 100.000 ejecuciones de cada uno de los dos targets de fuzz configurados.
- P2 pasó de nuevo con Miri ampliado sobre `low_rank_math`, `linalg`, `trust_region` y `transport`, TSan sobre `--all-targets` y snapshot SHA-256 pre/post sin mutación del checkout.
- P3 pasó sus 17 pruebas de aseguramiento obligatorias, incluidos los dos modelos deterministas acotados, concurrencia real de `engine_authority`, carreras/adversarios de filesystem y recuperación del corpus.

Última cobertura P2 medida dentro de la corrida P3:

| Componente | Líneas | Funciones | Regiones |
| --- | ---: | ---: | ---: |
| `src/engine/runtime.rs` | 80,18% | 77,12% | 81,49% |
| `src/engine/transition.rs` | 75,80% | 70,00% | 76,40% |
| `src/engine/support.rs` | 83,77% | 81,40% | 85,71% |
| `src/engine/analysis.rs` | 85,30% | 79,69% | 86,37% |
| `src/isolated_execution.rs` | 87,33% | 82,86% | 89,25% |
| `src/digest.rs` | 98,97% | 98,18% | 98,50% |
| **Global** | **86,97%** | **80,24%** | **87,62%** |

El receipt de aquella ejecución P3 se generó fuera del checkout. Como la puerta se ejecutó antes de crear el commit final, su campo `head_commit` identifica el padre `c8bbb6e...`; la vinculación con `43588d4...` se comprobó después comparando el manifiesto SHA-256 completo y los metadatos del checkout, que coincidieron exactamente. Por tanto, el commit `43588d4...` contiene el mismo snapshot de archivos que pasó P3, aunque el receipt original no fue reescrito para fingir un HEAD posterior.

La definición detallada de las puertas y los límites de esta evidencia están en [`quality/README.md`](quality/README.md).

## Toolchain

`rust-toolchain.toml` fija el toolchain estable `1.96.0` con `clippy` y `rustfmt`. Las puertas que necesitan Miri o sanitizadores usan además el nightly fijado por sus propios scripts de calidad y verifican su commit de toolchain.

`Cargo.toml` declara `rust-version = "1.85"` como MSRV del paquete. Esa declaración no sustituye una prueba de la suite completa con Rust 1.85; las puertas actuales se ejecutan con el toolchain estable fijado en `1.96.0`.

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

Los helpers internos `secure_dir` y `secure_file` no crean rutas: cuando se aplican a una ruta existente, fijan respectivamente los modos `0700` y `0600`.

## Compilación

Compilación release con dependencias bloqueadas y sin resolución de red:

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
adaptive-learning-cycle start <session-id> <learning-target.json> <policy.json>
adaptive-learning-cycle next <session-id>
adaptive-learning-cycle assimilate <session-id> <experiment-evidence.json>
adaptive-learning-cycle show <session-id>
adaptive-learning-cycle controller-train <controller-training-dataset.json> <controller-policy.json> <controller-binding.json>
adaptive-learning-cycle controller-show <session-id>
adaptive-learning-cycle controller-compose <invocation.json>
```

Las operaciones persistentes usan `TIDEX_PRIVATE_ROOT`. `controller-compose` exige una invocación confinada en la raíz privada.

### `autonomous-learning-plan`

No necesita la raíz privada porque calcula un plan puro desde un fichero de entrada:

```bash
autonomous-learning-plan learning-target.json
```

### `ledger-diagnose`

No define opciones de CLI propias. La implementación actual no inspecciona `argv`, por lo que argumentos adicionales se ignoran. Verifica el ledger de la raíz configurada y emite sus eventos y cabeza verificada:

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

P3 es acumulativa sobre P2 y añade model checking determinista acotado, pruebas concurrentes/recovery obligatorias y un recibo de aseguramiento SHA-256 externo al checkout. El snapshot que la superó está congelado en `43588d43d76269258efd6928b098369030f049cb`. Esta evidencia no debe describirse como “verificación formal universal”.

## Firma de release (P4)

La frontera de firma está en `quality/sign-release.sh`. Opera únicamente sobre un directorio de release fuera del checkout, exige `release-manifest.json` y `SHA256SUMS`, verifica primero todos los checksums y crea firmas OpenPGP detached ASCII-armored para ambos ficheros. Cuando se proporciona además el `.tar.zst` generado por P4, verifica su checksum externo y firma también el archive y su fichero `.sha256`.

La clave nunca se selecciona de forma implícita. Debe indicarse con su fingerprint completo de 40 hexadecimales:

```bash
export TIDEX_RELEASE_GPG_KEY=<fingerprint-completo>
quality/sign-release.sh /ruta/al/release /ruta/al/release.tar.zst
```

El fingerprint debe corresponder a una clave o subclave secreta con capacidad de firma disponible en el `GNUPGHOME` activo. Si la clave necesita passphrase en automatización, puede proporcionarse mediante un fichero externo al checkout con permisos `0600` o `0400` usando `TIDEX_RELEASE_GPG_PASSPHRASE_FILE`. El script no genera claves, no elige una por defecto, no firma artefactos con checksums inválidos y no sustituye una firma preexistente que no verifique con la clave autorizada.

La infraestructura de firma no decide la licencia legal del producto. `Cargo.toml` continúa sin declarar `license`/`license-file`; una distribución pública debe resolver esa decisión por separado.

## Dependencias y licencias

`deny.toml` controla las licencias y fuentes permitidas de dependencias. La allowlist actual incluye Apache-2.0, Apache-2.0 WITH LLVM-exception, MIT, Unicode-3.0 y Unlicense. El paquete raíz no declara actualmente un campo `license` en `Cargo.toml`; por tanto no debe inferirse una licencia del propio producto a partir de la política de dependencias.

La base RustSec usada por las puertas es una snapshot local. Una ejecución reproducible offline demuestra ausencia de avisos respecto de esa snapshot concreta, no respecto de vulnerabilidades publicadas con posterioridad.
