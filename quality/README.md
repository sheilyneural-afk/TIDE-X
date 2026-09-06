# Puerta 0: aptitud de una entrega

`quality/gate0-release.sh` es la puerta local de entrega. Usa el toolchain
estable fijado por `rust-toolchain.toml`, resuelve los dos grafos con sus
`Cargo.lock` y sin red, y construye en un directorio temporal fuera del
proyecto que elimina al terminar. Comprueba formato, Clippy con advertencias
como errores, todas las pruebas, auditoría de vulnerabilidades y las políticas
de dependencias tanto del núcleo como del arnés de fuzzing.

`.cargo/config.toml` fija `build.target-dir = "/tmp/tidex-cargo-target"`. Un
`cargo check`, `cargo test` o `cargo build` manual, y también `cargo metadata`
sobre `fuzz/Cargo.toml`, no pueden crear `target/` dentro del checkout. La
puerta lo verifica con `CARGO_TARGET_DIR` ausente. Las propias puertas siguen
usando destinos temporales aislados; ese valor por defecto solo cubre el uso
interactivo. `cargo fuzz coverage` es la excepción conocida: si no se pasa
`--target-dir` o `CARGO_TARGET_DIR`, escribe `./target` relativo al cwd, así
que esa campaña no se lanza sobre el árbol fuente.

El arnés aplica una política separada únicamente para admitir de forma
explícita la licencia permisiva NCSA del runtime LLVM libFuzzer; esa licencia
no queda admitida globalmente para el binario de producción.

También rechaza residuos conocidos y exige que `models/`, `state/`,
`artifacts/` y `evaluations/` no existan dentro del checkout. Los almacenes de
ejecución se configuran fuera del árbol fuente mediante `TIDEX_PRIVATE_ROOT`;
la puerta no consulta ni modifica ningún almacén residente.

```text
quality/gate0-release.sh
```

`quality/gate0-empty-state.sh` instala el binario real en dos directorios
temporales distintos y le entrega dos raíces privadas vacías mediante
`TIDEX_PRIVATE_ROOT`. La prueba exige en ambas instalaciones el rechazo exacto
`integrity:active_skill_bank_missing`, código de salida 2, salida estándar
vacía, diagnósticos idénticos y cero ficheros creados. Esto demuestra arranque
fail-closed y portabilidad sin consultar ni modificar el estado residente; no
depende de bind mounts, de una ruta canónica del checkout ni de una
inicialización que TIDE-X todavía no ofrece.

`cargo audit --no-fetch` usa la copia local de la base RustSec: garantiza que
la puerta no cambie de evidencia durante la ejecución, pero la vigencia de esa
copia debe controlarse por separado en el proceso de actualización.

# Puerta 1: tooling de calidad

La comprobación reproducible está en `quality/gate1-tooling.sh`. Usa sólo las
dependencias fijadas por los dos `Cargo.lock`, trabaja sin red para resolver
dependencias, limita cada fuzzer a un proceso y separa cobertura, Miri, ASan,
LSan, TSan y cada fuzzer en destinos temporales independientes para impedir mezclas
ABI. Una instantánea de rutas, modos, tamaños y SHA-256 antes y después hace
fallar la puerta si cualquier herramienta modifica el checkout.

El Nightly autorizado queda fijado por el commit de `rustc`
`0ed41eb4142dda2df61eb1145a312c1a9d62eb56` (Nightly 2026-09-04). El script se
detiene si el alias local `nightly` deriva a otro compilador. La actualización
de esa referencia exige una revisión explícita de Miri, sanitizadores y de las
advertencias de incompatibilidad futura. Miri conserva su aislamiento y explora
varias semillas sobre pruebas existentes del núcleo matemático. La puerta
impide que la cobertura demostrada retroceda: líneas 74 %, funciones 70 % y
regiones 75 %. Los umbrales pueden elevarse mediante
`QUALITY_MIN_LINE_COVERAGE`, `QUALITY_MIN_FUNCTION_COVERAGE` y
`QUALITY_MIN_REGION_COVERAGE`, pero nunca reducirse en una entrega. Puerta 2,
descrita más abajo, fija el siguiente nivel certificado sin reducir estos
contratos ni excluir módulos difíciles.

ASan y LSan son controles separados. Integrar la detección de fugas dentro de
ASan hace que toda la suite dependa de que el host permita `ptrace`. La puerta
prueba LSan de manera independiente: si el monitor del host lo bloquea, lo
declara explícitamente y ASan continúa sin fingir cobertura de fugas. Una
máquina de certificación compatible debe ejecutar con `QUALITY_REQUIRE_LSAN=1`;
en ese modo, no disponer de LSan bloquea la puerta. Una fuga real o cualquier
otro fallo de LSan siempre hace fallar la ejecución.

Ejecución acotada por defecto:

```text
quality/gate1-tooling.sh
```

Para aumentar la campaña sin cambiar el procedimiento:

```text
QUALITY_FUZZ_RUNS=1000000 quality/gate1-tooling.sh
```

# Puerta 2: verificación crítica de producción

`quality/gate2-verification.sh` es una puerta de certificación acumulativa. No
sustituye Puerta 0 ni Puerta 1: ejecuta ambas y después añade controles que no
pueden omitirse en P2. La campaña mínima de fuzzing queda fijada en 100.000
ejecuciones por objetivo; una variable de entorno sólo puede elevarla, nunca
reducirla.

P2 exige cobertura global mínima de 82 % de líneas, 75 % de funciones y 80 %
de regiones. Además impone mínimos de líneas sobre autoridades críticas para
impedir que el promedio global oculte una zona de sombra: `engine/runtime.rs`
80 %, `engine/transition.rs` 75 %, `engine/support.rs` 80 %,
`engine/analysis.rs` 85 %, `isolated_execution.rs` 85 % y `digest.rs` 95 %.
La cobertura se obtiene de un único `cargo llvm-cov --workspace --all-targets`
y se valida directamente desde el JSON emitido por LLVM.

La puerta repite Clippy con `-D warnings`, exige que LSan pueda ejecutarse y
termine limpio (un host que lo bloquee no puede certificar P2), ejecuta Miri con
procedencia estricta, alineación simbólica y múltiples semillas sobre
`low_rank_math`, `linalg`, `trust_region` y `transport`, y ejecuta
ThreadSanitizer sobre **todos los objetivos de prueba** del workspace, no
mediante un filtro de nombres. Toda la compilación se dirige a `/tmp`, y una
instantánea SHA-256 completa del checkout antes y después hace fallar P2 ante
cualquier mutación de fuentes o artefactos.

Ejecución de certificación:

```text
quality/gate2-verification.sh
```

Para aumentar, pero nunca rebajar, la campaña de fuzzing heredada por P1:

```text
QUALITY_FUZZ_RUNS=1000000 quality/gate2-verification.sh
```

### Resultados medidos de Puerta 2

La ejecución canónica de `quality/gate2-verification.sh` sobre el commit de certificación arrojó los siguientes resultados medidos con `cargo llvm-cov --workspace --all-targets --json`:

| Módulo Crítico / Superficie | Líneas Reales | Piso Exigido | Funciones | Regiones | Estado Auditoría |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `src/engine/runtime.rs` | **80.18%** | $\ge 80.00\%$ | 77.12% | 81.49% | **SUPERADO** |
| `src/engine/transition.rs` | **75.58%** | $\ge 75.00\%$ | 66.67% | 76.27% | **SUPERADO** |
| `src/engine/support.rs` | **83.77%** | $\ge 80.00\%$ | 81.40% | 85.71% | **SUPERADO** |
| `src/engine/analysis.rs` | **85.30%** | $\ge 85.00\%$ | 79.69% | 86.37% | **SUPERADO** |
| `src/isolated_execution.rs` | **87.33%** | $\ge 85.00\%$ | 82.86% | 89.25% | **SUPERADO** |
| `src/digest.rs` | **98.97%** | $\ge 95.00\%$ | 98.18% | 98.50% | **SUPERADO** |
| **Total Global Workspace** | **86.91%** | $\ge 82.00\%$ | **80.14%** | **87.57%** | **SUPERADO** |

- **Miri**: pasó sin errores sobre `low_rank_math`, `linalg`, `trust_region` y `transport` bajo `-Zmiri-strict-provenance`, `-Zmiri-symbolic-alignment-check` y semillas `0..8`.
- **ThreadSanitizer**: pasó sobre todos los targets; la librería ejecutó 401 tests y también se ejecutaron los binarios e integraciones, incluido `tests/brain.rs`.
- **Fuzzing**: 100.000 ejecuciones en `multi-case-solver` y `persisted-inputs` sin fallo del target en esa campaña acotada.


# Puerta 3: aseguramiento de concurrencia y fallos

`quality/gate3-assurance.sh` inicia la capa P3 de aseguramiento y es estrictamente acumulativa: P3 sólo puede pasar después de P2. Añade model checking determinista de decisiones usadas por producción, pruebas adversariales de concurrencia y recuperación, y un recibo SHA-256 fuera del checkout.

El primer modelo explora las intercalaciones de dos escritores sobre `CanonicalEngineHead`. Con `engine_authority.lock`, toda ejecución terminal forma una única cadena de revisiones; al retirar deliberadamente el lock, el mismo modelo debe encontrar un schedule de *lost update*, demostrando que el lock es una condición de autoridad necesaria. El segundo modelo enumera `live={absent,prior,new,foreign}` por `archive={absent,present}` y los puntos de crash desde `IntentRecorded` hasta `CommitSealed`, exigiendo rollback, restauración, replay explícito o rechazo fail-closed. `ReceiptSealed` usa la ruta de receipt autenticado.

La puerta también fija pruebas que no pueden desaparecer sin romper P3: writers concurrentes, reemplazo de inode, publicación/movimiento atómico, doble avance canónico y recovery real del corpus. El recibo registra HEAD, snapshot del checkout, toolchains, digests de Gate2/Gate3, métricas P2 y la lista de pruebas P3. `QUALITY_RECEIPT_PATH` puede elegir un destino externo; se rechaza escribirlo dentro del checkout.

Esta evidencia es model checking **acotado** de la máquina de estados y sus decisiones de producción. No es una prueba universal del kernel, filesystem o hardware. `loom` no se incorpora mientras no exista una dependencia fijada y disponible offline.

Ejecución:

```text
quality/gate3-assurance.sh
```

## Límite de la evidencia

Las campañas acotadas prueban ausencia de fallos únicamente sobre las entradas
ejecutadas. No demuestran ausencia universal de defectos. La puerta de entrega
debe conservar por separado la evidencia exacta de versión, configuración y
duración de cada campaña.

### Tipos de cobertura

**100 % de líneas**: Se alcanza ejecutando cada línea alcanzable. No demuestra
corrección; una prueba puede ejecutar una línea sin comprobar su resultado.

**100 % de ramas/condiciones**: Más fuerte; exige cubrir cada decisión
verdadera/falsa y combinaciones relevantes.

**100 % de requisitos y contratos críticos**: Obligatorio para TIDE‑X. Cada
invariante, rechazo fail‑closed, transición, recuperación y autoridad debe
tener prueba positiva y adversarial.

**Ausencia universal de fallos**: No se obtiene con un porcentaje. Para partes
acotadas se necesitan pruebas formales o model checking; para el resto, fuzzing
continuo, pruebas diferenciales, metamórficas, sanitizadores y evaluación
independiente.

### Estrategia de evolución

La puerta actual es sólida como infraestructura; la evidencia de TIDE‑X todavía
debe crecer:

1. Mantener los mínimos P2 y elevar por etapas el núcleo alcanzable hacia
   90 → 95 → 100 sin excluir módulos para mejorar el promedio.
2. Medir también ramas/condiciones cuando la instrumentación estable lo permita;
   P2 ya exige funciones y regiones además de líneas.
3. Mantener bajo Miri `low_rank_math`, `linalg`, `trust_region` y `transport`, y
   ampliar a otros módulos puros únicamente cuando sus fronteras sean compatibles
   con el intérprete.
4. Añadir model checking de concurrencia y fallos (loom o equivalente) para
   CAS, cabezas canónicas, publicación atómica y recuperación.
5. Añadir verificación formal a kernels e invariantes críticos donde sea
   viable.
6. Mantener fuzzing continuo por tiempo, no creer que un número finito
   "termina" el espacio.
7. Actualizar RustSec en una tarea con red, firmar la fecha y digest de la
   snapshot y después ejecutar la puerta reproducible sin red.
8. Guardar un recibo de calidad con commit, toolchains, configuración,
   duración, cobertura y digests de resultados.

El umbral no se sube con pruebas vacías ni excluyendo código difícil. El mapa
exacto de líneas y ramas sin cubrir se convierte en trabajo verificable.
