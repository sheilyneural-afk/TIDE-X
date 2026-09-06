# Puerta 0: aptitud de una entrega

`quality/gate0-release.sh` es la puerta local de entrega. Usa el toolchain
estable fijado por `rust-toolchain.toml`, resuelve los dos grafos con sus
`Cargo.lock` y sin red, y construye en un directorio temporal fuera del
proyecto que elimina al terminar. Comprueba formato, Clippy con advertencias
como errores, todas las pruebas, auditoría de vulnerabilidades y las políticas
de dependencias tanto del núcleo como del arnés de fuzzing.

`.cargo/config.toml` fija `build.target-dir = "/tmp/tidex-cargo-target"`. Sin
un override explícito de Cargo, `cargo check`, `cargo test`, `cargo build` y la
resolución del target de `fuzz/Cargo.toml` quedan fuera del checkout. P0 verifica
la resolución del `target_directory` con `CARGO_TARGET_DIR` ausente. Las propias
puertas usan además destinos temporales aislados bajo `/tmp`. P0-P3 no ejecutan
`cargo fuzz coverage` sobre el árbol fuente.

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
dependencias, ejecuta los dos targets de fuzz de forma secuencial y separa cobertura, Miri, ASan,
LSan, TSan y fuzzing en destinos temporales independientes para impedir mezclas
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
descrita más abajo, fija el siguiente nivel interno de verificación sin reducir estos
contratos ni excluir módulos difíciles.

ASan y LSan son controles separados. Integrar la detección de fugas dentro de
ASan hace que toda la suite dependa de que el host permita `ptrace`. La puerta
prueba LSan de manera independiente: si el monitor del host lo bloquea, lo
declara explícitamente y ASan continúa sin fingir cobertura de fugas. Un
host de verificación que pretenda ejecutar P2 debe usar `QUALITY_REQUIRE_LSAN=1`;
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

`quality/gate2-verification.sh` es una puerta interna de verificación acumulativa. No
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
termine limpio (un host que lo bloquee no puede superar P2), ejecuta Miri con
procedencia estricta, alineación simbólica y múltiples semillas sobre
`low_rank_math`, `linalg`, `trust_region` y `transport`, y ejecuta
ThreadSanitizer sobre **todos los objetivos de prueba** del workspace, no
mediante un filtro de nombres. Toda la compilación se dirige a `/tmp`, y una
instantánea SHA-256 completa del checkout antes y después hace fallar P2 ante
cualquier mutación de fuentes o artefactos.

Ejecución de verificación P2:

```text
quality/gate2-verification.sh
```

Para aumentar, pero nunca rebajar, la campaña de fuzzing heredada por P1:

```text
QUALITY_FUZZ_RUNS=1000000 quality/gate2-verification.sh
```

### Última medición acumulativa de Puerta 2

Puerta 2 quedó congelada inicialmente en `c8bbb6e8694ecf41df7b82c4962ddfedeeed3dda`. La ejecución P3 volvió a ejecutar P2 sobre el candidato que después se congeló exactamente como `43588d43d76269258efd6928b098369030f049cb`. El receipt P3 conserva las métricas de esa repetición:

| Módulo crítico / superficie | Líneas | Piso P2 | Funciones | Regiones | Estado P2 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `src/engine/runtime.rs` | **80.18%** | >= 80.00% | 77.12% | 81.49% | **SUPERADO** |
| `src/engine/transition.rs` | **75.80%** | >= 75.00% | 70.00% | 76.40% | **SUPERADO** |
| `src/engine/support.rs` | **83.77%** | >= 80.00% | 81.40% | 85.71% | **SUPERADO** |
| `src/engine/analysis.rs` | **85.30%** | >= 85.00% | 79.69% | 86.37% | **SUPERADO** |
| `src/isolated_execution.rs` | **87.33%** | >= 85.00% | 82.86% | 89.25% | **SUPERADO** |
| `src/digest.rs` | **98.97%** | >= 95.00% | 98.18% | 98.50% | **SUPERADO** |
| **Global** | **86.97%** | >= 82.00% | **80.24%** | **87.62%** | **SUPERADO** |

En esa misma repetición, la librería ejecutó 404 tests. Miri pasó sobre `low_rank_math`, `linalg`, `trust_region` y `transport`; ThreadSanitizer pasó sobre `--all-targets`; y cada target de fuzz (`multi-case-solver` y `persisted-inputs`) completó 100.000 ejecuciones en la campaña mínima de P2. Estas cifras describen esa ejecución acotada, no una ausencia universal de defectos.

# Puerta 3: aseguramiento de concurrencia y fallos

`quality/gate3-assurance.sh` define la capa P3 de aseguramiento y es estrictamente acumulativa: P3 sólo puede pasar después de P2. Añade model checking determinista de decisiones usadas por producción, pruebas adversariales de concurrencia y recuperación, y un recibo SHA-256 fuera del checkout.

El primer modelo explora las intercalaciones de dos escritores sobre `CanonicalEngineHead`. Con `engine_authority.lock`, toda ejecución terminal forma una única cadena de revisiones; al retirar deliberadamente el lock, el mismo modelo debe encontrar un schedule de *lost update*. Esto demuestra, dentro del modelo acotado explorado, que el lock es una condición necesaria para la propiedad modelada. El segundo modelo enumera `live={absent,prior,new,foreign}` por `archive={absent,present}` y los puntos de crash desde `IntentRecorded` hasta `CommitSealed`, exigiendo rollback, restauración, replay explícito o rechazo fail-closed. `ReceiptSealed` usa la ruta de receipt autenticado.

La puerta también fija pruebas que no pueden desaparecer sin romper P3: writers concurrentes, reemplazo de inode, publicación/movimiento atómico, doble avance canónico y recovery real del corpus. El recibo registra HEAD, snapshot del checkout, toolchains, digests de Gate2/Gate3, métricas P2 y la lista de pruebas P3. `QUALITY_RECEIPT_PATH` puede elegir un destino externo; se rechaza escribirlo dentro del checkout.

Esta evidencia es model checking **acotado** de la máquina de estados y sus decisiones de producción. No es una prueba universal del kernel, filesystem o hardware. `loom` no se incorpora mientras no exista una dependencia fijada y disponible offline.

La ejecución P3 que produjo el receipt se realizó antes de crear el commit final: el receipt conserva `head_commit=c8bbb6e...` y el digest del candidato. Después se congeló exactamente ese snapshot como `43588d43d76269258efd6928b098369030f049cb`; el manifiesto SHA-256 completo y el manifiesto de metadatos del checkout del nuevo commit coincidieron con los registrados por P3. El receipt original no se altera retrospectivamente.

Ejecución:

```text
quality/gate3-assurance.sh
```

## P4: frontera de firma disponible

Como primera pieza de Release Readiness, `quality/sign-release.sh` implementa una frontera OpenPGP fail-closed. Rechaza releases dentro del checkout, symlinks en la ruta de release, manifests/checksums no regulares, rutas no canónicas o escapadas en `SHA256SUMS`, subjects no regulares, checksum mismatch, fingerprint ambiguo/no completo y claves secretas sin capacidad de firma.

La selección de autoridad es explícita mediante `TIDEX_RELEASE_GPG_KEY=<fingerprint de 40 hex>`. Las firmas se publican como `SHA256SUMS.asc` y `release-manifest.json.asc` sólo después de verificarlas contra el mismo fingerprint. Una firma ya existente se acepta únicamente si vuelve a verificar para el subject exacto y la clave autorizada. No existe generación automática ni fallback a otra clave.

La implementación fue probada con una clave secreta efímera creada en un `GNUPGHOME` temporal: rechazo sin clave, firma real, verificación exacta de ambas firmas, replay idempotente y rechazo tras alterar un payload protegido por `SHA256SUMS`. Esa prueba de desarrollo no es una firma de distribución; una release pública requiere una clave autorizada persistente configurada por el propietario del proyecto.

P4 todavía no está superada: faltan el bundle de los ocho binarios, manifiesto/SBOM definitivos, instalación/activación/rollback/desinstalación aislados y la puerta acumulativa `gate4`.

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

**Cobertura de requisitos y contratos críticos**: Es un objetivo de
aseguramiento del proyecto, no una propiedad que un porcentaje de código pueda
demostrar. Los invariantes, rechazos fail‑closed, transiciones, recuperaciones y
autoridades críticas deben ganar evidencia positiva y adversarial explícita.

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
4. Ampliar el model checking ya existente a más escritores, más interleavings y
   más puntos de fallo; incorporar `loom` o equivalente sólo cuando pueda fijarse
   y reproducirse offline.
5. Añadir verificación formal a kernels e invariantes críticos donde sea
   viable.
6. Mantener fuzzing continuo por tiempo, no creer que un número finito
   "termina" el espacio.
7. Actualizar RustSec en una tarea con red, firmar la fecha y digest de la
   snapshot y después ejecutar la puerta reproducible sin red.
8. Mantener el recibo de calidad P3 y, en futuras puertas, vincular cada ejecución
   a commit, toolchains, configuración, duración, cobertura y digests de resultados.

El umbral no se sube con pruebas vacías ni excluyendo código difícil. Los
informes de cobertura permiten convertir líneas, funciones y regiones no
cubiertas en trabajo verificable; P0-P3 no afirman cobertura universal de ramas.
