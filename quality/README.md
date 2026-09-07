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

## Puerta 4: Release Readiness técnica

`quality/gate4-release-readiness.sh` es acumulativa por evidencia sobre P3, no por reejecución ciega. Sólo se ejecuta sobre Git limpio, toma un snapshot de modos/tamaños/SHA-256 del checkout y exige el mismo snapshot al terminar. Antes de construir una release, `quality/verify-p3-reuse.sh` debe demostrar que el receipt P3 canónico sigue ligado a su snapshot histórico y que el conjunto cerrado de 101 inputs P0–P3 conserva exactamente el mismo digest y los mismos toolchains. Si cualquiera cambia, P4 falla con exigencia de rerun de Gate3; no inicia automáticamente otra campaña larga.

La evidencia reutilizable está versionada en `quality/evidence/p3/`: `receipt.json` conserva el receipt P3, `receipt.sha256` su identidad y `reuse-anchor.json` fija `43588d43d76269258efd6928b098369030f049cb`, su parent P2, el manifiesto histórico completo y el digest `33369800e83b4e7261a6eedc0449e650cd7dbec8ab093c46a979185a5162bfdc` de los 101 inputs reutilizables. El verificador reconstruye el manifiesto original desde objetos Git más los cuatro hashes ignorados históricos; no depende de `/tmp`.

Cuando P3 se reutiliza, Gate4 mantiene comprobaciones frescas y baratas: `cargo fmt --check`, metadata offline/locked, `cargo audit --no-fetch` para raíz y fuzz, y `cargo deny` para ambas políticas. No repite tests, 100k fuzz, Miri ni sanitizadores mientras los inputs P0–P3 sigan idénticos.

La producción del artefacto usa `quality/build-release-bundle.sh`. Cada invocación realiza dos builds independientes de los nueve binarios con Cargo offline/locked y falla si cualquier ejecutable difiere byte a byte. Después genera un SBOM SPDX 2.3 normalizado, `release-manifest.json`, `SHA256SUMS` y un `.tar.zst` determinista. El manifest liga commit/tree Git, `TIDEX_SOURCE_TREE_DIGEST`, toolchain, `Cargo.toml`, `Cargo.lock`, hashes de los nueve binarios, SBOM y hashes del builder, signer, verifier y gestor de instalación. Gate4 ejecuta el builder dos veces más y compara el contenedor completo, incluidos modos y SHA-256, para comprobar reproducibilidad a nivel de bundle.

`quality/verify-release.sh` es la autoridad común de estructura. Rechaza payloads extra, symlinks, ficheros escribibles por grupo/otros, binarios no ejecutables, subjects no declarados, hash/tamaño divergente, identidad SBOM incorrecta, miembros de archive inesperados y contenido del archive que no coincida con el directorio verificado. Con `TIDEX_RELEASE_REQUIRE_SIGNATURE=1`, exige firmas OpenPGP válidas para el fingerprint completo suministrado en `TIDEX_RELEASE_GPG_KEY`.

`quality/sign-release.sh` delega toda la estructura al verifier y sólo añade la frontera criptográfica. Exige una clave secreta seleccionada por fingerprint completo; firma `SHA256SUMS`, `release-manifest.json`, el archive y su checksum externo cuando se proporciona el transporte. No genera claves, no elige una por defecto y no reemplaza silenciosamente firmas existentes.

`quality/manage-release-installation.sh` mantiene `installation.json` y `activation.json` como autoridades fuera del checkout. Las releases viven en `releases/<release-id>`; `current` es únicamente un symlink derivado y puede reconstruirse desde `activation.json` sin avanzar su revisión. El gestor exige install-root no escribible por terceros, state-root privado, lock real no-symlink, target compatible, releases verificadas y autoridades no escribibles. `rollback` sólo usa una `previous` autenticada y `uninstall` rechaza la release activa y preserva siempre el state-root externo.

Para la evidencia de rollback, Gate4 no relabela fixtures: construye la release actual y una segunda identidad real desde `HEAD^` en un clon local temporal. Ambas se firman con una clave efímera confinada a `/tmp`, se instalan con verificación de firma obligatoria y se recorre previous -> current -> rollback -> current -> uninstall de previous. También se ejecuta el binario instalado contra el state-root de prueba y se exige el rechazo fail-closed `integrity:active_skill_bank_missing` cuando no existe un banco activo.

Una ejecución técnica exitosa genera un receipt `cerebro.tidex.release_readiness_receipt/v1` fuera del checkout y devuelve `result=technical-passed`. Esto **no equivale a promoción pública**: la identidad OpenPGP usada por la puerta es de prueba y se destruye al terminar. La promoción pública permanece separada y el receipt enumera blockers observados. En el estado actual del repositorio, `Cargo.toml` no declara `license` ni `license-file`, por lo que la puerta no inventa una licencia y registra `product_license_undeclared`; además registra que no se ha realizado una firma de distribución de producción.

Ejecución:

```text
quality/gate4-release-readiness.sh
```

`QUALITY_P4_RECEIPT_PATH` permite seleccionar un receipt externo. Gate4 rechaza cualquier destino dentro del checkout.

## Evidencia experimental V64: ReceiverCompiler sobre Transformers

`quality/experiments/v64_transformer_portability.py` ejecuta un benchmark determinista con dos `torch.nn.TransformerEncoder` de arquitecturas e inicializaciones diferentes. No descarga modelos y no constituye una afirmación de portabilidad LLM-scale. Su propósito es comprobar una propiedad más estrecha: una firma funcional de una skill del donante puede compilarse en coordenadas nativas del receptor sin suministrar a TIDE-X los parámetros del donante ni la solución directa del receptor para la skill held-out.

La evidencia canónica en `quality/evidence/v64/receipt.json` liga la ejecución al commit `e96f3aef94f63014a9ceec54176dc5194eeedb39`. Dos ejecuciones completas sobre ese snapshot produjeron bytes idénticos, SHA-256 `8845c7f80ce470d7d7186c6ed5582c7f9af88e364ef90592d877cbe8df0e94be`. El resultado observado fue `pass=true`, `mean_recovered_gain=0.9999307459756482`, `minimum_recovered_gain=0.9995205672621005`; también pasaron el control de fuga del oracle held-out y la comprobación SHA-256 de que los backbones congelados no cambiaron durante el entrenamiento de las skills.

El benchmark sigue siendo un micro-benchmark: A tiene 5.858 parámetros y B 18.402, y las realizaciones de skill viven en una base receptora de cuatro coordenadas sobre backbones congelados. Por tanto V64 demuestra una compilación cross-architecture en ese régimen, no migración de un fine-tune completo ni portabilidad universal entre LLMs.

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
8. Mantener los receipts P3/P4 y vincular cada ejecución a commit, toolchains,
   configuración, duración, cobertura cuando corresponda y digests de resultados.

El umbral no se sube con pruebas vacías ni excluyendo código difícil. Los
informes de cobertura permiten convertir líneas, funciones y regiones no
cubiertas en trabajo verificable; P0-P4 no afirman cobertura universal de ramas.
