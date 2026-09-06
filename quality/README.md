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
regiones 75 %. El siguiente objetivo explícito es superar primero el 80 % y
después elevar por etapas el núcleo alcanzable hacia el 100 %, añadiendo
pruebas reales y nunca excluyendo módulos. Los umbrales pueden elevarse mediante
`QUALITY_MIN_LINE_COVERAGE`, `QUALITY_MIN_FUNCTION_COVERAGE` y
`QUALITY_MIN_REGION_COVERAGE`, pero nunca reducirse en una entrega.

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

1. Elevar cobertura real del 74,45 % por etapas: 80 → 90 → 95 → 100 en el
   núcleo alcanzable.
2. Medir también ramas y funciones, no sólo líneas.
3. Ampliar Miri desde `low_rank_math` a todos los módulos puros y seguros para
   Miri; excluir sólo fronteras de SO justificadas.
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
