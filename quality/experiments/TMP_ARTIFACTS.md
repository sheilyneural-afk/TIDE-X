# Artefactos experimentales conservados fuera de /tmp

La copia persistente de los artefactos V65–V67, el snapshot anterior y las
exploraciones V69 está en:

`/home/yo/.local/share/tidex/research/preserved-tmp-20260908T174152Z`

Consultar `manifest.json`: los archivos incluidos se cotejan por SHA-256
contra el origen. Los logs son instantáneas y no autoridad científica.
Los originales de los modelos se mantienen para no romper recibos ni scripts.

## Integración con el código existente

No copiar las exploraciones de /tmp como una segunda implementación activa.
La implementación actual es `v69_target_update_free_compilation.py`:

| Exploración archivada | Punto existente que debe reutilizarse |
| --- | --- |
| v69_explore.py / v69_explore_semantic.py | donor_signature, pca_projection, receiver_prompt_features |
| v69_threshold.py / v69_threshold2.py | fit_calibration_basis, calibration_fold, replay_model |
| v69_calibration_supervised_basis.py | fit_calibration_basis y v69_calibration_retry.py |

Esta tabla identifica responsabilidades relacionadas; no afirma equivalencia
numérica entre algoritmos ni valida los resultados de la implementación actual.
Las exploraciones se conservan como antecedentes con resultados favorables y
desfavorables. Sus cálculos margen_base + características @ delta no sustituyen
un forward del checkpoint materializado.

Los scripts usan calibración supervisada aunque no haya un optimizador por
gradiente. Sus umbrales numéricos no son umbrales de confianza metacognitiva.

## Almacenamiento y limpieza

Código y tests en Git; pesos, adaptadores, deltas y recibos fuera del repositorio.
Los árboles de compilación referenciados por .cargo/config.toml y experimentos
no se eliminan mientras haya tests activos. No modificar recibos históricos para
cambiar sus rutas: conservar el origen y registrar el destino de la copia.

Los directorios antiguos de tests retirados de /tmp quedan en
`/home/yo/.local/share/tidex/research/tmp-quarantine-20260908`.
Su journal registra cada rename y permite restaurar la ruta de origen si está
libre. La cuarentena no libera espacio; no equivale a borrado definitivo.
