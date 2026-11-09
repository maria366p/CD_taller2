# Taller 2: predicción de abandono de clientes para FinAlpes

MINE 4101 Ciencia de Datos Aplicada, Universidad de los Andes.

Integrantes: María Paula Ospina 202123208

## De qué trata

FinAlpes es una fintech colombiana que quiere venderle a los bancos un moelo para prdecir qué clientes se van a ir. El banco usa la predicción para contactar a esos clientes con una oferta de retencion. En este taler se trabaja con una muestra de clientes de bancos europeos, se entrenan y comparan varios modelos, se interpreta el mejor y se calcula cuánto valor le daría a un banco colombiano de 20.000 clientes.

## Qué hay en el repositorio

`data/churn.csv`: el dataset del taller, con 10.045 registros y 11 columnas. Se usa tal cual lo entregó el curso.

`notebooks/Taller2_Churn.ipynb`: el único notebook, con los puntos 1 a 5 del taller. Se corre completo y en orden, de la primera a la última celda, y no necesita ningún otro archivo ni ninguna ejecución previa. Lee el dataset con la ruta `../data/churn.csv`, así que hay que abrirlo desde la carpeta `notebooks`. Ya viene ejecutado, con todas las salidas.

`informe/Presentacion_Taller2.pptx`: la presentación corta del punto 6, con 9 diapositivas. Muestra los hallazgos del modelo y los resultados, y terina con la recomendación finl para el negocio.

`DECLARACION_IA.md`: declaración de uso de IA generativa del punto 7.

`requirements.txt`: las librerías necesarias.

## Cómo está organizado el notebook

Sección 1, entendimiento y preparación de los datos. Estructura, distribuciones, calidad de los datos, tratamiento de duplicados, inconsistencias, valores inválidos, atípicos y nulos, relación de cada variable con el abandono, correlaciones y pruebas estadísticas (t de Welch, U de Mann Whitney y chi cuadrado con V de Cramér). Corresponde al punto 1.

Sección 2, entrenamiento. Codificación one hot, características derivadas, división en entrenamiento, validación y prueba, imputación y estandarización, y ocho modelos con búsqueda de hiperparámetros: regresión logística, árbol de decisión, KNN, Random Forest, Gradient Boosting, votación, stacking y una red neuronal. Corresponde al punto 2.

Sección 3, análisis cuantitativo. Matrices de confusión, exactitud, precisión, recall, F1, AUC y AP en entrenamiento y validación, elección del modelo con la validación, evaluación en prueba, comparación del conjunto original con el ampliado, comparación del mejor modelo básico con el mejor ensamble, intervalos de confianza por bootstrapping, pruebas A B pareadas, prueba t y McNemar. Corresponde al punto 3.

Sección 4, análisis cualitativo. Coeficientes de la regresión logística, curvas ICE, valores SHAP y LIME, con el comportamiento global y tres clientes individuales. Corresponde al punto 4.

Sección 5, generación de valor. Supuestos con sus fuentes, valor esperado con la fórmula de Provost, fórmula simplificada, precisión de equilibrio, elección del umbral con el impacto, punto de equilibrio, recall mínimo para recuperar la inversión en tres meses y sensibilidad al costo de la oferta y a la tasa de éxito. Corresponde al punto 5.

## Cómo está organizada la presentación

Diapositiva 1, portada. Diapositiva 2, el problema y los datos. Diapositiva 3, qué clientes abandonan. Diapositiva 4, los 8 modelos que se validaron, con su F1, precisión, recall y AUC en validación. Diapositiva 5, por qué se eligió el Random Forest, con su resultado en validación y en prueba. Diapositiva 6, por qué predice así, con lo que muestran SHAP, ICE y LIME. Diapositiva 7, cuánto vale, con los supuestos de costo e inversión. Diapositiva 8, cuándo sí se paga, según el costo por contacto y la tasa de éxito. Diapositiva 9, la recomendación final.

Todas las cifras de la presentación salen de las salidas del notebook. Las fuentes de los supuestos de costo son las que están al final de este archivo.

## Cómo correrlo

1. Instalar las librerías con `pip install -r requirements.txt`. Incluye TensorFlow y Keras para la red neuronal.
2. Abrir `notebooks/Taller2_Churn.ipynb` desde la carpeta `notebooks`.
3. Ejecutar todas las celdas en orden. En un equipo de 2 nucleos tarda cerca de 20 minutos por las búsquedas de hiperparámeros y el entrenamiento de las redes. En uno con más núcleos tarda menos.

Las semillas están fijas en 42, así que los resultados se repiten. Con otras versiones de las librerías o con otro procesador, la red neuronal puede dar diferencias pequeñas en los decimales.

## Resultado principal

El Random Forest con las variables originales es el mejor modelo en el F1 de validacion, 0,586, y en prueba tiene un F1 de 0,608 con un intervalo del 95% de 0,569 a 0,645, una precisión de 0,559, un recall de 0,666 y un AUC de 0,855. Mejora al mejor modelo basico, el árbol de decisión, que tiene un F1 de prueba de 0,545, y la diferencia es significativa en laprueba pareada. Las características derivadas no mejoran al Random Forest.

En el valor, la conclusión depende del costo por contacto y de la tasa de éxito. El costo por contacto base es de unos 160.600 pesos, de los cuales 159.100 son la oferta, que es exonerar la cuota de manejo durante 6 meses, y 1.500 son el canal. La inversión base es de unos 82,7 millones de pesos, el costo de un equipo de 3,5 personas durante 3 meses. Con esos valores y una tasa de éxito de 0,3, ningún modelo se paga: con el umbral de 0,7 el árbol de decisión pierde 0,39 millones al mes en prueba. Con un costo por contacto de unos 100.000 pesos o menos sí se paga, en menos de 12 meses con el Random Forest, y con la oferta baja, de unos 55.500 pesos, en 5 meses. Con el costo base se paga si la tasa de éxito es de 0,4 o más, en unos 17 meses. Con la oferta alta no se paga ni con una tasa de 0,5. El costo por contacto es un tope, porque un descuento solo lo paga el banco a quien lo acepta. El costo de la oferta, el canal y la inversión son supuestos por validar con FinAlpes y con el banco antes de tomar decisiones.

## Fuentes del código y de los supuestos

El código del notebook sigue las ayudas y actividades de las Prácticas de Machine Learning del curso, en https://datascience-uniandes.github.io/PracticasMachineLearning/, y los tutoriales de limpieza de datos y análisis exploratorio del curso. El análisis conceptual usa las presentaciones del curso sobre modelos supervisados, interpretabilidad, métricas avanzadas y generación de valor.

Los supuestos de la sección 5 que no da el enunciado se estimaron con datos colombianos, todos en pesos. Las fuentes son estas:

Cuotas de manejo de los bancos, del reporte trimestral de la Superintendencia Financiera, reportado por Noticias RCN el 29 de julio de 2026: https://www.noticiasrcn.com/economia/cuota-de-manejo-bancos-mas-caros-y-gratuitos-1045068. Las tarifas originales están en https://www.superfinanciera.gov.co/61279.

Salario mínimo de 2026 y auxilio de transporte, del decrto transitorio 0159 de 2026 del Ministerio del Trabajo, reportado por La República: https://www.larepublica.co/economia/mintrabajo-publico-decreto-transitorio-que-mantiene-minimo-de-1-75-millones-en-2026-4331878.

Costo para el empleador de un trabajador con salario mínimo, de Valora Analitik: https://www.valoraanalitik.com/cuanto-cuesta-a-una-empresa-un-trabajador-que-gana-el-salario-minimo/ y de Forbes Colombia: https://forbes.co/2026/09/24/economia-y-finanzas/salario-minimo-2026-esto-cuesta-realmente-contratar-a-un-trabajador-en-colombia/.

Salarios de recién grduados en múltiplos del salario mínimo, del Observatorio Laboral para la Educación del Ministerio de Educación, con los graduados de 2021: https://ole.mineducacion.gov.co/1769/articles-426952_recurso_1.pdf.

El equipo y la duración del proyecto son supuestos propios, dado que no se encontró una fuente confiable. Las secciones 1 a 4 no usan datos externos.

## Declaración de IA generativa

Ver `DECLARACION_IA.md`.
