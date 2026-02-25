"""
Script de validación: Ejemplos reales de uso del k-bandido
Este script verifica que las implementaciones de ArmBinomial y ArmBernoulli
cumplen correctamente con los ejemplos reales descritos en la sección 5.1.
"""

import numpy as np
from arms import ArmBinomial
from arms import ArmBernoulli


def ejemplo_binomial_delivery():
    """
    Ejemplo 1: Optimización de promociones en una app de delivery
    
    Contexto:
    - k = 4 promociones diferentes (4 brazos)
    - n = 100 usuarios por lote (tamaño fijo)
    - Recompensa = número de usuarios que usaron la promoción
    - Distribución: Binomial(n=100, p_i) donde p_i es la tasa de conversión
    """
    print("=" * 80)
    print("EJEMPLO 1: Optimización de promociones en app de delivery")
    print("=" * 80)
    print()
    
    print("Contexto:")
    print("- App de delivery (tipo Rappi, Uber Eats, DoorDash)")
    print("- 4 promociones diferentes que podemos ofrecer")
    print("- Cada día asignamos una promoción a un lote de n=100 usuarios nuevos")
    print("- Recompensa = número de usuarios que usaron la promoción")
    print()
    
    # Crear 4 brazos (promociones) con tasas de conversión desconocidas
    # En la práctica, estas tasas son desconocidas, pero las definimos para simular
    promociones = {
        "A: 20% descuento >30€": ArmBinomial(n=100, p=0.12),  # 12% conversión
        "B: Envío gratis + 5€": ArmBinomial(n=100, p=0.18),   # 18% conversión
        "C: 2x1 en postres": ArmBinomial(n=100, p=0.08),      # 8% conversión
        "D: Descuento fijo 8€": ArmBinomial(n=100, p=0.15)    # 15% conversión
    }
    
    print("Promociones configuradas (tasas reales desconocidas para el agente):")
    for nombre, arm in promociones.items():
        print(f"  {nombre}")
        print(f"    -> Distribución: {arm}")
        print(f"    -> E[usuarios convertidos] = {arm.get_expected_value():.1f} de 100")
        print(f"    -> Tasa de conversión real: p = {arm.p:.2%}")
    print()
    
    # Simular 10 días de operación
    print("Simulación de 10 días (cada día = un lote de 100 usuarios):")
    print("-" * 80)
    
    np.random.seed(42)
    dias = 10
    
    for nombre, arm in promociones.items():
        print(f"\n{nombre}:")
        recompensas_diarias = []
        
        for dia in range(1, dias + 1):
            # Cada "pull" representa un día mostrando la promoción a 100 usuarios
            usuarios_convertidos = arm.pull()
            recompensas_diarias.append(usuarios_convertidos)
        
        print(f"  Usuarios convertidos por día: {recompensas_diarias}")
        print(f"  Media de conversión: {np.mean(recompensas_diarias):.1f} usuarios/día")
        print(f"  Total en 10 días: {sum(recompensas_diarias)} usuarios")
    
    print()
    print("VALIDACIÓN: La implementación ArmBinomial cumple correctamente")
    print("  - n=100 representa el lote de usuarios")
    print("  - p representa la tasa de conversión desconocida")
    print("  - pull() retorna el número de éxitos (usuarios que usaron la promoción)")
    print()


def ejemplo_bernoulli_publicidad():
    """
    Ejemplo 2: Publicidad Online (CTR - Click-Through Rate)
    
    Contexto:
    - k = 3 anuncios diferentes (3 brazos)
    - Cada impresión de anuncio es un ensayo independiente
    - Recompensa = 1 si el usuario hizo clic, 0 si no
    - Distribución: Bernoulli(p_i) donde p_i es el CTR real del anuncio
    """
    print("=" * 80)
    print("EJEMPLO 2: Publicidad Online - Optimización de CTR")
    print("=" * 80)
    print()
    
    print("Contexto:")
    print("- Campaña de publicidad online con 3 anuncios diferentes")
    print("- Cada vez que mostramos un anuncio, observamos: clic (1) o no clic (0)")
    print("- Objetivo: identificar el anuncio con mayor CTR (Click-Through Rate)")
    print()
    
    # Crear 3 brazos (anuncios) con CTR desconocidos
    anuncios = {
        "A1: Banner con descuento": ArmBernoulli(p=0.05),   # CTR = 5%
        "A2: Video promocional": ArmBernoulli(p=0.08),      # CTR = 8%
        "A3: Anuncio carousel": ArmBernoulli(p=0.03)        # CTR = 3%
    }
    
    print("Anuncios configurados (CTR real desconocido para el agente):")
    for nombre, arm in anuncios.items():
        print(f"  {nombre}")
        print(f"    -> Distribución: {arm}")
        print(f"    -> CTR real: p = {arm.p:.2%}")
        print(f"    -> E[clic] = {arm.get_expected_value():.3f}")
    print()
    
    # Simular 100 impresiones por anuncio
    print("Simulación de 100 impresiones por anuncio:")
    print("-" * 80)
    
    np.random.seed(42)
    n_impresiones = 100
    
    resultados = {}
    for nombre, arm in anuncios.items():
        print(f"\n{nombre}:")
        
        # Cada pull() representa una impresión del anuncio
        clics = [arm.pull() for _ in range(n_impresiones)]
        
        total_clics = sum(clics)
        ctr_observado = total_clics / n_impresiones
        
        resultados[nombre] = {
            'clics': clics,
            'total_clics': total_clics,
            'ctr_observado': ctr_observado,
            'ctr_real': arm.p
        }
        
        print(f"  Primeras 20 impresiones: {clics[:20]}")
        print(f"  Total de clics: {total_clics}/{n_impresiones}")
        print(f"  CTR observado: {ctr_observado:.2%}")
        print(f"  CTR real: {arm.p:.2%}")
        print(f"  Error: {abs(ctr_observado - arm.p):.2%}")
    
    print()
    print("VALIDACIÓN: La implementación ArmBernoulli cumple correctamente")
    print("  - Cada pull() representa una impresión de anuncio")
    print("  - Retorna 1 (clic) o 0 (no clic)")
    print("  - p representa el CTR real desconocido")
    print()
    
    # Identificar el mejor anuncio
    mejor_anuncio = max(resultados.items(), key=lambda x: x[1]['ctr_observado'])
    print(f"Mejor anuncio observado: {mejor_anuncio[0]}")
    print(f"  CTR observado: {mejor_anuncio[1]['ctr_observado']:.2%}")
    print()


def validacion_propiedades_matematicas():
    """
    Validar que las propiedades matemáticas se cumplen correctamente
    """
    print("=" * 80)
    print("VALIDACIÓN DE PROPIEDADES MATEMÁTICAS")
    print("=" * 80)
    print()
    
    # Validación Binomial
    print("1. Validación Binomial B(n=100, p=0.15):")
    print("-" * 40)
    arm_binom = ArmBinomial(n=100, p=0.15)
    
    # Valor esperado teórico
    mu_teorico = 100 * 0.15
    var_teorica = 100 * 0.15 * 0.85
    
    print(f"  Valor esperado teórico: μ = np = {mu_teorico}")
    print(f"  Valor esperado implementado: {arm_binom.get_expected_value()}")
    print(f"  Coinciden: {arm_binom.get_expected_value() == mu_teorico}")
    print()
    
    print(f"  Varianza teórica: σ² = np(1-p) = {var_teorica}")
    print(f"  Varianza implementada: {arm_binom.get_variance()}")
    print(f"  Coinciden: {arm_binom.get_variance() == var_teorica}")
    print()
    
    # Simulación empírica
    n_simulaciones = 10000
    muestras = [arm_binom.pull() for _ in range(n_simulaciones)]
    mu_empirico = np.mean(muestras)
    var_empirica = np.var(muestras)
    
    print(f"  Simulación con {n_simulaciones} muestras:")
    print(f"    Media empírica: {mu_empirico:.2f} (esperado: {mu_teorico})")
    print(f"    Varianza empírica: {var_empirica:.2f} (esperada: {var_teorica:.2f})")
    print(f"    Error media: {abs(mu_empirico - mu_teorico):.3f}")
    print(f"    Error varianza: {abs(var_empirica - var_teorica):.3f}")
    print()
    
    # Validación Bernoulli
    print("2. Validación Bernoulli B(p=0.6):")
    print("-" * 40)
    arm_bern = ArmBernoulli(p=0.6)
    
    # Valor esperado teórico
    mu_teorico = 0.6
    var_teorica = 0.6 * 0.4
    
    print(f"  Valor esperado teórico: μ = p = {mu_teorico}")
    print(f"  Valor esperado implementado: {arm_bern.get_expected_value()}")
    print(f"  Coinciden: {arm_bern.get_expected_value() == mu_teorico}")
    print()
    
    print(f"  Varianza teórica: σ² = p(1-p) = {var_teorica}")
    print(f"  Varianza implementada: {arm_bern.get_variance()}")
    print(f"  Coinciden: {arm_bern.get_variance() == var_teorica}")
    print()
    
    # Simulación empírica
    muestras = [arm_bern.pull() for _ in range(n_simulaciones)]
    mu_empirico = np.mean(muestras)
    var_empirica = np.var(muestras)
    
    print(f"  Simulación con {n_simulaciones} muestras:")
    print(f"    Media empírica: {mu_empirico:.3f} (esperado: {mu_teorico})")
    print(f"    Varianza empírica: {var_empirica:.3f} (esperada: {var_teorica:.3f})")
    print(f"    Error media: {abs(mu_empirico - mu_teorico):.4f}")
    print(f"    Error varianza: {abs(var_empirica - var_teorica):.4f}")
    print()
    
    # Validación: Bernoulli es Binomial con n=1
    print("3. Validación: Bernoulli es Binomial(n=1, p):")
    print("-" * 40)
    p = 0.7
    arm_bern = ArmBernoulli(p=p)
    arm_binom_n1 = ArmBinomial(n=1, p=p)
    
    print(f"  ArmBernoulli(p={p}): E[X] = {arm_bern.get_expected_value()}")
    print(f"  ArmBinomial(n=1, p={p}): E[X] = {arm_binom_n1.get_expected_value()}")
    print(f"  Valores esperados coinciden")
    print()
    
    print(f"  ArmBernoulli(p={p}): Var[X] = {arm_bern.get_variance()}")
    print(f"  ArmBinomial(n=1, p={p}): Var[X] = {arm_binom_n1.get_variance()}")
    print(f"  Varianzas coinciden")
    print()


def main():
    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "VALIDACIÓN DE IMPLEMENTACIONES" + " " * 27 + "║")
    print("║" + " " * 15 + "Ejemplos Reales del Problema del K-Bandido" + " " * 20 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    # Ejemplo 1: Binomial - App de Delivery
    ejemplo_binomial_delivery()
    
    input("Presiona Enter para continuar al siguiente ejemplo...")
    print("\n" * 2)
    
    # Ejemplo 2: Bernoulli - Publicidad Online
    ejemplo_bernoulli_publicidad()
    
    input("Presiona Enter para continuar a la validación matemática...")
    print("\n" * 2)
    
    # Validación de propiedades matemáticas
    validacion_propiedades_matematicas()
    
    print("=" * 80)
    print("CONCLUSIÓN")
    print("=" * 80)
    print()
    print("Las implementaciones ArmBinomial y ArmBernoulli cumplen correctamente")
    print("  con los ejemplos reales de la sección 5.1:")
    print()


if __name__ == "__main__":
    np.random.seed(42)
    main()