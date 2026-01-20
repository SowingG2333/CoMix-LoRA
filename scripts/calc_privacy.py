import numpy as np

def calculate_epsilon(sigma, delta=1e-5):
    """
    根据高斯机制的标准界限计算 Epsilon。
    公式: sigma = sqrt(2 * ln(1.25/delta)) / epsilon
    => epsilon = sqrt(2 * ln(1.25/delta)) / sigma
    """
    if sigma <= 0:
        return float('inf')
    
    # 分子项 (constant factor based on delta)
    factor = np.sqrt(2 * np.log(1.25 / delta))
    return factor / sigma

def main():
    # 也就是你代码 overall_verify.py 中的配置
    NOISE_LEVELS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
    DELTA = 1e-5  # 这是一个常用的严谨值

    print(f"假设 Delta = {DELTA} (标准设定)")
    print("-" * 60)
    print(f"{'Sigma (Noise)':<15} | {'Epsilon (Privacy Budget)':<25} | {'Privacy Level'}")
    print("-" * 60)

    for sigma in NOISE_LEVELS:
        epsilon = calculate_epsilon(sigma, DELTA)
        
        # 简单的文字描述
        if epsilon == float('inf'):
            level = "No Privacy"
        elif epsilon > 10:
            level = "Weak"
        elif epsilon > 2:
            level = "Moderate"
        elif epsilon > 0.5:
            level = "Strong"
        else:
            level = "Very Strong (Model may break)"
            
        eps_str = f"{epsilon:.2f}" if epsilon != float('inf') else "∞"
        print(f"{sigma:<15.1f} | {eps_str:<25} | {level}")

    print("-" * 60)
    print(f"公式系数: sqrt(2 * ln(1.25 / {DELTA})) ≈ {np.sqrt(2 * np.log(1.25 / DELTA)):.4f}")
    print(f"速算公式: Epsilon ≈ 4.83 / Sigma")

if __name__ == "__main__":
    main()
