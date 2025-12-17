import numpy as np
import matplotlib.pyplot as plt

def regression(X, Y, degree = 1):
    """
    Perform polynomial regression of degree n on the given data points (X, Y).

    Parameters:
    X : array-like, shape (m,)
        The input data points.
    Y : array-like, shape (m,)
        The output data points.
    n : int
        The degree of the polynomial to fit.

    Returns:
    coeffs : ndarray, shape (n+1,)
        The coefficients of the fitted polynomial, ordered from highest degree to lowest.
    """
    # Ensure X and Y are numpy arrays
    X = np.asarray(X)
    Y = np.asarray(Y)

    # Create the Vandermonde matrix
    X = np.vander(X, N=degree + 1, increasing=True)

    XTX = np.dot(X.T, X)
    XTY = np.dot(X.T, Y)
    coeffs = np.dot(np.linalg.inv(XTX), XTY)

    return coeffs


if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    max_degree = 6
    x = [0.5, 1, 1.5, 2, 2.5]
    y = [1.35, 0.9, 1.3, 2, 3.5]
    ceoffs = {}
    for degree in range(1, max_degree + 1):
        ceoffs[degree] = regression(x, y, degree)

    # 补充的绘图代码
    plt.figure(figsize=(10, 6))
    
    # 绘制原始数据点
    plt.scatter(x, y, color='black', s=50, zorder=5, label='Data points')
    
    # 生成平滑的x值用于绘制曲线
    x_plot = np.linspace(min(x) - 0.5, max(x) + 0.5, 200)
    
    # 为不同阶数设置不同的颜色
    colors = plt.cm.viridis(np.linspace(0, 1, max_degree))
    
    # 绘制每个阶数的回归曲线
    for degree in range(1, max_degree + 1):
        # 计算多项式在x_plot上的值
        y_plot = np.polyval(ceoffs[degree][::-1], x_plot)  # 注意系数顺序需要反转
        
        plt.plot(x_plot, y_plot, 
                color=colors[degree-1], 
                linewidth=2, 
                label=f'Degree {degree}')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Polynomial Regression with Different Degrees')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(min(x) - 0.5, max(x) + 0.5)
    plt.ylim(min(y) - 1, max(y) + 1)
    plt.tight_layout()
    plt.show()

