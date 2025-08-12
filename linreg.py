import numpy as np
from sklearn.model_selection import train_test_split  # type: ignore


class GDRegressor:
    def __init__(self, alpha=0.001, n_iter=100, progress=True):  # progressbar
        self.alpha = alpha
        self.n_iter = n_iter
        self.progress = progress
        self.coef_ = []
        self.intercept_ = 0
        self.loss_history = []
        self.theta_history = []
        self.theta = 0

    def fit(self, X_train, y_train):
        """
        Подбирает коэффициенты линейной модели, ориентируясь на отклонение
        предсказаний от реальных данных по MSE с помощью градиентного спуска.

        y - реальное заначение, x - признак, h - предсказание
        сюда функцию потерь (потом на графике)
        """
        y_train = np.array(y_train)
        X_train = np.array(X_train)
        self.theta = np.zeros(X_train.shape[1] + 1)
        X_b = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        n = len(y_train)
        y_train = y_train.values if hasattr(y_train, "values") else y_train
        y_train = y_train.flatten()
        print(self.n_iter)

        for _ in range(self.n_iter):
            y_hat = X_b @ self.theta
            mse = y_hat - y_train
            j_mse_i = (1 / n) * (X_b.T @ mse)
            self.loss_history.append(np.mean(mse**2))
            self.theta = self.theta - self.alpha * j_mse_i
            self.theta_history.append(self.theta.copy())
        min_loss = np.argmin(self.loss_history)
        self.intercept_ = self.theta_history[min_loss][0]
        self.coef_ = self.theta_history[min_loss][1:]

    def predict(self, X_test):
        """
        Считает предсказание, подставив в линейную модель найденные коэффициенты.
        """
        #X_test = np.array(X_test)
        #return self.intercept_ + self.coef_ * X_test
        X_test = np.array(X_test).reshape(-1, 1)
        return X_test.dot(self.coef_.reshape(-1, 1)) + self.intercept_


def z_scaler(feature):
    """
    Проводит стандартизацию для X.
    Возвращает обновленную матрицу X.
    """
    mean = feature.mean()
    std = feature.std()
    return (feature - mean) / std


def min_max(feature):
    """
    Проводит минимаксную нормализацию для X.
    Возвращает обновленную матрицу X.
    """
    return (feature - feature.min()) / (feature.max() - feature.min())


def rmse(y, y_hat):
    """
    Считает и возвращает корень из среднеквадратичной ошибки.
    """
    y = np.array(y)
    y_hat = np.array(y_hat)
    mse = np.mean((y_hat - y) ** 2)
    return np.sqrt(mse)


def r_squared(y, y_hat):
    """
    Считает и возвращает коэффициент детерминации.
    """
    y = np.array(y)
    y_hat = np.array(y_hat)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot


def find_optimal_params(X, y):
    """
    Подбирает лучшие гиперпараметры для поданных модели данных.
    Нужно разбить данные на тренировочную и тестовую выборки,
    обучать модель с разными гиперпараметрами на тренировочной выборке, оценивать на тестовой.
    Вернуть нужно лучшие гиперпараметры: max_iter, alpha.
    Лучшие - это те, при которых на тестовой выборке R^2 ≥ 0.49, RMSE ≤ 6.45.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=18)

    best_alpha = None
    best_max_iter = None
    best_r2 = -np.inf
    best_rmse = np.inf

    alphas = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 1]
    max_iters = [2000, 5000, 8000, 10000, 11000, 11500, 12000]

    for alpha in alphas:
        for max_iter in max_iters:
            if max_iter is None or not isinstance(max_iter, int):
                print(f"Invalid max_iter: {max_iter}, skipping")
                continue
            try:
                model = GDRegressor(alpha=alpha, n_iter=max_iter)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                r2 = r_squared(y_test, y_pred)
                current_rmse = rmse(y_test, y_pred)

                if r2 > best_r2 and current_rmse < best_rmse:
                    best_r2 = r2
                    best_rmse = current_rmse
                    best_alpha = alpha
                    best_max_iter = max_iter

            except Exception as e:
                print(f"Error with alpha={alpha}, max_iter={max_iter}: {str(e)}")
                continue
    print(f"best iter: {best_max_iter}, {best_alpha}, {best_r2}, {best_rmse}")
    return best_max_iter, best_alpha
