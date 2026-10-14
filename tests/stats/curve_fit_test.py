import numpy as np
import pytest
from numpy.polynomial import Polynomial
from scipy import stats

from lithos.stats.curve_fit import fit as stats_fit
from lithos.stats.curve_fit import fit_linear_regression, fit_polynomial, fit_sine


def _linear_data(n=30, seed=7):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, n)
    y = 2.0 * x + 1.0 + rng.normal(0, 0.5, n)
    return x, y


def _quadratic_data(n=50, seed=3):
    rng = np.random.default_rng(seed)
    x = np.linspace(-5, 5, n)
    y = 0.5 * x**2 - 2.0 * x + 3.0 + rng.normal(0, 0.5, n)
    return x, y


class TestFitValidation:
    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            fit_linear_regression(x=np.arange(5.0), y=np.arange(4.0))

    def test_linear_requires_more_than_two_points(self):
        with pytest.raises(ValueError, match="more points"):
            fit_linear_regression(x=np.array([0.0, 1.0]), y=np.array([1.0, 2.0]))

    def test_polynomial_requires_more_than_degree_plus_one(self):
        with pytest.raises(ValueError, match="more points"):
            fit_polynomial(x=np.arange(4.0), y=np.arange(4.0), degree=3)

    def test_sine_requires_more_than_four_points(self):
        with pytest.raises(ValueError, match="more points"):
            fit_sine(x=np.arange(4.0), y=np.arange(4.0))

    def test_sine_constant_x_raises(self):
        with pytest.raises(ValueError, match="identical"):
            fit_sine(x=np.full(10, 2.0), y=np.arange(10.0))


class TestLinearFit:
    def test_returns_polynomial_with_interval(self):
        x, y = _linear_data()
        poly, fit_y, fit_x, ci = fit_linear_regression(x=x, y=y)
        assert isinstance(poly, Polynomial)
        np.testing.assert_allclose(fit_x, np.sort(x))
        assert fit_y.shape == fit_x.shape
        assert ci.shape == fit_x.shape
        assert np.all(ci > 0)

    def test_recovers_slope_and_intercept(self):
        x, y = _linear_data(n=200)
        poly, _, _, _ = fit_linear_regression(x=x, y=y)
        slope = (poly(x.max()) - poly(x.min())) / (x.max() - x.min())
        assert slope == pytest.approx(2.0, abs=0.1)
        assert poly(0.0) == pytest.approx(1.0, abs=0.5)

    def test_no_ci_func_returns_none(self):
        x, y = _linear_data()
        _, _, _, ci = fit_linear_regression(x=x, y=y, ci_func=None)
        assert ci is None


class TestConstantX:
    @pytest.mark.parametrize("degree", [1, 2, 3])
    def test_collapses_to_mean_with_mean_ci(self, degree):
        x = np.full(6, 3.0)
        y = np.array([1.0, 2.0, 4.0, 5.0, 8.0, 10.0])
        if degree == 1:
            _, fit_y, _, ci = fit_linear_regression(x=x, y=y)
        else:
            _, fit_y, _, ci = fit_polynomial(x=x, y=y, degree=degree)
        # slope 0 with the mean as intercept
        np.testing.assert_allclose(fit_y, y.mean())
        # the mean's CI: t * sd(ddof=1) / sqrt(n), dof = n - 1
        expected = stats.t.ppf(0.975, 5) * np.std(y, ddof=1) / np.sqrt(6)
        np.testing.assert_allclose(ci, expected)

    def test_prediction_interval_adds_scatter(self):
        x = np.full(6, 3.0)
        y = np.array([1.0, 2.0, 4.0, 5.0, 8.0, 10.0])
        ci = fit_linear_regression(x=x, y=y, ci_func="pi")[3]
        expected = stats.t.ppf(0.975, 5) * np.std(y, ddof=1) * np.sqrt(1 + 1 / 6)
        np.testing.assert_allclose(ci, expected)

    def test_bootstrap_constant_x(self):
        rng = np.random.default_rng(5)
        x = np.full(20, 2.0)
        y = rng.normal(3.0, 1.0, 20)
        ci = fit_linear_regression(x=x, y=y, ci_func="bootstrap_ci")[3]
        assert ci.shape == (2, 20)
        assert np.all(np.isfinite(ci))
        assert np.all(ci >= 0)


class TestPolynomialFit:
    def test_recovers_quadratic(self):
        x, y = _quadratic_data()
        poly, fit_y, fit_x, ci = fit_polynomial(x=x, y=y, degree=2)
        assert fit_y.shape == (50,)
        residuals = y - poly(x)
        assert np.sqrt(np.mean(residuals**2)) < 1.0
        assert ci.shape == (50,)
        assert np.all(ci > 0)

    def test_ci_uses_polynomial_dof(self):
        x, y = _quadratic_data()
        poly, _, fit_x, ci = fit_polynomial(x=x, y=y, degree=2)
        residuals = y - poly(x)
        n = x.size
        t = stats.t.ppf(0.975, n - 3)  # dof = n - (degree + 1)
        s_err = np.sqrt(np.sum(residuals**2) / (n - 3))
        sxx = np.sum((x - x.mean()) ** 2)
        expected = t * s_err * np.sqrt(1 / n + (fit_x - x.mean()) ** 2 / sxx)
        np.testing.assert_allclose(ci, expected)

    def test_dispatcher_polynomial_key(self):
        x, y = _quadratic_data()
        output = stats_fit("polynomial", x=x, y=y, degree=2)
        assert len(output) == 4
        assert output[3] is not None  # the FIT_DICT key typo is fixed


class TestSineFit:
    def test_fit_with_interval_is_finite(self):
        rng = np.random.default_rng(11)
        x = np.linspace(0, 4 * np.pi, 100)
        y = 2.0 * np.sin(x + 0.5) + 3.0 + rng.normal(0, 0.2, 100)
        popt, fit_y, fit_x, ci = fit_sine(x=x, y=y)
        assert popt.shape == (4,)
        assert popt[0] == pytest.approx(2.0, abs=0.3)
        assert fit_y.shape == (100,)
        assert ci.shape == (100,)
        assert np.all(np.isfinite(ci))

    def test_bootstrap_raises_for_sine(self):
        x = np.linspace(0, 4 * np.pi, 100)
        y = np.sin(x)
        with pytest.raises(ValueError, match="bootstrap_ci"):
            fit_sine(x=x, y=y, ci_func="bootstrap_ci")


class TestBootstrap:
    def test_linear_bootstrap_shape(self):
        x, y = _linear_data(20)
        ci = fit_linear_regression(x=x, y=y, ci_func="bootstrap_ci")[3]
        assert ci.shape == (2, 20)
        assert np.all(ci >= 0)

    def test_polynomial_bootstrap(self):
        x, y = _quadratic_data(30)
        ci = fit_polynomial(x=x, y=y, degree=2, ci_func="bootstrap_ci")[3]
        assert ci.shape == (2, 30)
        assert np.all(np.isfinite(ci))
