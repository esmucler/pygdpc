import torch
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal
from pygdpc.gdpc import GDPC
from pygdpc.tests.utils import gen_data

DECIMAL = 3


class TestRegression():
    @classmethod
    def setup_class(cls):
        cls.seed = 123

    test_params_results = [(1, 50, 10, 100, 10, 0.536), (0, 50, 10, 100, 10, 0.628)]

    @pytest.mark.parametrize("k, num_series, num_periods, epochs, batch_size, result", test_params_results)
    def test_regression(self, k, num_series, num_periods, epochs, batch_size, result):
        data = gen_data(seed=self.seed, k=k, num_series=num_series, num_periods=num_periods)
        torch.manual_seed(self.seed)
        gdpc = GDPC(k=k, epochs=epochs, batch_size=batch_size, rescale=False)
        gdpc.fit(data)
        assert_almost_equal(gdpc.mse, result, DECIMAL)