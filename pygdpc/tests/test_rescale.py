import torch
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal
from pygdpc.gdpc import GDPC
from pygdpc.tests.utils import gen_data

DECIMAL = 5


class TestRescale():
    @classmethod
    def setup_class(cls):
        cls.seed = 123

    test_params = [(1, 50, 50, 2, 50), (1, 50, 10, 2, 10), (1, 10, 50, 2, 50),
                   (0, 50, 50, 2, 50), (0, 50, 10, 2, 10), (0, 10, 50, 2, 50)]

    @pytest.mark.parametrize("k, num_series, num_periods, epochs, batch_size", test_params)
    def test_rescale(self, k, num_series, num_periods, epochs, batch_size):
        data = gen_data(seed=self.seed, k=k, num_series=num_series, num_periods=num_periods)
        torch.manual_seed(self.seed)
        gdpc = GDPC(k=k, epochs=epochs, batch_size=batch_size, rescale=False)
        gdpc.fit(data)
        torch.manual_seed(self.seed)
        gdpc_rescaled = GDPC(k=k, epochs=epochs, batch_size=batch_size, rescale=True)
        gdpc_rescaled.fit(data)
        assert_almost_equal(gdpc.mse, gdpc_rescaled.mse, DECIMAL)
        assert_almost_equal(np.mean(gdpc_rescaled.component), 0, DECIMAL)
        assert_almost_equal(np.std(gdpc_rescaled.component), 1, DECIMAL)
        assert_equal(gdpc.conv, gdpc_rescaled.conv)
        assert_equal(gdpc.niter, gdpc_rescaled.niter)

