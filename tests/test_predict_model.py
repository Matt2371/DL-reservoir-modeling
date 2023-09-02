#### TEST PREDICT_MODEL.PY ####
import unittest
from src.models.predict_model import *

class test_predict_model(unittest.TestCase):
    """Test predict model functions"""
    def test_flatten_rm_pad(self):
        """Test flattening and remove pad"""
        input_y = torch.ones((2, 5, 1)) # (batches, timesteps, 1)
        input_y[-1, 2:, :] = -1 # Add some pads
        input_y_hat = 5 * torch.ones((2, 5, 1))

        expected_y = torch.ones(7)
        expected_y_hat = 5 * torch.ones(7)

        # Get outputs
        output_y_hat, output_y = flatten_rm_pad(y_hat=input_y_hat, y=input_y, pad_value=-1)

        np.testing.assert_almost_equal(expected_y, output_y)
        np.testing.assert_almost_equal(expected_y_hat, output_y_hat)
