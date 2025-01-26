import unittest
import torch
import os
import tempfile
from omegaconf import DictConfig
import hydra

from src.models.yolos.yolos import Yolos, PostProcess


class TestYolosModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up any state for the tests"""

        # Hydrate config from a file (use your config path here)
        # Here I assume that the `main` function from your code is used to load the config via Hydra.
        @hydra.main(config_path="C:\\Users\\pietr\\PycharmProjects\\idus\\config", config_name="config",
                    version_base="1.3")
        def get_config(cfg: DictConfig):
            cls.cfg = cfg
            # Initialize model and post-processing module
            cls.model = Yolos(cls.cfg)
            cls.post_process = PostProcess()

        # Call the Hydra config loading function
        get_config()

    def test_input_channels(self):
        """Test that the model's input channels match the config"""
        input_channels = self.model.backbone.patch_embed.proj.weight.shape[1]
        self.assertEqual(input_channels, self.cfg.inChans,
                         f"Expected input channels {self.cfg.inChans}, but got {input_channels}")

    def test_forward_pass(self):
        """Test a forward pass of the model"""
        dummy_input = torch.randn(self.cfg.batchSize, self.cfg.inChans,
                                  self.cfg.targetHeight, self.cfg.targetWidth)
        outputs = self.model(dummy_input)

        self.assertIsInstance(outputs, dict)
        self.assertIn('class', outputs)
        self.assertIn('bbox', outputs)

        # Check dimensions for class
        self.assertEqual(outputs['class'].shape[0], self.cfg.batchSize)
        self.assertEqual(outputs['class'].shape[1], 100)
        self.assertEqual(outputs['class'].shape[2], self.cfg.numClass + 1)

        # Check dimensions for bbox
        self.assertEqual(outputs['bbox'].shape[0], self.cfg.batchSize)
        self.assertEqual(outputs['bbox'].shape[1], 100)
        self.assertEqual(outputs['bbox'].shape[2], 4)

    def test_post_process(self):
        """Test the post-processing step"""
        dummy_input = torch.randn(self.cfg.batchSize, self.cfg.inChans,
                                  self.cfg.targetHeight, self.cfg.targetWidth)
        outputs = self.model(dummy_input)

        target_sizes = torch.tensor([[self.cfg.targetHeight, self.cfg.targetWidth]] * self.cfg.batchSize)
        results = self.post_process(outputs, target_sizes)

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), self.cfg.batchSize)

        for i, result in enumerate(results):
            self.assertIn('scores', result)
            self.assertIn('labels', result)
            self.assertIn('boxes', result)
            self.assertEqual(result['boxes'].shape[-1], 4)

    def test_save_weights(self):
        """Test saving model weights to a temporary directory"""

        # Create a temporary directory for saving the model weights
        with tempfile.TemporaryDirectory() as temp_dir:
            weight_path = os.path.join(temp_dir, "test_model.pth")

            # Save model weights
            torch.save(self.model.state_dict(), weight_path)

            # Check if the file exists
            self.assertTrue(os.path.exists(weight_path), "Model weights were not saved correctly")
            print(f"Model weights saved to: {weight_path}")


if __name__ == '__main__':
    unittest.main()
