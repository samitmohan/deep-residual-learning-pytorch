import unittest
import torch
import models


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def dimension_test(model, n, in_dims: tuple, out_size, batch_size=16):
    def case(residual, option):
        def test(self):
            m = model(n, residual=residual, option=option).to(device)
            test_input = torch.rand(batch_size, in_dims[0], in_dims[1], in_dims[1]).to(device)
            result = m.forward(test_input)
            shape = result.shape

            self.assertEqual(len(shape), 2)
            self.assertEqual(shape[0], batch_size)
            self.assertEqual(shape[1], out_size)

            del m, test_input

        return test

    return type(f'TestDimensions{model.__name__}{n}', (unittest.TestCase,), {
        'test_plain': case(False, None),
        'test_residual_A': case(True, 'A'),
        'test_residual_B': case(True, 'B')
    })


TestDimensionsCifar20 = dimension_test(models.CifarResNet, 20, (3, 32), 10)
TestDimensionsCifar32 = dimension_test(models.CifarResNet, 32, (3, 32), 10)
TestDimensionsCifar44 = dimension_test(models.CifarResNet, 44, (3, 32), 10)
TestDimensionsCifar56 = dimension_test(models.CifarResNet, 56, (3, 32), 10)
TestDimensionsCifar110 = dimension_test(models.CifarResNet, 110, (3, 32), 10)

TestDimensionsImageNet18 = dimension_test(models.ImageNetResNet, 18, (3, 224), 1000)
TestDimensionsImageNet34 = dimension_test(models.ImageNetResNet, 34, (3, 224), 1000)
TestDimensionsImageNet50 = dimension_test(models.ImageNetResNet, 50, (3, 224), 1000)
TestDimensionsImageNet101 = dimension_test(models.ImageNetResNet, 101, (3, 224), 1000)
TestDimensionsImageNet152 = dimension_test(models.ImageNetResNet, 152, (3, 224), 1000)


class TestBlockExpansion(unittest.TestCase):
    def test_basic_block_expansion(self):
        self.assertEqual(models.BasicBlock.expansion, 1)

    def test_bottleneck_block_expansion(self):
        self.assertEqual(models.BottleneckBlock.expansion, 4)


class TestParameterCounts(unittest.TestCase):
    def test_resnet20_params(self):
        """ResNet-20 should have ~0.27M parameters (Table 6)."""
        m = models.CifarResNet(20, residual=True, option='A')
        count = sum(p.numel() for p in m.parameters())
        self.assertGreater(count, 250_000)
        self.assertLess(count, 300_000)


class TestForwardBackward(unittest.TestCase):
    def test_cifar_resnet20_train_step(self):
        """Smoke test: one forward-backward pass completes without error."""
        m = models.CifarResNet(20, residual=True, option='A').to(device)
        x = torch.randn(4, 3, 32, 32, device=device)
        y = torch.randint(0, 10, (4,), device=device)
        out = m(x)
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()
        del m, x, y


if __name__ == '__main__':
    unittest.main(verbosity=2)
