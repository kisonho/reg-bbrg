import unittest


class Test0100(unittest.TestCase):
    def test_compile(self):
        import reg_bbrg

        # model build
        model = reg_bbrg.networks.build(1, 1)

        # compile manager
        manager = reg_bbrg.compile(model, lambda_adv=0.5)
        self.assertIsInstance(manager, reg_bbrg.AdversarialDiffusionManager)

    def test_import(self):
        import reg_bbrg

        bbdm_version = reg_bbrg.VERSION
        self.assertGreaterEqual(bbdm_version, "0.1a")
        pass
