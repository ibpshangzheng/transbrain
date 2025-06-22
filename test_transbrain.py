import unittest
import pandas as pd
import numpy as np
import transbrain as tb
import os
from pathlib import Path

current_file = Path(__file__).resolve()
base_dir = current_file.parent


class TestTransBrain(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.Transformer = tb.trans.SpeciesTrans('bn')

    def test_fetch_human_bn_atlas(self):
        atlas = tb.atlas.fetch_human_atlas(atlas_type='bn',region_type='all')
        self.assertIn('atlas', atlas)
        self.assertIn('atlas_data', atlas)
        self.assertIn('region_info', atlas)
        self.assertIn('info_table', atlas)
    
    def test_fetch_human_dk_atlas(self):
        atlas = tb.atlas.fetch_human_atlas(atlas_type='dk',region_type='all')
        self.assertIn('atlas', atlas)
        self.assertIn('atlas_data', atlas)
        self.assertIn('region_info', atlas)
        self.assertIn('info_table', atlas)

    def test_fetch_human_aal_atlas(self):
        atlas = tb.atlas.fetch_human_atlas(atlas_type='aal',region_type='all')
        self.assertIn('atlas', atlas)
        self.assertIn('atlas_data', atlas)
        self.assertIn('region_info', atlas)
        self.assertIn('info_table', atlas)

    def test_fetch_mouse_atlas(self):
        atlas = tb.atlas.fetch_mouse_atlas(region_type='all')
        self.assertIn('atlas', atlas)
        self.assertIn('atlas_data', atlas)
        self.assertIn('region_info', atlas)
        self.assertIn('info_table', atlas)


    def test_mouse_to_human(self):
        df = pd.read_csv(base_dir.joinpath('transbrain/exampledata/mouse/mouse_all_example_data.csv'),index_col=0)
        check_data = np.load(base_dir.joinpath('tests/mouse_phenotype_in_human.npy'))

        result = self.Transformer.mouse_to_human(df, region_type='all', normalize=True)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, check_data.shape)
        self.assertFalse(result.isnull().values.any())


    def test_human_to_mouse(self):
        df = pd.read_csv(base_dir.joinpath('transbrain/exampledata/human/bn/human_bn_all_example_data.csv'),index_col=0)
        check_data = np.load(base_dir.joinpath('tests/human_phenotype_in_mouse.npy'))

        result = self.Transformer.human_to_mouse(df, region_type='all', normalize=True)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)
        self.assertEqual(result.shape, check_data.shape)
        self.assertFalse(result.isnull().values.any())

    def test_human_phenotype_extraction(self):
        path = base_dir.joinpath('transbrain/exampledata/human/human_example_phenotype_data.nii.gz')
        check_data = np.load(base_dir.joinpath('tests/human_phenotype_extracted.npy'))
        human_atlas = tb.atlas.fetch_human_atlas(atlas_type='bn',region_type='cortex')
        result = tb.base.get_region_phenotypes(path, atlas_dict = human_atlas, atlas_type='bn', region_type='cortex')
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)
        self.assertEqual(result.shape, check_data.shape)
        self.assertFalse(result.isnull().values.any())

    
    def test_mouse_phenotype_extraction(self):
        path = base_dir.joinpath('transbrain/exampledata/mouse/mouse_example_phenotype_data.nii.gz')
        check_data = np.load(base_dir.joinpath('tests/mouse_phenotype_extracted.npy'))
        mouse_atlas = tb.atlas.fetch_mouse_atlas(region_type='all')
        result = tb.base.get_region_phenotypes(path, atlas_dict = mouse_atlas, atlas_type='mouse', region_type='all')
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)
        self.assertEqual(result.shape, check_data.shape)
        self.assertFalse(result.isnull().values.any())


    def test_load_graph_embeddings(self):
        result = self.Transformer._load_embeddings()
        self.assertIsInstance(result, np.ndarray)
        self.assertGreater(result.size, 0)

if __name__ == '__main__':

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestTransBrain)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\n🎉 TransBrain installed successfully!!!")
    else:
        print("\n❌ Please check errors.")
