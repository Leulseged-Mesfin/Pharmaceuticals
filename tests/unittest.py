import unittest
import pandas as pd
import numpy as np
from your_module import percentage_missing_values, fill_null_values  # Replace `your_module` with the module name.

class TestDataProcessingFunctions(unittest.TestCase):
    def setUp(self):
        # Sample DataFrame for testing
        self.test_df = pd.DataFrame({
            'Category': ['A', 'B', np.nan, 'D'],
            'Value': [10, np.nan, 30, 40],
            'Count': [1, 2, np.nan, 4]
        })

    def test_percentage_missing_values(self):
        # Expected percentage missing values
        expected_output = (
            "The telecom contains Category    25.0\nValue       25.0\nCount       25.0\ndtype: float64% missing values."
        )
        result = percentage_missing_values(self.test_df)
        self.assertEqual(result, expected_output)

    def test_fill_null_values(self):
        # Copy the DataFrame for testing the fill_null_values function
        df_filled = self.test_df.copy()
        
        # Apply the function
        fill_null_values(df_filled)
        
        # Expected results
        expected_category = ['A', 'B', 'B', 'D']  # Forward filled for 'Category'
        expected_value = [10, 0, 30, 40]          # 0 filled for numeric types
        expected_count = [1, 2, 0, 4]            # 0 filled for numeric types
        
        self.assertListEqual(df_filled['Category'].tolist(), expected_category)
        self.assertListEqual(df_filled['Value'].tolist(), expected_value)
        self.assertListEqual(df_filled['Count'].tolist(), expected_count)

if __name__ == '__main__':
    unittest.main()
