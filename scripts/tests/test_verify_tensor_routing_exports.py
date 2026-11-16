"""Hand-calculated arrays independently exercise the export receipt oracle."""
import unittest
from scripts.verify_tensor_routing_exports import scalar_reference


def state_for(values, head_width):
    return {'layers.array-learn.weight': [[0., 0.] for _ in values],
            'layers.array-learn.bias': values,
            'layers.head.weight': [[1.] + [0.] * (head_width-1), [0.] * (head_width-1) + [1.]],
            'layers.head.bias': [1., -1.]}


class ScalarRoutingReferenceTests(unittest.TestCase):
    def test_split_rejoins_last_first_middle_within_each_row(self):
        result = scalar_reference('split', state_for(list(range(12)), 12), [3., -2.])
        self.assertEqual(result['routing'], {'part0': [0,1,6,7], 'part1': [2,3,8,9], 'part2': [4,5,10,11]})
        self.assertEqual(result['join-first'], [4,5,0,1,10,11,6,7])
        self.assertEqual(result['flat'], [4,5,0,1,2,3,10,11,6,7,8,9])
        self.assertEqual(result['output'], [5., 8.])

    def test_chunk_returns_five_parts_and_selects_final_column(self):
        result = scalar_reference('chunk', state_for(list(range(26)), 2), [0., 0.])
        self.assertEqual(list(result['routing']), ['part0','part1','part2','part3','part4'])
        self.assertEqual(result['routing']['part4'], [12,25])
        self.assertEqual(result['output'], [13.,24.])

    def test_gather_uses_original_winning_positions_and_first_tie(self):
        result = scalar_reference('gather', state_for([5,5,-2,8,9,1,3,3], 4), [0.,0.])
        self.assertEqual(result['positions'], {'out':[5,8,9,3], 'indices':[0,3,0,2]})
        self.assertEqual(result['routing'], [5,8,9,3])
        self.assertEqual(result['output'], [6.,2.])

    def test_einsum_is_row_column_product_and_uses_both_branches(self):
        state = state_for([], 4)
        del state['layers.array-learn.weight']; del state['layers.array-learn.bias']
        for name, values in [('a',[1,2,3,4,5,6]), ('b',[7,8,9,10,11,12])]:
            state[f'layers.{name}-learn.weight'] = [[0.,0.] for _ in values]
            state[f'layers.{name}-learn.bias'] = values
        result = scalar_reference('einsum', state, [0.,0.])
        self.assertEqual(result['routing'], [58,64,139,154])
        self.assertEqual(result['output'], [59.,153.])
        state['layers.a-learn.weight'][0] = [2.,-3.]
        self.assertEqual(scalar_reference('einsum',state,[2.,1.])['routing'], [65,72,139,154])

    def test_refuses_wrong_saved_state_or_nonfinite_features(self):
        with self.assertRaises(ValueError): scalar_reference('split',state_for([1.],12),[0.,0.])
        with self.assertRaises(ValueError): scalar_reference('chunk',state_for(list(range(26)),2),[float('nan'),0.])
        with self.assertRaises(ValueError): scalar_reference('unknown',{},[0.,0.])


if __name__ == '__main__': unittest.main()
