import unittest
import numpy as np
import ocr_dense_index as index
import ocr_textbook_corpus as corpus


class OCRIndexTests(unittest.TestCase):
    def test_old_vectors_remain_bit_identical(self):
        original = np.eye(2,384,dtype=np.float32)
        added = np.eye(1,384,k=2,dtype=np.float32)
        result = index.append_vectors(original,added,2)
        self.assertEqual(result[:2].tobytes(),original.tobytes())
        np.testing.assert_array_equal(result[2:],added)

    def test_nonfinite_or_unnormalized_vectors_fail(self):
        for bad in (np.zeros((1,384)),np.full((1,384),np.nan),np.ones((1,12))):
            with self.assertRaises(ValueError): index.append_vectors(np.eye(1,384),bad,1)

    def test_sparse_ocr_page_is_preserved_without_fabricated_passage(self):
        self.assertEqual(corpus.chunks('chemical structure diagram'),[])
        text = ' '.join(str(i) for i in range(151))
        chunks = corpus.chunks(text)
        self.assertEqual([v[0] for v in chunks],[0,110])
        self.assertEqual(len(chunks[0][1].split()),140)
        self.assertEqual(chunks[1][1].split()[0],'110')


if __name__ == '__main__': unittest.main()
