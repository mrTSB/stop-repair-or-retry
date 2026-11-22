import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import reserve_math_auxiliary as study
import collect_auxiliary_math_pairs as paired
from test_auxiliary_pairs import Remote, read_all


class MathReservationTests(unittest.TestCase):
    def test_public_projection_never_decodes_reference_values(self):
        raw = b'{"id":"public-id","question":"Public question","answer":{"secret":"DO_NOT_DECODE"}}\n'
        original = json.loads
        def guarded(value, *args, **kwargs):
            self.assertNotIn('DO_NOT_DECODE', str(value))
            return original(value, *args, **kwargs)
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'test-questions.jsonl'; path.write_bytes(raw)
            with patch('json.loads', side_effect=guarded):
                self.assertEqual(study.projected(path, {'id','question'}), [{'id':'public-id','question':'Public question'}])

    def test_selection_excludes_ids_and_stems_before_hash_ranking(self):
        rows = [{'id':str(i),'question':f'Unique arithmetic statement number {i} with token{i}', 'answer':''} for i in range(12)]
        index = study.reserve.DuplicateIndex(); index.add(study.reserve.normalized(rows[3]['question']))
        stem = study.hashlib.sha256(study.reserve.previous.stem(rows[2]['question']).encode()).hexdigest()
        with patch.object(study, 'COUNT', 4):
            selected, excluded, unused = study.choose(rows, {'0','1'}, {stem}, index)
        self.assertEqual(len(selected),4)
        self.assertFalse({r['id'] for r in selected}&{'0','1','2','3'})
        self.assertEqual(excluded['historical_reserved_or_near_duplicate'],4)
        self.assertEqual(unused,4)

    def test_all_seven_math_requests_preserve_original_protocol_and_failures(self):
        row = {'id':'math-new','domain':'math','subject':'math','question':'What is 3 plus 5?', 'answer':''}
        with TemporaryDirectory() as temporary:
            root = Path(temporary); remote = Remote(invalid=True)
            with (root/'raw').open('a') as raw, (root/'attempts').open('a') as attempts, (root/'bank').open('a') as bank:
                journal = paired.Journal(remote,raw,attempts)
                collector = paired.Collector({},bank,journal,'registered'); collector.one(row)
                tree = {}
                for path in paired.contract.PATHS:
                    payload = paired.payload(row,path,tree)
                    self.assertEqual(payload['prompt'],paired.old.solver_prompt(row,path,paired.paid_tree(path,tree),{}))
                    self.assertEqual(payload['options']['num_predict'],256 if path else 128)
                    self.assertEqual(payload['options']['temperature'],.65 if path else 0)
                    tree[path] = collector.known[row['id'],path]['response']
                    self.assertFalse(tree[path]['answer'])
            logical,raw,attempts = read_all(root)
            known,coverage = paired.restore([row],logical,raw,attempts,'registered',True)
            self.assertEqual(coverage['physical_calls'],7); self.assertEqual(len(known),7)
            self.assertEqual(len(remote.calls),7)
            logical[1]['response']['input_tokens'] += 1
            with self.assertRaises(ValueError): paired.restore([row],logical,raw,attempts,'registered',True)

    def test_math_collector_rejects_labels_and_nonmath_rows(self):
        row = {'id':'q','domain':'math','subject':'math','question':'Q','answer':'8'}
        with self.assertRaises(ValueError): paired.payload(row,'',{})
        with self.assertRaises(ValueError): paired.payload({**row,'answer':'','domain':'science'},'',{})


if __name__ == '__main__': unittest.main()
