import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from transport import resilient


class InfrastructureRetries(unittest.TestCase):
    def test_completed_bad_answer_is_returned_once(self):
        response={'done':True,'response':'invalid answer'}
        remote=Mock(return_value=response)
        with tempfile.TemporaryDirectory() as folder:
            result=resilient(SimpleNamespace(request=SimpleNamespace(remote=remote)),Path(folder)/'errors.jsonl')('/api/generate',{'model':'unit'})
            self.assertIs(result,response)
            self.assertEqual(remote.call_count,1)

    def test_transport_retry_keeps_exact_payload_and_records_no_prompt(self):
        remote=Mock(side_effect=[ConnectionError('Deadline exceeded'),{'done':True}])
        payload={'model':'unit','prompt':'private prompt','options':{'seed':19}}
        with tempfile.TemporaryDirectory() as folder,patch('transport.time.sleep'):
            journal=Path(folder)/'errors.jsonl'
            resilient(SimpleNamespace(request=SimpleNamespace(remote=remote)),journal)('/api/generate',payload)
            self.assertEqual(remote.call_count,2)
            self.assertEqual(remote.call_args_list[0],remote.call_args_list[1])
            self.assertNotIn('private prompt',journal.read_text())
            self.assertEqual(json.loads(journal.read_text())['attempt'],1)

    def test_model_or_protocol_error_is_not_retried(self):
        remote=Mock(side_effect=ValueError('Missing inference metadata'))
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                resilient(SimpleNamespace(request=SimpleNamespace(remote=remote)),Path(folder)/'errors.jsonl')('/api/generate',{})
            self.assertEqual(remote.call_count,1)


if __name__=='__main__':unittest.main()
