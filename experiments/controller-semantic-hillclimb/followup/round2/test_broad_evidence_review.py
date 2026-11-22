from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import broad_evidence_review_runner as run
from test_active_textbook_review import Retriever
from test_evidence_first_review import EvidenceRemote
from test_strong_late_review import fixture


class ChangedRetriever(Retriever):
    def retrieve(self,row):
        result=super().retrieve(row)
        for item in result:item['text'] += ' NEW BROAD FACT'
        return result


class BroadEvidenceTests(unittest.TestCase):
    def test_only_compatible_prefixes_are_reused_and_all_costs_remain(self):
        rows,trees,*_=fixture();prior=run.prior_runner
        with TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'evidence-first-textbook-root-review-pilot-v3';folder.mkdir()
            with prior.configured_runner(),patch.object(prior.spec.retrieval,'retriever',return_value=Retriever()), \
                    (folder/'train.jsonl').open('a') as bank,(folder/'raw-generations.jsonl').open('a') as raw,(folder/'attempts.jsonl').open('a') as attempts:
                journal=prior.Journal(EvidenceRemote(),raw,attempts);collector=prior.original.Collector(trees,{},bank,journal)
                for row in rows:collector.one(row)
            run.io.write(folder/'complete.json',{'digests':{p.name:run.io.sha(p) for p in folder.iterdir()}})
            args=SimpleNamespace(data=root/'data')
            with patch.object(run.spec,'public_inputs',return_value=(rows,trees)),patch.object(run.spec.retrieval,'retriever',return_value=Retriever()):
                logical,raw,attempts,full=run.reused_inputs(args)
                self.assertEqual((len(logical),len(raw),len(full)),(192,320,64))
            with run.configured_runner(),patch.object(run.spec,'public_inputs',return_value=(rows,trees)), \
                    patch.object(run.spec.retrieval,'retriever',return_value=ChangedRetriever()):
                logical,raw,attempts,full=run.reused_inputs(args)
                self.assertEqual((len(logical),len(raw),len(full)),(0,64,0))
                new=root/'new';new.mkdir();run.io.jsonl(new/'raw',raw);run.io.jsonl(new/'attempts',attempts)
                remote=EvidenceRemote()
                with (new/'raw').open('a') as raw_stream,(new/'attempts').open('a') as attempt_stream,(new/'bank').open('a') as bank:
                    journal=run.Journal(remote,raw_stream,attempt_stream,raw,attempts)
                    collector=run.original.Collector(trees,{},bank,journal)
                    for row in rows:collector.one(row)
                records=run.protocol.strict_records(new/'bank');physical=run.protocol.strict_records(new/'raw')
                _,coverage=run.spec.restore(rows,trees,records,physical,True)
                self.assertEqual(len(remote.calls),256)
                self.assertFalse(any(p['options']['num_predict']==64 for p in remote.calls))
                self.assertEqual(coverage['physical_calls'],320)
                for item in records:
                    if item['kind']=='judge':
                        run.spec.validate_review_charge(item['response'])
                        self.assertEqual(item['response']['physical_api_calls'],3)


if __name__=='__main__':unittest.main()
