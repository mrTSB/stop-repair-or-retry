import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import science_proposal_headroom as driver


class ProposalHeadroomTests(unittest.TestCase):
    def test_all_options_required_and_unresolved_attempt_blocks_resume(self):
        row={'id':'q','subject':'physics','answer':'','options':['one','two','three','four']}
        with TemporaryDirectory() as tmp:
            args=SimpleNamespace(output=Path(tmp),registration_sha256='r')
            attempt={'id':'q','public_sha256':driver.proposal.canonical_sha(row),'registration_sha256':'r'}
            driver.io.append(args.output/'attempts.jsonl',attempt)
            with self.assertRaisesRegex(ValueError,'Unresolved'):driver.restore(args,[row])
            proposal={'all_options':[{'letter':l,'calibrated_logit':float(i),'token_slots':20} for i,l in enumerate('ABCD')],
                'encoder_calls':4,'encoder_token_slots':80,'proposed_letter':'D'}
            item={'id':'q','proposal':proposal,'registration_sha256':'r'}
            driver.io.jsonl(args.output/'proposals.jsonl',[item])
            self.assertEqual(driver.restore(args,[row],True),{'q':item})
            bad=copy.deepcopy(item);bad['proposal']['all_options'].pop()
            driver.io.jsonl(args.output/'proposals.jsonl',[bad])
            with self.assertRaises(ValueError):driver.restore(args,[row],True)

    def test_calibrated_ties_are_alphabetical_and_invalid_slots_rejected(self):
        row={'id':'q','subject':'physics','answer':'','options':['one','two','three','four']}
        with TemporaryDirectory() as tmp:
            args=SimpleNamespace(output=Path(tmp),registration_sha256='r')
            driver.io.append(args.output/'attempts.jsonl',{'id':'q','public_sha256':driver.proposal.canonical_sha(row),'registration_sha256':'r'})
            item={'id':'q','registration_sha256':'r','proposal':{'all_options':[{'letter':l,'calibrated_logit':0.,'token_slots':20} for l in 'ABCD'],
                'encoder_calls':4,'encoder_token_slots':80,'proposed_letter':'A'}}
            driver.io.jsonl(args.output/'proposals.jsonl',[item]);driver.restore(args,[row],True)
            item['proposal']['proposed_letter']='B';driver.io.jsonl(args.output/'proposals.jsonl',[item])
            with self.assertRaises(ValueError):driver.restore(args,[row],True)
            item['proposal']['proposed_letter']='A';item['proposal']['all_options'][0]['token_slots']=513
            item['proposal']['encoder_token_slots']=573;driver.io.jsonl(args.output/'proposals.jsonl',[item])
            with self.assertRaises(ValueError):driver.restore(args,[row],True)


if __name__=='__main__':unittest.main()
